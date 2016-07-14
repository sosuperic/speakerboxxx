# Extract input linguistic features and output acoustic features from CMU Arctic data

WAVS_PATH = 'data/cmu_slt_arctic/wav/'
PHONE_LABELS_PATH = 'data/cmu_us_slt_arctic/lab/'
TRANSCRIPTS_PATH = 'data/cmuarctic.data.txt'

import os
import pprint
import nltk
import string
import pdb

class FeatureExtractor:
    def __init__(self):
        self.cmudict = nltk.corpus.cmudict.dict()

        # See the notes in _align_phonemes_and_words for the reasoning for the following
        self.cmudict['and'] = [['AE', 'N', 'D']]
        self.cmudict['whiz-zip-bang'] = [['W', 'IH', 'Z', 'Z', 'IH', 'P', 'B', 'AE', 'NG']]
        self.cmudict['lop-ear'] = [['L', 'AA', 'P', 'IH', 'R']]
        self.cmudict['16'] = [['S', 'IH', 'K', 'S', 'T', 'IY', 'N']]
        self.cmudict['1908'] = [['TH', 'N', 'AY', 'N', 'T', 'IY', 'N', 'OW', '3.22450', 'EY', 'T']]
        self.cmudict["mcfee's"] = [['M', 'AH', 'K', 'F', 'IY', 'Z']]
        self.cmudict['life-giving'] = [['L', 'AY', 'F', 'G', 'IH', 'V', 'IH', 'NG']]
        self.cmudict['life-conserving'] = [['L', 'AY', 'F', 'K', 'AH', 'N', 'S', 'ER', 'V',  'IH', 'NG']]
        self.cmudict['orange-green'] = [['AO', 'R', 'AH', 'N', 'JH', 'G', 'R', 'IY', 'N']]
        self.cmudict['gold-green'] = [['G', 'OW', 'L', 'D', 'G', 'R', 'IY', 'N']]

    ################################################################################################
    # Pre analysis
    ################################################################################################

    def check_phone_set(self):
        """
        Iterate through phonetic alignments to get the full set of phonemes

        Notes on results
        ----------------
        # Extra:
        # AX 
        # PAU - pause
        # SSIL? https://groups.google.com/forum/#!msg/comp.speech/GRMTOAF4AYo/k4OwKUmdcJ4J


        # This one has AX and PAU: http://festvox.org/bsv/c4711.html
        # AX -> AH? ; The: DH AX (in .lab) vs. DH AH (CMU pronunciation dictionary)
        # How to do case folding

        # http://www.festvox.org/docs/manual-1.4.3/festival_12.html
        """
        phones = set()
        for fn in os.listdir(PHONE_LABELS_PATH):
            if fn.endswith('lab') and fn.startswith('arctic'):
                lines = tuple(open(os.path.join(PHONE_LABELS_PATH, fn), 'rb'))
                for line in lines:
                    if not line.startswith('#'):
                        phones.add(line.split(' ')[2].strip('\n').upper())

        phones = sorted(list(phones))
        pprint.pprint(phones)
        print len(phones)

    ################################################################################################
    # Aligning phonemes to words in order to get word-level and syllable-level contexts for a given phoneme
    ################################################################################################

    def _remove_lexical_stress(self, phone):
        """
        AO1 -> AO
        """
        return ''.join([i for i in phone if not i.isdigit()])

    def _get_phones_from_transcript(self, transcript):
        """
        Return phonetic translation of each word using cmudict in transcript

        Parameters
        ----------
        transcript: 'author of the danger trail, philip steels, etc.'

        Returns
        -------
        word_phones: [(author, [AO, TH, ER]), (of, [AH, V]), ...]

        Notes
        -----
        Some words have multiple translations
        """
        word_phones = []
        for word in transcript.split(' '):
            word = word.strip(string.punctuation)
            if word == '':
                continue
            try:
                phones = self.cmudict[word][0]
                phones = [self._remove_lexical_stress(p) for p in phones]
            except KeyError as e:                       # Not every word will have a phoneme lookup
                phones = []
            word_phones.append((word, phones))
        return word_phones

    def _extract_phone_times(self, recording):
        """
        Return phones and the times for a given recording, from lab/ folder

        Parameters
        ----------
        Recording: str; 'arctic_a0001'

        Returns
        -------
        phone_times: [(AO, 1.16), (TH, 1.35), ...]
        """
        phone_align = tuple(open(os.path.join(PHONE_LABELS_PATH, recording + '.lab'), 'rb'))
        phone_times = []
        for line in phone_align:
            if line.startswith('#'):
                continue
            time = line.split(' ')[0]
            phone = line.split(' ')[2].strip('\n').upper()
            phone = self._remove_lexical_stress(phone)
            if phone == 'SSIL' or phone == 'PAU':       # These are non-standard, not in the CMU Dict phone set
                continue
            if phone == 'AX':                           # This is also not in CMU Dict. Appears to correspond to AH
                phone = 'AH'
            phone_times.append((phone, time))
        return phone_times

    def _align_phonemes_and_words(self, phone_times, word_phones):
        """
        Align phones to words for a given recording 

        Parameters
        ----------
        phone_times: [(AO, 1.16), (TH, 1.35), ...]
            These come from the CMUSphinx labelings in lab/
        word_phones: [(author, [AO, TH, ER]), (of, [AH, V]), ...]
            These are the phonetic translations for the words in a recording

        Returns
        -------
        word_phone_times: [(author, AO, 1.16), (author, TH, 1.35), (author, ER, 1.55), (of, AH, 1.89), ...]

        Methodology
        -----------
        Iterate over the phoneme-time tuples in phone_times. For each phoneme, try to match to the current 
        phoneme in the current word in word_phones. If it's match, add the (word, phoneme, time) tuple and 
        update the current phoneme and current word indices. *If it's a mismatch, keep adding phonemes with
        the current word until the phoneme matches the first phoneme of the next word*. 

        Notes
        -----
        Major problem with current methodology (major as in it's a pretty serious theoretical flaw, but it
            *only* fails for 7 of the 1000+ cases):

        If the first phoneme of the next word is ALSO a mismatch, then this breaks down.
        This can happen when the phoneme is mis-matched, or there simply isn't a phoneme translation of the next word.

        3 out of 7 of the failure cases have to do with text-normalization for hyphens
        2 out of 7 of the failure cases have to do with text-normalization for numbers

        (Hacky) solution: In __init__, manually encoded the phonetic translations of the words that were causing failure

        Failure cases:
        - arctic_a0299: ('tomfoolery', []), ('and', [u'AH', u'N', u'D']),
        - whiz-zip-bang,  lop-ear
        - arctic_a0563: [('mrs', [u'M', u'IH', u'S', u'IH', u'Z']), ("mcfee's", []),
        - arctic_a0438: at sea, monday, march 16, 1908.
        - arctic_a0439: at sea, wednesday, march 18, 1908.
        - b0421: life-giving life-conserving
        - b0425: there were orange-green, gold-green, and a copper-green.

        """
        word_phone_times = []

        wp_w_idx, wp_p_idx = 0, 0                                 # index for iterating over word_phones
        i = 0
        while i < len(phone_times):
            p, t = phone_times[i]
            w = word_phones[wp_w_idx][0]
            phones = word_phones[wp_w_idx][1]

            if len(phones) == 0 or p != phones[wp_p_idx]:   # No word-phone translation (e.g. rifle-shot, []), or mismatch
                if wp_w_idx == len(word_phones) - 1:        # At the last word, append the rest of the phonemes with current word
                    for p,t in phone_times[i:]:
                        word_phone_times.append((w, p, t))
                    return word_phone_times

                else:                                       # Add all phonemes with current word until next match
                    word_phone_times.append((w, p, t))      # Append current phoneme to with current word

                    # TODO: handle last phoneme, though this is most likely subsumed by the above if clause
                    i += 1                                  # Move to next phoneme
                    p, t = phone_times[i]                   

                    wp_w_idx += 1                           # Update pointers so we can get first phoneme of next word
                    wp_p_idx = 0
                    first_phoneme_next_word = word_phones[wp_w_idx][1][wp_p_idx]
                    # TODO: problem: when first phoneme of next word is a mismatch as well...
                    while p != first_phoneme_next_word:         # Add phoneme with current word
                        word_phone_times.append((w, p, t))      # w is still current word
                        i += 1                                  # Move on to next phoneme
                        p, t = phone_times[i]
            else:                                           # Add the phoneme and update the pointers for word_phones
                word_phone_times.append((w, p, t))
                if wp_p_idx == len(phones) - 1:
                    wp_w_idx += 1
                    wp_p_idx = 0
                else:
                    wp_p_idx += 1
                i += 1

        return word_phone_times



    def align_phonemes_and_words(self):
        """
        Align phones to words for each recording

        Returns
        -------
        word_phone_times: [
            ('arctic_a0001',
             [(author, AO, 1.16), (author, TH, 1.35), (author, ER, 1.55), (of, AH, 1.89), ...]
            ),
            ('arctic_a0002',
                ...
            )
        """
        recording_word_phone_times = []

        lines = tuple(open(TRANSCRIPTS_PATH, 'r'))
        for l in lines:
            # Get the phones and times for this recording
            recording = l.split('"')[0][2:].strip()
            # recording = 'arctic_a0299'  
            phone_times = self._extract_phone_times(recording)

            # Get the word and cmudict translated phones for this recording
            transcript = l.split('"')[1].strip('\n').lower()
            # transcript = "i tell you i am disgusted with this adventure tomfoolery and rot."
            word_phones = self._get_phones_from_transcript(transcript)

            # pdb.set_trace()
            word_phone_times = self._align_phonemes_and_words(phone_times, word_phones)
            print word_phone_times
            print ''
            assert(len(word_phone_times) == len(phone_times))
            recording_word_phone_times.append((recording, word_phone_times))

        return recording_word_phone_times


    ################################################################################################
    # Context building 
    ################################################################################################

    def group_phonemes_by_syllable(self, phonemes):
        """
        Given set of phonemes
        """    
        pass

    ################################################################################################
    # Feature building
    ################################################################################################

if __name__ == '__main__':
    fe = FeatureExtractor()
    # fe.check_phone_set()
    fe.align_phonemes_and_words()
