# Extract input linguistic features and output acoustic features from CMU Arctic data

WAVS_PATH = 'data/cmu_us_slt_arctic/wav/'
PHONE_LABELS_PATH = 'data/cmu_us_slt_arctic/lab/'
TRANSCRIPTS_PATH = 'data/cmuarctic.data.txt'

import os
import pprint
import nltk
import string
import pdb
from collections import OrderedDict

import pyworld as pw
import numpy as np
from scipy.io.wavfile import read

class FeatureExtractor:
    def __init__(self):
        self.cmudict = nltk.corpus.cmudict.dict()
        self.tagdict = nltk.data.load('help/tagsets/upenn_tagset.pickle')
        self.univeral_tagset = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ',
            'DET', 'NUM', 'PRT', 'X', '.']
        self.universal_tagset_to_idx = {tag: i for i, tag in enumerate(self.univeral_tagset)}
        self.phoneset = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
            'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S',
            'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
        self.phoneset_to_idx = {p: i for i, p in enumerate(self.phoneset)}

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
        self.cmudict['springy'] = [['S', 'P', 'R', 'IH', 'NG', 'IY']]

        # Used for quick lookup when constructing positional & count word-in-utterance features
        self.rec_to_utt = {}

        # Used for features
        self.word_to_syl = {}

        self._create_word_phone_contexts()
        self._add_syllable_context()
        self._add_pos_to_context()

        self._extract_phoneme_durations()
        self._create_linguistic_features()
        # self._extract_audio_features()

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
            time = float(line.split(' ')[0])
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

    def _create_word_phone_contexts(self):
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
        rec_word_phone_times = []

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
            self.rec_to_utt[recording] = transcript

            # pdb.set_trace()
            word_phone_times = self._align_phonemes_and_words(phone_times, word_phones)
            # print word_phone_times
            # print ''
            assert(len(word_phone_times) == len(phone_times))
            rec_word_phone_times.append((recording, word_phone_times))

        self.rec_phone_contexts = rec_word_phone_times


    ################################################################################################
    # Context building 
    ################################################################################################

    # These are directly copied and pasted from w0rdplay/syllabizer
    def is_vowel(self, phoneme):
        return not set('AEIOU').isdisjoint(set(phoneme))

    def _count_number_of_syllables_in_phonemes(self, phonemes):
        return len([p for p in phonemes if self.is_vowel(p)])

    def _phoneme_syllabize(self, phonemes):
        """
        Main helper function for phoneme_syllabize
        """
        def next_phoneme_is_vowel(phonemes, cur_i):
            next_i = cur_i + 1
            if next_i >= len(phonemes):
                return False
            else:
                return self.is_vowel(phonemes[next_i])

        if self._count_number_of_syllables_in_phonemes(phonemes) == 1:
            return [phonemes]

        syllables = []
        cur_syllable = []
        prev_phoneme_is_vowel = False
        i = 0
        while i < len(phonemes):
            p = phonemes[i]
            if self.is_vowel(p):
                if prev_phoneme_is_vowel:
                    syllables.append(cur_syllable)
                    cur_syllable = [p]
                else:
                    cur_syllable.append(p)
                prev_phoneme_is_vowel = True
            else:
                if next_phoneme_is_vowel(phonemes, i):
                    syllables.append(cur_syllable)
                    cur_syllable = [p]
                    prev_phoneme_is_vowel = False
                elif prev_phoneme_is_vowel:
                    cur_syllable.append(p)
                    prev_phoneme_is_vowel = False
                else:
                    syllables.append(cur_syllable)
                    cur_syllable = [p]
                    while (i + 1 < len(phonemes)) and (not self.is_vowel(p)):   # Append until next vowel
                        i += 1
                        p = phonemes[i]
                        cur_syllable.append(p)

                    if sum([1 for p in cur_syllable if self.is_vowel(p)]) == 0: # Didn't hit any more vowels
                        syllables[-1] += cur_syllable                           # Test case: apologized
                        cur_syllable = []                                       # Clear so it gets filtered out

                    prev_phoneme_is_vowel = True
            i += 1
        syllables.append(cur_syllable)

        # Filter out initial empty syllable, e.g. hello
        syllables = [syl for syl in syllables if len(syl) > 0]

        return syllables

    def _add_syllable_context(self):
        """
        Add syllable context to existing rec - word, phone, time contexts

        Sets
        ----
        ('charged', 'CH', '2.32450'), -> ('charged', ['CH', 'AA', 'R', 'JH', 'D'], 'CH', '2.32450')
        """ 
        for i in range(len(self.rec_phone_contexts)):
            rec, word_phone_times = self.rec_phone_contexts[i]
            word_syllable_phone_times = []

            # word_phones['author'] = [AO, TH, ER]
            word_phones = OrderedDict()                       # Maintain order of insertion, i.e. times
            for word, phone, time in word_phone_times:
                if word in word_phones:
                    word_phones[word].append(phone)
                else:
                    word_phones[word] = [phone]

            # [ [[AO], [TH, ER]], ... ]
            word_syllables = []                             # [[]]
            for word, phones in word_phones.items():        # Split phonemes for that word into syllables
                syl = self._phoneme_syllabize(phones)
                word_syllables.append(syl)
                self.word_to_syl[word] = syl

            w_idx = 0                                       # Index of current word in word_syllables
            syl_idx = 0                                     # Index of current syllable in current word
            p_idx = 0                                       # Index of current phoneme in current syllable
            # pdb.set_trace()
            for word, phone, time in word_phone_times:      # Glue together words, syllables, and phonemes
                # print word_syllables[w_idx]
                cur_syl = word_syllables[w_idx][syl_idx]
                word_syllable_phone_times.append( (word, cur_syl, phone, time) )
                # print word, cur_syl, phone, time

                at_end_of_word = syl_idx + 1 == len(word_syllables[w_idx])
                at_end_of_syllable = p_idx + 1 == len(cur_syl)
                if at_end_of_syllable and at_end_of_word:
                    w_idx += 1
                    syl_idx = 0
                    p_idx = 0
                elif at_end_of_syllable:
                    syl_idx += 1
                    p_idx = 0
                else:
                    p_idx += 1

            self.rec_phone_contexts[i] = (rec, word_syllable_phone_times)

    ################################################################################################
    # POS tagging
    ################################################################################################
    
    # These are copied and pasted directly from w0rdplay/pos_tagger.py
    def tokenize(self, text):
        return nltk.word_tokenize(text)

    def pos_tag(self, tokenized):
        return nltk.pos_tag(tokenized)

    def pos_tag_simplified(self, tokenized):
        tagged = self.pos_tag(tokenized)
        simplified = [(word, nltk.map_tag('en-ptb', 'universal', tag)) for word, tag in tagged]
        return simplified

    def _add_pos_to_context(self):
        """
        Sets
        ----
        ('charged', ['CH', 'AA', 'R', 'JH', 'D'], 'CH', '2.32450') -> ('VBD', charged', ['CH', 'AA', 'R', 'JH', 'D'], 'CH', '2.32450')
        """
        tagged = {}
        for i in range(len(self.rec_phone_contexts)):
            print 'pos', i
            pos_plus_contexts = []
            rec, contexts = self.rec_phone_contexts[i]
            for word, syl, phone, time in contexts:
                if word in tagged:
                    tag = tagged[word]
                else:
                    tag = self.pos_tag_simplified(self.tokenize(word))[0][1]   # list of (word, tags)
                pos_plus_contexts.append( [tag, word, syl, phone, time] )
            self.rec_phone_contexts[i] = [rec, pos_plus_contexts]

            # if i == 100:
            #     break

    ################################################################################################
    # Feature building - phoneme encoding, POS encoding, counts, etc.
    ################################################################################################

    def _create_linguistic_features(self):
        # TODO: include neighbors or not?
        """
        Create feature vectors for all utterances. Base unit is at phoneme level.

        Features
        --------
        Categorical:
        - [x] phoneme identity
        - [x] POS identity
        - [x] vowel phoneme in current syllable

        Position
        --------
        - [x] position of current phoneme in syllable
        - [x] position of current syllable in word
        - [x] position of current word in utterance

        Counts
        ------
        - [x] number of phonemes in syllable
        - [x] number of syllables in word
        - [x] number of words in utterance

        Other
        -----
        - [x] Duration of current utterance?
        - For acoustic only I believe:
            - Unidirectional: coarse-coded position of the current frame in the current phoneme,
            - MSFT: position of the current frame of the current phone
        """

        # pprint.pprint(self.rec_phone_contexts)
        # pprint.pprint(self.rec_phone_durations)
        for i in range(len(self.rec_phone_contexts)):
            rec, contexts = self.rec_phone_contexts[i]
            w_in_utt_idx = 0
            syl_in_w_idx = 0
            p_in_syl_idx = 0
            for j in range(len(contexts)):
                tag, word, syl, phone, time = contexts[j]

                # Categorical
                phone_encoded = np.zeros(len(self.phoneset))
                phone_encoded[self.phoneset_to_idx[phone]] = 1
                
                tag_encoded = np.zeros(len(self.univeral_tagset))
                tag_encoded[self.universal_tagset_to_idx[tag]] = 1
                
                print tag, word, syl, phone
                syl_vowel = [p for p in syl if self.is_vowel(p)][0]
                syl_vowel_encoded = np.zeros(len(self.phoneset))
                syl_vowel_encoded[self.phoneset_to_idx[syl_vowel]] = 1

                # Position
                pos_p_in_syl = np.array([p_in_syl_idx])
                pos_syl_in_word = np.array([syl_in_w_idx])

                utt_words = [w.strip(string.punctuation) for w in self.rec_to_utt[rec].split(' ')]
                utt_words = [w for w in utt_words if len(w) > 0]
                pos_w_in_utt = np.array([w_in_utt_idx])

                # Counts
                num_p_in_syl = np.array([len(syl)])
                num_syl_in_word = np.array([len(self.word_to_syl[word])])   # TODO: What if words have different pronunciations? Can that happen?
                num_words_in_utt = np.array([len(utt_words)])

                # Other
                utt_dur = np.array([sum([d for p, d in self.rec_phone_durations[i][1]])])

                # Concatenate
                features = np.hstack([
                    phone_encoded,
                    tag_encoded,
                    syl_vowel_encoded,
                    pos_p_in_syl,
                    pos_syl_in_word,
                    pos_w_in_utt,
                    num_p_in_syl,
                    num_syl_in_word,
                    num_words_in_utt,
                    utt_dur
                    ])

                # print w_in_utt_idx, syl_in_w_idx,  p_in_syl_idx
                # print utt_dur
                # print features

                # Similar logic as in _add_syllable_context()
                at_end_of_word = syl_in_w_idx + 1 == len(self.word_to_syl[word])
                at_end_of_syl = p_in_syl_idx + 1 == len(syl)
                if at_end_of_word:
                    w_in_utt_idx += 1
                    syl_in_w_idx = 0
                    p_in_syl_idx = 0
                elif at_end_of_syl:
                    syl_in_w_idx += 1
                    p_in_syl_idx = 0
                else:
                    p_in_syl_idx += 1


    ################################################################################################
    # 
    # ACOUSTIC FEATURES
    # 
    ################################################################################################
    def _wavread(self, filename):
        SHORT_MAX = 32767   # What is this?

        fs, x = read(filename)
        x = x.astype(np.float) / SHORT_MAX
        return x, fs

    def _extract_audio_features(self):
        """
        Extract target features every 5ms for acoustic model
        """
        pyDioOpt = pw.pyDioOption(
            f0_floor=50, 
            f0_ceil=600, 
            channels_in_octave=2,
            frame_period=5, 
            speed=1)
        eps = 1e-10

        features = []
        for fn in os.listdir(WAVS_PATH):
            if fn.endswith('wav'):
                rec = fn.replace('.wav', '')
                print rec
                fp = os.path.join(WAVS_PATH, fn)
                x, fs = self._wavread(fp)
                f0, sp, ap, pyDioOpt = pw.wav2world(x, fs, pyDioOpt=pyDioOpt)    # use default options

                # num_frames = math.ceil(num_samples / sample_rate * s_to_ms / frame_period_window)
                #            = math.ceil(x.shape[0] / fs * 1000 / 5)
                # shapes: (num_frames, ), (num_frames, 513), (num_frames, 513)
                f0 = np.log(f0 + eps)
                sp = np.log(sp + eps)
                ap = np.log(ap + eps)

                frames = []
                for time in range(f0.shape[0]):
                    voiced = True
                    if abs(f0[time] - np.log(eps)) < 1e-4:                       # unvoiced in this frame
                        voiced = False
                    frames.append([voiced, f0[time], sp[time, :], ap[time, :]])

                print frames[0]
                features.append([rec, frames])

    def _extract_phoneme_durations(self):
        """
        Extract durations of each phoneme for duration model

        Notes
        -----
        We ignored the PAU (pause) phonemes earlier, but we need it to get the length of the last
        actual phoneme. Doing it here instead of keeping the PAU's and hassling with it when
        extracting contexts to keep it clean.
        """
        # print self.rec_phone_contexts[0]
        # pdb.set_trace()
        self.rec_phone_durations = []
        for rec, contexts in self.rec_phone_contexts:
            durations = []
            for i in range(len(contexts) - 1):                  # - 1 because last duration will be calculated specially
                tag, word, syl, phone, time = contexts[i]
                duration = contexts[i+1][4] - time
                durations.append((phone, duration))

            # Last actual phone: we look up
            last_phone = contexts[-1][3]
            last_phone_start_time = contexts[-1][4]
            phone_align = tuple(open(os.path.join(PHONE_LABELS_PATH, rec + '.lab'), 'rb'))
            phone_align = phone_align[::-1]                     # Reverse order - some have multiple PAU's
            for i, line in enumerate(phone_align):
                prev_phone = phone_align[i+1].split(' ')[2].strip('\n').upper()
                prev_phone = self._remove_lexical_stress((prev_phone))
                prev_phone = 'AH' if prev_phone == 'AX' else prev_phone
                if prev_phone == last_phone:
                    durations.append((last_phone, float(line.split(' ')[0]) - last_phone_start_time))
                    break
            self.rec_phone_durations.append((rec, durations))

    def align_input_and_duration_features(self):
        """
        Features for duration model (one time-step):
        x = linguistic feature around one phoneme
        y = float value for duration
        """
        pass

    def align_input_and_acoustic_features(self):
        """
        Features for acoustic model (one time-step):
        x = linguistic feature around one phoneme
        y = log f0, sp, ap

        if f0 is 0 at that 5ms multiple, then interpolate

        x will be the same linguistic feature for [phoneme duration / 5 ms]
        """
        pass


if __name__ == '__main__':
    fe = FeatureExtractor()
    # pprint.pprint(fe.rec_phone_contexts)
    # pprint.pprint(fe.rec_phone_durations)
