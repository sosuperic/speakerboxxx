# Extract input linguistic features and output features from utterance segmented Blizzard 2013 data

WORKING_DIR = '../speakerboxxx'
ALIGNER_DIR = '../Prosodylab-Aligner'
WAVS_PATH = '../Prosodylab-Aligner/data/blizzard2013/train/segmented/wavn'
PROMPTS_PATH = '../Prosodylab-Aligner/data/blizzard2013/train/segmented/prompts.gui'

LINGUISTIC_INPUTS_PATH = 'data/processed/blizzard2013/linguistic_inputs'
DURATION_TARGETS_PATH = 'data/processed/blizzard2013/duration_targets'

import os
import pprint
import nltk
import string
import pdb
import h5py
from collections import OrderedDict

import textgrid as tg
import numpy as np

class BlizzardFeatureExtractor:
    def __init__(self):
        self.tagdict = nltk.data.load('help/tagsets/upenn_tagset.pickle')
        self.universal_tagset = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ',
            'DET', 'NUM', 'PRT', 'X', '.']
        self.universal_tagset_to_idx = {tag: i for i, tag in enumerate(self.universal_tagset)}

        self.phoneset = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
            'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S',
            'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
        self.phoneset_to_idx = {p: i for i, p in enumerate(self.phoneset)}

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    def parse_and_save_prompts(self):
        """
        Parse prompts by extracting and normalizing sentences and save to WAVS_PATH
        """
        print('Parsing and saving prompts to .lab files')
        f = open(PROMPTS_PATH, 'rb')
        id, prompt = '', ''
        i = 0
        for line in f.readlines():
            if i % 3 == 0:
                id = ''.join(line.split()).replace('\n', '')    # CA-BB-01-01
            elif i % 3 == 1:
                # Normalize line 
                line = line.translate(string.maketrans("",""), string.punctuation)
                line = ' '.join(line.split())       # replace whitespace with space
                line = line.replace('\n', '')
                line = line.upper()

                # Write to .lab file
                lab_f = open(os.path.join(WAVS_PATH, '{}.lab'.format(id)), 'wb')
                lab_f.write(line)
                lab_f.close()
            i += 1
        f.close()

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    def force_align_phonemes_and_words(self):
        """
        Force align to get phoneme and word timings. Also save these to WAVS_PATH
        """
        print('Force aligning phonemes and words')

        cmd_cd_to_aligner_dir = 'cd {}'.format(ALIGNER_DIR)
        cmd_align = 'python3 -m aligner -a {}'.format(WAVS_PATH)
        oov_txt_path = os.path.join(ALIGNER_DIR, 'OOV.txt')

        # First align to get OOV words
        print('First force-align pass to see if any oov words')
        if os.path.exists(oov_txt_path):       # clean up from previous runs
            os.remove(oov_txt_path)
        os.chdir(ALIGNER_DIR)
        os.system(cmd_align)

        if os.path.exists(oov_txt_path):
            # Create set of all OOV words
            oov = open(oov_txt_path).readlines()
            oov = set([w.strip('\n') for w in oov])
            print('{} oov words'.format(len(oov)))

            # Get ids of prompts that contain a oov word
            print('Finding ids of prompts that contain a oov word')
            oov_ids = []
            f = open(PROMPTS_PATH, 'rb')
            id, prompt = '', ''
            i = 0
            for line in f.readlines():
                if i % 3 == 0:
                    id = ''.join(line.split()).replace('\n', '')    # CA-BB-01-01
                elif i % 3 == 1:
                    # Normalize line 
                    line = line.translate(string.maketrans("",""), string.punctuation)
                    line = ' '.join(line.split())       # replace whitespace with space
                    line = line.replace('\n', '')
                    line = line.upper()

                    for word in line.split():
                        if word in oov:
                            oov_ids.append(id)
                            break
                i += 1
            f.close()
            print('{} ids with a oov word'.format(len(oov_ids)))

            # Remove labs and wavs that contain a oov word
            print('Removing wavs and labs that contain a oov word')
            i = 0
            j = 0
            for id in oov_ids:
                # print(id)
                lab_path = os.path.join(WAVS_PATH, '{}.lab'.format(id))
                wav_path = os.path.join(WAVS_PATH, '{}.wav'.format(id))
                # print(lab_path)
                # print(wav_path)
                if os.path.exists(lab_path):
                    os.remove(lab_path)
                if os.path.exists(wav_path):
                    os.remove(wav_path)

            print('Second force-align pass')
            os.chdir(ALIGNER_DIR)
            os.system(cmd_align)

        else:
            print('No oov words. Done')

        # Cleanup
        os.chdir(WORKING_DIR)

    def _get_textgrid_paths(self):
        files = [f for f in os.listdir(WAVS_PATH) if 'TextGrid' in f]
        paths = [os.path.join(WAVS_PATH, f) for f in files]
        return paths

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    
    #####################################################################################################################   
    # These are copied and pasted directly from w0rdplay/pos_tagger.py
    def _tokenize(self, text):
        return nltk.word_tokenize(text)

    def _pos_tag(self, tokenized):
        return nltk.pos_tag(tokenized)

    def _pos_tag_simplified(self, tokenized):
        tagged = self._pos_tag(tokenized)
        simplified = [(word, nltk.map_tag('en-ptb', 'universal', tag)) for word, tag in tagged]
        return simplified
    def _remove_lexical_stress(self, phone):
        """
        AO1 -> AO
        """
        return ''.join([i for i in phone if not i.isdigit()])

    def _is_vowel(self, phone):
        return not set('AEIOU').isdisjoint(set(phone))

    def _count_number_of_syllables_in_phonemes(self, phonemes):
        return len([p for p in phonemes if self._is_vowel(p)])

    def _syllabize_phones(self, phonemes):
        """
        Convert list of phonemes into syllables
        """
        def next_phoneme_is_vowel(phonemes, cur_i):
            next_i = cur_i + 1
            if next_i >= len(phonemes):
                return False
            else:
                return self._is_vowel(phonemes[next_i])

        if self._count_number_of_syllables_in_phonemes(phonemes) == 1:
            return [phonemes]

        syllables = []
        cur_syllable = []
        prev_phoneme_is_vowel = False
        i = 0
        while i < len(phonemes):
            p = phonemes[i]
            if self._is_vowel(p):
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
                    while (i + 1 < len(phonemes)) and (not self._is_vowel(p)):   # Append until next vowel
                        i += 1
                        p = phonemes[i]
                        cur_syllable.append(p)

                    if sum([1 for p in cur_syllable if self._is_vowel(p)]) == 0: # Didn't hit any more vowels
                        syllables[-1] += cur_syllable                           # Test case: apologized
                        cur_syllable = []                                       # Clear so it gets filtered out

                    prev_phoneme_is_vowel = True
            i += 1
        syllables.append(cur_syllable)

        # Filter out initial empty syllable, e.g. hello
        syllables = [syl for syl in syllables if len(syl) > 0]

        return syllables

    #####################################################################################################################   
    def _get_clean_phone_seq(self, phones_intervals):
        """
        Return cleaned list of tuples (start, end, phone)

        Parameters
        ----------
        phones_intervals: TextGrid interval tier

        Notes
        -----
        - ignore sil phone, which occurs at beg and end
        - Handle '' phones by splitting their time: half goes to previous phone, next goes to next phone
            - '' phones are silences in the middle
            - note that this means it may still be the second, or the second to last phone, following and 
            - followed by a sil phone, respectively
        - remove lexical stress on vowel phonemes
        """
        # First pass to splitting '' phone time and remove them
        for i, interval in enumerate(phones_intervals):
            halfway = (interval.maxTime + interval.minTime) / 2.0
            if interval.mark == '':
                phones_intervals[i-1].maxTime = halfway
                phones_intervals[i+1].minTime = halfway
                phones_intervals.removeInterval(interval)

        # Second pass to remove 'sil', and clean phones
        cleaned = []
        for i, interval in enumerate(phones_intervals):
            start, end, phone = interval.minTime, interval.maxTime, interval.mark
            if phone == 'sil':
                continue
            if self._is_vowel(phone):
                phone = self._remove_lexical_stress(phone)
            cleaned.append([start, end, phone])

        return cleaned

    def _get_clean_words_seq(self, words_intervals):
        """
        Return cleaned list of tuples (start, end, word)

        Parameters
        ----------
        words_intervals: TextGrid interval tier

        Notes
        -----
        - ignore sil word, which occurs at beg and end
        - Handle 'sp' words by splitting their time: half goes to previous word, next goes to next word
            - 'sp' words are silences in the middle
            - note that this means it may still be the second, or the second to last word, following and 
            - followed by a sil word, respectively
        """
        # First pass to splitting '' word time and remove them
        for i, interval in enumerate(words_intervals):
            halfway = (interval.maxTime + interval.minTime) / 2.0
            if interval.mark == 'sp':
                words_intervals[i-1].maxTime = halfway
                words_intervals[i+1].minTime = halfway
                words_intervals.removeInterval(interval)

        # Second pass to remove 'sil', and clean phones
        cleaned = []
        for i, interval in enumerate(words_intervals):
            start, end, word = interval.minTime, interval.maxTime, interval.mark
            if word == 'sil':
                continue
            if self._is_vowel(word):
                word = self._remove_lexical_stress(word)
            cleaned.append([start, end, word])

        return cleaned

    #####################################################################################################################   
    def _get_phone2word(self, cleaned_phones, cleaned_words):
        """
        Match timings to return phone2word map

        Parameters
        ----------
        cleaned_words: list of (start, end, words)
        cleaned_phones: list of (start, end, phones)

        Returns
        -------
        phone2word: list of (phone, word)
            - phones in the same order
        word2phones: list of (word, [p1, p2, p3, p4, p5])
        """
        phone2word = []
        word2phones = []

        w_idx = 0
        cur_phones = []
        for _, p_end, p in cleaned_phones:
            w = cleaned_words[w_idx][2]
            phone2word.append((p,w))
            cur_phones.append(p)
            if p_end == cleaned_words[w_idx][1]:
                word2phones.append((w, cur_phones))
                cur_phones = []
                w_idx += 1

        import pprint
        # pprint.pprint(phone2word)
        # pprint.pprint(word2phones)
        # print()
        # print()
        # print()
        
        return phone2word, word2phones

    def _get_word2syllables(self, word2phones):
        """
        Map word to syllables

        Parameters
        ----------
        word2phones: list of (word, [p1, p2, p3, p4, p5])

        Returns
        -------
        word2syllables: list of (word, [[p1,p2], [p3,p4,p5]])
            - syllables is list of list of phones 
        """

        word2syllables = [(w, self._syllabize_phones(phones)) for w, phones in word2phones]
        # syl2word = {}
        # for word, syls in word2syls.items():
        #     for syl in syls:
        #         syl2word[syl] = word

        # import pprint
        # pprint.pprint(word2syllables)
        # print()
        # print()
        # print()

        return word2syllables

    def _get_phone2syl(self, cleaned_phones, word2phones, word2syllables):
        """
        Map word to syllable it's contained in
        """
        all_syllables = [word2syllables[i][1] for i in range(len(word2phones))]
        all_syllables = [syl for word in all_syllables for syl in word]      # flatten

        phone2syl = []
        syl_idx = 0
        num_p_in_cur_syl = 0
        for _, _, p in cleaned_phones:
            syl = all_syllables[syl_idx]
            phone2syl.append((p, syl))
            num_p_in_cur_syl += 1
            if num_p_in_cur_syl == len(syl):
                num_p_in_cur_syl = 0
                syl_idx += 1

        # import pprint
        # pprint.pprint([word2phones[i][0] for i in range(len(word2phones))])
        # Check 1
        # pprint.pprint(all_syllables)
        # Check 2
        # pprint.pprint(phone2syl)
        # print()
        # print()
        # print()
        # for phone, word in phone2word:
        #     syllables = 


        return phone2syl

    #####################################################################################################################   
    def create_and_save_linguistic_input_features(self):
        """
        Create feature vectors for all utterances. Base unit is at phoneme level.

        Features
        --------
        Categorical:
        - [x] quinphone identity: 39 * 5
        - [x] POS identity: 12
        - [x] vowel phoneme in current syllable: 39

        Position
        --------
        - [x] position of current phoneme in syllable
        - [x] position of current syllable in word
        - [x] position of current word in utterance

        Counts
        ------
        - [x] number of phones in syllable
        - [x] number of syllables in word
        - [x] number of words in utterance
        - Note: requires the following maps: phone2syl, phone2word, word2syllables
            - phone2syl to get syl, which is list of phones, to get number of phones in syl
            - phone2word to get word, then word2syllables to get number of syllables in word

        Other
        -----
        - [x] Duration of current utterance?
        - [x] For acoustic only:
            - Unidirectional: coarse-coded position of the current frame in the current phoneme,
            - MSFT: position of the current frame of the current phone
        """
        print('Creating linguistic features')

        feature_size = 6*len(self.phoneset) + len(self.universal_tagset) + 8    # leave extra bit for frame position
        textgrid_paths = self._get_textgrid_paths()
        for path_idx, path in enumerate(textgrid_paths):     # one recording
            print('@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n@\n{}, {}'.format(path_idx, path))
            id = os.path.splitext(os.path.basename(path))[0]
            grid = tg.TextGrid()
            grid.read(path)
            phones_interval_tier = grid.getFirst('phones')
            words_interval_tier = grid.getFirst('words')

            cleaned_words = self._get_clean_words_seq(words_interval_tier)
            cleaned_phones = self._get_clean_phone_seq(phones_interval_tier)

            phone2word, word2phones = self._get_phone2word(cleaned_phones, cleaned_words)
            word2syllables = self._get_word2syllables(word2phones)
            phone2syl = self._get_phone2syl(cleaned_phones, word2phones, word2syllables)

            w_in_utt_idx = 0
            syl_in_w_idx = 0
            p_in_syl_idx = 0
            features = np.zeros([len(cleaned_phones), feature_size])                  # num_phonemes x feature_size
            for i, interval in enumerate(cleaned_phones):                       # iterate over phones
                start, end, phone = interval[0], interval[1], interval[2]
                word = phone2word[i][1]
                syl = phone2syl[i][1]

                ##### Categorical
                # Quinphone identities
                quinphone_encoded = np.zeros(5 * len(self.phoneset))
                q_indices = range(i-2, i+3)
                q_phones = []
                for window_idx, q_idx in enumerate(q_indices):
                    if q_idx < 0 or q_idx >= len(cleaned_phones):
                        continue
                    q_phone = cleaned_phones[q_idx][2]
                    q_phone_idx = self.phoneset_to_idx[q_phone]
                    q_phone_encoded = np.zeros(len(self.phoneset))
                    q_phone_encoded[q_phone_idx] = 1
                    quinphone_encoded[window_idx * len(self.phoneset): (window_idx + 1) * len(self.phoneset)] = q_phone_encoded
                    q_phones.append(q_phone)

                # POS identity
                tag = self._pos_tag_simplified(self._tokenize(word))[0][1]  # list of (word, tags)
                tag_encoded = np.zeros(len(self.universal_tagset))
                tag_encoded[self.universal_tagset_to_idx[tag]] = 1

                # Vowel phoneme in current syllable
                syl_vowel_encoded = np.zeros(len(self.phoneset))
                syl_vowel = [p for p in syl if self._is_vowel(p)]
                if len(syl_vowel) > 0:
                    syl_vowel = syl_vowel[0]
                    syl_vowel_encoded[self.phoneset_to_idx[syl_vowel]] = 1

                ##### Position
                pos_p_in_syl = np.array([p_in_syl_idx])
                pos_syl_in_word = np.array([syl_in_w_idx])
                pos_w_in_utt = np.array([w_in_utt_idx])

                ##### Counts
                num_p_in_syl = np.array([len(syl)])
                num_syl_in_word = np.array([len(word2syllables[w_in_utt_idx][1])])
                num_words_in_utt = np.array([len(cleaned_words)])

                ##### Other
                utt_dur = np.array([cleaned_words[-1][1] - cleaned_words[0][0]])

                ##### Concatenate
                frame_features = np.hstack([
                    quinphone_encoded,
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
                features[i,:feature_size-1] = frame_features        # -1 for acoustic frame position


                ##### Sanity check
                print 'utt', ' '.join([w for _, _, w in cleaned_words])
                print word, syl, phone
                print w_in_utt_idx, syl_in_w_idx, p_in_syl_idx
                print 'num_syl_in_word', len(word2syllables[w_in_utt_idx][1])
                print 'utt_dur', cleaned_words[-1][1] - cleaned_words[0][0]
                print 'quinphones', q_phones
                print ''

                ##### Counters to get positional features
                at_end_of_syl = p_in_syl_idx + 1 == len(syl)
                # TODO: at_end_of_word seems wrong
                at_end_of_word = (syl_in_w_idx + 1 == len(word2syllables[w_in_utt_idx][1])) and at_end_of_syl
                # at_end_of_word = (syl_in_w_idx + 1 == len(self.word_to_syl[word])) and at_end_of_syl
                if at_end_of_word:
                    w_in_utt_idx += 1
                    syl_in_w_idx = 0
                    p_in_syl_idx = 0
                elif at_end_of_syl:
                    syl_in_w_idx += 1
                    p_in_syl_idx = 0
                else:
                    p_in_syl_idx += 1

            # # Save features to disk
            with h5py.File(os.path.join(LINGUISTIC_INPUTS_PATH, id + '.h5'), 'w') as hf:
                hf.create_dataset('x', data=features)

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    def create_and_save_duration_target_features(self):
        print('Creating and saving duration target features')
        i = 0
        textgrid_paths = self._get_textgrid_paths()
        for i, path in enumerate(textgrid_paths):
            id = os.path.splitext(os.path.basename(path))[0]
            print(i, id)
            grid = tg.TextGrid()
            grid.read(path)
            phones_intervals = grid.getFirst('phones')

            # Get duration of each non-'sil', non-'' phoneme
            durations = []
            cleaned_phones = self._get_clean_phone_seq(phones_intervals)
            for start, end, phone in cleaned_phones:
                durations.append(end - start)

            duration_targets = np.array(durations)
            duration_targets = np.expand_dims(duration_targets, 1)      # (num_phonemes i.e. seq_len, ) -> (num_phonemes, 1)

            with h5py.File(os.path.join(DURATION_TARGETS_PATH, id + '.h5'), 'w') as hf:
                hf.create_dataset('y', data=duration_targets)
            i += 1

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    def create_and_save_acoustic_target_features(self):
        print('Creating and saving acoustic target features')

        # Get window of valid audio for which to extract audio features
        tmp_fn = 'recording_windows_for_audio_features.txt'
        print('Getting valid windows (ignoring initial and ending silences) for each recording and saving to tmp file')
        textgrid_paths = self._get_textgrid_paths()
        tmp_f = open(tmp_fn, 'wb')
        for i, path in enumerate(textgrid_paths):
            id = os.path.splitext(os.path.basename(path))[0]
            grid = tg.TextGrid()
            grid.read(path)
            words_intervals = grid.getFirst('words')
            cleaned_words = self._get_clean_words_seq(words_intervals)
            start = cleaned_words[0][0]
            end = cleaned_words[-1][1]
            if i == len(textgrid_paths)-1:
                tmp_f.write('{},{},{}'.format(id, start, end))
            else:
                tmp_f.write('{},{},{}\n'.format(id, start, end))

        # Execute julia program to actually extract and save features
        print('Executing julia program, which will read tmp file and extract audio features')
        os.system('julia BlizzardAcousticFeatureExtractor.jl')


        # # Clean up
        # print('Cleaning up: delete tmp file')
        # os.remove(tmp_fn)

if __name__ == '__main__':
    bfe = BlizzardFeatureExtractor()

    # Run once to get TextGrids
    # bfe.parse_and_save_prompts()
    # bfe.force_align_phonemes_and_words()

    # Create and save features
    # bfe.create_and_save_linguistic_input_features()

    # bfe.create_and_save_duration_target_features()
    
    bfe.create_and_save_acoustic_target_features()
