import pandas as pd
import numpy as np
from os.path import join
from utils import preprocess_sentence


class Data:
    def __init__(self, pair_id, sentence_1, sentence_2, duplicated, one_hot):
        self._pair_id = int(pair_id)
        self._sentence_1 = sentence_1
        self._sentence_2 = sentence_2
        self._label = duplicated
        self._oneh_label = one_hot

    def __repr__(self):
        if isinstance(self._sentence_1, str):
            sentence_1_summary = ' '.join(self._sentence_1.split()[:5]) + '...'
            sentence_2_summary = ' '.join(self._sentence_2.split()[:5]) + '...'
            return '({} - {}): {}'.format(sentence_1_summary, sentence_2_summary, self._label)
        else:
            sentence_1_summary = ', '.join([str(v) for v in self._sentence_1[:5]])
            sentence_2_summary = ', '.join([str(v) for v in self._sentence_2[:5]])
            return '([{}, ...] - [{}, ...] ({})): {}'.format(sentence_1_summary, sentence_2_summary,
                                                             self.sentence_1.shape[0], self._label)

    @property
    def sentence_1(self):
        return self._sentence_1

    @sentence_1.setter
    def sentence_1(self, value):
        self._sentence_1 = value

    @property
    def sentence_2(self):
        return self._sentence_2

    @sentence_2.setter
    def sentence_2(self, value):
        self._sentence_2 = value

    @property
    def label(self):
        return self._label

    @property
    def pair_id(self):
        return self._pair_id

class Corpus:
    def __init__(self, **kwargs):
        self._sim_data = []
        self._non_sim_data = []
        self._test_data = []
        self._data_frame = None

        self._corpora_path = join(kwargs['corpus_path'], kwargs['partitions_path'])
        self._partition = kwargs['partition']
        self.load_kaggle(kwargs['preprocess'])

    @property
    def sim_data(self):
        return self._sim_data

    @property
    def non_sim_data(self):
        return self._non_sim_data

    @property
    def test_data(self):
        return self._test_data

    def load_sim_quora(self, preprocess=False):
        # If two questions are duplicated, they are similar
        # Load the similar questions or duplicate in this case
        sim_questions = self._data_frame[self._data_frame['is_duplicate'] == 1][
            ['id', 'question1', 'question2']].values.tolist()
        sim_tags = self._data_frame[self._data_frame['is_duplicate'] == 1][
            'is_duplicate'].values.tolist()
        assert len(sim_questions) == len(sim_tags)
        assert len(set(sim_tags)) == 1
        assert next(iter(set(sim_tags))) == 1

        # Save each pair of similar questions within the data class
        for question, tag in zip(sim_questions, sim_tags):
            if preprocess:
                q1 = preprocess_sentence(question[1])
                q2 = preprocess_sentence(question[2])
            else:
                q1 = question[1]
                q2 = question[2]

            if q1 and q2:
                self._sim_data.append(Data(question[0], q1, q2, tag, [0, 1]))

    def load_non_sim_quora(self, preprocess=False):
        # Load the non similar questions or the non duplicated ones
        non_sim_questions = self._data_frame[self._data_frame['is_duplicate'] == 0][
            ['id', 'question1', 'question2']].values.tolist()
        non_sim_tags = self._data_frame[self._data_frame['is_duplicate'] == 0]['is_duplicate'].values.tolist()
        assert len(non_sim_questions) == len(non_sim_tags)
        assert len(set(non_sim_tags)) == 1
        assert next(iter(set(non_sim_tags))) == 0

        # Save each pair of non similar questions within the data class
        for question, tag in zip(non_sim_questions, non_sim_tags):
            # There are two errors in the dataset.
            # Ids: 105796, 201871 doesn't have a pair of questions.
            # This condition prevent storing this value
            if preprocess:
                q1 = preprocess_sentence(question[1])
                q2 = preprocess_sentence(question[2])
            else:
                q1 = question[1] if isinstance(question[1], str) else None
                q2 = question[2] if isinstance(question[2], str) else None
            if q1 and q2:
                self._non_sim_data.append(Data(question[0], q1, q2, tag, [1, 0]))

    def load_kaggle(self, preprocess):
        self._data_frame = pd.read_csv(self._corpora_path, header=0)
        print(self._data_frame.keys())
        if self._partition == 'test':
            self._test_data = []
            for q_id, q1, q2 in self._data_frame[['test_id', 'question1', 'question2']].values:
                # print(q_id, q1, q2)
                if isinstance(q1, str) and isinstance(q2, str):
                    if preprocess:
                        q1 = preprocess_sentence(q1)
                        q2 = preprocess_sentence(q2)
                    self._test_data.append(Data(q_id, q1, q2))
                elif isinstance(q1, str) and not isinstance(q2, str):
                    self._test_data.append(Data(q_id, q1, ""))
                elif not isinstance(q1, str) and isinstance(q2, str):
                    self._test_data.append(Data(q_id, "", q2))
                else:
                    print(q_id)
            assert len(self._test_data) == 2345796

        else:
            self.load_sim_quora(preprocess=preprocess)
            self.load_non_sim_quora(preprocess=preprocess)

    def to_index(self, vocab_processor):
        if self.test_data:
            for data in self.test_data:
                data.sentence_1 = np.array(list(vocab_processor.transform([data.sentence_1]))[0])
                data.sentence_2 = np.array(list(vocab_processor.transform([data.sentence_2]))[0])
        else:
            for data in self.non_sim_data:
                data.sentence_1 = np.array(list(vocab_processor.transform([data.sentence_1])))[0]
                data.sentence_2 = np.array(list(vocab_processor.transform([data.sentence_2])))[0]
            for data in self.sim_data:
                data.sentence_1 = np.array(list(vocab_processor.transform([data.sentence_1])))[0]
                data.sentence_2 = np.array(list(vocab_processor.transform([data.sentence_2])))[0]
