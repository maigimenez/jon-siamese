import pandas as pd
from os.path import join

class Data:
    def __init__(self, tweet_a, tweet_b, similarity=None, one_hot=None):
        # TODO: change tweet a/b for text a/b since it can be tweets or questions
        self._tweet_a = tweet_a
        self._tweet_b = tweet_b
        self._label = similarity
        self._oneh_label = one_hot

    def __repr__(self):
        tweet_a_summary = ' '.join(self._tweet_a.split()[:5]) + '...'
        tweet_b_summary = ' '.join(self._tweet_b.split()[:5]) + '...'
        return '({} - {}): {}'.format(tweet_a_summary, tweet_b_summary, self._label)

    @property
    def tweet_a(self):
        return self._tweet_a

    @tweet_a.setter
    def tweet_a(self, value):
        self._tweet_a = value

    @property
    def tweet_b(self):
        return self._tweet_b

    @tweet_b.setter
    def tweet_b(self, value):
        self._tweet_b = value

    @property
    def label(self):
        return self._label

    @property
    def oneh_label(self):
        return self._oneh_label


class Corpus:
    def __init__(self, corpus_name, **kwargs):
        self._sim_data = []
        self._non_sim_data = []
        self._test_data = None
        self._data_frame = None

        if corpus_name == 'similarity':
            self._sim_path = kwargs['sim_path']
            self._non_sim_path = kwargs['non_sim_path']
            self.load_similarity()

        if corpus_name == 'quora':
            self._corpora_path = kwargs['corpus_path']
            if 'partition' in kwargs.keys():
                self._partition = kwargs['partition']
                self._partitions_path = kwargs['partitions_path']
            else:
                self._partition = None
            self.load_quora()

        if corpus_name == 'kaggle':
            self._corpora_path = join(kwargs['corpus_path'], kwargs['partitions_path'])
            self._partition = kwargs['partition']
            self.load_kaggle()

    @property
    def sim_data(self):
        return self._sim_data

    @property
    def non_sim_data(self):
        return self._non_sim_data

    @property
    def test_data(self):
        return self._test_data

    def load_similarity(self):
        """ Load the Similarity dataset buit by Jesús Alonso
        and encapsulate it using this class

        """

        # TODO check that the filepath that the user has passed is correct
        # Load data into a dataframe.
        pd_similar = pd.read_csv(self._sim_path,
                                 names=['tweet_1', 'tweet_2', 'similarity'])
        pd_not_similar = pd.read_csv(self._non_sim_path,
                                     names=['tweet_1', 'tweet_2', 'similarity'])
        self._data_frame = pd.concat([pd_similar, pd_not_similar])

        # Load the similar tweets
        sim_tweets = pd_similar[['tweet_1', 'tweet_2']].values.tolist()
        sim_tags = pd_similar['similarity'].values.tolist()
        assert len(sim_tweets) == len(sim_tags)
        assert len(set(sim_tags)) == 1
        assert next(iter(set(sim_tags))) == 1

        # Save each pair of similar tweet within the data class
        for tws, tag in zip(sim_tweets, sim_tags):
            self._sim_data.append(Data(tws[0], tws[1], tag, [0, 1]))

        # Load the non similar tweets
        not_sim_tweets = pd_not_similar[['tweet_1', 'tweet_2']].values.tolist()
        not_sim_tags = pd_not_similar['similarity'].values.tolist()
        assert len(not_sim_tweets) == len(not_sim_tags)
        assert len(set(not_sim_tags)) == 1
        assert next(iter(set(not_sim_tags))) == 0

        # Save each pair of non similar tweet within the data class
        for tws, tag in zip(not_sim_tweets, not_sim_tags):
            self._non_sim_data.append(Data(tws[0], tws[1], tag, [1, 0]))

        # assert len(self._sim_data) == 5538
        # assert len(self._non_sim_data) == 16379


    def load_sim_quora(self):
        # Load the similar questions or duplicate in this case
        sim_questions = self._data_frame[self._data_frame['is_duplicate'] == 1][
            ['question1', 'question2']].values.tolist()
        sim_tags = self._data_frame[self._data_frame['is_duplicate'] == 1][
            'is_duplicate'].values.tolist()
        assert len(sim_questions) == len(sim_tags)
        assert len(set(sim_tags)) == 1
        assert next(iter(set(sim_tags))) == 1

        # Save each pair of similar questions within the data class
        for question, tag in zip(sim_questions, sim_tags):
            self._sim_data.append(Data(question[0], question[1], tag, [0, 1]))

    def load_non_sim_quora(self):
        # Load the non similar questions or the non duplicated ones
        non_sim_questions = self._data_frame[self._data_frame['is_duplicate'] == 0][
            ['question1', 'question2']].values.tolist()
        non_sim_tags = self._data_frame[self._data_frame['is_duplicate'] == 0]['is_duplicate'].values.tolist()
        assert len(non_sim_questions) == len(non_sim_tags)
        assert len(set(non_sim_tags)) == 1
        assert next(iter(set(non_sim_tags))) == 0

        # Save each pair of non similar questions within the data class
        for question, tag in zip(non_sim_questions, non_sim_tags):
            # There are two errors in the dataset.
            # Ids: 105796, 201871 doesn't have a pair of questions.
            # This condition prevent storing this value
            if isinstance(question[0], str) and isinstance(question[1], str):
                self._non_sim_data.append(Data(question[0], question[1], tag, [0, 1]))

    def load_quora(self):
        self._data_frame = pd.read_csv(self._corpora_path, sep='\t', header=0)
        self.load_sim_quora()
        self.load_non_sim_quora()

        # TODO modify it to return all the partitions at once.
        # Load the partitions
        if self._partition:
            QUORA_PARTITION_PATH = join(self._partitions_path, self._partition + '.tsv')
            patition_ids = [int(line.strip().split('\t')[-1]) for line in open(QUORA_PARTITION_PATH)]
            if self._partition == 'train':
                assert len(patition_ids) == 384348
            else:
                assert len(patition_ids) == 10000

            # Replace the dataframe with only the rows from this partition and reload
            self._data_frame = self._data_frame.loc[self._data_frame['id'].isin(patition_ids)]
            assert len(patition_ids) == self._data_frame.shape[0]

            self.load_sim_quora()
            self.load_non_sim_quora()

    def load_kaggle(self):
        self._data_frame = pd.read_csv(self._corpora_path, header=0)
        if self._partition == 'test':
            self._test_data = []
            for q1, q2 in self._data_frame[['question1', 'question2']].values:
                self._test_data.append(Data(q1, q2))
            assert len(self._test_data) == 2345796
        else:
            self.load_sim_quora()
            self.load_non_sim_quora()

