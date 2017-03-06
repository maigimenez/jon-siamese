import pandas as pd


class Data:
    def __init__(self, tweet_a, tweet_b, similarity, one_hot):
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
    def __init__(self, corpus_name, sim_path, non_sim_path):
        self._sim_data = []
        self._non_sim_data = []
        self._data_frame = None

        if corpus_name == 'similarity':
            self._sim_path = sim_path
            self._non_sim_path = non_sim_path
            self._one_hot = {0: [0, 1], 1: [1, 0]}
            self.load_similarity()

        if corpus_name == 'quora':
            # TODO
            self.load_quora()

    @property
    def sim_data(self):
        return self._sim_data

    @property
    def non_sim_data(self):
        return self._non_sim_data

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

        # There is no need to use one-hot encoding for the siamese networks
        # # Encode the similar tags using a one-hot representation
        # sim_tags_encoded = [self._one_hot[tag] for tag in sim_tags]
        # assert len(sim_tags_encoded) == len(sim_tags)
        # assert len(sim_tweets) == len(sim_tags_encoded)

        # Save each pair of similar tweet within the data class
        for tws, tag in zip(sim_tweets, sim_tags):
            self._sim_data.append(Data(tws[0], tws[1], tag, [0, 1]))

        # Load the non similar tweets
        not_sim_tweets = pd_not_similar[['tweet_1', 'tweet_2']].values.tolist()
        not_sim_tags = pd_not_similar['similarity'].values.tolist()
        assert len(not_sim_tweets) == len(not_sim_tags)
        assert len(set(not_sim_tags)) == 1
        assert next(iter(set(not_sim_tags))) == 0

        # There is no need to use one-hot encodding for the siamese networks
        # # Encode the non similar tags using a one-hot representation
        # not_sim_tags_encoded = [self._one_hot[tag] for tag in not_sim_tags]
        # assert len(not_sim_tags_encoded) == len(not_sim_tags)
        # assert len(not_sim_tweets) == len(not_sim_tags_encoded)

        # Save each pair of non similar tweet within the data class
        for tws, tag in zip(not_sim_tweets, not_sim_tags):
            self._non_sim_data.append(Data(tws[0], tws[1], tag, [1, 0]))

            # assert len(self._sim_data) == 5538
            # assert len(self._non_sim_data) == 16379

    def load_quora(self):
        # TODO
        pass