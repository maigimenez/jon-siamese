class Data:
    def __init__(self, pair_id, sentence_1, sentence_2, duplicated=None, one_hot=None):
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

    @property
    def oneh_label(self):
        return self._oneh_label