import tensorflow as tf
from utils import preprocess_sentence
from data import Data
from random import shuffle, randrange
from utils import build_vocabulary, write_tfrecord
from os.path import join, isdir
from os import makedirs

import pickle
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

class Corpus:
    def __init__(self, corpus_name, path, preprocess=None, max_len=None):
        """ Load the data set in a class

        Arguments:
        :param preprocess (bool): True if not pre-process will be applied
                                  False if not pre-process will be applied

        :param max_len (int): -1 if all sentences will be pre-processed
                              N sentences longer than N will be pre-processed
        """
        self._name = corpus_name
        self._sim_data = []
        self._non_sim_data = []
        self._path = path
        self._preprocess = preprocess
        self._max_len = max_len

        if corpus_name == 'ibm':
            self.load_ibm(path, preprocess)
        elif corpus_name == 'p4p':
            self.load_p4p(path)

    @property
    def sim_data(self):
        return self._sim_data

    @property
    def non_sim_data(self):
        return self._non_sim_data

    def save_data(self, preprocess, max_len, qid, q1, q2, label):
        if preprocess:
            q1 = preprocess_sentence(q1, max_len)
            q2 = preprocess_sentence(q2, max_len)
        # This is a non-duplicate sentence -> dissimilar
        if label == '0':
            self._non_sim_data.append(Data(qid, q1, q2, label, [0, 1]))
        # This is a duplicate sentence -> similar
        else:
            self._sim_data.append(Data(qid, q1, q2, label, [1, 0]))

    def shuffle(self):
        shuffle(self._sim_data)
        shuffle(self._non_sim_data)

    def to_index_data(self, data, vocab_processor):
        data.sentence_1 = np.array(list(vocab_processor.transform([data.sentence_1])))[0]
        data.sentence_2 = np.array(list(vocab_processor.transform([data.sentence_2])))[0]
        return data

    def to_index(self, vocab_processor, data=None):
        for data in self.non_sim_data:
            data.sentence_1 = np.array(list(vocab_processor.transform([data.sentence_1])))[0]
            data.sentence_2 = np.array(list(vocab_processor.transform([data.sentence_2])))[0]
        for data in self.sim_data:
            data.sentence_1 = np.array(list(vocab_processor.transform([data.sentence_1])))[0]
            data.sentence_2 = np.array(list(vocab_processor.transform([data.sentence_2])))[0]

    def create_vocabularies(self,  num_sim_sentences, num_nonsim_sentences,
                            partitions_path=None):
        """ Create and save the vocabularies

        :param partitions_path: path where the binarized files should be saved
            if this is not present the vocabularies won't be saved.
        :return: the vocabulary processor.

        """
        vocab_non_sim = self._non_sim_data[:num_nonsim_sentences]
        vocab_sim = self._sim_data[:num_sim_sentences]
        vocab_processor, sequence_length = build_vocabulary(vocab_sim,
                                                            vocab_non_sim)
        if partitions_path:
            if not isdir(partitions_path):
                makedirs(partitions_path)
            pickle.dump(vocab_processor, open(join(partitions_path,
                                                   "vocab.train"), "wb"))
            pickle.dump(sequence_length, open(join(partitions_path,
                                                   "sequence.len"), "wb"))

        return vocab_processor
