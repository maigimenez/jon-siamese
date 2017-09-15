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
        :param preprocess: None if not pre-process will be applied
                           -1 if all sentences will be pre-processed
                           N sentences longer than N will be pre-processed
        """
        self._sim_data = []
        self._non_sim_data = []

        if corpus_name == 'ibm':
            self.load_ibm(path, preprocess)
        elif corpus_name == 'quora':
            self.load_quora(path, preprocess, max_len)
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

    def load_ibm(self, path, preprocess):
        # TODO The ids of the IBM partitions with the ones released by Quora.
        with open(path) as dataset_file:
            for line in dataset_file:
                label, q1, q2, qid = line.strip().split('\t')
                self.save_data(preprocess, qid, q1, q2, label)

    def load_quora(self, path, preprocess, max_len):
        with open(path) as dataset_file:
            next(dataset_file)
            aux_line = ''
            for line in dataset_file:
                line_strip = line.strip().split('\t')
                # If some field is missing do not consider this entry.
                if len(line_strip) == 6:
                    qid, _, _, q1, q2, label = line.strip().split('\t')
                    self.save_data(preprocess, max_len, qid, q1, q2, label)

                else:
                    aux_line += line
                    strip_aux = aux_line.strip().split('\t')
                    if len(strip_aux) == 6:
                        qid, _, _, q1, q2, label = aux_line.strip().split('\t')
                        aux_line = ''
                        self.save_data(preprocess, max_len, qid, q1, q2, label)

    def load_p4p(self, path):
        # TODO Get the preprocess values from the console arguments
        preprocess, max_len = False, None
        with open(path) as dataset_file:
            dataset_reader = csv.reader(dataset_file, delimiter='\t')
            next(dataset_reader)
            for qid, line in enumerate(dataset_reader):
                if len(line) == 9:
                    sen_1, sen_2, tag = line[0], line[1], line[2]
                    if tag == 'Plagio':
                        self.save_data(preprocess, max_len, str(qid)+'_1', sen_1, sen_2, '1')
                    else:
                        self.save_data(preprocess, max_len, str(qid)+'_2', sen_1, sen_2, '0')


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
        print(type(num_sim_sentences), type(num_nonsim_sentences))
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

    def balance_partitions(self, partitions_path,  one_hot=False, split_files=False):
        """ Write a balance set of sentences """
        vocab_processor = self.create_vocabularies(133263, 231027, partitions_path)
        # Create and save the  TRAIN FILE
        if not split_files:
            writer = tf.python_io.TFRecordWriter(join(partitions_path, "train.tfrecords"))
        else:
            writer_sim = tf.python_io.TFRecordWriter(join(partitions_path, "train_sim.tfrecords"))
            writer_dis = tf.python_io.TFRecordWriter(join(partitions_path, "train_dis.tfrecords"))

        lines = 0
        # Mixed part: similar and non similar sentences
        for i in range(133263):
            # Write a non similar sentence
            data = self._non_sim_data[i]
            data_idx = self.to_index_data(data, vocab_processor)
            if not split_files:
                write_tfrecord(writer, data_idx, one_hot)
            else:
                write_tfrecord(writer_dis, data_idx, one_hot)

            # Write a similar sentence
            data = self._sim_data[i]
            data_idx = self.to_index_data(data, vocab_processor)
            if not split_files:
                write_tfrecord(writer, data_idx, one_hot)
            else:
                write_tfrecord(writer_sim, data_idx, one_hot)
            lines += 2

        # Remaining dissimilar sentences and random similar ones
        for i in range(133263, 231027):
            # Write a non similar sentence
            data = self._non_sim_data[i]
            data_idx = self.to_index_data(data, vocab_processor)
            if not split_files:
                write_tfrecord(writer, data_idx, one_hot)
            else:
                write_tfrecord(writer_dis, data_idx, one_hot)

            # Write a random similar sentence
            data_idx = self._sim_data[randrange(0, 133263)]
            if not split_files:
                write_tfrecord(writer, data_idx, one_hot)
            else:
                write_tfrecord(writer_sim, data_idx, one_hot)
            lines += 2
        print("Saved {} data examples for training".format(lines))

        # Create and save the  DEV FILE
        writer = tf.python_io.TFRecordWriter(join(partitions_path, "dev.tfrecords"))
        lines = 0
        # Mixed part: similar and non similar sentences
        for i, j in zip(range(231027, 239027), range(133263, 141263)):
            data = self._non_sim_data[i]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)

            data = self._sim_data[j]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)

            lines += 2

        # Remaining dissimilar dev sentences
        for i in range(239027, 243027):
            data = self._non_sim_data[i]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)
            lines += 1
        print("Saved {} data examples for development".format(lines))

        # Create and save the  TEST FILE
        writer = tf.python_io.TFRecordWriter(join(partitions_path, "test.tfrecords"))

        lines = 0
        # Mixed part: similar and non similar sentences
        for i, j in zip(range(243027, 251027), range(141263, 149263)):
            data = self._non_sim_data[i]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)

            data = self._sim_data[j]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)
            lines += 2

        # Remaining dissimilar test sentences
        for i in range(251027, 255027):
            data = self._non_sim_data[i]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)
            lines += 1
        print("Saved {} data examples for testing".format(lines))

    def write_partitions_mixed(self, partitions_path, one_hot=False):
        """ Create the partitions and write them in csv """

        vocab_processor = self.create_vocabularies(133263, 231027, partitions_path)

        # Create and save the  TRAIN FILE
        writer = tf.python_io.TFRecordWriter(join(partitions_path, "train.tfrecords"))
        lines = 0
        for i in range(133263):
            # Write a non similar sentence
            data = self._non_sim_data[i]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)

            # Write a similar sentence
            data = self._sim_data[i]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)
            lines += 2

        for i in range(133263, 231027):
            data = self._non_sim_data[i]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)
            lines += 1
        print("Saved {} data examples for training".format(lines))

        # Create and save the  DEV FILE
        writer = tf.python_io.TFRecordWriter(join(partitions_path, "dev.tfrecords"))
        lines = 0
        # Mixed part: similar and non similar sentences
        for i, j in zip(range(231027, 239027), range(133263, 141263)):
            data = self._non_sim_data[i]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)

            data = self._sim_data[j]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)
            lines += 2

        for i in range(239027, 243027):
            data = self._non_sim_data[i]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)
            lines += 1
        print("Saved {} data examples for development".format(lines))

        # Create and save the  TEST FILE
        writer = tf.python_io.TFRecordWriter(join(partitions_path, "test.tfrecords"))
        lines = 0
        # Mixed part: similar and non similar sentences
        for i, j in zip(range(243027, 251027), range(141263, 149263)):
            data = self._non_sim_data[i]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)

            data = self._sim_data[j]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)
            lines += 2

        for i in range(251027, 255027):
            data = self._non_sim_data[i]
            data_idx = self.to_index_data(data, vocab_processor)
            write_tfrecord(writer, data_idx, one_hot)
            lines += 1
        print("Saved {} data examples for testing".format(lines))

    # TODO: To be deleted
    def make_partitions_quora(self):
        self.shuffle()
        vocab_non_sim = self._non_sim_data[:231027]
        vocab_sim = self._sim_data[:133263]
        vocab_processor, sequence_length = build_vocabulary(vocab_sim,
                                                            vocab_non_sim)
        train_non_sim = [self.to_index_data(data, vocab_processor)
                         for data in self._non_sim_data[:207026]]
        train_sim = [self.to_index_data(data, vocab_processor)
                     for data in self._sim_data[:117262]]
        dev_non_sim = [self.to_index_data(data, vocab_processor)
                       for data in self._non_sim_data[207027:231027]]
        dev_sim = [self.to_index_data(data, vocab_processor)
                   for data in self._sim_data[117263:133263]]
        test_non_sim = [self.to_index_data(data, vocab_processor)
                        for data in self._non_sim_data[231027:]]
        test_sim = [self.to_index_data(data, vocab_processor)
                    for data in self._sim_data[133263:]]

        return train_non_sim, train_sim, dev_non_sim, dev_sim, \
               test_non_sim, test_sim, vocab_processor, sequence_length

