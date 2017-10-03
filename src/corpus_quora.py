import tensorflow as tf
from os.path import join
from random import randrange

from corpus import Corpus
from utils import write_tfrecord


class CorpusQuora(Corpus):
    def __init__(self, path, preprocess=None, max_len=None):
        Corpus.__init__(self, 'quora', path, preprocess=None, max_len=None)
        self.load()

    def load(self):
        with open(self._path) as dataset_file:
            next(dataset_file)
            aux_line = ''
            for line in dataset_file:
                line_strip = line.strip().split('\t')
                # If some field is missing do not consider this entry.
                if len(line_strip) == 6:
                    qid, _, _, q1, q2, label = line.strip().split('\t')
                    self.save_data(self._preprocess, self._max_len, qid, q1, q2, label)

                else:
                    aux_line += line
                    strip_aux = aux_line.strip().split('\t')
                    if len(strip_aux) == 6:
                        qid, _, _, q1, q2, label = aux_line.strip().split('\t')
                        aux_line = ''
                        self.save_data(self._preprocess, self._max_len, qid, q1, q2, label)


    def balance_partitions(self, partitions_path,  one_hot=False,
                           split_files=False):
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

        # DEPRECATED
        def make_partitions(self):
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

