import tensorflow as tf
from os.path import join
from random import randrange

from corpus import Corpus
from utils import write_tfrecord


class CorpusMS(Corpus):
    def __init__(self, path, preprocess=None, max_len=None):
        Corpus.__init__(self, 'microsoft', path, preprocess=None, max_len=None)
        self.load()

    def load(self):
        id_pair = 0
        with open(self._path) as corpus_file:
            next(corpus_file)
            for line in corpus_file:
                label, id1, id2, q1, q2 = line.strip().split('\t')
                # id_pair = id1 + '-' + id2
                # 1 similar / 0 disimilar
                self.save_data(self._preprocess, self._max_len, id_pair, q1, q2, label)
                id_pair += 1

    def write_tensors(self, partitions_path, partitions_set,
                      one_hot=False, vocab_processor=None,
                      split_files=False):

        writer, writer_dis, writer_sim = None, None, None
        if not vocab_processor:
            vocab_processor = self.create_vocabularies(len(self._sim_data),
                                                       len(self._non_sim_data),
                                                       partitions_path)
        lines = 0
        if partitions_set == 'train':
            if not split_files:
                writer = tf.python_io.TFRecordWriter(join(partitions_path, "train.tfrecords"))
            else:
                writer_sim = tf.python_io.TFRecordWriter(join(partitions_path, "train_sim.tfrecords"))
                writer_dis = tf.python_io.TFRecordWriter(join(partitions_path, "train_dis.tfrecords"))

            # I am assuming that I have more similar than disimilar sentences.
            # This is true in the MS dataset

            for i in range(len(self._non_sim_data)):
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

            for i in range(len(self._non_sim_data), len(self._sim_data)):
                # Write a similar sentence
                data = self._sim_data[i]
                data_idx = self.to_index_data(data, vocab_processor)
                if not split_files:
                    write_tfrecord(writer, data_idx, one_hot)
                else:
                    write_tfrecord(writer_sim, data_idx, one_hot)

                # Write a random dissimilar sentence
                data_idx = self._sim_data[randrange(len(self._non_sim_data))]
                if not split_files:
                    write_tfrecord(writer, data_idx, one_hot)
                else:
                    write_tfrecord(writer_sim, data_idx, one_hot)
                lines += 2

            print("Saved {} data examples for training".format(lines))

        else:
            writer = tf.python_io.TFRecordWriter(join(partitions_path, partitions_set+'.tfrecords'))
            for data in self._sim_data:
                data_idx = self.to_index_data(data, vocab_processor)
                write_tfrecord(writer, data_idx, one_hot)
                lines += 1
            for data in self._non_sim_data:
                data_idx = self.to_index_data(data, vocab_processor)
                write_tfrecord(writer, data_idx, one_hot)
                lines += 1

            print("Saved {} data examples for {}".format(lines, partitions_set))

        return vocab_processor