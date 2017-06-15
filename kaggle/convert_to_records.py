import tensorflow as tf
from corpus import Corpus
from utils import build_vocabulary
import pickle

if __name__ == "__main__":
    # Filepaths where the dataset is found
    KAGGLE_PATH = '/home/mgimenez/Dev/corpora/Quora/Kaggle/'
    TRAIN_KAGGLE = 'train.csv'
    TEST_KAGGLE = 'test.csv'

    # Read the dataset
    train = Corpus(corpus_path=KAGGLE_PATH, partition='train', partitions_path=TRAIN_KAGGLE, preprocess=True)
    print(len(train.sim_data), len(train.non_sim_data), train.sim_data[0],  train.non_sim_data[0], train._data_frame.shape)
    pickle.dump(train, open("kaggle.train", "wb"))

    test = Corpus(corpus_path=KAGGLE_PATH, partition='test', partitions_path=TEST_KAGGLE, preprocess=True)
    print(len(test.test_data), test._data_frame.shape)
    pickle.dump(test, open("kaggle.test", "wb"))

    vocab_processor, sequence_length = build_vocabulary(train)
    pickle.dump(vocab_processor, open("vocab.train", "wb"))
    pickle.dump(sequence_length, open("sequence_length", "wb"))

    train.to_index(vocab_processor)
    # print('TRAIN', len(train.sim_data), len(train.non_sim_data), train.sim_data[0], train.non_sim_data[0])
    test.to_index(vocab_processor)
    # print('TEST', type(test.test_data), len(test.test_data))

    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for data in train.sim_data:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[data.label])),
                    'sentence_1': tf.train.Feature(int64_list=tf.train.Int64List(value=data.sentence_1.astype("int64"))),
                    'sentence_2': tf.train.Feature(int64_list=tf.train.Int64List(value=data.sentence_2.astype("int64"))),
                }))
        writer.write(example.SerializeToString())
    for data in train.non_sim_data:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[data.label])),
                    'sentence_1': tf.train.Feature(int64_list=tf.train.Int64List(value=data.sentence_1.astype("int64"))),
                    'sentence_2': tf.train.Feature(int64_list=tf.train.Int64List(value=data.sentence_2.astype("int64"))),
                }))
        writer.write(example.SerializeToString())

    writer = tf.python_io.TFRecordWriter("test.tfrecords")
    for i, data in enumerate(test.test_data):
        print(i)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'pair_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[data.pair_id])),
                    'sentence_1': tf.train.Feature(int64_list=tf.train.Int64List(value=data.sentence_1.astype("int64"))),
                    'sentence_2': tf.train.Feature(int64_list=tf.train.Int64List(value=data.sentence_2.astype("int64"))),
                }))
        writer.write(example.SerializeToString())
