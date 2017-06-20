import tensorflow as tf
from corpus import Corpus
from utils import build_vocabulary
import pickle
import argparse
from os.path import join, isdir
from os import makedirs


def get_arguments():
    parser = argparse.ArgumentParser(description='Create TF Records from Kaggle CSV')
    parser.add_argument('--data', metavar='d', type=str,
                        help='Path where the Kaggle dataset is.',
                        dest='dataset_path')

    parser.add_argument('--out', metavar='o', type=str,
                        help='Path where the tf records will be saved.',
                        dest='output_path')

    parser.add_argument('-p', action='store_true', default=False,
                        help='Apply a pre-processing phase',
                        dest='preprocess')

    parser.add_argument('-o', action='store_true', default=False,
                        help='Save the one-hot encoding',
                        dest='one_hot')

    args = parser.parse_args()
    return args.dataset_path, args.output_path, args.preprocess, args.one_hot

if __name__ == "__main__":
    dataset_path, output_path, preprocess, one_hot = get_arguments()

    # Filepaths where the dataset is found
    # KAGGLE_PATH = '/home/mgimenez/Dev/corpora/Quora/Kaggle/'
    TRAIN_KAGGLE = join(dataset_path, 'train.csv')
    TEST_KAGGLE = join(dataset_path, 'test.csv')

    # Read the dataset
    train = Corpus(corpus_path=dataset_path, partition='train', partitions_path=TRAIN_KAGGLE, preprocess=preprocess)
    print(len(train.sim_data), len(train.non_sim_data), train.sim_data[0],  train.non_sim_data[0], train._data_frame.shape)
    # pickle.dump(train, open("kaggle.train", "wb"))

    test = Corpus(corpus_path=dataset_path, partition='test', partitions_path=TEST_KAGGLE, preprocess=preprocess)
    print(len(test.test_data), test._data_frame.shape)
    # pickle.dump(test, open("kaggle.test", "wb"))

    if not isdir(output_path):
        makedirs(output_path)

    vocab_processor, sequence_length = build_vocabulary(train)
    pickle.dump(vocab_processor, open(join(output_path, "vocab.train"), "wb"))
    pickle.dump(sequence_length, open(join(output_path, "sequence.len"), "wb"))

    train.to_index(vocab_processor)
    test.to_index(vocab_processor)

    # Write the TF records
    writer = tf.python_io.TFRecordWriter(join(output_path, "train.tfrecords"))
    for data in train.sim_data:
        data_label = data.oneh_label if one_hot else [data.label]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=data_label)),
                    'sentence_1': tf.train.Feature(int64_list=tf.train.Int64List(value=data.sentence_1.astype("int64"))),
                    'sentence_2': tf.train.Feature(int64_list=tf.train.Int64List(value=data.sentence_2.astype("int64"))),
                }))
        writer.write(example.SerializeToString())
    for data in train.non_sim_data:
        data_label = data.oneh_label if one_hot else [data.label]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=data_label)),
                    'sentence_1': tf.train.Feature(int64_list=tf.train.Int64List(value=data.sentence_1.astype("int64"))),
                    'sentence_2': tf.train.Feature(int64_list=tf.train.Int64List(value=data.sentence_2.astype("int64"))),
                }))
        writer.write(example.SerializeToString())


    writer = tf.python_io.TFRecordWriter(join(output_path, "test.tfrecords"))
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
