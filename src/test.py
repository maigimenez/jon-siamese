import tensorflow as tf
import numpy as np
import pickle
from time import time
from os.path import join, abspath, curdir
import csv
import argparse
from utils import read_flags
from siamese import Siamese

def get_arguments():
    parser = argparse.ArgumentParser(description='Test a Siamese Architecture')
    parser.add_argument('--tf', metavar='r', type=str,
                        help='Path where the tfrecords are',
                        dest='tf_path')
    parser.add_argument('--model', metavar='m', type=str,
                        help='Path where the trained model is',
                        dest='model_path')
    parser.add_argument('--flags', metavar='f', type=str,
                        help='Path where the flags for training are',
                        dest='flags_path')

    args = parser.parse_args()
    return args.tf_path, args.model_path, args.flags_path

def load_binarize_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def read_sample(filename_queue, num_labels, sequence_len):
    reader = tf.TFRecordReader()
    _, serialized_sample = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_sample,
        features={
            'label': tf.FixedLenFeature([num_labels], tf.int64),
            'sentence_1': tf.FixedLenFeature([sequence_len], tf.int64),
            'sentence_2': tf.FixedLenFeature([sequence_len], tf.int64)
        })
    return features['label'], features['sentence_1'], features['sentence_2']


def input_pipeline(filepath, batch_size, num_labels, sequence_len, num_epochs=None):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filepath], num_epochs=num_epochs)
        pair_id, sentence_1, sentence_2 = read_sample(filename_queue, num_labels, sequence_len)
        pair_batch, sentences_1_batch, sentences_2_batch = tf.train.shuffle_batch([pair_id, sentence_1, sentence_2],
                                                                                  batch_size=batch_size,
                                                                                  num_threads=1,
                                                                                  capacity=1000 + 3 * FLAGS.batch_size,
                                                                                  min_after_dequeue=1000)
    return pair_batch, sentences_1_batch, sentences_2_batch

if __name__ == "__main__":

    tf_path, model_path, flags_path = get_arguments()
    FLAGS = read_flags(flags_path)

    # Import the parameters binarized
    TEST_PATH = join(tf_path, 'test.tfrecords')
    vocab_processor_path = join(tf_path, 'vocab.train')
    vocab_processor = load_binarize_data(vocab_processor_path)
    sequence_length_path = join(tf_path, 'sequence.len')
    seq_len = load_binarize_data(sequence_length_path)

    # TODO this is a parameter
    one_hot =  False
    n_labels = 2 if one_hot else 1

  # TEST THE SYSTEM
    with tf.Graph().as_default():
        label_batch, test_1_batch, test_2_batch = input_pipeline(filepath=TEST_PATH,
                                                                batch_size=1,
                                                                num_labels=n_labels,
                                                                sequence_len=seq_len,
                                                                num_epochs=1)

        print(type(label_batch), type(test_1_batch), type(test_2_batch))
        siamese = Siamese(seq_len,
                          vocab_size=len(vocab_processor.vocabulary_),
                          embedding_size=FLAGS.embedding_dim,
                          filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                          num_filters=FLAGS.num_filters,
                          margin=1.0,
                          threshold=1.5,
                          fully=True,
                          hash_size=128)

        init_op = tf.global_variables_initializer()
        init_again = tf.local_variables_initializer()

        predictions = {}
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init_op)
            sess.run(init_again)

            # Restore the model
            saver = tf.train.Saver()
            model_abspath = abspath(join(curdir, model_path, "model.ckpt"))
            save_path = saver.save(sess, model_abspath)
            saver.restore(sess, save_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            test_sample, hits = 0, 0

            try:
                while not coord.should_stop():
                    test_1, test_2, test_label = sess.run([test_1_batch, test_2_batch, label_batch])
                    # TEST CLASSIFICATION
                    loss, attraction, repulsion, dis, acc = \
                            sess.run([siamese.loss, siamese.attraction_loss,
                                      siamese.repulsion_loss, siamese.distance,
                                      siamese.accuracy],
                                     feed_dict={
                                         siamese.left_input: test_1,
                                         siamese.right_input: test_2,
                                         siamese.label: test_label,
                                     })
                    log_str = "(#{0: <5} - {6}) - Loss: {1:.4f} - " \
                                  "(a: {2:.3f} - r: {3:.3f} - " \
                                  "d: {4:.4f}, accuracy:{5:.4f})"
                    print(log_str.format(test_sample, loss,
                                         attraction[0][0],
                                         repulsion[0][0],
                                         dis[0], acc,
                                         test_label[0][0],))
                    test_sample += 1
                    if acc == 1:
                        hits += 1
            except tf.errors.OutOfRangeError:
                print("Done testing!")
            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()

            print(hits, test_sample, hits/test_sample)


