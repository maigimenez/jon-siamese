import tensorflow as tf
import numpy as np
import pickle
from time import time
from os.path import join, abspath, curdir
import csv

from siamese import Siamese


def default_flags():
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # GPU related configuration
    FLAGS.allow_soft_placement = True
    FLAGS.log_device_placement = False

    # Hyperparameters
    FLAGS.embedding_dim = 300
    FLAGS.filter_sizes = '3,4,5'
    FLAGS.num_filters = 50

    # Training parameters
    FLAGS.batch_size = 100
    FLAGS.num_epochs = 1
    # FLAGS.evaluate_every = 1
    # FLAGS.checkpoint_every = 1
    # FLAGS.shuffle_epochs = True
    # FLAGS.mixed = False

    print("Default parameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print(" - {}={}".format(attr.upper(), value))
    print("")

    return FLAGS

def load_binarize_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def read_sample(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_sample = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_sample,
        features={
            'pair_id': tf.FixedLenFeature([], tf.int64),
            'sentence_1': tf.FixedLenFeature([sequence_length], tf.int64),
            'sentence_2': tf.FixedLenFeature([sequence_length], tf.int64)
        })
    return features['pair_id'], features['sentence_1'], features['sentence_2']


def input_pipeline(filepath, batch_size, num_epochs=None):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filepath], num_epochs=num_epochs)
        pair_id, sentence_1, sentence_2 = read_sample(filename_queue)
        pair_batch, sentences_1_batch, sentences_2_batch = tf.train.shuffle_batch([pair_id, sentence_1, sentence_2],
                                                                                  batch_size=batch_size,
                                                                                  num_threads=1,
                                                                                  capacity=1000 + 3 * FLAGS.batch_size,
                                                                                  min_after_dequeue=1000)
    return pair_batch, sentences_1_batch, sentences_2_batch


if __name__ == "__main__":
    RECORDS_PATH = '/home/mgimenez/Dev/projects/jon-siamese/kaggle/bin_data/preprocess60'
    TEST_PATH = join(RECORDS_PATH, 'test.tfrecords')

    vocab_processor_path = join(RECORDS_PATH, 'vocab.train')
    vocab_processor = load_binarize_data(vocab_processor_path)

    sequence_length_path = join(RECORDS_PATH, 'sequence_length')
    sequence_length = load_binarize_data(sequence_length_path)


    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    FLAGS = default_flags()
    with tf.Graph().as_default():
        pair_batch, test_1_batch, test_2_batch = input_pipeline(filepath=TEST_PATH,
                                                                batch_size=1,
                                                                num_epochs=1)
        print(type(pair_batch), type(test_1_batch), type(test_2_batch))
        siamese = Siamese(sequence_length,
                          vocab_size=len(vocab_processor.vocabulary_),
                          embedding_size=FLAGS.embedding_dim,
                          filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                          num_filters=FLAGS.num_filters,
                          margin=1.0,
                          threshold=1.5,
                          fully=True,
                          hash_size=128)

        init_op = tf.initialize_all_variables()
        init_again = tf.initialize_local_variables()

        predictions = {}
        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(init_again)
            saver = tf.train.Saver()
            save_path = saver.save(sess, "models/model.ckpt")

            saver.restore(sess, save_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            test_batch = 0

            try:
                with open('predictions_aux.csv', 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['test_id', 'is_duplicate'])

                    while not coord.should_stop():
                        test_1, test_2, test_id = sess.run([test_1_batch, test_2_batch, pair_batch])
                        # print(a, type(a), a[0], type(a[0]))
                        loss, attraction, repulsion, dis, pr = \
                             sess.run([siamese.loss, siamese.attraction_loss,
                                       siamese.repulsion_loss, siamese.distance, siamese.predictions],
                                      feed_dict={
                                          siamese.left_input: test_1,
                                          siamese.right_input: test_2,
                                          siamese.is_training: False})
                        # print("(Test #{0: <5}) - Prediction: {5} - "
                        #       "Loss: {1:.4f} - "
                        #       "(a: {2:.3f} - r: {3:.3f} - "
                        #        "d: {4:.4f})".format(test_batch, loss, np.mean(attraction),
                        #                             np.mean(repulsion), np.mean(dis), pr))
                        test_batch += 1
                        print(test_batch)
                        predictions[test_id[0]] = pr[0]
                        csv_writer.writerow([test_id[0], pr[0]])
                        # print("{}\t{}\n".format(list(sess.run([pair_batch])[0])[0], int(pr[0])))

            except tf.errors.OutOfRangeError:
                print("Done testing!")
            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()
