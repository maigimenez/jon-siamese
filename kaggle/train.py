import tensorflow as tf
import numpy as np
import pickle
from time import time
from os.path import join, abspath, curdir

from siamese_pr import Siamese


def default_flags():
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # GPU related configuration
    FLAGS.allow_soft_placement = True
    FLAGS.log_device_placement = False

    # Hyperparameters
    FLAGS.embedding_dim = 300
    FLAGS.filter_sizes = '3,4,5,7'
    FLAGS.num_filters = 50

    # Training parameters
    FLAGS.batch_size = 100
    FLAGS.num_epochs = 10
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
        features={'label': tf.FixedLenFeature([], tf.int64),
                  'sentence_1': tf.FixedLenFeature([sequence_length], tf.int64),
                  'sentence_2': tf.FixedLenFeature([sequence_length], tf.int64)})
    return features['label'], features['sentence_1'], features['sentence_2']


def input_pipeline(filepath, batch_size, num_epochs=None):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filepath], num_epochs=num_epochs)
        label, sentence_1, sentence_2 = read_sample(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in four threads to avoid being a bottleneck.
        label_batch, sentences_1_batch, sentences_2_batch = tf.train.shuffle_batch([label, sentence_1, sentence_2],
                                                                                   batch_size=batch_size,
                                                                                    num_threads=4,
                                                                                    capacity=1000 + 3 * FLAGS.batch_size,
                                                                                    # Ensures a minimum amount of shuffling of examples.
                                                                                    min_after_dequeue=1000)
        return label_batch, sentences_1_batch, sentences_2_batch


if __name__ == "__main__":
    RECORDS_PATH = '/home/mgimenez/Dev/projects/jon-siamese/kaggle/bin_data/preprocess60'
    TRAIN_PATH = join(RECORDS_PATH, 'train.tfrecords')

    vocab_processor_path = join(RECORDS_PATH, 'vocab.train')
    vocab_processor = load_binarize_data(vocab_processor_path)
    sequence_length_path = join(RECORDS_PATH, 'sequence_length')
    sequence_length = load_binarize_data(sequence_length_path)

    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    FLAGS = default_flags()

    with tf.Graph().as_default():
        label_batch, sentences_1_batch, sentences_2_batch = input_pipeline(filepath=TRAIN_PATH,
                                                                           batch_size=FLAGS.batch_size,
                                                                           num_epochs=FLAGS.num_epochs)
        print(type(label_batch), type(sentences_1_batch), type(sentences_2_batch))

        siamese = Siamese(sequence_length,
                          vocab_size=len(vocab_processor.vocabulary_),
                          embedding_size=FLAGS.embedding_dim,
                          filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                          num_filters=FLAGS.num_filters,
                          margin=1.0,
                          threshold=1.5,
                          hash_size=None)
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   100000, 0.96, staircase=True)
        # train_op = tf.train.MomentumOptimizer(0.0001, 0.95, use_nesterov=True).minimize(siamese.loss,
        #                                                                                 global_step=global_step)
        train_op = tf.train.MomentumOptimizer(0.0001, 0.95, use_nesterov=True).minimize(siamese.loss,
                                                                                        global_step=global_step)
        init_op = tf.initialize_all_variables()
        init_again = tf.initialize_local_variables()

        # sess = tf.Session()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(init_again)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            step = 0
            try:
                while not coord.should_stop():
                    # TRAIN CLASSIFICATION
                    # _, loss, attraction, repulsion, dis, acc = \
                    #     sess.run([train_op, siamese.loss, siamese.attraction_loss,
                    #               siamese.repulsion_loss, siamese.distance,
                    #               siamese.accuracy],
                    #              feed_dict={
                    #                  siamese.left_input: sess.run(sentences_1_batch),
                    #                  siamese.right_input: sess.run(sentences_2_batch),
                    #                  siamese.label: sess.run(label_batch),
                    #                  siamese.is_training: True})
                    #
                    # print("(#{0: <5} - {6}) - Loss: {1:.4f} - "
                    #       "(a: {2:.3f} - r: {3:.3f} - "
                    #       "d: {4:.4f}, accuracy:{5:.4f})".format(sess.run(global_step),
                    #                                              loss, np.mean(attraction),
                    #                                              np.mean(repulsion),
                    #                                              np.mean(dis), acc,
                    #                                              np.mean(sess.run(label_batch))))


                    # TRAIN WITH PROBABILITIES
                    _, loss, acc = \
                        sess.run([train_op, siamese.loss, siamese.accuracy],
                                 feed_dict={
                                     siamese.left_input: sess.run(sentences_1_batch),
                                     siamese.right_input: sess.run(sentences_2_batch),
                                     siamese.label: sess.run(label_batch),
                                     siamese.is_training: True})

                    print("(#{0: <5} - {3}) - Loss: {1:.4f} - "
                          "Accuracy:{2:.4f}".format(sess.run(global_step), loss, acc,
                                                    np.mean(sess.run(label_batch))))

                    #             step += 1


            except tf.errors.OutOfRangeError:
                print("Done training!")
            finally:
                coord.request_stop()

            coord.join(threads)
            save_path = saver.save(sess, "model.ckpt")
            print("Model saved in file: %s" % save_path)
