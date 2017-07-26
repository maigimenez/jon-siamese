import tensorflow as tf
import pickle
from os.path import join, abspath, curdir
import matplotlib.pyplot as plt

import argparse
from utils import read_flags
from siamese import Siamese
from os import makedirs

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
                                                                                  capacity=1000 + 3 * batch_size,
                                                                                  min_after_dequeue=1000)
    return pair_batch, sentences_1_batch, sentences_2_batch


def test_model(tf_path, model_path, flags_path):

    # Import the parameters binarized
    test_tfrecors = join(tf_path, 'test.tfrecords')
    vocab_processor_path = join(tf_path, 'vocab.train')
    vocab_processor = load_binarize_data(vocab_processor_path)
    sequence_length_path = join(tf_path, 'sequence.len')
    seq_len = load_binarize_data(sequence_length_path)
    FLAGS = read_flags(flags_path)
    # TODO Remove this from the siamese class
    fully_layer = True if FLAGS.hash_size else False
    # TODO this is a parameter
    one_hot = False
    n_labels = 2 if one_hot else 1


    # TEST THE SYSTEM
    with tf.Graph().as_default():

        label_batch, test_1_batch, test_2_batch = input_pipeline(filepath=test_tfrecors,
                                                                 batch_size=1,
                                                                 num_labels=n_labels,
                                                                 sequence_len=seq_len,
                                                                 num_epochs=1)

        print(type(label_batch), type(test_1_batch), type(test_2_batch))
        siamese = Siamese(sequence_length=seq_len,
                          vocab_size=len(vocab_processor.vocabulary_),
                          embedding_size=FLAGS.embedding_dim,
                          filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                          num_filters=FLAGS.num_filters,
                          margin=FLAGS.margin)

        init_op = tf.global_variables_initializer()
        init_again = tf.local_variables_initializer()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Initialize variables
            sess.run(init_op)
            sess.run(init_again)

            # Restore the model
            saver.restore(sess, join(model_path, "model.ckpt"))

            # Create the coordinators to read the test data
            coord = tf.train.Coordinator()

            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            test_sample, hits = 0, 0

            try:
                while not coord.should_stop():
                    test_1, test_2, test_label = sess.run([test_1_batch, test_2_batch, label_batch])
                    # TEST CLASSIFICATION
                    loss, attraction, repulsion, dis, acc = \
                        sess.run([siamese.loss, siamese.attr,
                                  siamese.rep, siamese.distance,
                                  siamese.accuracy],
                                 feed_dict={
                                     siamese.left_input: test_1,
                                     siamese.right_input: test_2,
                                     siamese.labels: test_label,
                                     siamese.is_training: False
                                 })

                    with open(join(model_path, 'test.log'), 'a') as log_file:
                        log_str = "(#{0: <5} - {6}) - Loss: {1:.4f} - " \
                                  "(a: {2:.3f} - r: {3:.3f} - " \
                                  "d: {4:.4f}, accuracy:{5:.4f})\n"
                        log_file.write(log_str.format(test_sample, loss,
                                                      attraction[0][0],
                                                      repulsion[0][0],
                                                      dis[0], acc,
                                                      test_label[0][0], ))

                    with open(join(model_path, 'distances.log'), 'a') as log_file:
                        log_str = "{}\t{}\n"
                        log_file.write(log_str.format(dis[0],
                                                      test_label[0][0]))
                    test_sample += 1
                    if acc == 1:
                        hits += 1
            except tf.errors.OutOfRangeError:
                print("Done testing!")
            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()

            with open(join(model_path, 'results.txt'), 'w') as results_file:
                results_file.write("Accuracy: {} ({}/{})".format(hits / test_sample, hits, test_sample))

            print("Results saved in: {}".format(join(model_path, 'results.txt')))

def best_score(threshold, dissimilar, similar):
    hits = sum([1 for s in dissimilar if s > threshold]) + sum([1 for s in similar if s <= threshold])
    return hits/(len(similar)+len(dissimilar))

def plot_distances(model_path):
    similar, dissimilar = [], []
    with open(join(model_path, 'distances.log')) as dist_file:
        for line in dist_file:
            dist, tag = line.strip().split('\t')
            if tag == '1':
                similar.append(float(dist))
            else:
                dissimilar.append(float(dist))

    plt.hist(similar, color='r', alpha=0.5, label='Similar')
    plt.hist(dissimilar, color='b', alpha=0.5, label='Dissimilar')
    plt.title("Siamese distances in test")
    plt.xlabel("Distances")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(join(model_path, 'test.pdf'))

    scores = {}
    for i in range(int(min(similar)), int(max(dissimilar))):
        for j in range(10):
            decimal = j/10
            scores[i+decimal] = best_score(i+decimal, dissimilar, similar)

    best_accuracy, best_threshold = -1, None
    for k, v in scores.items():
        if v > best_accuracy:
            best_accuracy = v
            best_threshold = k
    worst_accuracy = min(scores.values())

    with open(join(model_path, 'results.txt'), 'a') as results_file:
        log_str = "\nThe best accuracy is {} with threshold {}. And the worst {}"
        results_file.write(log_str.format(best_accuracy, best_threshold, worst_accuracy))


if __name__ == "__main__":
    tf_path, model_path, flags_path = get_arguments()
    test_model(tf_path, model_path, flags_path)
    plot_distances(model_path)
