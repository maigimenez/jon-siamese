import numpy as np
import tensorflow as tf
from os.path import join, abspath, curdir, exists
from time import time
from collections import Counter

from corpus import Corpus
from utils import build_vocabulary, shuffle_epochs, batch_iter_mixed, batch_iter
from siamese import Siamese


def from_text_index(corpus, vocab_processor):
    for data in corpus.non_sim_data:
        data.tweet_a = np.array(list(vocab_processor.transform([data.tweet_a])))[0]
        data.tweet_b = np.array(list(vocab_processor.transform([data.tweet_b])))[0]
    for data in corpus.sim_data:
        data.tweet_a = np.array(list(vocab_processor.transform([data.tweet_a])))[0]
        data.tweet_b = np.array(list(vocab_processor.transform([data.tweet_b])))[0]


def load_similarity_corpus():
    TRAIN_SIM_PATH = '/home/mgimenez/Dev/projects/jon/dataset/isis_tr_sim.csv'
    TRAIN_NOT_SIM_PATH = '/home/mgimenez/Dev/projects/jon/dataset/isis_tr_notsim.csv'
    train_corpus = Corpus('similarity', TRAIN_SIM_PATH, TRAIN_NOT_SIM_PATH)

    VAL_SIM_PATH = '/home/mgimenez/Dev/projects/jon/dataset/isis_val_sim.csv'
    VAL_NOT_SIM_PATH = '/home/mgimenez/Dev/projects/jon/dataset/isis_val_notsim.csv'
    val_corpus = Corpus('similarity', VAL_SIM_PATH, VAL_NOT_SIM_PATH)

    TEST_SIM_PATH = '/home/mgimenez/Dev/projects/jon/dataset/isis_test_sim.csv'
    TEST_NOT_SIM_PATH = '/home/mgimenez/Dev/projects/jon/dataset/isis_test_notsim.csv'
    test_corpus = Corpus('similarity', TEST_SIM_PATH, TEST_NOT_SIM_PATH)

    # Build the vocabulary
    vocab_processor, sequence_length = build_vocabulary(train_corpus)

    # Convert the words to the lookup index
    from_text_index(train_corpus, vocab_processor)
    from_text_index(test_corpus, vocab_processor)
    from_text_index(val_corpus, vocab_processor)

    return train_corpus, val_corpus, test_corpus, sequence_length, vocab_processor


def default_flags():
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # GPU related configuration
    FLAGS.allow_soft_placement = True
    FLAGS.log_device_placement = False

    # Hyperparameters
    FLAGS.embedding_dim = 300
    FLAGS.filter_sizes = '3,4,5'
    FLAGS.num_filters = 100
    FLAGS.num_epochs = 10

    # Training parameters
    FLAGS.batch_size = 100
    FLAGS.num_epochs = 1
    FLAGS.evaluate_every = 1
    FLAGS.checkpoint_every = 1
    FLAGS.shuffle_epochs = True
    FLAGS.mixed = False

    print("Default parameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print(" - {}={}".format(attr.upper(), value))
    print("")

    return FLAGS


def build_data_feed(corpus):
    right_sentences = []
    left_sentences = []
    labels = []

    sim_data = np.array(corpus.sim_data)
    for sample in sim_data:
        right_sentences.append(sample.tweet_a)
        left_sentences.append(sample.tweet_b)
        labels.append(sample.label)

    non_sim_data = np.array(corpus.non_sim_data)
    for sample in non_sim_data:
        right_sentences.append(sample.tweet_a)
        left_sentences.append(sample.tweet_b)
        labels.append(sample.label)

    return right_sentences, left_sentences, labels


def test_step(sess, siamese, test_corpus, merged):
    """
    Evaluates model on a dev set
    """
    test_left_sentences, test_right_sentences, test_sim_labels = build_data_feed(test_corpus)
    feed_dict = {siamese.left_input: test_left_sentences,
                 siamese.right_input: test_right_sentences,
                 siamese.label: test_sim_labels}
    loss, distance, accuracy, summary_str = sess.run([siamese.loss, siamese.distance, siamese.accuracy, merged],
                                                     feed_dict)

    print("--> TEST")
    print("\t     Loss: {0:.4f} (d: {1:.4f}) accuracy: {2:.4f}\n".format(loss, np.mean(distance), accuracy))


def dev_step(sess, siamese, val_left_sentences, val_right_sentences, val_sim_labels, merged, epoch):
    # EVALUATE
    loss, d, accuracy, summary_str = sess.run([siamese.loss, siamese.distance, siamese.accuracy, merged],
                                              feed_dict={siamese.left_input: val_left_sentences,
                                                         siamese.right_input: val_right_sentences,
                                                         siamese.label: val_sim_labels})
    print("--> (VAL epoch #{0})".format(epoch))
    print("\t     Loss: {0:.4f} (d: {1:.4f}) Accuracy: {2:.4f}\n".format(loss, np.mean(d), accuracy))


def train_siamese(margin, threshold, sequence_length, vocab_processor, train_corpus, val_corpus, test_corpus, config_flags=None):
    if not config_flags:
        FLAGS = default_flags()
    else:
        FLAGS = config_flags

    val_left_sentences, val_right_sentences, val_sim_labels = build_data_feed(val_corpus)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            siamese = Siamese(sequence_length,
                              vocab_size=len(vocab_processor.vocabulary_),
                              embedding_size=FLAGS.embedding_dim,
                              filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                              num_filters=FLAGS.num_filters,
                              margin=margin,
                              threshold=threshold)

            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       100000, 0.96, staircase=True)

            train_step = tf.train.MomentumOptimizer(0.0001, 0.95, use_nesterov=True).minimize(siamese.loss,
                                                                                              global_step=global_step)

            print()
            sess.run(tf.global_variables_initializer())

            # setup tensorboard
            # Output directory for models and summaries
            timestamp = str(int(time()))
            out_dir = abspath(join(curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for trainable variables
            print('Trainable variables')
            vars_summary = []
            for var in tf.trainable_variables():
                print(' - {}'.format(var.name))
                # tf.summary.histogram(var.op.name, var)
                vars_summary.append(tf.summary.histogram("{}/grad/hist".format(var.op.name), var))
            vars_summaries_merged = tf.summary.merge(vars_summary)

            step = tf.summary.scalar('step', global_step)
            loss_summary = tf.summary.scalar('loss', siamese.loss)
            atraction_summary = tf.summary.histogram('attraction_loss', siamese.attraction_loss)
            merged = tf.summary.merge([loss_summary, atraction_summary, vars_summaries_merged])
            # merged = tf.summary.merge_all()
            # tf.summary.scalar('within_loss', siamese.within_loss)
            print()

            writer = tf.summary.FileWriter('train.log', sess.graph)
            print()

            sim_data = np.array(train_corpus.sim_data)
            non_sim_data = np.array(train_corpus.non_sim_data)
            data_size = len(sim_data) + len(non_sim_data)
            num_batches_per_epoch = int(data_size / FLAGS.batch_size) + 1
            print("Num batches per epoch: {} ({})\n".format(num_batches_per_epoch, data_size))

            for epoch in range(FLAGS.num_epochs):
                print("-------------------------------- EPOCH {} ---------------------------".format(epoch))

                # Prepare the batches
                if FLAGS.mixed and FLAGS.shuffle_epochs:
                    shuffle_data = shuffle_epochs(sim_data, non_sim_data, mixed=True)
                    batches = batch_iter_mixed(shuffle_data, FLAGS.batch_size, num_batches_per_epoch)
                elif FLAGS.mixed and not FLAGS.shuffle_epochs:
                    data = np.append(sim_data, non_sim_data)
                    batches = batch_iter_mixed(data, FLAGS.batch_size, num_batches_per_epoch)
                elif not FLAGS.mixed and FLAGS.shuffle_epochs:
                    shuffled_sim_data, shuffled_non_sim_data = shuffle_epochs(sim_data, non_sim_data)
                    batches = batch_iter(shuffled_sim_data, shuffled_non_sim_data, FLAGS.batch_size,
                                         num_batches_per_epoch)
                else:
                    batches = batch_iter(sim_data, non_sim_data, FLAGS.batch_size, num_batches_per_epoch)

                # TRAIN A BATCH
                sim_distances, non_sim_distances = [], []
                for cur_batch, batch in enumerate(batches):
                    batch_data, batch_type = batch[0], batch[1]
                    right_sentences = [sample.tweet_a for sample in batch_data]
                    left_sentences = [sample.tweet_b for sample in batch_data]
                    sim_labels = [sample.label for sample in batch_data]
                    # print(Counter(sim_labels))
                    assert len(right_sentences) == len(left_sentences) == len(sim_labels)
                    _, loss, attraction, repulsion, d, accuracy, predictions, correct, summary_str = sess.run(
                        [train_step, siamese.loss,
                         siamese.attraction_loss, siamese.repulsion_loss,
                         siamese.distance, siamese.accuracy,
                         siamese.predictions,
                         siamese.correct_predictions, merged],
                        feed_dict={siamese.left_input: left_sentences,
                                   siamese.right_input: right_sentences,
                                   siamese.label: sim_labels})
                    writer.add_summary(summary_str, epoch)
                    print("(#{0: <7}) - Loss: {1:.4f} (a: {2:.4f} - r: {3:.4f}"
                          "- d: {4:.4f}, accuracy:{5:.4f})".format(batch_type, loss,
                                                                   np.mean(attraction),
                                                                   np.mean(repulsion),
                                                                   np.mean(d),
                                                                   accuracy))
                    # print(predictions)
                    # print(sim_labels)
                    # print(correct)
                    if batch_type == 'SIM':
                        sim_distances.append(d)
                    else:
                        non_sim_distances.append(d)
                print('---------------------> sim: {} -  non sim: {}'.format(np.array(sim_distances).mean(),
                                                                             np.array(non_sim_distances).mean()))

                dev_step(sess, siamese, val_left_sentences, val_right_sentences, val_sim_labels, merged, epoch)
            del val_left_sentences, val_right_sentences, val_sim_labels, val_corpus
            del train_corpus

            test_step(sess, siamese, test_corpus, merged)


def simmilarty_experiment(margin=1.5, threshold=1.0, config_flags=None):
    train, val, test, sequence_len, vocab_processor = load_similarity_corpus()
    train_siamese(margin, threshold, sequence_len, vocab_processor, train, val, test, config_flags)

if __name__ == "__main__":
    simmilarty_experiment()