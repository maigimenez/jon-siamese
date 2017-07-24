import tensorflow as tf
import numpy as np
from os.path import join, abspath, curdir
import pickle
from os import makedirs
from time import time

from siamese import Siamese
from utils import shuffle_epochs, batch_iter, get_dev_data, read_flags


def parse_function(example_tf, num_labels=1, sequence_len=2):
    features = {'label': tf.FixedLenFeature([num_labels], tf.int64),
                'sentence_1': tf.FixedLenFeature([sequence_len], tf.int64),
                'sentence_2': tf.FixedLenFeature([sequence_len], tf.int64)}

    parsed_features = tf.parse_single_example(example_tf, features)

    return parsed_features['label'], parsed_features['sentence_1'], parsed_features['sentence_2']

def read_sample(filename_queue, num_labels, sequence_len):
    reader = tf.TFRecordReader()
    _, serialized_sample = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_sample,
        features={'label': tf.FixedLenFeature([num_labels], tf.int64),
                  'sentence_1': tf.FixedLenFeature([sequence_len], tf.int64),
                  'sentence_2': tf.FixedLenFeature([sequence_len], tf.int64)})
    return features['label'], features['sentence_1'], features['sentence_2']


def input_pipeline(filepath, batch_size, num_labels, sequence_len, num_epochs=None):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filepath], num_epochs=num_epochs)
        label, sentence_1, sentence_2 = read_sample(filename_queue, num_labels, sequence_len)

        # Shuffle the examples and collect them into batch_size batches.
        label_batch, \
        sentences_1_batch, \
        sentences_2_batch = tf.train.shuffle_batch([label, sentence_1, sentence_2],
                                                   batch_size=batch_size,
                                                   num_threads=4,
                                                   capacity=1000 + 3 * batch_size,
                                                   min_after_dequeue=1000)
        return label_batch, sentences_1_batch, sentences_2_batch

def load_binarize_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_all_records(filepath,  num_labels, sequence_len):
    """ Read all the records from a file with tfrecords """
    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer([filepath], num_epochs=1)
        label, sentence_1, sentence_2 = read_sample(filename_queue, num_labels, sequence_len)
        init_op = tf.global_variables_initializer()
        init_again = tf.local_variables_initializer()
        sess.run(init_op)
        sess.run(init_again)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        dev_lab, dev_1, dev_2 = [], [], []
        try:
            while not coord.should_stop():
                aux_label, aux_s1, aux_s2 = sess.run([label, sentence_1, sentence_2])
                dev_lab.append(aux_label)
                dev_1.append(aux_s1)
                dev_2.append(aux_s2)
        except tf.errors.OutOfRangeError:
            print('Done reading the dev dataset')
        finally:
            coord.request_stop()
        coord.join(threads)

        return dev_lab, dev_1, dev_2


def train_siamese_fromtf(tf_path, config_flags, out_dir=None, one_hot=False, verbose=False):
    """ Train a Siamese NN using a tfrecords as an input"""

    tf.logging.set_verbosity(tf.logging.INFO)

    # Create the directory where the training will be saved
    if not out_dir:
        timestamp = str(int(time()))
        out_dir = abspath(join(curdir, "models", timestamp))
        makedirs(out_dir, exist_ok=True)

    # Load the records
    train_path = join(tf_path, 'train.tfrecords')
    vocab_processor_path = join(tf_path, 'vocab.train')
    vocab_processor = load_binarize_data(vocab_processor_path)
    sequence_length_path = join(tf_path, 'sequence.len')
    seq_len = load_binarize_data(sequence_length_path)

    # Read the configuration flags
    FLAGS = read_flags(config_flags)
    # TODO Remove this from the siamese class
    n_labels = 2 if one_hot else 1
    print('--------', n_labels)

    with tf.Graph().as_default():

        label_batch, sentences_1_batch, sentences_2_batch = input_pipeline(filepath=train_path,
                                                                           batch_size=FLAGS.batch_size,
                                                                           num_labels=n_labels,
                                                                           sequence_len=seq_len,
                                                                           num_epochs=FLAGS.num_epochs)
        siamese = Siamese(sequence_length=seq_len,
                          vocab_size=len(vocab_processor.vocabulary_),
                          embedding_size=FLAGS.embedding_dim,
                          filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                          num_filters=FLAGS.num_filters,
                          margin=FLAGS.margin)

        global_step = tf.Variable(0, trainable=False)

        # learning_rate = tf.placeholder(tf.float32, shape=[])
        # train_op = tf.train.GradientDescentOptimizer(
        #     learning_rate=learning_rate).minimize(siamese.loss)

        # optimizer = tf.train.AdamOptimizer(0.2)
        # grads_and_vars = optimizer.compute_gradients(siamese.loss)
        # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # starter_learning_rate = 0.1
        # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
        #                                            100000, 0.96, staircase=True)

        train_op = tf.train.MomentumOptimizer(0.01, 0.5, use_nesterov=True)
        train_op = train_op.minimize(siamese.loss, global_step=global_step)

        init_op = tf.global_variables_initializer()
        init_again = tf.local_variables_initializer()

        saver = tf.train.Saver()
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)

        sess = tf.Session(config=session_conf)
        with sess.as_default() as sess:
            if verbose:
                tf.summary.histogram('embedding', siamese.W_embedding)
                tf.summary.histogram('tensor_left', siamese.left_siamese)
                # tf.summary.histogram('tensor_left_z', tf.nn.zero_fraction(siamese.left_siamese))
                tf.summary.histogram('tensor_right', siamese.right_siamese)
                # tf.summary.histogram('tensor_right_z', tf.nn.zero_fraction(siamese.right_siamese))
                tf.summary.histogram('distance', siamese.distance)

                tf.summary.scalar('loss', siamese.loss)
                tf.summary.scalar('distance', siamese.distance[0])
                tf.summary.scalar('attraction', siamese.attr[0][0])
                tf.summary.scalar('repulsion', siamese.rep[0][0])

                summary_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter('./train', sess.graph)

            sess.run(init_op)
            sess.run(init_again)

            # Show which variables are going to be train
            variables_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            for k, v in zip(variables_names, values):
                print("Variable: ", k, "- Shape: ", v.shape)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            try:
                while not coord.should_stop():
                    # print('--------------------------------------------------------------')
                    labels, s1, s2 = sess.run([label_batch, sentences_1_batch, sentences_2_batch])
                    current_step = tf.train.global_step(sess, global_step)
                    if verbose:
                        _, summaries, loss, attraction, repulsion, distance, l, r, W, mp, le, re = \
                            sess.run([train_op, summary_op,
                                      siamese.loss, siamese.attr,
                                      siamese.rep, siamese.distance,
                                      siamese.left_siamese, siamese.right_siamese,
                                      siamese.W_embedding,
                                      siamese.left_embedded, siamese.right_embedded],
                                     feed_dict={
                                         siamese.left_input: s1,
                                         siamese.right_input: s2,
                                         siamese.labels: labels,
                                         # learning_rate: 0.01
                                         siamese.is_training: True
                                     })
                        summary_writer.add_summary(summaries, global_step=current_step)

                    else:
                        _, loss, attraction, repulsion, distance = \
                            sess.run([train_op,
                                      siamese.loss, siamese.attr,
                                      siamese.rep, siamese.distance],
                                     feed_dict={
                                         siamese.left_input: s1,
                                         siamese.right_input: s2,
                                         siamese.labels: labels,
                                         # learning_rate: 0.01
                                         siamese.is_training: True
                                     })

                        with open(join(out_dir, 'train.log'), 'a') as log_file:
                            log_str = "(#{0: <5} - {1}) - Loss: {2:.4f} - " \
                                      "(a: {3:.3f} - r: {4:.3f} - " \
                                      "d: {5:.4f}) \n"
                            log_file.write(log_str.format(current_step,
                                                          np.mean(labels),
                                                          loss,
                                                          np.mean(attraction),
                                                          np.mean(repulsion),
                                                          np.mean(distance)))

            except tf.errors.OutOfRangeError:
                print("Done training!")
            finally:
                coord.request_stop()

            coord.join(threads)

            # Save the model
            if not out_dir:
                timestamp = str(int(time()))
                out_dir = abspath(join(curdir, "models", timestamp))
                makedirs(out_dir, exist_ok=True)

            with open(join(out_dir, 'parameters.txt'), 'w') as param_file:
                param_file.write("Default parameters: \n")
                for attr, value in sorted(FLAGS.__flags.items()):
                    param_file.write(" - {}={}\n".format(attr.upper(), value))

            save_path = saver.save(sess, join(out_dir, "model.ckpt"))
            print("Model saved in file: {}".format(save_path))
            return out_dir


def train_strep(sess, train_op, siamese, s1, s2, labels, learning_rate,
                learn_rate, out_dir, current_step):
    _, loss, attraction, repulsion, distance, accuracy = \
        sess.run([train_op,
                  siamese.loss, siamese.attr,
                  siamese.rep, siamese.distance,
                  siamese.accuracy],
                 feed_dict={
                     siamese.left_input: s1,
                     siamese.right_input: s2,
                     siamese.labels: labels,
                     learning_rate: learn_rate,
                     siamese.is_training: True
                 })

    with open(join(out_dir, 'train.log'), 'a') as log_file:
        log_str = "(#{0: <5} - {1}) - Loss: {2:.4f} - " \
                  "(a: {3:.3f} - r: {4:.3f} - " \
                  "d: {5:.4f}, accuracy:{6:.3f}) \n"
        log_file.write(log_str.format(current_step,
                                      np.mean(labels),
                                      loss,
                                      np.mean(attraction),
                                      np.mean(repulsion),
                                      np.mean(distance),
                                      accuracy))


def dev_step(sess, siamese, val_left_sentences, val_right_sentences, val_sim_labels, epoch):
    # EVALUATE
    loss, d, accuracy, summary_str = sess.run([siamese.loss, siamese.distance, siamese.accuracy],
                                              feed_dict={siamese.left_input: val_left_sentences,
                                                         siamese.right_input: val_right_sentences,
                                                         siamese.label: val_sim_labels})
    print("--> (VAL epoch #{0})".format(epoch))
    print("\t     Loss: {0:.4f} (d: {1:.4f}) Accuracy: {2:.4f}\n".format(loss, np.mean(d), accuracy))


def train_siamese(train_non_sim, train_sim, dev_non_sim, dev_sim,
                  vocab_processor, sequence_len, config_flags=None):
    """ Train a siamese NN """
    FLAGS = read_flags(config_flags)
    val_left_sentences, val_right_sentences, val_sim_labels = get_dev_data(dev_sim, dev_non_sim)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)

        sess = tf.Session(config=session_conf)
        # TODO Remove this from the siamese class
        if not FLAGS.hash_size:
            fully_layer = False
        else:
            fully_layer = True

        with sess.as_default():
            print('HASH TRAIN  ----->', FLAGS.hash_size)
            siamese = Siamese(sequence_len,
                              vocab_size=len(vocab_processor.vocabulary_),
                              embedding_size=FLAGS.embedding_dim,
                              filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                              num_filters=FLAGS.num_filters,
                              margin=FLAGS.margin,
                              threshold=FLAGS.threshold,
                              fully=fully_layer,
                              hash_size=FLAGS.hash_size)

            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       100000, 0.96, staircase=True)
            train_step = tf.train.MomentumOptimizer(0.0001, 0.95, use_nesterov=True).minimize(siamese.loss,
                                                                                              global_step=global_step)

            print()
            sess.run(tf.global_variables_initializer())
            data_size = len(train_sim) + len(train_non_sim)
            num_batches_per_epoch = int(data_size / FLAGS.batch_size) + 1
            print("Num batches per epoch: {} ({})\n".format(num_batches_per_epoch, data_size))

            train_sim = np.array(train_sim)
            train_non_sim = np.array(train_non_sim)

            for epoch in range(FLAGS.num_epochs):
                print("-------------------------------- EPOCH {} ---------------------------".format(epoch))
                # Prepare the batches
                if FLAGS.shuffle_epochs:
                    shuffled_sim_data, shuffled_non_sim_data = shuffle_epochs(train_sim, train_non_sim)
                    batches = batch_iter(shuffled_sim_data, shuffled_non_sim_data, FLAGS.batch_size,
                                         num_batches_per_epoch)
                else:
                    batches = batch_iter(train_sim, train_non_sim, FLAGS.batch_size, num_batches_per_epoch)

                # TRAIN A BATCH
                sim_distances, non_sim_distances = [], []
                for cur_batch, batch in enumerate(batches):
                    batch_data, batch_type = batch[0], batch[1]
                    right_sentences = [sample.sentence_1 for sample in batch_data]
                    left_sentences = [sample.sentence_2 for sample in batch_data]
                    sim_labels = [sample.label for sample in batch_data]
                    # print(Counter(sim_labels))
                    # print(len(right_sentences))
                    assert len(right_sentences) == len(left_sentences) == len(sim_labels)

                    _, loss, attraction, repulsion, d, accuracy, predictions, correct = sess.run(
                        [train_step, siamese.loss,
                         siamese.attraction_loss, siamese.repulsion_loss,
                         siamese.distance, siamese.accuracy,
                         siamese.predictions,
                         siamese.correct_predictions],
                        feed_dict={siamese.left_input: left_sentences,
                                   siamese.right_input: right_sentences,
                                   siamese.label: sim_labels})

                    print("(#{0: <7}) - Loss: {1:.4f} (a: {2:.4f} - r: {3:.4f}"
                          "- d: {4:.4f}, accuracy:{5:.4f})".format(batch_type, loss,
                                                                   np.mean(attraction),
                                                                   np.mean(repulsion),
                                                                   np.mean(d),
                                                                   accuracy))
                    if batch_type == 'SIM':
                        sim_distances.append(d)
                    else:
                        non_sim_distances.append(d)
                print('---------------------> sim: {} -  non sim: {}'.format(np.array(sim_distances).mean(),
                                                                             np.array(non_sim_distances).mean()))
                print(len(val_sim_labels))
                dev_step(sess, siamese, val_left_sentences, val_right_sentences, val_sim_labels, epoch)
                print('Working dev step')
