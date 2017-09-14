import tensorflow as tf
import numpy as np
from os.path import join, abspath, curdir
from os import makedirs
from time import time

from siamese import Siamese
from double_siamese import DoubleSiamese
from utils import shuffle_epochs, batch_iter, get_dev_data, load_binarize_data


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
                                                   capacity=100 + 3 * batch_size,
                                                   min_after_dequeue=100)
        return label_batch, sentences_1_batch, sentences_2_batch

def train_double_siamese(tf_path, flags, num_epochs, out_dir=None, init_embeddings=None):
    tf.logging.set_verbosity(tf.logging.INFO)

    # Create the directory where the training will be saved
    if not out_dir:
        timestamp = str(int(time()))
        out_dir = abspath(join(curdir, "models", timestamp))
        makedirs(out_dir, exist_ok=True)

    # Load the records
    train_sim_path = join(tf_path, 'train_sim.tfrecords')
    train_dis_path = join(tf_path, 'train_dis.tfrecords')
    vocab_processor_path = join(tf_path, 'vocab.train')
    vocab_processor = load_binarize_data(vocab_processor_path)
    sequence_length_path = join(tf_path, 'sequence.len')
    seq_len = load_binarize_data(sequence_length_path)

    n_labels = 1

    with tf.Graph().as_default():
        # Get similar sentences batch
        slabel_batch, s1_batch, s2_batch = input_pipeline(filepath=train_sim_path,
                                                          batch_size=flags.batch_size,
                                                          num_labels=n_labels,
                                                          sequence_len=seq_len,
                                                          num_epochs=num_epochs)
        # Get non-similar sentences batch
        dlabel_batch, d1_batch, d2_batch = input_pipeline(filepath=train_dis_path,
                                                          batch_size=flags.batch_size,
                                                          num_labels=n_labels,
                                                          sequence_len=seq_len,
                                                          num_epochs=num_epochs)
        double_siam = DoubleSiamese(sequence_length=seq_len,
                                    vocab_size=len(vocab_processor.vocabulary_),
                                    embedding_size=flags.embedding_dim,
                                    filter_sizes=list(map(int, flags.filter_sizes.split(","))),
                                    num_filters=flags.num_filters,
                                    margin=flags.margin)

        global_step = tf.Variable(0, trainable=False)

        train_op = tf.train.MomentumOptimizer(0.01, 0.5, use_nesterov=True)
        train_op = train_op.minimize(double_siam.loss, global_step=global_step)

        init_op = tf.global_variables_initializer()
        init_again = tf.local_variables_initializer()

        saver = tf.train.Saver()
        session_conf = tf.ConfigProto(allow_soft_placement=flags.allow_soft_placement,
                                      log_device_placement=flags.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default() as sess:
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
                    slabels, s1, s2 = sess.run([slabel_batch, s1_batch, s2_batch])
                    dlabels, d1, d2 = sess.run([dlabel_batch, d1_batch, d2_batch])
                    current_step = tf.train.global_step(sess, global_step)
                    _, loss = sess.run([train_op, double_siam.loss],
                                       feed_dict={
                                           double_siam.sim_branch.left_input: s1,
                                           double_siam.sim_branch.right_input: s2,
                                           double_siam.sim_branch.labels: slabels,
                                           double_siam.disim_branch.left_input: d1,
                                           double_siam.disim_branch.right_input: d2,
                                           double_siam.disim_branch.labels: dlabels,
                                           double_siam.sim_branch.is_training: True,
                                           double_siam.disim_branch.is_training: True
                         })


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
                for attr, value in sorted(flags.__flags.items()):
                    param_file.write(" - {}={}\n".format(attr.upper(), value))

            save_path = saver.save(sess, join(out_dir, "model.ckpt"))
            print("Model saved in file: {}".format(save_path))
            return out_dir


def train_siamese_fromtf(tf_path, flags, num_epochs, out_dir=None, one_hot=False,
                         verbose=False, init_embeddings=None):
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

    # TODO Remove this from the siamese class
    n_labels = 2 if one_hot else 1
    print('--------', n_labels)

    with tf.Graph().as_default():

        label_batch, sentences_1_batch, sentences_2_batch = input_pipeline(filepath=train_path,
                                                                           batch_size=flags.batch_size,
                                                                           num_labels=n_labels,
                                                                           sequence_len=seq_len,
                                                                           num_epochs=num_epochs)
        siamese = Siamese(sequence_length=seq_len,
                          vocab_size=len(vocab_processor.vocabulary_),
                          embedding_size=flags.embedding_dim,
                          filter_sizes=list(map(int, flags.filter_sizes.split(","))),
                          num_filters=flags.num_filters,
                          margin=flags.margin)

        global_step = tf.Variable(0, trainable=False)

        # learning_rate = tf.placeholder(tf.float32, shape=[])
        # train_op = tf.train.GradientDescentOptimizer(
        #     learning_rate=learning_rate).minimize(siamese.loss)

        # optimizer = tf.train.AdamOptimizer(0.2)
        # grads_and_vars = optimizer.compute_gradients(siamese.loss)
        # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        starter_learning_rate = 0.01
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   1000000, 0.95, staircase=False)
        train_op = tf.train.MomentumOptimizer(learning_rate, 0.5, use_nesterov=True)

        # train_op = tf.train.MomentumOptimizer(0.01, 0.5, use_nesterov=True)
        train_op = train_op.minimize(siamese.loss, global_step=global_step)

        init_op = tf.global_variables_initializer()
        init_again = tf.local_variables_initializer()

        saver = tf.train.Saver()
        session_conf = tf.ConfigProto(allow_soft_placement=flags.allow_soft_placement,
                                      log_device_placement=flags.log_device_placement)

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

            # Load embeddings
            if init_embeddings is not None:
                sess.run(siamese.W_embedding.assign(init_embeddings))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            try:
                while not coord.should_stop():
                    labels, s1, s2 = sess.run([label_batch, sentences_1_batch, sentences_2_batch])
                    current_step = tf.train.global_step(sess, global_step)
                    if verbose:
                        train_step_verbose(sess, train_op, summary_op, summary_writer,
                                           siamese, s1, s2, labels, current_step)

                    else:
                        train_step(sess, train_op, siamese, s1, s2, labels, out_dir, current_step)

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
                for attr, value in sorted(flags.__flags.items()):
                    param_file.write(" - {}={}\n".format(attr.upper(), value))

            save_path = saver.save(sess, join(out_dir, "model.ckpt"))
            print("Model saved in file: {}".format(save_path))
            return out_dir


def train_step_verbose(sess, train_op, summary_op, summary_writer,
                       siamese, s1, s2, labels, current_step):
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


def train_step(sess, train_op, siamese, s1, s2, labels, out_dir, current_step):
    _, loss, attraction, repulsion, distance, accuracy = \
        sess.run([train_op,
                  siamese.loss, siamese.attr,
                  siamese.rep, siamese.distance,
                  siamese.accuracy],
                 feed_dict={
                     siamese.left_input: s1,
                     siamese.right_input: s2,
                     siamese.labels: labels,
                     # learning_rate: learn_rate,
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

