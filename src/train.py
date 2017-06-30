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

def train_siamese_fromtf(tf_path, config_flags, one_hot=False):
    """ Train a Siamese NN using a tfrecords as an input"""
    # Load the records
    train_path = join(tf_path, 'train.tfrecords')
    vocab_processor_path = join(tf_path, 'vocab.train')
    vocab_processor = load_binarize_data(vocab_processor_path)
    sequence_length_path = join(tf_path, 'sequence.len')
    seq_len = load_binarize_data(sequence_length_path)

    # Read the configuration flags
    FLAGS = read_flags(config_flags)
    # TODO Remove this from the siamese class
    fully_layer = True if FLAGS.hash_size else False
    n_labels = 2 if one_hot else 1
    print('--------', n_labels)

    # Load the dev records
    dev_path = join(tf_path, 'dev.tfrecords')
    # dev_labels, dev_s1, dev_s2 = get_all_records(dev_path, n_labels, seq_len)

    with tf.Graph().as_default():

        label_batch, sentences_1_batch, sentences_2_batch = input_pipeline(filepath=train_path,
                                                                           batch_size=FLAGS.batch_size,
                                                                           num_labels=n_labels,
                                                                           sequence_len=seq_len,
                                                                           num_epochs=FLAGS.num_epochs)
        #
        # dev_labels, dev_sentences_1, dev_sentences_2 = input_pipeline(filepath=dev_path,
        #                                                               batch_size=FLAGS.batch_size,
        #                                                               num_labels=n_labels,
        #                                                               sequence_len=seq_len,
        #                                                               num_epochs=FLAGS.num_epochs)
        #
        print('HASH TRAIN  ----->', FLAGS.hash_size)
        siamese = Siamese(sequence_length=seq_len,
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
        train_op = tf.train.MomentumOptimizer(0.0001, 0.95, use_nesterov=True).minimize(siamese.loss,
                                                                                        global_step=global_step)
        init_op = tf.global_variables_initializer()
        init_again = tf.local_variables_initializer()

        saver = tf.train.Saver()
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)

        sess = tf.Session(config=session_conf)
        with sess.as_default() as sess:
            sess.run(init_op)
            sess.run(init_again)

            # TODO la funcion map no la detecta y por lo tanto NO VA NADA
            # training_dataset = tf.contrib.data.TFRecordDataset([train_path])
            # # training_dataset = training_dataset.map(lambda x: parse_function(x, n_labels, seq_len))
            # training_dataset = training_dataset.map(lambda x: x)
            #
            # # training_dataset = training_dataset.shuffle(buffer_size=10000)
            # # training_dataset = training_dataset.repeat().batch(100)
            #
            # validation_dataset = tf.contrib.data.TFRecordDataset([train_path])
            # # training_dataset = tf.contrib.data.TFRecordDataset([train_path]).map(lambda x: )
            # # validation_dataset = tf.contrib.data.TFRecordDataset([train_path]).map(
            # #     lambda x: parse_function(x, n_labels, seq_len))
            # iterator = tf.contrib.data.Iterator.from_structure(training_dataset.output_types,
            #                                                    training_dataset.output_shapes)
            # next_element = iterator.get_next()
            #
            # training_init_op = iterator.make_initializer(training_dataset)
            # validation_init_op = iterator.make_initializer(training_dataset)
            #
            # # Run 20 epochs in which the training dataset is traversed, followed by the
            # # validation dataset.
            # for _ in range(1):
            #     # Initialize an iterator over the training dataset.
            #     sess.run(training_init_op)
            #     for _ in range(1):
            #         a = sess.run(next_element)
            #         # parse_function(a, n_labels, seq_len)
            #         print(a)
            #     #
            #     # # Initialize an iterator over the validation dataset.
            #     # sess.run(validation_init_op)
            #     # for _ in range(1):
            #     #     sess.run(next_element)



            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            step = 0
            try:
                while not coord.should_stop():
                    # print('--------------------------------------------------------------')
                    label, s1, s2 = sess.run([label_batch, sentences_1_batch, sentences_2_batch])
                    step += 1
                    print(step, label.shape, step%1000)
                    # print(sess.run(sentences_1_batch).shape, sess.run(sentences_2_batch).shape,
                    #       sess.run(label_batch).shape)

                    _, loss, attraction, repulsion, dis, acc = \
                        sess.run([train_op, siamese.loss, siamese.attraction_loss,
                                  siamese.repulsion_loss, siamese.distance,
                                  siamese.accuracy],
                                 feed_dict={
                                     siamese.left_input: s1,
                                     siamese.right_input: s2,
                                     siamese.label: label,
                                     })
                    log_str = "(#{0: <5} - {6}) - Loss: {1:.4f} - " \
                              "(a: {2:.3f} - r: {3:.3f} - " \
                              "d: {4:.4f}, accuracy:{5:.4f})"
                    print(log_str.format(sess.run(global_step), loss,
                                         np.mean(attraction),
                                         np.mean(repulsion),
                                         np.mean(dis), acc,
                                         np.mean(sess.run(label_batch))))

                    # TODO Dev
                    # if not step % 10:
                    #     print('--------------------------------------------------------------')
                    #     coord_dev = tf.train.Coordinator()
                    #     threads = tf.train.start_queue_runners(coord=coord_dev, sess=sess)
                    #     devstep = 0
                    #     try:
                    #         while not coord_dev.should_stop():
                    #             label, s1, s2 = sess.run([dev_labels, dev_sentences_1, dev_sentences_2])
                    #             devstep += 1
                    #             print(devstep, label.shape)
                    #     except tf.errors.OutOfRangeError:
                    #         print("Done dev!")
                    #     finally:
                    #         coord.request_stop()
                    #
                    #     coord.join(threads)


            except tf.errors.OutOfRangeError:
                print("Done training!")
            finally:
                coord.request_stop()

            coord.join(threads)

            # Save the model
            timestamp = str(int(time()))
            out_dir = abspath(join(curdir, "models", timestamp))
            makedirs(out_dir, exist_ok=True)

            with open(join(out_dir, 'parameters.txt'), 'w') as param_file:
                param_file.write("Default parameters: \n")
                for attr, value in sorted(FLAGS.__flags.items()):
                    param_file.write(" - {}={}\n".format(attr.upper(), value))

            save_path = saver.save(sess, join(out_dir, "model.ckpt"))
            print("Model saved in file: {}".format(save_path))



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
