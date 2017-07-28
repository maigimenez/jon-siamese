import tensorflow as tf
from os.path import join

from utils import read_flags, load_binarize_data, input_pipeline_test, best_score
from siamese import Siamese


def dev_step(tf_path, model_path, flags_path, current_step):

    # Import the parameters binarized
<<<<<<< HEAD
    test_tfrecors = join(tf_path, 'dev.tfrecords')
=======
    test_tfrecors = join(tf_path, 'test.tfrecords')
>>>>>>> f0d45aaf2f49a673e0ca6502a2c6da2696e13740
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

    distances_filename = join(model_path, 'dev_' + str(current_step) + '_distances.log')
    log_filename = join(model_path, 'dev_' + str(current_step) + '.log')

    # TEST THE SYSTEM
    with tf.Graph().as_default():

        label_batch, test_1_batch, test_2_batch = input_pipeline_test(filepath=test_tfrecors,
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
                    loss, attraction, repulsion, dis = \
                        sess.run([siamese.loss, siamese.attr,
                                  siamese.rep, siamese.distance],
                                 feed_dict={
                                     siamese.left_input: test_1,
                                     siamese.right_input: test_2,
                                     siamese.labels: test_label,
                                     siamese.is_training: False
                                 })

                    with open(log_filename, 'a') as log_file:
                        log_str = "(#{0: <5} - {5}) - Loss: {1:.4f} - " \
                                  "(a: {2:.3f} - r: {3:.3f} - " \
                                  "d: {4:.4f})\n"
                        log_file.write(log_str.format(test_sample, loss,
                                                      attraction[0][0],
                                                      repulsion[0][0],
                                                      dis[0],
                                                      test_label[0][0], ))

                    with open(distances_filename, 'a') as dist_file:
                        log_str = "{}\t{}\n"
                        dist_file.write(log_str.format(dis[0],
                                                       test_label[0][0]))
                    test_sample += 1

            except tf.errors.OutOfRangeError:
                print("Done evaluating!")
            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()

            with open(join(model_path, 'dev.txt'), 'a') as results_file:
                results_file.write("Accuracy: {} ({}/{})".format(hits / test_sample, hits, test_sample))

            print("Results saved in: {}".format(join(model_path, 'dev.txt')))
            find_threshold(model_path)


def find_threshold(model_path):
    similar, dissimilar = [], []
    with open(join(model_path, 'distances.log')) as dist_file:
        for line in dist_file:
            dist, tag = line.strip().split('\t')
            if tag == '1':
                similar.append(float(dist))
            else:
                dissimilar.append(float(dist))

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

    with open(join(model_path, 'dev.txt'), 'a') as results_file:
        log_str = "\nThe best accuracy is {} with threshold {}. And the worst {}"
        results_file.write(log_str.format(best_accuracy, best_threshold, worst_accuracy))