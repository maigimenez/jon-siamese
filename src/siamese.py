import tensorflow as tf
from utils import contrastive_loss


class Siamese:

    # Create model
    def __init__(self, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters, margin):
        with tf.name_scope("embeddings") as embeddings_scope:
            self.filter_sizes = filter_sizes
            self.embedding_size = embedding_size
            self.num_filters = num_filters
            self.W_embedding = tf.Variable(tf.random_uniform([vocab_size, self.embedding_size], -1.0, 1.0),
                                           trainable=True, name="W_embedding")
            self.is_training = tf.placeholder(tf.bool, [], name='is_training')

        with tf.variable_scope("siamese") as siam_scope:
            # 1ST LAYER: Embedding layer
            with tf.variable_scope("embeddings-siamese") as input_scope:
                self.left_input = tf.placeholder(tf.int32, [None, sequence_length], name='left')
                left_embedded_words = tf.nn.embedding_lookup(self.W_embedding, self.left_input)
                self.left_embedded = tf.expand_dims(left_embedded_words, -1, name='left_embeddings')
                print('  ---> EMBEDDING LEFT: ', self.left_embedded)

                self.right_input = tf.placeholder(tf.int32, [None, sequence_length], name='right')
                right_embedded_words = tf.nn.embedding_lookup(self.W_embedding, self.right_input)
                self.right_embedded = tf.expand_dims(right_embedded_words, -1, name='right_embeddings')
                print('  ---> EMBEDDING RIGHT: ', self.right_embedded)

            self.left_siamese = self.subnet(self.left_embedded, 'left', False)
            print("---> SIAMESE TENSOR: ", self.left_siamese)
            siam_scope.reuse_variables()
            self.right_siamese = self.subnet(self.right_embedded, 'right', True)
            print("---> SIAMESE TENSOR: ", self.right_siamese)

        with tf.name_scope("similarity"):
            print('\n ----------------------- JOIN SIAMESE ----------------------------')
            self.labels = tf.placeholder(tf.int32, [None, 1], name='labels')
            self.labels = tf.to_float(self.labels)
            print('---> LABELS: ', self.labels)

            with tf.variable_scope("loss"):
                self.margin = tf.get_variable('margin', dtype=tf.float32,
                                              initializer=tf.constant(margin, shape=[1]),
                                              trainable=False)
                self.loss, self.attr, \
                self.rep, self.distance, self.maxpart = contrastive_loss(self.labels,
                                                           self.left_siamese,
                                                           self.right_siamese,
                                                           self.margin)

        with tf.name_scope("prediction"):
            # TODO Este es un parámetro de configuración
            self.threshold = tf.get_variable('threshold', dtype=tf.float32,
                                             initializer=tf.constant(1.0, shape=[1]))
            self.predictions = tf.less_equal(self.distance, self.threshold)
            self.predictions = tf.cast(self.predictions, 'float32')
            self.correct_predictions = tf.equal(self.predictions, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

    def subnet(self, input_sentence, sub_name, reuse=False):
        with tf.name_scope("subnet_"+sub_name):
            print('\n ----------------------- SUBNET ----------------------------')
            # 2ND LAYER: Convolutional + ReLU + Max-pooling
            pooled_outputs, conv_outputs = [], []
            for filter_size in self.filter_sizes:
                with tf.variable_scope("conv-{}".format(filter_size)) as conv_scope:
                    name = 'conv-' + str(filter_size)
                    output_conv = self.conv_layer(input_sentence,
                                                  filter_size, self.embedding_size,
                                                  1, self.num_filters, name)
                    output_conv = tf.contrib.layers.batch_norm(output_conv,
                                                               is_training=self.is_training,
                                                               center=True,
                                                               scale=False,
                                                               trainable=True,
                                                               updates_collections=None,
                                                               scope='bn',
                                                               decay=0.9)
                    conv_outputs.append(output_conv)
                    pooled = self.max_pool_layer(output_conv, name)
                    pooled_outputs.append(pooled)

            # 3RD LAYER: CONCAT ALL THE POOLED FEATURES
            num_filters_total = self.num_filters * len(self.filter_sizes)
            h_pool = tf.concat(pooled_outputs, 1)
            print('  ---> CONCAT:', h_pool)

            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total], name='re_sim')
            print('  ---> RESHAPE:', h_pool_flat)

        return h_pool_flat

    def conv_layer(self, input, kernel_height, kernel_width, channels, kernels_num, name):
        # Define the convolution layer
        filter_shape = [kernel_height, kernel_width, channels, kernels_num]
        W_init = tf.truncated_normal_initializer(stddev=0.1)
        W = tf.get_variable(name + '_W', dtype=tf.float32, initializer=W_init,
                            shape=filter_shape)
        b_init = tf.constant(0.1, shape=[kernels_num])
        b = tf.get_variable(name + '_b', dtype=tf.float32, initializer=b_init)
        strides = [1, 1, 1, 1]
        cnn = tf.nn.conv2d(input, W, strides=strides, padding='VALID', name=name+'_cnn')
        activation = tf.nn.tanh(tf.nn.bias_add(cnn, b), name=name+'_relu')
        print('    ---> CNN: ', activation)
        return activation

    def max_pool_layer(self, output_conv, name):
        conv_heigh = output_conv.get_shape()[1]
        pool_size = [1, conv_heigh, 1, 1]

        pooled = tf.nn.max_pool(output_conv, ksize=pool_size, strides=[1, 1, 1, 1],
                                padding='VALID', name=name+'_pool')
        print('    ---> MAXPOOL: ', pooled)
        return pooled



