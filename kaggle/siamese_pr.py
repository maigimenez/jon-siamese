import tensorflow as tf

def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensors
    """

    with tf.name_scope('euclidean_distance') as scope:
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), 1))
    return distance


class SiameseSubnet:
    def __init__(self, input_x, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters,
                 hash_size=64):
        self.input_x = input_x
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.hash_size = hash_size

    def create_model(self, subnet_name, reuse=True):
        # 1ST LAYER: Embedding layer
        with tf.name_scope("embedding"):
            self.W_embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                           trainable=True, name="W_embedding")
            self.embedded_words = tf.nn.embedding_lookup(self.W_embedding, self.input_x)
            self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)
            print('\n---> EMBEDDING: ', self.embedded_words_expanded)

        # 2ND LAYER: Convolutional + ReLU + Max-pooling
        pooled_outputs, conv_outputs = [], []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-pool-%s" % filter_size) as conv_scope:
                # Define the convolution layer
                kernel_shape = [filter_size, self.embedding_size]
                init_weights = tf.contrib.layers.xavier_initializer_conv2d()
                output_conv = tf.contrib.layers.conv2d(self.embedded_words_expanded,
                                                       num_outputs=self.num_filters,
                                                       kernel_size=kernel_shape,
                                                       activation_fn=tf.nn.relu,
                                                       padding='VALID',
                                                       weights_initializer=init_weights,
                                                       scope=conv_scope, reuse=reuse)
                print('---> CONV: ', output_conv)
                conv_outputs.append(output_conv)
                conv_heigh = output_conv.get_shape()[1]
                pool_size = [1, conv_heigh, 1, 1]
                pooled = tf.nn.max_pool(output_conv, ksize=pool_size, strides=[1, 1, 1, 1],
                                        padding='VALID', name="pool")
                print('---> MAXPOOL: ', pooled)
                pooled_outputs.append(pooled)

        # 3RD LAYER: CONCAT ALL THE POOLED FEATURES
        num_filters_total = self.num_filters * len(self.filter_sizes)
        # self.h_pool = tf.concat(pooled_outputs, len(self.filter_sizes)-1)
        self.h_pool = tf.concat(pooled_outputs, 1)
        print('---> HPOOL:', self.h_pool)

        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print('---> CONCAT:', self.h_pool_flat)

        if self.hash_size:
            with tf.variable_scope("fully") as fully_scope:
                print('---> HASH TRAIN SUBNET ', self.hash_size)
                fully_shape = [self.h_pool_flat.get_shape().as_list()[1], self.hash_size]
                w_fully = tf.get_variable("w_fully", fully_shape,
                                          initializer=tf.random_normal_initializer())
                b_fully = tf.get_variable("b_fully", [self.hash_size],
                                          initializer=tf.constant_initializer(0.1))


                # self.fully = tf.nn.sigmoid(tf.matmul(self.h_pool_flat, w_fully) + b_fully)
                self.fully = tf.nn.relu(tf.matmul(self.h_pool_flat, w_fully) + b_fully)
                print('---> HASH-FULLY:', self.fully)

            return self.h_pool_flat, self.fully
        else:
            return self.h_pool_flat, None

class Siamese:
    def populate_labels(self):
        label_training = tf.placeholder(tf.int32, [None], name='label')  # 1 if same, 0 if different
        label_training = tf.to_float(label_training)
        return label_training

    def __init__(self, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters,
                 margin=1.5, threshold=1.0,hash_size=64):
        # TODO Study how to modify this in order to not run out of GPU memory
        self.is_training = tf.placeholder(tf.bool, [], name='is_training')

        with tf.variable_scope("siamese") as siamese_scope:
            print('HASH TRAIN SIAMESE -->', hash_size)
            self.left_input = tf.placeholder(tf.int32, [None, sequence_length], name='left')
            self.right_input = tf.placeholder(tf.int32, [None, sequence_length], name='right')

            self.left_siamese = SiameseSubnet(self.left_input, sequence_length, vocab_size,
                                              embedding_size, filter_sizes, num_filters, hash_size)
            self.left_tensor, self.fully_left = self.left_siamese.create_model('left', reuse=False)
            siamese_scope.reuse_variables()
            self.right_siamese = SiameseSubnet(self.right_input, sequence_length, vocab_size,
                                               embedding_size, filter_sizes, num_filters, hash_size)
            self.right_tensor, self.fully_right = self.right_siamese.create_model('right', reuse=True)

            h1_units = 256
            h1_shape = [self.connected_fully.get_shape().as_list()[1], h1_units]
            w_h1 = tf.get_variable("w_h1", h1_shape, initializer=tf.random_normal_initializer())
            b_h1 = tf.get_variable("b_fully", [h1_units], initializer=tf.constant_initializer(0.1))

            h2_units = 2
            h2_shape = [h1_units, h2_units]
            w_h2 = tf.get_variable("w_h1", h2_shape, initializer=tf.random_normal_initializer())
            b_h2 = tf.get_variable("b_fully", [h2_units], initializer=tf.constant_initializer(0.1))

            # TODO create the version where you can concatenated the last layer with the hash
            if not hash_size:
                self.connected_fully = tf.concat([self.left_tensor, self.right_tensor], 1)
                print('\n---> FULLY:', self.connected_fully, self.connected_fully.get_shape().as_list()[1])
                h1 = tf.nn.relu(tf.matmul(self.connected_fully, w_h1) + b_h1)
                print(h1)
                h2 = tf.nn.softmax(tf.matmul(self.connected_fully, w_h2) + b_h2)
                print(h2)

            # print(tf.concat([self.fully_left, self.fully_right], 1))
            # print(tf.concat([self.fully_left, self.fully_right], 0))
            # self.connected_fully = tf.concat([self.fully_left, self.fully_right], 1)
            #
            # print("FULLY ", self.connected_fully)
            # print(tf.layers.dense(inputs=self.connected_fully, units=256, activation=tf.nn.relu))

            # dense  = tf.layers.dense(inputs=self.connected_fully, units=256, activation=tf.nn.relu)
            # outputs = tf.layers.dense(inputs=self.dense, units=2, activation=tf.nn.softmax)
            # y = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
            # self.predictions = tf.nn.softmax(self.connected_fully)

            # print("ARGMAX 0: ", tf.argmax(outputs, 0))
            # print("ARGMAX 1: ", tf.argmax(outputs, 1))

            # empty_labels = tf.constant(0, tf.float32)
            # self.label = tf.cond(self.is_training,
            #                      self.populate_labels,
            #                      lambda: empty_labels)
            # print("---> PREDICTIONS ", self.predictions, self.label)
            #
            #
            #
            #
            # self.correct_predictions = tf.equal(self.predictions, self.label)
            # print("---> CORRECT:", self.correct_predictions,
            #       self.correct_predictions * tf.log(self.predictions),
            #       -tf.reduce_sum(self.correct_predictions * tf.log(self.predictions),
            #                      reduction_indices=[1])
            #       )
            #
            # # define the loss function
            # self.loss = tf.reduce_mean(-tf.reduce_sum(self.correct_predictions * tf.log(self.predictions),
            #                                           reduction_indices=[1]))
            # print("---> LOSS ", self.loss)
            # self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))
