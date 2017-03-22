import tensorflow as tf

def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensors
    """

    with tf.name_scope('euclidean_distance') as scope:
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), 1))
    return distance


class SiameseSubnet:
    def __init__(self, input_x, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters):
        self.input_x = input_x
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

    def create_model(self, subnet_name, reuse=True):
        # 1ST LAYER: Embedding layer
        with tf.name_scope("embedding"):
            self.W_embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                           trainable=True, name="W_embedding")
            self.embedded_words = tf.nn.embedding_lookup(self.W_embedding, self.input_x)
            self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)
            print('---> EMBEDDING: ', self.embedded_words_expanded)

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
                conv_outputs.append(output_conv)

                print('---> CONV: ', output_conv)
                conv_heigh = output_conv.get_shape()[1]
                pool_size = [1, conv_heigh, 1, 1]
                pooled = tf.nn.max_pool(output_conv, ksize=pool_size, strides=[1, 1, 1, 1],
                                        padding='VALID', name="pool")
                print('---> MAXPOOL: ', pooled)
                pooled_outputs.append(pooled)

        # 3RD LAYER: CONCAT ALL THE POOLED FEATURES
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, len(self.filter_sizes))
        print('---> HPOOL:', self.h_pool)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print('---> CONCAT:', self.h_pool_flat)
        return self.embedded_words_expanded, self.h_pool_flat, conv_outputs


class Siamese:
    def __init__(self, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters,
                 margin=1.5, threshold=1.0, contrastive_fully=False, hash_size=64):
        # TODO Study how to modify this in order to not run out of GPU memory
        with tf.device('/cpu:0'):
            with tf.name_scope("siamese"):
                self.left_input = tf.placeholder(tf.int32, [None, sequence_length], name='left')
                self.right_input = tf.placeholder(tf.int32, [None, sequence_length], name='right')

                self.left_siamese = SiameseSubnet(self.left_input, sequence_length, vocab_size,
                                                  embedding_size, filter_sizes, num_filters)
                self.left_embedded, self.left_tensor, self.left_conv = self.left_siamese.create_model('left', reuse=False)
                # self.left_sigmoid = tf.nn.sigmoid(self.left_tensor)

                self.right_siamese = SiameseSubnet(self.right_input, sequence_length, vocab_size,
                                                   embedding_size, filter_sizes, num_filters)
                self.right_embedded, self.right_tensor, self.right_conv = self.right_siamese.create_model('right', reuse=True)
                # self.right_sigmoid = tf.nn.sigmoid(self.right_tensor)

            with tf.name_scope("similarity"):
                self.label = tf.placeholder(tf.int32, [None], name='label')  # 1 if same, 0 if different
                self.label = tf.to_float(self.label)
                print('---->', self.label)

                self.margin = margin
                if contrastive_fully:
                    # Fully connected layer
                    fully_shape = [self.right_tensor.get_shape().as_list()[1], hash_size]
                    w_fully_r = tf.Variable(tf.truncated_normal(fully_shape, stddev=0.1), name="W_right")
                    b_fully_r = tf.Variable(tf.constant(0.1, shape=[hash_size]), name="b_right")
                    self.fully_right = tf.nn.sigmoid(tf.matmul(self.right_tensor, w_fully_r) + b_fully_r)
                    print('RIGHT Fully', self.fully_right)

                    w_fully_l = tf.Variable(tf.truncated_normal(fully_shape, stddev=0.1), name="W_left")
                    b_fully_l = tf.Variable(tf.constant(0.1, shape=[hash_size]), name="b_left")
                    self.fully_left = tf.nn.sigmoid(tf.matmul(self.left_tensor, w_fully_l) + b_fully_l)
                    print('RIGHT Left', self.fully_left)

                    (self.loss, self.attraction_loss,
                     self.repulsion_loss, self.distance) = self.contrastive_loss(self.margin,
                                                                                 self.fully_left,
                                                                                 self.fully_right)
                    print("---> FULLY")
                else:
                    (self.loss, self.attraction_loss,
                     self.repulsion_loss, self.distance) = self.contrastive_loss(self.margin,
                                                                                 self.left_tensor,
                                                                                 self.right_tensor)
                    print("---> SIAMESE TENSOR")

            with tf.name_scope("prediction"):
                threshold = tf.constant(threshold)
                self.predictions = tf.less_equal(self.distance, threshold)
                print(self.predictions)
                self.predictions = tf.cast(self.predictions, 'float32')
                print(self.predictions)
                print(self.label)
                self.correct_predictions = tf.equal(self.predictions, self.label)
                print("CORRECT:", self.correct_predictions)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))


    def contrastive_loss(self, margin, left_tensor, right_tensor):
        """
        L(W)= (1-y)LS(d^2) + y*LD*{max(0, margin - d)}^2
        similares cerquita, disimilares lejos -> stma busca equilibrio
        LS: similares siempre junto d^2. Atracción. Distancia en el espacio reducido
        LD: Repulsión.
        margen penaliza sólo las que estén dentro de un radio.
        pensar en el valor apropiado del margen.


        Compute the contrastive loss as in as in
            https://gitlab.idiap.ch/biometric/xfacereclib.cnn/blob/master/xfacereclib/cnn/scripts/experiment.py#L156
            With Y = [-1 +1] --> [POSITIVE_PAIR NEGATIVE_PAIR]
            L = log( m + exp( Y * d^2)) / N

            Compute the contrastive loss as in
            http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

            L = 0.5 * (Y) * D^2 + 0.5 * (1-Y) * {max(0, margin - D)}^2
            OR MAYBE THAT
            L = 0.5 * (1-Y) * D^2 + 0.5 * (Y) * {max(0, margin - D)}^2

        Return the loss operation
        """
        with tf.name_scope("contrastive_loss"):
            label = tf.to_float(self.label)
            one = tf.constant(1.0)

            d = compute_euclidean_distance(left_tensor, right_tensor)
            attraction_loss = tf.multiply(label, tf.square(d))
            max_part = tf.square(tf.maximum(margin - d, 0))
            repulsion_loss = tf.multiply(one - label, max_part)
            # between_class = tf.exp(tf.mul(one-label, tf.square(d)))  # (Y)*(d^2)
            # max_part = tf.square(tf.maximum(margin-d, 0))
            # within_class = tf.mul(label, max_part)  # (Y) * max((margin - d)^2, 0)

            # loss = 0.5 * tf.reduce_mean(within_class + between_class)
            loss = 0.5 * tf.reduce_mean(attraction_loss + repulsion_loss)

        return loss, attraction_loss, repulsion_loss, d
