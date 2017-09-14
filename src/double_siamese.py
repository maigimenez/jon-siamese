import tensorflow as tf
from utils import contrastive_loss
from siamese import Siamese

class DoubleSiamese:

    # Create model
    def __init__(self, sequence_length, vocab_size, embedding_size,
                 filter_sizes, num_filters, margin):
        with tf.variable_scope("branches") as double_scope:
            self.sim_branch = Siamese(sequence_length, vocab_size,
                                      embedding_size, filter_sizes, num_filters, margin)
            double_scope.reuse_variables()
            self.disim_branch = Siamese(sequence_length, vocab_size,
                                        embedding_size, filter_sizes, num_filters, margin)

            # TODO: Modify this to minimize the AUR
            self.loss = tf.reduce_mean(self.sim_branch.loss + self.disim_branch.loss,
                                  name="loss_branches")
