from os.path import join
from corpus import Corpus
import argparse

from utils import build_vocabulary
from train import train_siamese, train_siamese_fromtf


def get_arguments():
    parser = argparse.ArgumentParser(description='Train a Siamese Architecture')
    parser.add_argument('--flags', metavar='f', type=str,
                        help='Path where the flags for training are',
                        dest='flags_path')

    args = parser.parse_args()
    return args.flags_path


def load_ibm():
    """ Load the train and dev datasets """
    IBM_PATH = '/home/mgimenez/Dev/corpora/Quora/IBM'
    TRAIN_PATH = join(IBM_PATH, 'train.tsv')
    train = Corpus('ibm', TRAIN_PATH)
    DEV_PATH = join(IBM_PATH, 'dev.tsv')
    dev = Corpus('ibm', DEV_PATH)
    TEST_PATH = join(IBM_PATH, 'test.tsv')
    test = Corpus('ibm', TEST_PATH)

    vocab_processor, seq_len = build_vocabulary(train.sim_data,
                                                train.non_sim_data)
    train.to_index(vocab_processor)
    dev.to_index(vocab_processor)
    test.to_index(vocab_processor)

    return train.non_sim_data, train.sim_data, \
           dev.non_sim_data, dev.sim_data, \
           test.sim_data, test.non_sim_data, \
           vocab_processor, seq_len


def load_quora():
    QUORA_PATH = '/home/mgimenez/Dev/corpora/Quora/quora_duplicate_questions.tsv'
    dataset = Corpus('quora', QUORA_PATH)

    train_non_sim, train_sim, dev_non_sim, dev_sim, \
    test_non_sim, test_sim, \
    vocab_processor, seq_len = dataset.make_partitions_quora()

    return train_non_sim, train_sim, dev_non_sim, dev_sim, \
           test_non_sim, test_sim, vocab_processor, seq_len

def quoraTF_experiment(flags_path):
    tensors_path = '../tfrecords'
    train_siamese_fromtf(tensors_path, flags_path)

if __name__ == "__main__":

    flags_path = get_arguments()
    quoraTF_experiment(flags_path)

    # load_ibm()

    # train_non_sim, train_sim, \
    # dev_non_sim, dev_sim, \
    # test_non_sim, test_sim, vocab_processor, seq_len = load_quora()
    # train_siamese(train_non_sim, train_sim, dev_non_sim, dev_sim, vocab_processor, seq_len)

