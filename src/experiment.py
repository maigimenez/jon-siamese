import argparse
from os.path import join, abspath, curdir
from os import makedirs
from time import time

from corpus import Corpus
from utils import build_vocabulary, write_flags, read_flags
from train import train_siamese_fromtf
from test import test_model
from validation import dev_step
from convert_to_records import create_tfrecods

def get_arguments():
    parser = argparse.ArgumentParser(description='Train a Siamese Architecture')
    parser.add_argument('--flags', metavar='f', type=str,
                        help='Path where the flags for training are',
                        dest='flags_path')
    parser.add_argument('--data', metavar='d', type=str,
                        help='Path where the Quora dataset is.',
                        dest='dataset_path')
    parser.add_argument('--tf', metavar='f', type=str,
                        help='Path where tfrecords are located',
                        dest='tf_path')
    args = parser.parse_args()
    return args.flags_path, args.dataset_path, args.tf_path


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


def quoraTF_default(flags_path, tf_path, out_dir=None):
    flags = read_flags(flags_path)
    num_epochs = flags.num_epochs
    evaluate_epochs = flags.evaluate_epochs
    for i in range(0, num_epochs, evaluate_epochs):
        # Train n epochs and then evaluate the system
        if not out_dir:
            out_dir = train_siamese_fromtf(tf_path, flags, evaluate_epochs)
        else:
            train_siamese_fromtf(tf_path, flags, evaluate_epochs, out_dir)

        dev_step(tf_path, out_dir, flags_path, i)
    test_model(tf_path, out_dir, flags_path)


def quoraTF_experiments(tensors_path):

    embeddings = ['300', '50', '100']
    filters = ['3,4,5', '1,2,3', '3,5,7', '3,7,9']
    number_of_filters = ['50', '100', '200']
    hashes = ['None']
    margins = ['15', '20', '30']
    thresholds = ['1.5']
    epochs = ['5', '10', '20']

    hyperparams = {'Embeddings': None,
                   'Filter sizes': None,
                   'Number of filters': None,
                   'Hash size': None,
                   'Margin': None,
                   'Threshold': None}
    params = {'Number of epochs': None}

    for e in embeddings:
        hyperparams['Embeddings'] = e
        for f in filters:
            hyperparams['Filter sizes'] = f
            for n in number_of_filters:
                hyperparams['Number of filters'] = n
                for h in hashes:
                    hyperparams['Hash size'] = h
                    for m in margins:
                        hyperparams['Margin'] = m
                        for t in thresholds:
                            hyperparams['Threshold'] = t
                            for ep in epochs:
                                params['Number of epochs'] = ep

                                # Save the params
                                current_model = str(int(time()))
                                out_dir = abspath(join("models", current_model))
                                makedirs(out_dir, exist_ok=True)
                                flags_path = join(out_dir, 'config.flags')
                                write_flags(hyperparams, params, flags_path)
                                print('The model is saved in this path: {}'.format(out_dir))

                                quoraTF_default(flags_path, tensors_path, out_dir)

                                # # Train the model
                                # train_siamese_fromtf(tensors_path, flags_path, out_dir)
                                #
                                # # Test the model
                                # test_model(tensors_path, out_dir, flags_path)

if __name__ == "__main__":

    flags_path, dataset_path, tf_path = get_arguments()

    if flags_path:
        quoraTF_default(flags_path, tf_path)
    else:
        # TODO: Generate them
        # create_tfrecods(dataset_path, tensors_path, False, False)
        quoraTF_experiments(tf_path)
