import tensorflow as tf
from corpus import Corpus
from utils import build_vocabulary
import pickle
import argparse
from os.path import join, isdir
from os import makedirs


def get_arguments():
    parser = argparse.ArgumentParser(description='Create TF Records from Kaggle CSV')
    parser.add_argument('--data', metavar='d', type=str,
                        help='Path where the Quora dataset is.',
                        dest='dataset_path')

    parser.add_argument('--out', metavar='o', type=str,
                        help='Path where the tf records will be saved.',
                        dest='output_path')

    parser.add_argument('-p', action='store_true', default=False,
                        help='Apply a pre-processing phase',
                        dest='preprocess')

    parser.add_argument('-o', action='store_true', default=False,
                        help='Save the one-hot encoding',
                        dest='one_hot')

    args = parser.parse_args()
    return args.dataset_path, args.output_path, args.preprocess, args.one_hot

def save_partitions(dataset_path, csv_path):
    dataset = Corpus('quora', dataset_path)
    dataset.make_partitions_quora(csv_path)

if __name__ == "__main__":
    # TODO the pre-process is not applied
    dataset_path, output_path, preprocess, one_hot = get_arguments()
    # Create and save the partitions
    dataset = Corpus('quora', dataset_path)
    print(len(dataset.sim_data), len(dataset.non_sim_data))
    dataset.write_partitions_mixed(output_path, one_hot)
