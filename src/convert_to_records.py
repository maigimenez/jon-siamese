from corpus import Corpus
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Create TF Records from Quora CSV')
    parser.add_argument('--corpus', metavar='c', type=str,
                        help='Select which dataset is going to be converted [quora | p4p]',
                        dest='corpus')

    parser.add_argument('--data', metavar='d', type=str,
                        help='Path where the Quora dataset is.',
                        dest='dataset_path')

    parser.add_argument('--out', metavar='o', type=str,
                        help='Path where the tf records will be saved.',
                        dest='output_path')

    parser.add_argument('-p', type=str,
                        help='Apply a pre-processing phase (All, No, '
                             'Max. length of sentences to be processed)',
                        dest='preprocess')

    parser.add_argument('-o', action='store_true', default=False,
                        help='Save the one-hot encoding',
                        dest='one_hot')

    parser.add_argument('-b', action='store_true', default=False,
                        help='Save a balanced dataset',
                        dest='balanced')

    args = parser.parse_args()
    return args.corpus, args.dataset_path, args.output_path, args.preprocess, \
           args.one_hot, args.balanced


def create_tfrecods(dataset_path, output_path, preprocess, max_len, one_hot, balanced):
    dataset = Corpus('quora', dataset_path, preprocess, max_len)
    if not balanced:
        dataset.write_partitions_mixed(output_path, one_hot)
    else:
        dataset.balance_partitions(output_path+'balance', one_hot)

if __name__ == "__main__":

    # Load the arguments from console
    corpus, dataset_path, output_path, preprocess_flag, one_hot, balanced = get_arguments()
    max_len = None

    # Convert the preprocess flag to boolean values and get the max length of the sentence to preprocess
    if preprocess_flag.lower() == 'no':
        preprocess = False
    else:
        preprocess = True
        if preprocess_flag.lower() == 'all':
            max_len = -1
        else:
            max_len = int(preprocess_flag)

    # Create and save the partitions
    dataset = Corpus(corpus, dataset_path, preprocess, max_len)
    print('Read {} similarity sencenteces and {} disimilar.'.format(
        len(dataset.sim_data), len(dataset.non_sim_data)))

    if balanced:
        split_files = True
        dataset.balance_partitions(output_path, one_hot, split_files=True)
    else:
        dataset.write_partitions_mixed(output_path, one_hot)