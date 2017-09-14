from corpus import Corpus
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Create TF Records from Quora CSV')
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

    args = parser.parse_args()
    return args.dataset_path, args.output_path, args.preprocess, args.one_hot


def create_tfrecods(dataset_path, output_path, preprocess, max_len, one_hot, balanced):
    dataset = Corpus('quora', dataset_path, preprocess, max_len)
    if not balanced:
        dataset.write_partitions_mixed(output_path, one_hot)
    else:
        dataset.balance_partitions(output_path+'balance', one_hot)

if __name__ == "__main__":
    # TODO the pre-process is not applied
    dataset_path, output_path, preprocess_flag, one_hot = get_arguments()
    max_len = None
    if preprocess_flag.lower() == 'no':
        preprocess = False
    else:
        preprocess = True
        if preprocess_flag.lower() == 'all':
            max_len = -1
        else:
            max_len = int(preprocess_flag)

    # Create and save the partitions
    dataset = Corpus('quora', dataset_path, preprocess, max_len)
    print('Read {} similarity sencenteces and {} disimilar.'.format(
        len(dataset.sim_data), len(dataset.non_sim_data)))
    # TODO Include a flag to decide whether to create a balance file or not,
    # and to create one or several balanced files
    # dataset.write_partitions_mixed(output_path, one_hot)
    # print('------------------ BALANCE --------------------------')
    split_files = True
    dataset.balance_partitions(output_path, one_hot, split_files=True)
