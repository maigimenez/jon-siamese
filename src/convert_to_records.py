from corpus_ms import CorpusMS
from corpus_quora import CorpusQuora
import argparse
from os.path import join


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
    if not preprocess_flag:
        preprocess = False
    elif preprocess_flag.lower() == 'no':
        preprocess = False
    else:
        preprocess = True
        if preprocess_flag.lower() == 'all':
            max_len = -1
        else:
            max_len = int(preprocess_flag)

    if corpus == 'quora':
        # Create and save the partitions
        dataset = CorpusQuora(dataset_path, preprocess, max_len)
        print('Read {} similarity sencenteces and {} disimilar.'.format(
            len(dataset.sim_data), len(dataset.non_sim_data)))
        if balanced:
            dataset.balance_partitions(output_path, one_hot, split_files=True)
        else:
            dataset.write_partitions_mixed(output_path, one_hot)

    elif corpus == 'microsoft':
        # Load the train
        train = CorpusMS(join(dataset_path, 'msr_paraphrase_train.txt'),
                       preprocess, max_len)
        print('[TRAIN] Read {} similarity sencenteces and {} disimilar.'.format(
            len(train.sim_data), len(train.non_sim_data)))
        vocabulary = train.write_tensors(output_path, 'train', split_files=balanced)

        # Load the test
        test = CorpusMS(join(dataset_path, 'msr_paraphrase_test.txt'),
                      preprocess, max_len)
        print('[TEST] Read {} similarity sencenteces and {} disimilar.'.format(
            len(test.sim_data), len(test.non_sim_data)))
        test.write_tensors(output_path, 'test', vocab_processor=vocabulary)
