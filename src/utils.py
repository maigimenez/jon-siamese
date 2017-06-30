from nltk import word_tokenize
from nltk.corpus import stopwords
from tensorflow.contrib import learn
import configparser
import tensorflow as tf
import numpy as np


def preprocess_sentence(sentence, max_len=60):
    """ Pre-process a sentence: remove stop words and lowercase the sentence
    Arguments:
        :param sentence: sentence to pre-process
        :param max_len: -1 if all sentences will be pre-processed
                        N sentences longer than N will be pre-process.
    """

    en_stopwords = stopwords.words('english')
    if isinstance(sentence, str):
        if max_len != -1 and len(sentence) > max_len:
            return ' '.join([token.lower() for token in word_tokenize(sentence)
                             if token not in en_stopwords])
        else:
            return ' '.join([token.lower() for token in word_tokenize(sentence)])
    return None


def build_vocabulary(sim_data, non_sim_data):
    """" Build vocabulary, the lookup table and transform the text """

    train_texts = []
    for data in sim_data:
        # TODO tokenize some words like @usernames?
        train_texts.append(data.sentence_1)
        train_texts.append(data.sentence_2)
    for data in non_sim_data:
        train_texts.append(data.sentence_1)
        train_texts.append(data.sentence_2)

    max_document_length = max([len(x.split()) for x in train_texts])
    print("The max. document length is: {}".format(max_document_length))
    # Creates the lookup table: Maps documents to sequences of word ids.
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab_processor = vocab_processor.fit(train_texts)
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

    return vocab_processor, max_document_length


def read_flags(config_filepath=None):
    """ Read the flags for the NN from a config file """

    config = configparser.ConfigParser()
    if not config_filepath:
        config_filepath = 'default.flags'
    config.read(config_filepath)

    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # GPU related configuration
    if config['GPU']['allow_soft_placement'] == 'True':
        FLAGS.allow_soft_placement = True
    else:
        FLAGS.allow_soft_placement = False
    if config['GPU']['log_device_placement'] == 'True':
        FLAGS.log_device_placement = True
    else:
        FLAGS.log_device_placement = False

    # Hyperparameters
    FLAGS.embedding_dim = int(config['Hyperparameters']['Embeddings'])
    FLAGS.filter_sizes = config['Hyperparameters']['Filter sizes']
    FLAGS.num_filters = int(config['Hyperparameters']['Number of filters'])
    hash_size = config['Hyperparameters']['Hash size']
    if hash_size == 'None':
        FLAGS.hash_size = None
    else:
        FLAGS.hash_size = int(hash_size)
    FLAGS.margin = float(config['Hyperparameters']['Margin'])
    FLAGS.threshold = float(config['Hyperparameters']['Threshold'])

    # Training parameters
    FLAGS.batch_size = int(config['Training']['Batch size'])
    FLAGS.num_epochs = int(config['Training']['Number of epochs'])
    if config['Training']['Shuffle epochs'] == 'True':
        FLAGS.shuffle_epochs = True
    else:
        FLAGS.shuffle_epochs = False

    print("Default parameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print(" - {}={}".format(attr.upper(), value))
    print("")

    return FLAGS


def shuffle_epochs(sim_data, non_sim_data, mixed=False):
    if mixed:
        data = np.append(sim_data, non_sim_data)
        shuffle_indices = np.random.permutation(np.arange(len(sim_data)+len(non_sim_data)))
        return data[shuffle_indices]

    shuffle_sim_indices = np.random.permutation(np.arange(len(sim_data)))
    shuffled_sim_data = sim_data[shuffle_sim_indices]
    shuffle_non_sim_indices = np.random.permutation(np.arange(len(non_sim_data)))
    shuffled_non_sim_data = non_sim_data[shuffle_non_sim_indices]
    return shuffled_sim_data, shuffled_non_sim_data


def batch_iter(sim_data, non_sim_data, batch_size, num_batches_per_epoch):
    data_size = len(sim_data) + len(non_sim_data)
    # Shuffle the data each epoch.
    shuffled_sim_data, shuffled_non_sim_data = shuffle_epochs(sim_data, non_sim_data)

    for batch_num in range(num_batches_per_epoch):
        # It's an odd batch -> return sim_data while there is still available
        # It's an even batch -> return non_sim_data while there is still available
        # Yields handles this.
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        if start_index < len(sim_data):
            # print('({}) SIM -> [{} - {}]'.format(batch_num, start_index, end_index))
            sim_data_batch = shuffled_sim_data[start_index:end_index]
            # If the last batch has not the same size grab the N firsts items.
            # print(start_index, len(sim_data_batch))
            if len(sim_data_batch) < batch_size:
                new_end_idx = batch_size - len(sim_data_batch)
                yield (np.append(sim_data_batch, shuffled_sim_data[:new_end_idx]), 'SIM')
            else:
                yield (sim_data_batch, 'SIM')

        if start_index < len(non_sim_data):
            # print('({}) NON SIM -> [{} - {}]'.format(batch_num, start_index, end_index))
            non_sim_data_batch = shuffled_non_sim_data[start_index:end_index]
            # If the last batch has not the same size grab the N firsts items.
            if len(non_sim_data_batch) < batch_size:
                new_end_idx = batch_size - len(non_sim_data_batch)
                yield (np.append(non_sim_data_batch, shuffled_non_sim_data[:new_end_idx]), 'NON SIM')
            else:
                yield (non_sim_data_batch, 'NON SIM')


def get_dev_data(sim_data, non_sim_data):
    right_sentences, left_sentences, labels = [], [], []
    sim_data = np.array(sim_data)
    for sample in sim_data:
        right_sentences.append(sample.sentence_1)
        left_sentences.append(sample.sentence_2)
        labels.append(sample.label)

    non_sim_data = np.array(non_sim_data)
    for sample in non_sim_data:
        right_sentences.append(sample.sentence_1)
        left_sentences.append(sample.sentence_2)
        labels.append(sample.label)

    return right_sentences, left_sentences, labels


def write_tfrecord(writer, data, one_hot):
    """ Write a TF record """

    data_label = data.oneh_label if one_hot else [int(data.label)]
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'pair_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[data.pair_id])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=data_label)),
                'sentence_1': tf.train.Feature(int64_list=tf.train.Int64List(value=data.sentence_1.astype("int64"))),
                'sentence_2': tf.train.Feature(int64_list=tf.train.Int64List(value=data.sentence_2.astype("int64"))),
            }))
    writer.write(example.SerializeToString())