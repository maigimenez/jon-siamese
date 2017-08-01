from nltk import word_tokenize
from nltk.corpus import stopwords
from tensorflow.contrib import learn
import configparser
import tensorflow as tf
import numpy as np
import pickle


def best_score(threshold, dissimilar, similar):
    """ Returns the accuracy achieved  using a determined threshold"""
    hits = sum([1 for s in dissimilar if s > threshold]) + sum([1 for s in similar if s <= threshold])
    return hits/(len(similar)+len(dissimilar))


def read_sample(filename_queue, num_labels, sequence_len):
    reader = tf.TFRecordReader()
    _, serialized_sample = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_sample,
        features={
            'label': tf.FixedLenFeature([num_labels], tf.int64),
            'sentence_1': tf.FixedLenFeature([sequence_len], tf.int64),
            'sentence_2': tf.FixedLenFeature([sequence_len], tf.int64)
        })
    return features['label'], features['sentence_1'], features['sentence_2']


def input_pipeline_test(filepath, batch_size, num_labels, sequence_len,
                        num_epochs=None):
    """ Creates batches of data in a queue and returns a batch of ids and sentences """
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filepath], num_epochs=num_epochs)
        label, sentence_1, sentence_2 = read_sample(filename_queue, num_labels, sequence_len)
        labels, sentences_1_batch, sentences_2_batch = tf.train.shuffle_batch([label, sentence_1,
                                                                               sentence_2],
                                                                              batch_size=batch_size,
                                                                              num_threads=1,
                                                                              capacity=1000 + 3 * batch_size,
                                                                              min_after_dequeue=100)
    return labels, sentences_1_batch, sentences_2_batch

def load_binarize_data(path):
    """ Given a file load its binarized data.
    Arguments:
        :param path: path where the pickle file is located.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def preprocess_sentence(sentence, max_len=60):
    """ Pre-process a sentence: remove stop words and lowercase the sentence
    Arguments:
        :param sentence: sentence to pre-process
        :param max_len: -1 if all sentences will be pre-processed
                        N sentences longer than N will be pre-process.
    """
    en_stopwords = stopwords.words('english')
    if isinstance(sentence, str):
        if max_len == -1 or len(sentence) > max_len:
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


def write_flags(hyperparams, params, flags_path):
    config = configparser.ConfigParser()
    config['GPU'] = {'allow_soft_placement': 'True',
                     'log_device_placement': 'False'}

    config['Hyperparameters'] = {}
    for key, value in hyperparams.items():
        config['Hyperparameters'][key] = value

    config['Training'] = {}
    for key, value in params.items():

        config['Training'][key] = value
    config['Training']['Batch size'] = '100'
    config['Training']['Evaluate every n epochs'] = '5'
    config['Training']['Shuffle epochs'] = 'True'
    with open(flags_path, 'w') as configfile:
        config.write(configfile)


def read_flags(config_filepath=None):
    """ Read the flags for the NN from a config file """

    config = configparser.ConfigParser()
    if not config_filepath:
        config_filepath = '../config/default.flags'
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
    FLAGS.evaluate_epochs = int(config['Training']['Evaluate every n epochs'])
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


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensors
    """
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), 1))


def contrastive_loss(labels, left_tensor, right_tensor, margin):
    """
    L(W)= (1-y)LS(d^2) + y*LD*{max(0, margin - d)}^2
    similares cerquita, disimilares lejos -> stma busca equilibrio
    LS: similares siempre junto d^2. Atracción. Distancia en el espacio reducido
    LD: Repulsión.
    margen penaliza sólo las que estén dentro de un radio.
    pensar en el valor apropiado del margen.


    Compute the contrastive loss as in
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        L = 0.5 * (1-Y) * D^2 + 0.5 * (Y) * {max(0, margin - D)}^2

    Return the loss operation
    """

    # TODO Cuidado que están al revés respecto al papel
    y = tf.subtract(1.0, labels, name="1-yi")
    not_y = tf.to_float(labels)

    d = compute_euclidean_distance(left_tensor, right_tensor)
    max_part = tf.square(tf.maximum(tf.subtract(margin, d), 0))

    attraction_loss = tf.multiply(not_y, tf.square(d))
    repulsion_loss = tf.multiply(y, max_part)
    # TODO Change the weights between attraction and repulsion instead of 0.5*attraction + 0.5*loss
    loss = tf.reduce_mean(attraction_loss + repulsion_loss, name="loss")

    # loss = tf.reduce_mean(tf.add(attraction_loss, repulsion_loss),
    #                       name="loss")

    return loss, attraction_loss, repulsion_loss, d, max_part
