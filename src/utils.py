from tensorflow.contrib import learn
import numpy as np
from collections import Counter


def build_vocabulary(train_corpus):
    """" Build vocabulary, the lookup table and transform the text """

    train_texts = []
    for data in train_corpus.non_sim_data:
        # TODO tokenize some words like @usernames?
        train_texts.append(data.tweet_a)
        train_texts.append(data.tweet_b)
    for data in train_corpus.sim_data:
        train_texts.append(data.tweet_a)
        train_texts.append(data.tweet_b)

    max_document_length = max([len(x.split()) for x in train_texts])
    print("The max. document length is: {}".format(max_document_length))
    # Creates the lookup table: Maps documents to sequences of word ids.
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab_processor = vocab_processor.fit(train_texts)
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

    return vocab_processor, max_document_length


def shuffle_epochs(sim_data, non_sim_data, mixed=False):
    if mixed:
        data = np.append(sim_data, non_sim_data)
        shuffle_indices = np.random.permutation(np.arange(len(sim_data)+len(non_sim_data)))
        print(type(data[shuffle_indices]), data[shuffle_indices].shape)
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
            print(start_index, len(sim_data_batch))
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


def batch_iter_mixed(data, batch_size, num_batches_per_epoch):
    data_size = len(data)

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        if start_index < data_size:
            data_batch = data[start_index:end_index]
            # labels = [d.label for d in data_batch]
            # print(Counter(labels), start_index, end_index)
            if len(data_batch) < batch_size:
                new_end_idx = batch_size - len(data_batch)
                yield (np.append(data_batch, data[:new_end_idx]), '-')
            else:
                yield (data_batch, '-')