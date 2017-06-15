from nltk import word_tokenize
from nltk.corpus import stopwords
from tensorflow.contrib import learn


def preprocess_sentence(sentence, max_len=60):
    en_stopwords = stopwords.words('english')
    if isinstance(sentence, str):
        if len(sentence) > max_len:
            return ' '.join([token.lower() for token in word_tokenize(sentence)
                             if token not in en_stopwords])
        else:
            return ' '.join([token.lower() for token in word_tokenize(sentence)])
    return None


def build_vocabulary(train_corpus):
    """" Build vocabulary, the lookup table and transform the text """

    train_texts = []
    for data in train_corpus.non_sim_data:
        # TODO tokenize some words like @usernames?
        train_texts.append(data.sentence_1)
        train_texts.append(data.sentence_2)
    for data in train_corpus.sim_data:
        train_texts.append(data.sentence_1)
        train_texts.append(data.sentence_2)

    max_document_length = max([len(x.split()) for x in train_texts])
    print("The max. document length is: {}".format(max_document_length))
    # Creates the lookup table: Maps documents to sequences of word ids.
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab_processor = vocab_processor.fit(train_texts)
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

    return vocab_processor, max_document_length