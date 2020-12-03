from collections import Counter
from utils import Indexer
import numpy as np
from typing import List
import re
import nltk
import itertools

class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (List[string]): list of words
        label (int): 0 or 1 (0 = negative, 1 = positive)
    """

    def __init__(self, words, label):
        self.words = words
        self.label = label

    def __repr__(self):
        return repr(self.words) + "; label=" + repr(self.label)

    def __str__(self):
        return self.__repr__()

def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and cleans the sentences and forms
    SentimentExamples.

    NOTE: Compared to Assignment 1, we lowercase the data for you. This is because the GloVe embeddings don't
    distinguish case and so can only be used with lowercasing.

    :param infile: file to read from
    :return: a list of SentimentExamples parsed from the file
    """
    f = open(infile)
    exs = []
    for line in f:
        if len(line.strip()) > 0:
            fields = line.split("\t")
            if len(fields) != 2:
                fields = line.split()
                label = 0 if "0" in fields[-1] else 1
                ll = [[nltk.word_tokenize(w), ' '] for w in fields[:-1].lower().split()]
                sent = list(itertools.chain(*list(itertools.chain(*ll))))
            else:
                label = 0 if "0" in fields[-1] else 1
                ll = [[nltk.word_tokenize(w), ' '] for w in fields[0].lower().split()]
                sent = list(itertools.chain(*list(itertools.chain(*ll))))
            exs.append(SentimentExample(sent, label))
            
    f.close()
    return exs

class WordEmbeddings:
    """
    Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
    word in the indexer. The 0 vector is returned if an unknown word is queried.
    """
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
        """
        Returns the embedding for a given word
        :param word: The word to look up
        :return: The UNK vector if the word is not in the Indexer or the vector otherwise
        """
        word_idx = self.word_indexer.index_of(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of("UNK")]

def read_word_embeddings(embeddings_file: str) -> WordEmbeddings:
    """
    Loads the given embeddings (ASCII-formatted) into a WordEmbeddings object. Augments this with an UNK embedding
    that is the 0 vector. Reads in all embeddings with no filtering -- you should only use this for relativized
    word embedding files.
    :param embeddings_file: path to the file containing embeddings
    :return: WordEmbeddings object reflecting the words and their embeddings
    """
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    # Make position 0 a PAD token, which can be useful if you
    word_indexer.add_and_get_index("PAD")
    # Make position 1 the UNK token
    word_indexer.add_and_get_index("UNK")
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx+1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            word_indexer.add_and_get_index(word)
            # Append the PAD and UNK vectors to start. Have to do this weirdly because we need to read the first line
            # of the file to see what the embedding dim is
            if len(vectors) == 0:
                vectors.append(np.zeros(vector.shape[0]))
                vectors.append(np.zeros(vector.shape[0]))
            vectors.append(vector)
    f.close()
    # Turn vectors into a 2-D numpy array
    return WordEmbeddings(word_indexer, np.array(vectors))

#################
# You probably don't need to interact with this code unles you want to relativize other sets of embeddings
# to this data. Relativization = restrict the embeddings to only have words we actually need in order to save memory.
# Very advantageous, though it requires knowing your dataset in advance, so it couldn't be used in a production system
# operating on streaming data.
def relativize(file, outfile, word_counter):
    """
    Relativize the word vectors to the given dataset represented by word counts
    :param file: word vectors file
    :param outfile: output file
    :param word_counter: Counter of words occurring in train/dev/test data
    :return:
    """
    f = open(file)
    o = open(outfile, 'w')
    voc = []
    for line in f:
        word = line[:line.find(' ')]
        if word_counter[word] > 0:
            # print("Keeping word vector for " + word)
            voc.append(word)
            o.write(line)
    for word in word_counter:
        if word not in voc:
            count = word_counter[word]
            if count > 1:
                print("Missing " + word + " with count " + repr(count))
    f.close()
    o.close()

def relativize_sentiment_data():
    # Count all words in the train, dev, and *test* sets. Note that this use of looking at the test set is legitimate
    # because we're not looking at the labels, just the words, and it's only used to cache computation that we
    # otherwise would have to do later anyway.
    word_counter = Counter()
    for ex in read_sentiment_examples("data/amazon_cells_labelled.txt"):
        for word in ex.words:
            word_counter[word] += 1
    for ex in read_sentiment_examples("data/imdb_labelled.txt"):
        for word in ex.words:
            word_counter[word] += 1
    for ex in read_sentiment_examples("data/yelp_labelled.txt"):
        for word in ex.words:
            word_counter[word] += 1
    relativize("data/glove.6B.300d.txt", "data/glove.6B.300d-relativized.txt", word_counter)

if __name__=="__main__":
    relativize_sentiment_data()
    exit()
    import sys
    embs = read_word_embeddings("data/glove.6B.300d-relativized.txt")
    query_word_1 = sys.argv[1]
    query_word_2 = sys.argv[2]
    if embs.word_indexer.index_of(query_word_1) == -1:
        print("%s is not in the indexer" % query_word_1)
    elif embs.word_indexer.index_of(query_word_2) == -1:
        print("%s is not in the indexer" % query_word_2)
    else:
        emb1 = embs.get_embedding(query_word_1)
        emb2 = embs.get_embedding(query_word_2)
        print("cosine similarity of %s and %s: %f" % (query_word_1, query_word_2, np.dot(emb1, emb2)/np.sqrt(np.dot(emb1, emb1) * np.dot(emb2, emb2))))
