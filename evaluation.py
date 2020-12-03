from models import load_model
import nltk
import itertools
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from sentiment_data import *

def prepare_input(sentence):
    word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
    max_sequence_length = 50

    ll = [[nltk.word_tokenize(w), ' '] for w in sentence.lower().split()]
    words = list(itertools.chain(*list(itertools.chain(*ll))))
    
    input = torch.LongTensor()
    input_words = map(lambda x: x if x != " " else "PAD", ["PAD"] + words[:-1])
    indexes = []

    for word in input_words[:max_sequence_length]:
        idx = word_embeddings.word_indexer.index_of(word)
        if idx == -1:
            idx = 1
        indexes.append(idx)

    if max_sequence_length - len(words) > 0:
        for i in range(max_sequence_length - len(words)):
            indexes.append(word_embeddings.word_indexer.index_of("PAD"))

    indexes = torch.LongTensor(indexes)[None]
    input = torch.cat((input, indexes), dim=0)

    return input


if __name__ == "__main__":
    #get output from training data
    train_exs = read_sentiment_examples("data/amazon_cells_labelled.txt")
    word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
    max_sequence_length = 50
    labels = []
    model = load_model()

    data = torch.FloatTensor()
    for ex in train_exs:
        words = ex.words[:max_sequence_length]
        label = ex.label
        labels.append(label)

        input = torch.LongTensor()
        input_words = map(lambda x: x if x != " " else "PAD", ["PAD"] + words[:-1])
        indexes = []

        for word in input_words:
            idx = word_embeddings.word_indexer.index_of(word)
            if idx == -1:
                idx = 1
            indexes.append(idx)

        if max_sequence_length - len(words) > 0:
            for i in range(max_sequence_length - len(words)):
                indexes.append(word_embeddings.word_indexer.index_of("PAD"))

        indexes = torch.LongTensor(indexes)[None]
        input = torch.cat((input, indexes), dim=0)

        z = model.get_latent_vector(input)[None,:]
        data = torch.cat((data, z), dim=0)

    #create plot from latent vectors
    data = data.detach().numpy()
    data = pd.DataFrame(data)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data_scaled)

    pca_data_df = pd.DataFrame(pca_data)
    
    neg_x = []
    neg_y = []
    pos_x = []
    pos_y =[]

    for i, row in pca_data_df.iterrows():
        label = labels[i]
        x, y = row[0], row[1]
        
        if not label:
            neg_x.append(x)
            neg_y.append(y)
        else:
            pos_x.append(x)
            pos_y.append(y)

    plt.scatter(neg_x, neg_y, c="r")
    plt.scatter(pos_x, pos_y, c="g")

    plt.savefig("amazon_pca.png")