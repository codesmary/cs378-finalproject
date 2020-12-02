from models import VariationalAutoencoder
import torch
import time
import random
from sentiment_data import *

def train_vae(train_exs: List[SentimentExample], word_embeddings: WordEmbeddings):
    matrix_len = 5020
    emb_dim = 300
    weights_matrix = torch.zeros(matrix_len, emb_dim)

    for i in range(len(word_embeddings.word_indexer.objs_to_ints)):
        word = word_embeddings.word_indexer.get_object(i)
        weights_matrix[i,:] = torch.from_numpy(word_embeddings.get_embedding(word)).float()

    max_sequence_length = 200
    num_epochs = 1
    lr = 1e-3
    model = VariationalAutoencoder(weights_matrix, max_sequence_length)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(num_epochs):
        print("Epoch", epoch+1)
        model.train()

        ex_indices = list(range(len(train_exs)))
        random.shuffle(ex_indices)

        #TODO run on all training examples
        for ex in train_exs[:1]:
            words = ex.words
            input = torch.LongTensor()
            target = torch.LongTensor()

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

            target_words = map(lambda x: x if x != " " else "PAD", words)
            indexes = []

            for word in target_words:
                idx = word_embeddings.word_indexer.index_of(word)
                if idx == -1:
                    idx = 1
                indexes.append(idx)

            if max_sequence_length - len(words) > 0:
                for i in range(max_sequence_length - len(words)):
                    indexes.append(word_embeddings.word_indexer.index_of("PAD"))

            indexes = torch.LongTensor(indexes)
            target = torch.cat((target, indexes), dim=0)

            output = model(input)

            l = loss(output, target)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        model.eval()

    return model

if __name__ == "__main__":
    train_exs = read_sentiment_examples("data/amazon_cells_labelled.txt")
    dev_exs = read_sentiment_examples("data/yelp_labelled.txt")
    word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")

    start = time.time()
    model = train_vae(train_exs, word_embeddings)
    train_eval_time = time.time() - start
    print("Time for training and evaluation: %.2f seconds" % train_eval_time)