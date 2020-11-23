from models import LSTM
import torch
import time
import random
from sentiment_data import *
import nltk
import json
from heapq import heappush, heapreplace

def sample_random(model, word_embeddings, max_length: int = 20):
    st = ''
    for i in range(max_length):
        prob = torch.FloatTensor(model.get_next_word_log_probs(st))
        idx = torch.distributions.Categorical(logits=prob).sample()
        idx = idx.item()
        next_word = word_embeddings.word_indexer.get_object(idx)
        if next_word == "PAD":
            next_word = " "
        st += next_word
        if next_word == '.':
            break
    return st

# class TopNHeap:
#     """
#     A heap that keeps the top N elements around
#     h = TopNHeap(2)
#     h.add(1)
#     h.add(2)
#     h.add(3)
#     h.add(0)
#     print(h.elements)
#     > [2,3]

#     """
#     def __init__(self, N):
#         self.elements = []
#         self.N = N

#     def add(self, e):
#         from heapq import heappush, heapreplace
#         if len(self.elements) < self.N:
#             heappush(self.elements, e)
#         elif self.elements[0] < e:
#             heapreplace(self.elements, e)

#     def get_all(self):
#         return self.elements

# def beam_search(model, word_embeddings, beam_size: int, n_results: int = 10, max_length: int = 20, average_log_likelihood: bool = False):
#     heap = TopNHeap(beam_size)
#     sent = TopNHeap(n_results)
#     heap.add((0,' '))
#     beams= [heap]
#     for t in range(max_length):
#         beams.append(TopNHeap(beam_size))
#         for beam in beams[t].get_all():
#             scr, str = beam
#             for i in range(len(word_embeddings.word_indexer.objs_to_ints)):
#                 next_word = word_embeddings.word_indexer.get_object(i)
#                 if next_word == "PAD":
#                     next_word = " "
#                 new_str = str + next_word
#                 prob = model.get_next_word_log_probs(str)
#                 if average_log_likelihood:
#                     new_scr = (scr*len(str) + prob[i].item())/(t+1)
#                 else:
#                     new_scr = scr + prob[i].item()
#                 beams[t+1].add((new_scr,new_str))
#                 if new_str[-1] == '.' or len(new_str) == max_length:
#                     sent.add((new_scr,new_str))
#     res = []
#     for s in sent.get_all():
#         res.append(s[1])
#     return res

class RNNLanguageModel:
    def __init__(self, model, word_embeddings):
        self.model = model
        self.word_embeddings = word_embeddings

    def get_next_word_log_probs(self, context):
        if context == " " or context == "":
            input_words = ["PAD"]
        else:
            ll = [[nltk.word_tokenize(w), ' '] for w in context.lower().split()]
            sent = list(itertools.chain(*list(itertools.chain(*ll))))
            input_words = map(lambda x: x if x != " " else "PAD", sent)
        input = torch.LongTensor()
        indexes = []
        for word in input_words:
            idx = word_embeddings.word_indexer.index_of(word)
            if idx == -1:
                idx = 1
            indexes.append(idx)
        indexes = torch.LongTensor(indexes)[None]
        input = torch.cat((input, indexes), dim=0)

        output = self.model(input)[0,:]
        return output.detach().numpy()

    def get_log_prob_sequence(self, next_words, context):
        prob = 0
        output = self.get_next_word_log_probs(context)

        ll = [[nltk.word_tokenize(w), ' '] for w in next_words.lower().split()]
        sent = list(itertools.chain(*list(itertools.chain(*ll))))
        next_words = map(lambda x: x if x != " " else "PAD", sent)
        
        for word in next_words:
            idx = word_embeddings.word_indexer.index_of(word)
            if idx == -1:
                idx = 1
            prob += output[idx]
            context += word
            output = self.get_next_word_log_probs(context)
        
        return prob

def train_lm(train_exs: List[SentimentExample], word_embeddings: WordEmbeddings):
    matrix_len = 5020
    emb_dim = 300
    weights_matrix = torch.zeros(matrix_len, emb_dim)

    for i in range(len(word_embeddings.word_indexer.objs_to_ints)):
        word = word_embeddings.word_indexer.get_object(i)
        weights_matrix[i,:] = torch.from_numpy(word_embeddings.get_embedding(word)).float()

    emb_dim = 5
    num_epochs = 20
    model = LSTM(weights_matrix)
    lr = 1e-3
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(num_epochs):
        print("Epoch", epoch+1)
        model.train()

        ex_indices = list(range(len(train_exs)))
        random.shuffle(ex_indices)

        for ex in train_exs:
            input = torch.LongTensor()
            target = torch.LongTensor()

            for i in range(0, len(ex.words)-emb_dim, emb_dim):
                input_words = map(lambda x: x if x != " " else "PAD", ["PAD"] + ex.words[i+1:i+emb_dim])
                indexes = []
                for word in input_words:
                    idx = word_embeddings.word_indexer.index_of(word)
                    if idx == -1:
                        idx = 1
                    indexes.append(idx)
                indexes = torch.LongTensor(indexes)[None]
                input = torch.cat((input, indexes), dim=0)

                target_words = map(lambda x: x if x != " " else "PAD", ex.words[i:i+emb_dim])
                indexes = []
                for word in target_words:
                    idx = word_embeddings.word_indexer.index_of(word)
                    if idx == -1:
                        idx = 1
                    indexes.append(idx)
                indexes = torch.LongTensor(indexes)
                target = torch.cat((target, indexes), dim=0)
            
            output = model(input)

            l = loss(output, target)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        model.eval()

    return RNNLanguageModel(model, word_embeddings)


def print_evaluation(dev_exs, lm, word_embeddings, output_bundle_path="output.json"):
    """
    Runs both the sanity check and also runs the language model on the given text and prints three metrics: log
    probability of the text under this model (treating the text as one log sequence), average log probability (the
    previous value divided by sequence length), and perplexity (averaged "branching favor" of the model)
    :param lm: model to evaluate
    :param output_bundle_path: the path to print the output bundle to, in addition to printing it
    """
    text = ""
    num_words = 0
    for ex in dev_exs[:10]:
        for word in ex.words:
            text += word
            num_words += 1
        text += " "
        num_words += 1
    
    log_prob = lm.get_log_prob_sequence(text, " ")
    avg_log_prob = log_prob/num_words
    perplexity = np.exp(-log_prob / num_words)
    data = {'log_prob': log_prob, 'avg_log_prob': avg_log_prob, 'perplexity': perplexity}
    print("=====Results=====")
    print(json.dumps(data, indent=2))
    with open(output_bundle_path, 'w') as outfile:
        json.dump(data, outfile)

if __name__ == "__main__":
    train_exs = read_sentiment_examples("data/amazon_cells_labelled.txt")
    dev_exs = read_sentiment_examples("data/yelp_labelled.txt")
    word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")

    start = time.time()
    model = train_lm(train_exs, word_embeddings)
    train_eval_time = time.time() - start
    print("Time for training and evaluation: %.2f seconds" % train_eval_time)

    print_evaluation(dev_exs, model, word_embeddings)

    #fix sampling
    for i in range(10):
        s = sample_random(model, word_embeddings)
        log_prob = model.get_log_prob_sequence(s, " ")
        print(s, log_prob)
    print()

    # for s in beam_search(model, word_embeddings, 10):
    #     log_prob = model.get_log_prob_sequence(s, " ")
    #     print(s, log_prob)
    # print()

    # for s in beam_search(model, word_embeddings, 10, average_log_likelihood=True):
    #     log_prob = model.get_log_prob_sequence(s, " ")
    #     print(s, log_prob)