from models import load_model
import torch
import nltk
import json
from language import *
from sentiment_data import *

class LanguageModel:
    def __init__(self, model, word_embeddings, max_sequence_length):
        self.model = model
        self.word_embeddings = word_embeddings
        self.max_sequence_length = max_sequence_length

    def get_next_word_log_probs_conditioned_on_z(self, context, z):
        if context == " " or context == "":
            input_words = ["PAD"]
        else:
            ll = [[nltk.word_tokenize(w), ' '] for w in context.lower().split()]
            sent = list(itertools.chain(*list(itertools.chain(*ll))))
            input_words = list(map(lambda x: x if x != " " else "PAD", sent))
        input = torch.LongTensor()
        indexes = []
        for word in input_words:
            idx = word_embeddings.word_indexer.index_of(word)
            if idx == -1:
                idx = 1
            indexes.append(idx)

        if self.max_sequence_length - len(input_words) > 0:
            for i in range(self.max_sequence_length - len(input_words)):
                indexes.append(word_embeddings.word_indexer.index_of("PAD"))

        indexes = torch.LongTensor(indexes)[None]
        input = torch.cat((input, indexes), dim=0)

        output = self.model(input, training=False, z=z)[0,:]
        return output.detach().numpy()

    def get_next_word_log_probs(self, context):
        if context == " " or context == "":
            input_words = ["PAD"]
        else:
            ll = [[nltk.word_tokenize(w), ' '] for w in context.lower().split()]
            sent = list(itertools.chain(*list(itertools.chain(*ll))))
            input_words = list(map(lambda x: x if x != " " else "PAD", sent))
        input = torch.LongTensor()
        indexes = []
        for word in input_words:
            idx = word_embeddings.word_indexer.index_of(word)
            if idx == -1:
                idx = 1
            indexes.append(idx)

        if self.max_sequence_length - len(input_words) > 0:
            for i in range(self.max_sequence_length - len(input_words)):
                indexes.append(word_embeddings.word_indexer.index_of("PAD"))

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

    def get_log_prob_sequence_conditioned_on_z(self, next_words, context, z):
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
            output = self.get_next_word_log_probs_conditioned_on_z(context, z)
        
        return prob

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
    for ex in dev_exs:
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
    dev_exs = read_sentiment_examples("data/yelp_labelled.txt")[:100]
    word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
    max_sequence_length = 50
    model = load_model()

    neg_data = torch.FloatTensor()
    pos_data = torch.FloatTensor()

    for ex in train_exs:
        words = ex.words[:max_sequence_length]
        label = ex.label

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

        z = model.get_latent_vector(input)
        
        if label == 0:
            neg_data = torch.cat((neg_data, z[None,:]), dim=0)
        else:
            pos_data = torch.cat((pos_data, z[None,:]), dim=0)

    neg_data_mean = neg_data.mean(dim=0)
    pos_data_mean = pos_data.mean(dim=0)

    language_model = LanguageModel(model, word_embeddings, max_sequence_length)
    
    #Output log prob and perplexity
    print_evaluation(dev_exs, language_model, word_embeddings)

    #Output random samples
    print("purely random samples")
    for i in range(10):
        s = sample_random(language_model, word_embeddings)
        log_prob = language_model.get_log_prob_sequence(s, " ")
        print(s, log_prob)

    print()

    #Output samples conditioned on average negative vector
    print("conditioned negatively")
    for i in range(10):
        s = sample_random_conditioned_on_z(language_model, word_embeddings, neg_data_mean)
        log_prob = language_model.get_log_prob_sequence_conditioned_on_z(s, " ", neg_data_mean)
        print(s, log_prob)

    print()

    #Output samples conditioned on average positive vector
    print("conditioned positively")
    for i in range(10):
        s = sample_random_conditioned_on_z(language_model, word_embeddings, pos_data_mean)
        log_prob = language_model.get_log_prob_sequence_conditioned_on_z(s, " ", pos_data_mean)
        print(s, log_prob)