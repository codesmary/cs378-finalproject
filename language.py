import torch

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