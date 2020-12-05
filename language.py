import torch

def sample_random(lm, word_embeddings, max_length: int = 10):
    st = ''
    for i in range(max_length):
        prob = torch.FloatTensor(lm.get_next_word_log_probs(st))
        idx = torch.distributions.Categorical(logits=prob).sample()
        idx = idx.item()
        next_word = word_embeddings.word_indexer.get_object(idx)
        if next_word == "PAD":
            next_word = " "
        st += next_word
        if next_word == '.':
            break
    return st

def sample_random_conditioned_on_z(lm, word_embeddings, z, max_length: int = 10):
    st = ''
    for i in range(max_length):
        prob = torch.FloatTensor(lm.get_next_word_log_probs_conditioned_on_z(st, z))
        idx = torch.distributions.Categorical(logits=prob).sample()
        idx = idx.item()
        next_word = word_embeddings.word_indexer.get_object(idx)
        if next_word == "PAD":
            next_word = " "
        st += next_word
        if next_word == '.':
            break
    return st