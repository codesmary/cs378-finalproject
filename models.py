import torch
from os import path

def create_embedding_layer(weights_matrix, non_trainable=True):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({"weight": weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class LSTM(torch.nn.Module):
    def __init__(self, weights_matrix):
        super().__init__()
        self.hidden = 150
        self.num_layers = 2
        self.embedding, num_embeddings, emb_dim = create_embedding_layer(weights_matrix)
        self.rnn = torch.nn.LSTM(emb_dim, self.hidden, num_layers=self.num_layers, dropout=0.2, batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden, num_embeddings),
            torch.nn.LogSoftmax(dim=1)
        )
        self.init_weight()

    def init_weight(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, X):
        embedded_input = self.embedding(X)
        batch_size = embedded_input.shape[0]
        if len(embedded_input.shape) == 2:
            embedded_input = embedded_input.unsqueeze(1)
        init_state = (torch.zeros(2, batch_size, self.hidden),
                      torch.zeros(2, batch_size, self.hidden))
        output, (hidden_state, cell_state) = self.rnn(embedded_input, init_state)
        output = self.classifier(output.reshape(output.size(0)*output.size(1), output.size(2)))
        return output

class DilatedCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, state):
        raise NotImplementedError()

class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X):
        enc_output = self.encoder(enc_X)
        dec_state = self.decoder_init_state(enc_output)
        return self.decoder(dec_X, dec_state)

def save_model(model):
    if isinstance(model, VariationalAutoencoder):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'autoencoder.th'))
    raise ValueError("Model type '%s' not supported!" % str(type(model)))