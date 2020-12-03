import torch
from torch import save
from torch import load
from os import path
from torch.nn.utils import weight_norm
from sentiment_data import *

class LSTM(torch.nn.Module):
    def __init__(self, emb_dim, hidden=150, layers=2, dropout=0.2, latent_dim=2):
        super().__init__()
        self.hidden = hidden
        self.num_layers = layers
        self.rnn = torch.nn.LSTM(emb_dim, self.hidden, num_layers=self.num_layers, dropout=dropout, batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden, latent_dim),
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
        batch_size = X.shape[0]
        init_state = (torch.zeros(2, batch_size, self.hidden),
                      torch.zeros(2, batch_size, self.hidden))
        output, (hidden_state, cell_state) = self.rnn(X, init_state)
        output = self.classifier(output.reshape(output.size(0)*output.size(1), output.size(2)))
        
        return output

class DilatedCNN(torch.nn.Module):
    class CausalConv1dBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.ConstantPad1d((2*dilation,0),0),
                weight_norm(torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.ConstantPad1d((2*dilation,0),0),
                weight_norm(torch.nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(0.1)
            )
            self.resize = torch.nn.Conv1d(in_channels, out_channels, 1)
            self.relu = torch.nn.LeakyReLU()

        def forward(self, x):
            residual = x
            x = self.network(x)
            residual = self.resize(residual)
            return self.relu(x + residual)

    def __init__(self, max_sequence_length, num_embeddings, layers=[600,600]):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.num_embeddings = num_embeddings
        
        c = max_sequence_length
        L = []
        total_dilation = 1
        for l in layers:
            L.append(self.CausalConv1dBlock(c, l, 3, total_dilation))
            total_dilation *= 2
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Conv1d(c, max_sequence_length * num_embeddings, 1)
    
    def forward(self, X, enc_state):
        #"condition" on z
        new_x = torch.zeros((1, X.shape[1], X.shape[2] + 1))
        new_x[0,:,0] = enc_state
        new_x[0,:,1:] = X

        X = self.network(new_x)
        X = self.classifier(X)
        X = X.mean(dim=[2])
        X = X.reshape((self.max_sequence_length, self.num_embeddings))
        return X

def create_embedding_layer(weights_matrix, non_trainable=True):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({"weight": weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, weights_matrix, max_sequence_length):
        super().__init__()
        self.embedding, num_embeddings, emb_dim = create_embedding_layer(weights_matrix)
        self.encoder = LSTM(emb_dim)
        self.decoder = DilatedCNN(max_sequence_length, num_embeddings)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        sigma = torch.exp(log_var/2)
        eps = torch.randn_like(sigma)
        sample = mu + (sigma * eps)
        return sample

    def forward(self, X, training=True, z=None):
        embedded_input = self.embedding(X)

        if training and not z:
            output = self.encoder(embedded_input)
            mu, log_var = output[:,0], output[:,1]
            sample = self.reparameterize(mu, log_var)
        elif not training and z:
            sample = z
        else:
            raise Exception("Must be in training or testing mode")

        return self.decoder(embedded_input, sample)

    def get_latent_vector(self, X):
        embedded_input = self.embedding(X)

        output = self.encoder(embedded_input)
        mu, log_var = output[:,0], output[:,1]
        sample = self.reparameterize(mu, log_var)

        return sample
       

def save_model(model):
    if isinstance(model, VariationalAutoencoder):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'autoencoder.th'))
    raise ValueError("Model type '%s' not supported!" % str(type(model)))

def load_model():
    word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
    max_sequence_length = 50
    matrix_len = 5020
    emb_dim = 300
    weights_matrix = torch.zeros(matrix_len, emb_dim)

    for i in range(len(word_embeddings.word_indexer.objs_to_ints)):
        word = word_embeddings.word_indexer.get_object(i)
        weights_matrix[i,:] = torch.from_numpy(word_embeddings.get_embedding(word)).float()

    r = VariationalAutoencoder(weights_matrix, max_sequence_length)
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'autoencoder.th'), map_location='cpu'))
    return r