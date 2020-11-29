import torch
from os import path
from torch.nn.utils import weight_norm

def create_embedding_layer(weights_matrix, non_trainable=True):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({"weight": weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class LSTM(torch.nn.Module):
    def __init__(self, weights_matrix, hidden=150, layers=2, dropout=0.2, latent_dim=2):
        super().__init__()
        self.hidden = hidden
        self.num_layers = layers
        self.embedding, num_embeddings, emb_dim = create_embedding_layer(weights_matrix)
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
        """
        :param X: [batch_size, seq_len] tensor
        :return: context vector z with shape of [batch_size, latent_variable_size]
        """
        embedded_input = self.embedding(X)
        batch_size = embedded_input.shape[0]
        if len(embedded_input.shape) == 2:
            embedded_input = embedded_input.unsqueeze(1)
        init_state = (torch.zeros(2, batch_size, self.hidden),
                      torch.zeros(2, batch_size, self.hidden))
        output, (hidden_state, cell_state) = self.rnn(embedded_input, init_state)
        output = self.classifier(output.reshape(output.size(0)*output.size(1), output.size(2)))
        return output

#TODO add embedding
#TODO condition on "state" z
#concatenating z with every word embedding of the decoder input
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

    def __init__(self, layers=[600,600]):
        super().__init__()
        c = 5020
        L = []
        total_dilation = 1
        for l in layers:
            L.append(self.CausalConv1dBlock(c, l, 3, total_dilation))
            total_dilation *= 2
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Conv1d(c, 5020, 1)
    
    def forward(self, X, z):
        x = F.pad(x, (1, 0), 'constant', 0)
        x = self.network(x)
        x = self.classifier(x)
        return x

class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, weights_matrix):
        super().__init__()
        self.encoder = LSTM(weights_matrix)
        self.decoder = DilatedCNN()

    def forward(self, enc_X, dec_X):
        z = self.encoder(enc_X)
        return self.decoder(dec_X, z)

def save_model(model):
    if isinstance(model, VariationalAutoencoder):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'autoencoder.th'))
    raise ValueError("Model type '%s' not supported!" % str(type(model)))