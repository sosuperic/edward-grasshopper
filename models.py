# model.py

"""
Classes:
- InfluencersLSTM: Pass GMM components from Artist into LSTM to create final hidden state representing influencers  
- ArtistGAN
"""

import torch
from torch.autograd import Variable
import torch.nn as nn

from artist_embeddings import Artist

class InfluencersLSTM(nn.Module):
    """
    LSTM that that takes Artist GMM embeddings as input.
    Output embedding representing influencers is the final hidden state (+ cell state?)
    
    Public attributes:
    - emb_size: dimensionality of linear layers applied to input, before LSTM
    - hidden_size: hidden size of LSTM
    - n_layers: number of layers in LSTM
    - lstm: nn LSTM
    """
    def __init__(self, emb_size, hidden_size, n_layers=1):
        super(InfluencersLSTM, self).__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers)   # input_size, hidden_size, num_layers

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, x):
        """
        Inputs:
        - x: expected (seq_len, batch, input_size)
        """
        # Save so we can reshape back after linear layer
        seq_len, batch_size, input_size = x.size(0), x.size(1), x.size(2)

        # Apply linear layer to input first
        # Have to reshape so that linear is applied to each sequence+batch item separately
        x = x.view(seq_len, batch_size * input_size)            # reshape to (seq_len, batch * input_size)
        x = nn.Linear(batch_size * input_size, batch_size * self.emb_size)(x)     # (seq_len, batch * emb_size)
        x = x.view(seq_len, batch_size, self.emb_size)                     # (seq_len, batch, emb_size)

        # Pass into LSTM
        output, (h_n, c_n) = self.lstm(x)
        # output: (seq_len, batch, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # c_n: (num_layers * num_directions, batch, hidden_size)
        influencer_emb = torch.cat((h_n, c_n), 2)       # (num_layers * num_directions, batch, 2 * hidden_size)
        influencer_emb = influencer_emb.view(batch_size, -1)    # (batch, final_emb_size)

        return influencer_emb

def test_influencers_lstm():
    """Pass in mock input to InfluencersLSTM"""
    mock_input = Variable(torch.randn(9, 1, Artist.component_size))
    # 9 mixture components, e.g. 3 influencers, each with 3 components
    # 1 batch size (num artists), batching will occur at image level (GAN model)
    # Artist.comp_emb_size: emb_size of one component of Artist

    emb_size = 16
    hidden_size = 32
    n_layers = 1
    influencers_lstm = InfluencersLSTM(emb_size, hidden_size, n_layers)
    influencers_emb = influencers_lstm(mock_input)

    print influencers_emb

if __name__ == '__main__':
    test_influencers_lstm()