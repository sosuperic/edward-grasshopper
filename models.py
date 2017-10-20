# model.py

"""
Classes:
- InfluencersLSTM: Pass GMM components from Artist into LSTM to create final hidden state representing influencers  
- ArtistGAN

Available functions:
- test_influencers_lstm: Pass in mock input to InfluencersLSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn

from artist_embeddings import Artist


####################################################################################################################
# InfluencersLSTM
####################################################################################################################

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
        return (Variable(torch.zeros(1, 1, self.hidden_size)),
                Variable(torch.zeros(1, 1, self.hidden_size)))

    def forward(self, x, init_h, init_c):
        """
        Inputs:
        - x: PackedSequence
        """
        # Apply linear layer to data and then re-pack it
        # Note: we are manually creating PackedSequence here using already packed x.data (and applying a linear
        # layer to it). x.data is 2D
        # This is despite the docs saying not to create it manually, but rather use pack_padded_sequence, which
        # we did do to create x originally. But this seems the most straight forward way
        lin_layer = nn.Linear(x.data.size(1), self.emb_size)
        packed = nn.utils.rnn.PackedSequence(lin_layer(x.data), x.batch_sizes)

        # Pass into LSTM
        output, (h_n, c_n) = self.lstm(packed, (init_h, init_c))
        # output: (seq_len, batch, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # c_n: (num_layers * num_directions, batch, hidden_size)
        influencer_emb = torch.cat((h_n, c_n), 2)       # (num_layers * num_directions, batch, 2 * hidden_size)

        return influencer_emb

    def get_final_emb_size(self):
        """Return final embedding size, including layers + directions (see end of forward())"""
        size = self.n_layers * 1 * 2 * self.hidden_size
        return size

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

####################################################################################################################
# Utils
####################################################################################################################
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

####################################################################################################################
# ArtistGAN
####################################################################################################################
#
# class ArtistGANFactory(object):
#     """
#     The ArtistGAN is really a cDCGAN.
#     This class simply acts as a wrapper to produce the Generator and Discriminator, loss, etc.
#     """
#     def __init__(self):
#         super(ArtistGANFactory, self).__init__()
#
#     @staticmethod
#     def get_generator(z_size, d):
#         return Generator(z_size, d)
#
#     @staticmethod
#     def get_discriminator(d):
#         return Discriminator(d)
#
#     @staticmethod
#     def get_loss(d):
#         """
#         TODO: replace with Wasserstein loss?
#         """
#         return nn.BCELoss()

# G(z)
class Generator(nn.Module):
    # initializers
    def __init__(self, z_size, num_filters, influencers_emb_size):
        super(Generator, self).__init__()

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

        # First layers operating on z
        self.conv1_z = nn.ConvTranspose2d(z_size, num_filters * 4, 4, 1, 0)
        self.bn1_z = nn.BatchNorm2d(num_filters * 4)
        self.conv2_z = nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 4, 2, 1)
        self.bn2_z = nn.BatchNorm2d(num_filters * 2)

        # First layers operating on influencers embedding (in parallel with above)
        self.conv1_inf = nn.ConvTranspose2d(influencers_emb_size, num_filters * 4, 4, 1, 0)
        self.bn1_inf = nn.BatchNorm2d(num_filters * 4)
        self.conv2_inf = nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 4, 2, 1)
        self.bn2_inf = nn.BatchNorm2d(num_filters * 2)

        # After fusion of z and influencers
        self.conv3 = nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(num_filters * 2)
        self.conv4 = nn.ConvTranspose2d(num_filters * 2, num_filters, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(num_filters)
        self.conv5 = nn.ConvTranspose2d(num_filters, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, z, influencers_emb):
        """
        Inputs:
        - z: (batch, z_size, 1, 1)
        - influencers_emb (batch, embedding size, 1, 1)
        """
        z = self.relu(self.bn1_z(self.conv1_z(z)))
        z = self.relu(self.bn2_z(self.conv2_z(z)))

        influencers_emb = self.relu(self.bn1_inf(self.conv1_inf(influencers_emb)))
        influencers_emb = self.relu(self.bn2_inf(self.conv2_inf(influencers_emb)))

        # Fuse
        x = torch.cat([z, influencers_emb], 1)      # (batch, filters, x, y)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        x = self.tanh(x)

        return x

class Discriminator(nn.Module):

    def __init__(self, num_filters, num_labels):
        super(Discriminator, self).__init__()

        self.num_filters = num_filters

        self.conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(num_filters * 2)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(num_filters * 4)
        self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(num_filters * 8)
        self.conv5 = nn.Conv2d(num_filters * 8, num_filters, 4, 1, 0)

        self.discrim_linear = nn.Linear(num_filters, 1)
        self.sigmoid = nn.Sigmoid()

        self.aux_linear = nn.Linear(num_filters, num_labels)
        self.softmax = nn.Softmax()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.conv5(x)

        x = x.view(-1, self.num_filters)

        discrim = self.discrim_linear(x)
        discrim = self.sigmoid(discrim)

        aux = self.aux_linear(x)
        aux = self.softmax(aux)

        return discrim, aux

if __name__ == '__main__':
    test_influencers_lstm()