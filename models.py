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
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, x):
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
        output, (h_n, c_n) = self.lstm(packed)
        # output: (seq_len, batch, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # c_n: (num_layers * num_directions, batch, hidden_size)
        influencer_emb = torch.cat((h_n, c_n), 2)       # (num_layers * num_directions, batch, 2 * hidden_size)

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

class ArtistGANFactory(object):
    """
    The ArtistGAN is really a cDCGAN.
    This class simply acts as a wrapper to produce the Generator and Discriminator, loss, etc.
    """
    def __init__(self):
        super(ArtistGANFactory, self).__init__()

    @staticmethod
    def get_generator(d):
        return Generator(d)

    @staticmethod
    def get_discriminator(d):
        return Discriminator(d)

    @staticmethod
    def get_loss(d):
        """
        TODO: replace with Wasserstein loss?
        """
        return nn.BCELoss()

# G(z)
class Generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d * 4, 4, 1, 0)      # in_channels (z is noise), out_channels; why *4?
        self.deconv1_1_bn = nn.BatchNorm2d(d*4)
        self.deconv1_2 = nn.ConvTranspose2d(2, d*4, 4, 1, 0)          # This is on y, or in our case (influecnerlstm embedding)
        self.deconv1_2_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        # self.deconv4 = nn.ConvTranspose2d(d, 3, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1) #    (3, )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label):
        x = F.leaky_relu(self.deconv1_1_bn(self.deconv1_1(input)), 0.2)
        y = F.leaky_relu(self.deconv1_2_bn(self.deconv1_2(label)), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        # x = F.tanh(self.deconv4(x))
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = F.tanh(self.deconv5(x))

        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(3, d/2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(2, d/2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        # self.conv4 = nn.Conv2d(d*4, 1, 4, 1, 0)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        # x = F.sigmoid(self.conv4(x))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

if __name__ == '__main__':
    test_influencers_lstm()