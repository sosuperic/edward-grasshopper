# model.py

"""
Classes:
- InfluencersLSTM: Pass GMM components from Artist into LSTM to create final hidden state representing influencers  
- ArtistGAN

Available functions:
- test_influencers_lstm: Pass in mock input to InfluencersLSTM
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as weight_init

from artist_embeddings import Artist


####################################################################################################################
# InfluencersLSTM
####################################################################################################################

class InfluencersLSTM(nn.Module):
    """
    LSTM that that takes Artist GMM embeddings as input.
    Output embedding representing influencers is the final hidden state (+ cell state?)
    
    Public attributes:
    - x_emb_dim: dimensionality of input to linear layers
    - emb_size: dimensionality of linear layers applied to input, before LSTM
    - hidden_size: hidden size of LSTM
    - n_layers: number of layers in LSTM
    - lstm: nn LSTM
    """
    def __init__(self, x_emb_dim, emb_size, hidden_size, n_layers=1):
        super(InfluencersLSTM, self).__init__()

        self.x_emb_dim = x_emb_dim
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lin_layer = nn.Linear(x_emb_dim, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers)   # input_size, hidden_size, num_layers

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, batch_size, self.hidden_size)),
                Variable(torch.zeros(1, batch_size, self.hidden_size)))

    def forward(self, x, lengths, init_h, init_c):
        """
        Inputs:
            - x: Tensor of size [max_len, batch, emb_dim]
            - lengths: list of length batch (used to mask x), i.e. number of influencers for each artist in batch
                values are in [1, max_len]
        """
        # Apply linear layer to data and then create PackedSequence, which masks
        # if torch.cuda.is_available():
        #     self.lin_layer.cuda()
        #     self.lstm.cuda()

        x = self.lin_layer(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False)

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
class Generator(nn.Module):
    # initializers
    def __init__(self, z_size, num_filters, influencers_emb_size):
        super(Generator, self).__init__()

        self.tanh = nn.Tanh()

        self.lin1 = nn.Linear(z_size + influencers_emb_size, num_filters[0])

        self.conv2 = nn.ConvTranspose2d(num_filters[0], num_filters[1], 5, 2)
        self.bn2 = nn.BatchNorm2d(num_filters[1])
        self.conv3 = nn.ConvTranspose2d(num_filters[1], num_filters[2], 5, 2)
        self.bn3 = nn.BatchNorm2d(num_filters[2])
        self.conv4 = nn.ConvTranspose2d(num_filters[2], num_filters[3], 5, 2)
        self.bn4 = nn.BatchNorm2d(num_filters[3])
        self.conv5 = nn.ConvTranspose2d(num_filters[3], num_filters[4], 5, 2)
        self.bn5 = nn.BatchNorm2d(num_filters[4])
        self.conv6 = nn.ConvTranspose2d(num_filters[4], 3, 4, 2)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, z, influencers_emb):
        """
        Inputs:
        - z: (batch, z_size, 1, 1)
        - influencers_emb (batch, embedding size, 1, 1)
        """
        batch_size = z.size(0)

        # Combine noise and influencers embedding, pass through linear layer, reshape
        comb = torch.cat([z, influencers_emb], dim=1)
        comb = comb.view(batch_size, -1)
        comb = self.lin1(comb)                                  # (batch, num_filters[0])
        comb = comb.view(comb.size(0), comb.size(1), 1, 1)      # (batch, num_filters[0], 1, 1)

        # Pass through conv layers
        comb = F.leaky_relu(self.bn2(self.conv2(comb)), 0.2)
        comb = F.leaky_relu(self.bn3(self.conv3(comb)), 0.2)
        comb = F.leaky_relu(self.bn4(self.conv4(comb)), 0.2)
        comb = F.leaky_relu(self.bn5(self.conv5(comb)), 0.2)
        comb = self.tanh(self.conv6(comb))

        return comb

class Discriminator(nn.Module):

    def __init__(self, num_filters, num_labels):
        super(Discriminator, self).__init__()

        self.num_filters = num_filters
        self.num_labels = num_labels

        self.conv1 = nn.Conv2d(3, num_filters[0], 3, 2)                    # 16 maps
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], 3, 1)      # 32
        self.bn2 = nn.BatchNorm2d(num_filters[1])
        self.conv3 = nn.Conv2d(num_filters[1], num_filters[2], 3, 2)    # 64
        self.bn3 = nn.BatchNorm2d(num_filters[2])
        self.conv4 = nn.Conv2d(num_filters[2], num_filters[3], 3, 1)  # 128
        self.bn4 = nn.BatchNorm2d(num_filters[3])
        self.conv5 = nn.Conv2d(num_filters[3], num_filters[4], 3, 2) # 256
        self.bn5 = nn.BatchNorm2d(num_filters[4])
        self.conv6 = nn.Conv2d(num_filters[4], num_filters[5], 3, 1) # 512
        self.bn6 = nn.BatchNorm2d(num_filters[5])

        # Calculate size after passing x through above convolution layers
        # Need this to define linear layer
        # h1 = math.floor()
        # Just hardcode for now
        self.lin = nn.Linear(61952, num_labels + 1)# plus 1 for discrim # [16, 32, 64, 128, 256, 512]
        # self.lin = nn.Linear(15488, num_labels + 1) # [16, 32, 64, 64, 128, 128]
        # self.lin = nn.Linear(1936, num_labels + 1) # [16, 32, 64, 64, 32, 16]

        # Kind of following Imagenet AC-GAN paper
        # self.conv1 = nn.Conv2d(3, num_filters, 3, 2)                    # 16 maps
        # self.conv2 = nn.Conv2d(num_filters, num_filters * 2, 3, 1)      # 32
        # self.bn2 = nn.BatchNorm2d(num_filters * 2)
        # self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 3, 2)  # 64
        # self.bn3 = nn.BatchNorm2d(num_filters * 4)
        # self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 3, 1)  # 128
        # self.bn4 = nn.BatchNorm2d(num_filters * 8)
        # self.conv5 = nn.Conv2d(num_filters * 8, num_filters * 16, 3, 2) # 256
        # self.bn5 = nn.BatchNorm2d(num_filters * 16)
        # self.conv6 = nn.Conv2d(num_filters * 16, num_filters * 32, 3, 1) # 512
        # self.bn6 = nn.BatchNorm2d(num_filters * 32)

        # self.linear = nn.Linear(num_filters * 32, num_labels + 1)   # plus 1 for discrim
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.2)

        x = x.view(batch_size, -1)
        # print x.size()
        x = self.lin(x)  # (1, 203)

        # Use first element for discriminator, rest for auxiliary classifier
        discrim = self.sigmoid(x[:,0])          # (batch, )
        discrim = torch.unsqueeze(discrim, 1)   # (batch, 1)
        aux = self.softmax(x[:,1:])             # (batch, num_labels)

        return discrim, aux


if __name__ == '__main__':
    test_influencers_lstm()