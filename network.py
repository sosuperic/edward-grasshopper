# network.py

"""
Available classes:
- Network
"""

from datetime import datetime
import os
import pickle
from tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from config import SAVED_MODELS_PATH, WIKIART_ARTIST_EMBEDDINGS_PATH
from datasets import get_wikiart_data_loader
from models import InfluencersLSTM, ArtistGANFactory

###############################################################################
# NETWORK
###############################################################################
class Network(object):
    """
    Class used for training and evaluation
    """
    def __init__(self, hp):
        self.hp = hp

        # Define / set up models
        self.influencers_lstm = InfluencersLSTM(hp.emb_size, hp.hidden_size)
        self.artist_G = ArtistGANFactory.get_generator(hp.d)
        self.artist_D = ArtistGANFactory.get_discriminator(hp.d)
        self.models = [self.influencers_lstm, self.artist_G, self.artist_D]

        # Initialize models
        self.artist_G.weight_init(mean=0.0, std=0.02)
        self.artist_D.weight_init(mean=0.0, std=0.02)

        # Move to cuda
        if torch.cuda.is_available():
            for model in self.models:
                model.cuda()

    def train(self):
        """Train on training data split"""
        for model in self.models:
            model.train()

        # Set up loss

        # Adam optimizer
        LSTM_optimizer = optim.Adam(self.influencers_lstm.parameters(), lr=self.hp.lr, betas=(0.5, 0.999))
        G_optimizer = optim.Adam(self.artist_G.parameters(), lr=self.hp.lr, betas=(0.5, 0.999))
        D_optimizer = optim.Adam(self.artist_D.parameters(), lr=self.hp.lr, betas=(0.5, 0.999))

        # Get Tensorboard summary writer
        writer = SummaryWriter('runs/' + datetime.now().strftime('%B%d_%H-%M-%S'))

        # Get data_loaders
        train_data_loader = get_wikiart_data_loader(self.hp.batch_size, self.hp.img_size)
        # valid_data_loader = get_you_data_loader('valid', hp.batch_size)

        for e in range(self.hp.max_nepochs):
            for train_idx, inputs in enumerate(train_data_loader):
            # for train_idx, (inputs, labels) in enumerate(train_data_loader):
                # Convert raw Tensors to nn Variables
                # inputs, labels = self.turn_inputs_labels_to_vars(inputs, labels)

                LSTM_optimizer.zero_grad()
                G_optimizer.zero_grad()
                D_optimizer.zero_grad()

                # Pass through network and get loss
                # outputs = self.model(inputs)
                # loss = criterion(outputs, labels)
                # print 'Loss: {}'.format(self.var2numpy(loss))

                # Optimize
                # loss.backward()
                # LSTM_optimizer.step()
                # G_optimizer.step()
                # D_optimizer.step()

                # Write loss to Tensorboard
                # if train_idx % 10 == 0:
                #     writer.add_scalar('loss', loss.clone().cpu().data.numpy(), e * train_data_loader.__len__() + train_idx)

            # Save
            # if e % self.hp.save_every_nepochs == 0:
            #     torch.save(self.model.state_dict(), os.path.join(SAVED_MODELS_PATH, 'e{}.pt'.format(e)))

            # # Evaluate on validation split
            # for valid_idx, (inputs, labels) in enumerate(valid_data_loader):
            #     inputs, labels = self.turn_inputs_labels_to_vars(inputs, labels)
            #     outputs = self.model(inputs)
            #     _, top_predicted_labels = torch.max(outputs, 1)

            # Write parameters to Tensorboard every epoch
            # for name, param in self.model.named_parameters():
            #     writer.add_histogram(name, param.clone().cpu().data.numpy(), e * train_data_loader.__len__() + train_idx)

        writer.close()

    def generate(self, influencers):
        """
        Inputs:
        - influencers: comma-separated list of influencers passed in by command line
        """
        # Load models
        # self.load_model()self.hp.load_model_fp

        for model in self.models:
            model.eval()

        # Load influencer embeddings, pass into InfluencerLSTM
        influencers = influencers.split(',')
        influencer_embs = []
        for influencer in influencers:
            emb_path = os.path.join(WIKIART_ARTIST_EMBEDDINGS_PATH, influencer, 'embedding.pkl')
            emb = pickle.load(open(emb_path, 'r'))
            for mode_idx in range(emb.shape[0]):
                influencer_embs.append(emb[mode_idx])
        influencers_emb = self.influencers_lstm(influencer_embs)

        # Get noise vector, other vectors for Generator
        # Concate with influencers_emb
        # output = self.artist_G(input)

        # Save img

    ##############
    # Utils
    ##############
    def load_model(self, models_fp):
        """Load model parameters from given filename"""
        print 'Loading model parameters from {}'.format(models_fp)
        # self.model.load_state_dict(torch.load(fp))

    def turn_inputs_labels_to_vars(self, inputs, labels):
        """Wrap tensors in Variables for autograd; also move to gpu potentially"""
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        return inputs, labels

    def var2numpy(self, var):
        """Return numpy matrix of data in Variable. Separate util function because CUDA tensors require
        moving to cpu first"""
        if var.is_cuda:
            return var.data.cpu().numpy()
        else:
            return var.data.numpy()
