# network.py

"""
Available classes:
- Network
"""

from datetime import datetime
import json
import numpy as np
import os
import pickle
from tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils

from config import SAVED_MODELS_PATH, ARTIST_EMB_DIM ,\
    WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH, TRAIN_ON_ALL
from datasets import get_wikiart_data_loader, WikiartDataset
from models import InfluencersLSTM, Generator, Discriminator

###############################################################################
# NETWORK
###############################################################################
class Network(object):
    """
    Class used for training and evaluation
    
    Public attributes:
    - hp:
    - _artist2info
    """
    def __init__(self, hp):
        self.hp = hp
        self._artist2info = None

        # Define / set up models
        self.influencers_lstm = InfluencersLSTM(ARTIST_EMB_DIM, hp.lstm_emb_size, hp.lstm_hidden_size)
        self.artist_G = Generator(hp.z_size, hp.g_num_filters, self.influencers_lstm.get_final_emb_size())
        self.artist_D = Discriminator(hp.d_num_filters, WikiartDataset.get_num_valid_artists(TRAIN_ON_ALL))
        self.models = [self.influencers_lstm, self.artist_G, self.artist_D]

        # Initialize models
        self.artist_G.weight_init(mean=0.0, std=0.02)
        self.artist_D.weight_init(mean=0.0, std=0.02)

        # Load weights potentially

        # Move to cuda
        if torch.cuda.is_available():
            for model in self.models:
                model.cuda()

    # @property
    # def artist2info(self):
    #     if self._artist2info is None:
    #         self._artist2info = pickle.load(open(WIKIART_ARTIST_TO_INFO_PATH, 'r'))
    #     return self._artist2info

    def get_influencers_emb(self, artist_batch):
        """
        Return:
            - x: Tensor of size [max_len, batch, emb_dim]
            - lengths: list of length batch (used to mask x), i.e. number of influencers for each artist in batch
                values are in [1, max_len]
            - sorted_artists (sorted by descending sequence length)
        
        Notes:
            - A sequence for *one* artist is the sequence of GMM means representing its influencers. This is already
            pre-computed and saved in datasets.py as a (num_means, emb_size) matrix
        """
        influencers_batch = []                          # stores tuples of (emb, length, artist)
        max_len = 0
        for artist in artist_batch:
            # Load emb for that artist
            emb_path = os.path.join(WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH, artist, 'embedding.pkl')
            emb = pickle.load(open(emb_path, 'r'))      # numpy (num_means, emb_size)

            # Update max length
            if emb.shape[0] > max_len:
                max_len = emb.shape[0]

            emb_len_artist = [emb, emb.shape[0], artist]
            influencers_batch.append(emb_len_artist)

        # Sort by sequence length in descending order
        influencers_batch = sorted(influencers_batch, key=lambda x: -x[1])
        lengths = [length for emb, length, artist in influencers_batch]
        sorted_artists = [artist for emb, length, artist in influencers_batch]

        # Now that we have the max length, create a matrix that of size (max_seq_len, batch, emb_size)
        batch_size = len(artist_batch)
        emb_size = influencers_batch[0][0].shape[1]
        x = np.zeros((max_len, batch_size, emb_size))
        for i, (emb, _, _) in enumerate(influencers_batch):
            padded = np.zeros((max_len, emb_size))
            padded[:len(emb),:] = emb
            x[:,i,:] = padded

        # Convert to Variable
        x = Variable(torch.Tensor(x))

        if torch.cuda.is_available():
            x = x.cuda()

        return x, lengths, sorted_artists

    def get_artists_targets(self, artist_batch):
        """
        Return Variable for batch of artists: (num_artists), where value is index in [0, num_artists-1]
        """
        targets = []
        for artist in artist_batch:
            target = WikiartDataset.get_artist_one_hot_index(artist, TRAIN_ON_ALL)
            targets.append(target)
        targets = np.array(targets)
        targets = Variable(torch.Tensor(targets))
        return targets

    def train(self):
        """Train on training data split"""
        for model in self.models:
            model.train()

        # Load models
        if self.hp.load_lstm_fp is not None:
            print 'Loading LSTM'
            self.influencers_lstm.load_state_dict(torch.load(self.hp.load_lstm_fp))
        if self.hp.load_G_fp is not None:
            print 'Loading G'
            self.artist_G.load_state_dict(torch.load(self.hp.load_G_fp))
        if self.hp.load_D_fp is not None:
            print 'Loading D'
            self.artist_D.load_state_dict(torch.load(self.hp.load_D_fp))

        # If all paths are given, then keep on saving things to that directory
        # (Assumes all paths are from same directory)
        # Else make a new directory
        if (self.hp.load_lstm_fp is not None) and \
                (self.hp.load_G_fp is not None) and \
                (self.hp.load_D_fp is not None):
            out_dir = os.path.dirname(self.hp.load_lstm_fp)
            out_dir_imgs = os.path.join(out_dir, 'imgs')
            print 'Checkpoints will continue to be saved to: {}'.format(out_dir)
            cur_epoch = self.hp.cur_epoch
        else:
            # Make directory to store outputs
            out_dir = os.path.join(SAVED_MODELS_PATH, datetime.now().strftime('%B%d_%H-%M-%S'))
            out_dir_imgs = os.path.join(out_dir, 'imgs')
            os.mkdir(out_dir)
            os.mkdir(out_dir_imgs)
            print 'Checkpoints will be saved to: {}'.format(out_dir)
            cur_epoch = 0

            # Save hyperparmas
            with open(os.path.join(out_dir, 'hp.json'), 'w') as f:
                json.dump(vars(self.hp), f)

        # Get Tensorboard summary writer
        writer = SummaryWriter(out_dir)

        # Get data_loaders
        train_data_loader = get_wikiart_data_loader(self.hp.batch_size, self.hp.img_size, TRAIN_ON_ALL)
        # valid_data_loader = get_you_data_loader('valid', hp.batch_size)

        # Set up loss
        D_discrim_criterion = nn.BCELoss()
        D_aux_criterion = nn.CrossEntropyLoss()

        # Adam optimizer
        LSTM_optimizer = optim.Adam(self.influencers_lstm.parameters(), lr=self.hp.lr_lstm, betas=(0.5, 0.999))
        G_optimizer = optim.Adam(self.artist_G.parameters(), lr=self.hp.lr_G, betas=(0.5, 0.999))
        D_optimizer = optim.Adam(self.artist_D.parameters(), lr=self.hp.lr_D, betas=(0.5, 0.999))

        # Test by creating images for same artist every _ iterations
        test_artist_batch = ['rembrandt', 'pablo-picasso']        # TODO: add more

        # Use fixed noise to generate images every _ iterations, i.e. same noise but model has been further trained
        # Define fake and real labels so we can re-use it / pre-allocate space
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1)
        fixed_noise = torch.FloatTensor(len(test_artist_batch), self.hp.z_size, 1, 1).normal_(0, 1)
        fixed_noise = Variable(fixed_noise)
        if torch.cuda.is_available():
            fixed_noise = fixed_noise.cuda()

        last_loss_D_fake_discrim = float('inf')
        last_loss_D_real_discrim = float('inf')
        for e in range(cur_epoch, self.hp.max_nepochs):
            data_iter = iter(train_data_loader)
            train_idx = 0
            # for train_idx, (img_batch, artist_batch) in enumerate():
            while train_idx < len(train_data_loader):
                # Get data batch
                (img_batch, artist_batch) = data_iter.next()
                batch_size = len(artist_batch)
                train_idx += 1

                # Move inputs to cuda if possible, TODO: did I move everything
                if torch.cuda.is_available():
                    img_batch = img_batch.cuda()

                # For each image, we have to get the artist and their influencer embeddings
                batch_influencers_emb, lengths, sorted_artists = self.get_influencers_emb(artist_batch)

                # Get one hot artist labels for auxiliary classification loss
                # one_hots = self.get_artists_one_hot(sorted_artists).type(torch.LongTensor)
                targets = self.get_artists_targets(sorted_artists).type(torch.LongTensor)
                if torch.cuda.is_available():
                    targets = targets.cuda()

                ######################################################################
                # 1) UPDATE D NETWORK: maximize log(D(x)) + log(1 - D(G(z)))
                ######################################################################
                self.artist_D.zero_grad()

                ######################################################################
                #       First update D with the real images, i.e. log(D(x))
                ######################################################################
                # discrim, aux = self.artist_D(Variable(img_batch))
                instance_noise = torch.normal(torch.zeros(img_batch.size()), torch.zeros(img_batch.size()).fill_(0.05))
                if torch.cuda.is_available():
                    instance_noise = instance_noise.cuda()
                discrim, aux = self.artist_D(Variable(img_batch + instance_noise))

                real_labels = torch.ones(batch_size, 1)
                if torch.cuda.is_available():
                    real_labels = real_labels.cuda()
                real_labels = Variable(real_labels)
                loss_D_real_discrim = D_discrim_criterion(discrim, real_labels)
                loss_D_real_aux = D_aux_criterion(aux, targets)
                loss_D_real = loss_D_real_discrim + loss_D_real_aux
                loss_D_real.backward()

                ######################################################################
                #       Next update D with fake images, i.e. log(1 - D(G(z)))
                ######################################################################
                noise = torch.FloatTensor(batch_size, self.hp.z_size, 1, 1).normal_(0, 1)
                noise = Variable(noise)
                if torch.cuda.is_available():
                    noise = noise.cuda()

                # Get influencers embedding
                # self.influencers_lstm.lstm.zero_grad()
                # self.influencers_lstm.lin_layer.zero_grad()
                lstm_init_h, lstm_init_c = self.influencers_lstm.init_hidden(batch_size)  # clear out hidden states
                if torch.cuda.is_available():
                    lstm_init_h = lstm_init_h.cuda()
                    lstm_init_c = lstm_init_c.cuda()
                batch_influencers_emb = self.influencers_lstm(batch_influencers_emb,
                                                              lengths,
                                                              lstm_init_h,
                                                              lstm_init_c)  # (num_layers * num_directions, batch, 2 * hidden_size)
                batch_influencers_emb = batch_influencers_emb.view(batch_size, -1, 1, 1)  # (batch, ..., 1, 1)

                # Pass through Generator, then get loss from discriminator and backprop
                fake_imgs = self.artist_G(noise, batch_influencers_emb)
                instance_noise = torch.normal(torch.zeros(img_batch.size()), torch.zeros(img_batch.size()).fill_(0.05))
                if torch.cuda.is_available():
                    instance_noise = instance_noise.cuda()
                fake_imgs = fake_imgs + Variable(instance_noise)
                fake_labels = Variable(torch.zeros(batch_size, 1))
                if torch.cuda.is_available():
                    fake_labels = fake_labels.cuda()
                discrim, aux = self.artist_D(fake_imgs.detach())
                loss_D_fake_discrim = D_discrim_criterion(discrim, fake_labels)
                loss_D_fake_aux = D_aux_criterion(aux, targets)
                loss_D_fake = loss_D_fake_discrim + loss_D_fake_aux
                loss_D_fake.backward()
                # D_G_z1 = output.data.mean()
                loss_D = loss_D_real + loss_D_fake

                # Don't perform gradient descent step if D is too well-trained
                if (last_loss_D_fake_discrim > 0.3) and (last_loss_D_real_discrim > 0.3):
                    D_optimizer.step()
                last_loss_D_fake_discrim = loss_D_fake_discrim.data[0]
                last_loss_D_real_discrim = loss_D_real_discrim.data[0]

                ######################################################################
                # 2) UPDATE G NETWORK: maximize log(D(G(z))), i.e. want D(G(z)) to be 1's
                ######################################################################
                self.influencers_lstm.zero_grad()
                self.artist_G.zero_grad()
                discrim, aux = self.artist_D(fake_imgs)  # now that D's been updated
                real_labels = Variable(torch.ones(batch_size, 1))  # fake labels are real for generator cost
                if torch.cuda.is_available():
                    real_labels = real_labels.cuda()
                loss_G_discrim = D_discrim_criterion(discrim,
                                                     real_labels)  # To train G, want D to think images are real, so use torch.ones. Minimize cross entropy between discrim and ones.
                loss_G_aux = D_aux_criterion(aux, targets)
                loss_G = loss_G_discrim + loss_G_aux
                loss_G.backward()
                LSTM_optimizer.step()
                G_optimizer.step()

                # Write loss to Tensorboard
                if train_idx % 10 == 0:
                    writer.add_scalar('loss_D_fake_discrim', loss_D_fake_discrim.clone().cpu().data.numpy(),
                                      e * train_data_loader.__len__() + train_idx)
                    writer.add_scalar('loss_D_fake_aux', loss_D_fake_aux.clone().cpu().data.numpy(),
                                      e * train_data_loader.__len__() + train_idx)
                    # writer.add_scalar('loss_D_fake', loss_D_fake.clone().cpu().data.numpy(),
                    #                   e * train_data_loader.__len__() + train_idx)
                    writer.add_scalar('loss_D_real_discrim', loss_D_real_discrim.clone().cpu().data.numpy(),
                                      e * train_data_loader.__len__() + train_idx)
                    writer.add_scalar('loss_D_real_aux', loss_D_real_aux.clone().cpu().data.numpy(),
                                      e * train_data_loader.__len__() + train_idx)
                    # writer.add_scalar('loss_D_real', loss_D_real.clone().cpu().data.numpy(),
                    #                   e * train_data_loader.__len__() + train_idx)
                    # writer.add_scalar('loss_D', loss_D.clone().cpu().data.numpy(),
                    #                   e * train_data_loader.__len__() + train_idx)
                    writer.add_scalar('loss_G_discrim', loss_G_discrim.clone().cpu().data.numpy(),
                                      e * train_data_loader.__len__() + train_idx)
                    writer.add_scalar('loss_G_aux', loss_G_aux.clone().cpu().data.numpy(),
                                      e * train_data_loader.__len__() + train_idx)
                    # writer.add_scalar('loss_G', loss_G.clone().cpu().data.numpy(),
                    #                   e * train_data_loader.__len__() + train_idx)


                # Save images, remembering to normalize back to [0,1]
                # Generate a fake one with fixed noise and for test artists
                # Also save model
                if train_idx % 100 == 0:
                    print 'Epoch {} - {}'.format(e, train_idx)

                    # Save some real images (sanity check)
                    vutils.save_image(img_batch[0:4], os.path.join(out_dir_imgs, 'real_e{}_{}.png'.format(e, train_idx)), normalize=True)

                    # Generate fake
                    lstm_init_h, lstm_init_c = self.influencers_lstm.init_hidden(len(test_artist_batch))  # clear out hidden states
                    test_batch_influencers_emb, test_lengths, test_sorted_artists = self.get_influencers_emb(test_artist_batch)
                    if torch.cuda.is_available():
                        test_batch_influencers_emb = test_batch_influencers_emb.cuda()
                        lstm_init_h = lstm_init_h.cuda()
                        lstm_init_c = lstm_init_c.cuda()
                    test_batch_influencers_emb = self.influencers_lstm(test_batch_influencers_emb,
                                                                       test_lengths,
                                                                       lstm_init_h,
                                                                       lstm_init_c)  # (num_layers * num_directions, batch, 2 * hidden_size)
                    test_batch_influencers_emb = test_batch_influencers_emb.view(len(test_artist_batch), -1, 1, 1)  # (batch, ..., 1, 1)
                    test_noise = torch.FloatTensor(len(test_artist_batch), self.hp.z_size, 1, 1).normal_(0, 1)
                    test_noise = Variable(test_noise)
                    if torch.cuda.is_available():
                        test_noise = test_noise.cuda()
                    fake_imgs_fixed = self.artist_G(fixed_noise, test_batch_influencers_emb)
                    fake_imgs_random = self.artist_G(test_noise, test_batch_influencers_emb)
                    vutils.save_image(fake_imgs_fixed.data, os.path.join(out_dir_imgs, 'fake_fixed_e{}_{}.png'.format(e, train_idx)), normalize=True)
                    vutils.save_image(fake_imgs_random.data,os.path.join(out_dir_imgs, 'fake_random_e{}_{}.png'.format(e, train_idx)), normalize=True)

                    # Write image to tensorboard
                    tboard_name = '_'.join(test_artist_batch)
                    tboard_image = vutils.make_grid(fake_imgs_fixed.data, normalize=True, scale_each=True)
                    writer.add_image(tboard_name  + '__fixed', tboard_image, train_idx)  # TODO: should this be e * n + train_idx?
                    tboard_image = vutils.make_grid(fake_imgs_random.data, normalize=True, scale_each=True)
                    writer.add_image(tboard_name + '__random', tboard_image, train_idx)  # TODO: should this be e * n + train_idx?

                    # Save
                    if e % self.hp.save_every_nepochs == 0:
                        torch.save(self.influencers_lstm.state_dict(), os.path.join(out_dir, 'lstm_e{}.pt'.format(e)))
                        torch.save(self.artist_G.state_dict(), os.path.join(out_dir, 'G_e{}.pt'.format(e)))
                        torch.save(self.artist_D.state_dict(), os.path.join(out_dir, 'D_e{}.pt'.format(e)))

            # Write parameters to Tensorboard every epoch
            for name, param in self.influencers_lstm.named_parameters():
                writer.add_histogram('lstm-' + name, param.clone().cpu().data.numpy(), e * train_data_loader.__len__() + train_idx)
            for name, param in self.artist_G.named_parameters():
                writer.add_histogram('G-' + name, param.clone().cpu().data.numpy(), e * train_data_loader.__len__() + train_idx)
            for name, param in self.artist_D.named_parameters():
                writer.add_histogram('D-' + name, param.clone().cpu().data.numpy(), e * train_data_loader.__len__() + train_idx)



        writer.close()

    def generate(self, influencers, n=16):
        """
        Inputs:
        - influencers: comma-separated list of influencers passed in by command line
        - n: number of images to generate
        """
        # Load models
        self.influencers_lstm.load_state_dict(torch.load(self.hp.load_lstm_fp))
        self.artist_G.load_state_dict(torch.load(self.hp.load_G_fp))
        self.artist_D.load_state_dict(torch.load(self.hp.load_D_fp))

        for model in self.models:
            model.eval()

        # batch_size = 1

        # Get influencers_emb
        influencers = influencers.split(',')
        # TODO: get_influcners_emb takes a batch of artists. Whereas influencers here is the influencers themselves
        # Different. Want to just get it and then repeat it n times
        batch_influencers_emb, lengths, sorted_artists = self.get_influencers_emb(influencers)
        lstm_init_h, lstm_init_c = self.influencers_lstm.init_hidden(len(influencers))  # clear out hidden states
        batch_influencers_emb = self.influencers_lstm(batch_influencers_emb,
                                                      lengths,
                                                      lstm_init_h,
                                                      lstm_init_c)  # (num_layers * num_directions, batch, 2 * hidden_size)
        batch_influencers_emb = batch_influencers_emb.view(1, -1, 1, 1)  # (batch, ..., 1, 1)

        # Get noise
        noise = torch.FloatTensor(n, self.hp.z_size, 1, 1).normal_(0, 1)
        noise = Variable(noise)

        fake_imgs = self.artist_G(noise, batch_influencers_emb)
