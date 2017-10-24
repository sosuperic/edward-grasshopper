# pretrain_lstm.py

"""
Pretrain InfluencersLSTM to classify artist. 

One linear layer projecting hidden state of LSTM into num_artists. 
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

from config import SAVED_MODELS_PATH, \
    WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH, HParams
from datasets import get_wikiart_data_loader, WikiartDataset
from models import InfluencersLSTM


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
        self.influencers_lstm = InfluencersLSTM(hp.lstm_emb_size, hp.lstm_hidden_size)
        self.linear = nn.Linear(2 * hp.lstm_hidden_size, WikiartDataset.get_num_valid_artists())
        # self.models = [self.influencers_lstm, self.linear]
        self.model = nn.Sequential(self.influencers_lstm, self.linear)

        # Move to cuda
        if torch.cuda.is_available():
            # for model in self.models:
            #     model.cuda()
            self.model.cuda()

    def get_influencers_emb(self, artist_batch):
        """
        Return Torch Variable / module using Packed sequence (max_seq_len, batch, emb_size), as well as 
        sorted_artists (sorted by sequence length)

        A sequence for *one* artist is the sequence of GMM means representing its influencers. This is already
        pre-computed and saved in datasets.py as a (num_means, emb_size) matrix
        """
        influencers_batch = []
        max_len = 0
        for artist in artist_batch:
            # Load emb for that artist
            emb_path = os.path.join(WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH, artist, 'embedding.pkl')
            emb = pickle.load(open(emb_path, 'r'))  # numpy (num_means, emb_size)

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
        input = np.zeros((max_len, batch_size, emb_size))
        for i, (emb, _, _) in enumerate(influencers_batch):
            padded = np.zeros((max_len, emb_size))
            padded[:len(emb), :] = emb
            input[:, i, :] = padded

        # Convert to Variable
        input = Variable(torch.Tensor(input))

        # Create packed sequence
        packed_sequence = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False)

        return packed_sequence, sorted_artists

    def get_artists_targets(self, artist_batch):
        """
        Return Variable for batch of artists: (num_artists), where value is index in [0, num_artists-1]
        """
        targets = []
        for artist in artist_batch:
            target = WikiartDataset.get_artist_one_hot_index(artist)
            targets.append(target)
        targets = np.array(targets)
        targets = Variable(torch.Tensor(targets))
        return targets

    def train(self):
        """Train on training data split"""
        # for model in self.models:
        #     model.train()
        self.model.train()

        # Load models
        if self.hp.load_lstm_fp is not None:
            self.load_model(self.hp.load_lstm_fp, self.hp.load_G_fp, self.hp.load_D_fp)

        # Make directory to store outputs
        out_dir = os.path.join(SAVED_MODELS_PATH, 'LSTM_' + datetime.now().strftime('%B%d_%H-%M-%S'))
        out_dir_imgs = os.path.join(out_dir, 'imgs')
        os.mkdir(out_dir)
        os.mkdir(out_dir_imgs)
        print 'Checkpoints will be saved to: {}'.format(out_dir)

        # Save hyperparmas
        with open(os.path.join(out_dir, 'hp.json'), 'w') as f:
            json.dump(vars(self.hp), f)

        # Get Tensorboard summary writer
        writer = SummaryWriter(out_dir)

        # Get data_loaders
        train_data_loader = get_wikiart_data_loader(self.hp.batch_size, self.hp.img_size, 'train')
        valid_data_loader = get_wikiart_data_loader(self.hp.batch_size, self.hp.img_size, 'valid')

        # Set up loss
        criterion = nn.CrossEntropyLoss()

        # Adam optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.hp.lr, betas=(0.5, 0.999))

        for e in range(self.hp.max_nepochs):
            for train_idx, (img_batch, artist_batch) in enumerate(train_data_loader):
                self.model.zero_grad()
                batch_size = len(artist_batch)

                # Get influencers embedding
                batch_influencers_emb, sorted_artists = self.get_influencers_emb(artist_batch)
                lstm_init_h, lstm_init_c = self.influencers_lstm.init_hidden()  # clear out hidden states
                if torch.cuda.is_available():
                    batch_influencers_emb = batch_influencers_emb.cuda()
                    lstm_init_h = lstm_init_h.cuda()
                    lstm_init_c = lstm_init_c.cuda()

                batch_influencers_emb = self.influencers_lstm(batch_influencers_emb,
                                                              lstm_init_h,
                                                              lstm_init_c)  # (num_layers * num_directions, batch, 2 * hidden_size)
                batch_influencers_emb = batch_influencers_emb.view(batch_size, -1)    # (batch, ...)
                output = self.linear(batch_influencers_emb)

                # Get one hot artist labels for auxiliary classification loss
                targets = self.get_artists_targets(sorted_artists).type(torch.LongTensor)
                loss = criterion(output, targets)
                predicted_class = torch.max(output, 1)[1]
                accuracy = (predicted_class == targets).sum().cpu().data.numpy() / float(batch_size) * 100

                # Backprop
                loss.backward()
                optimizer.step()

                # Write loss to Tensorboard
                if train_idx % 10 == 0:
                    writer.add_scalar('loss', loss.clone().cpu().data.numpy(),
                                      e * train_data_loader.__len__() + train_idx)
                    writer.add_scalar('accuracy', accuracy,
                                      e * train_data_loader.__len__() + train_idx)

                # Save images
                # Also save model
                if train_idx % 100 == 0:
                    print 'Epoch {} - {}'.format(e, train_idx)

                    # Save
                    if e % self.hp.save_every_nepochs == 0:
                        torch.save(self.influencers_lstm.state_dict(), os.path.join(out_dir, 'lstm_e{}.pt'.format(e)))
                        torch.save(self.linear.state_dict(), os.path.join(out_dir, 'linear_e{}.pt'.format(e)))

                # break

            # Evaluate on validation split every epoch
            valid_losses = []
            valid_accuracies = []
            for valid_idx, (img_batch, artist_batch) in enumerate(valid_data_loader):
                batch_size = len(artist_batch)
                batch_influencers_emb, sorted_artists = self.get_influencers_emb(artist_batch)
                lstm_init_h, lstm_init_c = self.influencers_lstm.init_hidden()
                batch_influencers_emb = self.influencers_lstm(batch_influencers_emb,
                                                              lstm_init_h,
                                                              lstm_init_c)
                batch_influencers_emb = batch_influencers_emb.view(batch_size, -1)  # (batch, ...)
                output = self.linear(batch_influencers_emb)
                targets = self.get_artists_targets(sorted_artists).type(torch.LongTensor)
                loss = criterion(output, targets)
                predicted_class = torch.max(output, 1)[1]
                accuracy = (predicted_class == targets).sum().data[0] / float(batch_size) * 100
                valid_losses.append(loss)
                valid_accuracies.append(accuracy)
                # break
            valid_loss = np.array(valid_losses).mean().data[0]
            valid_accuracy = np.array(valid_accuracies).mean()
            idx = (e + 1) * train_data_loader.__len__()     # plot it so it matches up with end of training
            writer.add_scalar('valid_loss', valid_loss, idx)
            writer.add_scalar('valid_accuracy', valid_accuracy, idx)

            # Write parameters to Tensorboard every epoch
            for name, param in self.influencers_lstm.named_parameters():
                idx = (e + 1) * train_data_loader.__len__()
                writer.add_histogram('lstm-' + name, param.clone().cpu().data.numpy(), idx)
                # writer.add_histogram('lstm-' + name + '/grad', param.clone().grad.numpy(), idx)

        writer.close()

    def test(self):
        """Test on held-out data split"""
        # for model in self.models:
        #     model.eval()
        self.model.eval()

        self.load_model(self.hp.load_lstm_fp, self.hp.load_linear_fp)

        test_data_loader = get_wikiart_data_loader(self.hp.batch_size, self.hp.img_size, 'test')
        criterion = nn.CrossEntropyLoss()
        losses = []
        accuracies = []


        for valid_idx, (img_batch, artist_batch) in enumerate(test_data_loader):
            batch_size = len(artist_batch)
            batch_influencers_emb, sorted_artists = self.get_influencers_emb(artist_batch)
            lstm_init_h, lstm_init_c = self.influencers_lstm.init_hidden()
            batch_influencers_emb = self.influencers_lstm(batch_influencers_emb,
                                                          lstm_init_h,
                                                          lstm_init_c)
            batch_influencers_emb = batch_influencers_emb.view(batch_size, -1)  # (batch, ...)
            output = self.linear(batch_influencers_emb)
            targets = self.get_artists_targets(sorted_artists).type(torch.LongTensor)
            loss = criterion(output, targets)
            predicted_class = torch.max(output, 1)[1]
            accuracy = (predicted_class == targets).sum().cpu().data.numpy() / float(batch_size) * 100
            losses.append(loss)
            accuracies.append(accuracy)
            break
        loss = np.array(losses).mean().data[0]
        accuracy = np.array(accuracies).mean()

        print 'Loss: {}'.format(loss)
        print 'Accuracy: {}'.format(accuracy)

    ##############
    # Utils
    ##############
    def load_model(self, lstm_fp, linear_fp):
        """Load model parameters from given filename"""
        # print 'Loading model parameters from {}'.format(models_fp)
        print 'Loading model parameters'
        self.influencers_lstm.load_state_dict(torch.load(lstm_fp))
        self.linear.load_state_dict(torch.load(linear_fp))
        print 'Done loading model parameters'

        # def turn_inputs_labels_to_vars(self, inputs, labels):
        #     """Wrap tensors in Variables for autograd; also move to gpu potentially"""
        #     if torch.cuda.is_available():
        #         inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        #     else:
        #         inputs, labels = Variable(inputs), Variable(labels)
        #     return inputs, labels
        #
        # def var2numpy(self, var):
        #     """Return numpy matrix of data in Variable. Separate util function because CUDA tensors require
        #     moving to cpu first"""
        #     if var.is_cuda:
        #         return var.data.cpu().numpy()
        #     else:
        #         return var.data.numpy()


if __name__ == '__main__':
    hparams = HParams()
    hparams.lr = 0.01
    network = Network(hparams)
    network.train()

    # hparams.load_lstm_fp = 'checkpoints/LSTM_October24_02-59-44/lstm_e0.pt'
    # hparams.load_linear_fp = 'checkpoints/LSTM_October24_02-59-44/linear_e0.pt'
    # network.test()