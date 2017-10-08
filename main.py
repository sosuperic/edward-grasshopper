import argparse
from datetime import datetime
import os
from PIL import Image
from tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import utils as vutils

WIKIART_DATASET_PATH = '/media/lsm/dude1/echu/art_datasets/wikiart/data/'

class WikiartDataset(Dataset):
    """
    Images from Wikiart Dataset -- used for prediction
    """

    def __init__(self, transform=None, artist=None):
        """
        Inputs
        ------
        split_name: 'train', 'valid', 'test' determines data per class
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform

        # Get filenames and (dummy?) labels
        self.path = os.path.join(WIKIART_DATASET_PATH, artist)
        fns = os.listdir(self.path)
        self.fns_labels = [(fn, 0) for fn in fns]

    def __len__(self):
        """Return number of data points"""
        return len(self.fns_labels)

    def __getitem__(self, idx):
        """Return idx-th image and label in dataset"""
        fn, label = self.fns_labels[idx]
        fp = os.path.join(self.path, fn)
        image = Image.open(fp)

        # Note: image is a (3,256,256) matrix, which is what we want because
        #       pytorch conv works on (batch, channels, width, height)
        if self.transform:
            image = self.transform(image)

        return (fp, image, label)

def get_wikiart_data_loader(artist, batch_size):
    """
    Return data loader for the wikiart Dataset. Uses CenterCrop instead of RandomCrop
    """
    transform = transforms.Compose([
        transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = WikiartDataset(transform=transform, artist=artist)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader