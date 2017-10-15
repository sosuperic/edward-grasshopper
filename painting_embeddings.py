# painting_embeddings.py

"""
Generate embedding representing painting that captures semantic and stylistic content.

Available classes:
- Painting

Available functions:
- test_artist_title_extraction: Test that artist and title are extracted correctly
- test_lazy_load: Test lazy load of image and image properties functionality
- test_one_painting: Create embedding for one painting
- test_extract_VGG16_activations: Pass image through VGG16 and extract selected features

"""

from config import WIKIART_TEST_IMG_PATH
import numpy as np
import os
from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class Painting(object):
    """Represents a painting stored as an image.

    Public attributes:
    - path: path to image (str)
    - artist: artist name extracted from path (str)
    - title: title of painting extracted from path (str)
    
    - img_mode: mode of Pillow Image (e.g. RBG, L). Denotes number of channels
    - img_size: size of Pillow Image (e.g. (750, 326))
    - img: loaded image as numpy matrix
    """
    emb_size = (2048,)

    def __init__(self, path, vgg16=None):
        self.path = os.path.normpath(path)
        self.artist = os.path.basename(os.path.dirname(self.path))
        self.title = os.path.basename(self.path).split('.')[0]

        self._img = None
        self._img_mode = None
        self._img_size = None

    ####################################################################################################################
    # Class properties and utils
    ####################################################################################################################
    @property
    def img(self):
        """Lazily load painting as Pillow Image"""
        if self._img is None:
            self.load_and_set_img_fields()
        return self._img

    @property
    def img_mode(self):
        if self._img_mode is None:
            self.load_and_set_img_fields()
        return self._img_mode

    @property
    def img_size(self):
        if self._img_size is None:
            self.load_and_set_img_fields()
        return self._img_size

    def load_and_set_img_fields(self):
        img = Image.open(self.path)
        self._img_mode = img.mode
        self._img_size = img.size
        self._img = np.asarray(img)

    @staticmethod
    def get_vgg16():
        """Return the model that can be used by multiple paintings"""
        return models.vgg16(pretrained=True)

    def is_valid_img_for_vgg16(self):
        """Return whether image has correct number of channels and is large enough"""
        correct_num_channels = self.img_mode == 'RGB'
        img_large_enough = (self.img_size[0] >= 224) and (self.img_size[1] >= 224)
        valid = correct_num_channels and img_large_enough
        return valid

    def get_img_variable_for_vgg16(self):
        """Return pytorch Variable with correct dimensions as input into vgg16"""
        transform = transforms.Compose([
            transforms.ToPILImage(),            # should save PILImage upon original loading or refactor somehow
            transforms.Scale(256),
            transforms.RandomCrop(224),
            transforms.ToTensor()])
        tensor = transform(self.img)            # CWH format
        tensor.unsqueeze_(0)                    # BCWH

        return Variable(tensor)

    ####################################################################################################################
    # Representation
    ####################################################################################################################
    def create_embedding(self, vgg16):

        vgg16 = VGG16Extractor(vgg16)
        x = self.get_img_variable_for_vgg16()
        print vgg16(x)

        content_emb = 2
        style_emb = 2

        # content_emb = self._create_content_embedding(vgg16)
        # style_emb = self._create_style_embedding(vgg16)
        embedding = [content_emb, style_emb]    # TODO: concat
        return embedding
    #
    # def _create_style_embedding(self, vgg16):
    #     self.img
    #     return 2
    #
    # def _create_content_embedding(self, vgg16):
    #     return 3


########################################################################################################################
# Helper to extract activations from pre-trained model
########################################################################################################################
class VGG16Extractor(nn.Module):
    """
    VGG16 has two Sequential containers: 'features', and 'classifier'. Traverse all modules, saving activations
    for the features we want to extract
    
    Public attributes:
    - pretrained: VGG16 model
    - features_names: list of names of features, e.g. ['features.4', 'classifier.3']
    """
    def __init__(self, pretrained, feature_names):
        super(VGG16Extractor, self).__init__()

        self.pretrained = pretrained
        self.feature_names = feature_names

    def forward(self, x):
        """
        Input:
        - x: (batch, channels, x, y)
        """
        features = []

        batch_size = x.size(0)

        # Pass input through features part of model
        for name, module in self.pretrained.features.named_modules():
            if len(module._modules) == 0:       # first module is the entire thing, check for leaf nodes
                full_name = 'features.{}'.format(name)
                x = module(x)
                if full_name in self.feature_names:
                    features.append(x)

        # Reshape for linear layer
        x = x.view(batch_size, -1)

        # Pass through classifier part of model (linear layers)
        for name, module in self.pretrained.classifier.named_modules():
            if len(module._modules) == 0:  # first module is the entire thing, check for leaf nodes
                full_name = 'classifier.{}'.format(name)
                x = module(x)
                if full_name in self.feature_names:
                    features.append(x)

        return features

########################################################################################################################
# Basic test cases
########################################################################################################################
def test_artist_title_extraction():
    """Test that artist and title are extracted correctly"""
    painting = Painting(WIKIART_TEST_IMG_PATH)
    print painting.path
    print painting.artist
    print painting.title

def test_lazy_load():
    """Test lazy load of image and image properties functionality"""
    painting = Painting(WIKIART_TEST_IMG_PATH)
    print painting.is_valid_img_for_vgg16()

def test_one_painting_emb():
    """Create embedding for one painting"""
    vgg16 = Painting.get_vgg16()
    painting = Painting(WIKIART_TEST_IMG_PATH)
    emb = painting.create_embedding(vgg16)

    print emb

def test_extract_VGG16_activations():
    """Pass image through VGG16 and extract selected features"""
    vgg16 = Painting.get_vgg16()
    vgg16_extractor = VGG16Extractor(vgg16, ['features.8'])
    painting = Painting(WIKIART_TEST_IMG_PATH)
    x = painting.get_img_variable_for_vgg16()
    features = vgg16_extractor(x)
    print features
    print len(features)

if __name__ == '__main__':
    # test_artist_title_extraction()
    # test_lazy_load()
    # test_one_painting_emb()
    test_extract_VGG16_activations()


    #     if torch.cuda.is_available():
    #         model = model.cuda()
