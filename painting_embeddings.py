# painting_embeddings.py

"""
Generate embedding representing painting that captures semantic and stylistic content.

Available classes:
- Painting

Available functions:
- test_artist_title_extraction: Test that artist and title are extracted correctly
- test_lazy_load: Test lazy load of image and image properties functionality
- test_one_painting: Create embedding for one painting

"""

from config import WIKIART_TEST_IMG_PATH
import numpy as np
import os
from PIL import Image
import torchvision.models as models

class Painting(object):
    """Represents a painting stored as an image.

    Public attributes:
    - path: path to image (str)
    - artist: artist name extracted from path
    - title: title of painting extracted from path
    
    - img_mode: mode of Pillow Image (e.g. RBG, L). Denotes number of channels
    - img_size: size of Pillow Image (e.g. (750, 326))
    - img: loaded image as numpy matrix
    """

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


    ####################################################################################################################
    # Representation
    ####################################################################################################################
    def create_embedding(self, vgg16):
        content_emb = self._create_content_embedding(vgg16)
        style_emb = self._create_style_embedding(vgg16)
        embedding = [content_emb, style_emb]    # TODO: concat
        return embedding

    def _create_style_embedding(self, vgg16):
        self.img
        return 2

    def _create_content_embedding(self, vgg16):
        return 3

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


if __name__ == '__main__':
    test_artist_title_extraction()
    test_lazy_load()
    test_one_painting_emb()

    #     if torch.cuda.is_available():
    #         model = model.cuda()
