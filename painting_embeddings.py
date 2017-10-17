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

- save_raw_content_and_style_embs()
    - Calculate content and raw style vectors for all images
- fit_and_save_PCA():
    - Use all raw content and style embeddings to fit a content_PCA and style_PCA
- save_PCA_content_and_style_embs():
    - Use saved PCA to save content and style embeddings with a reduced dimension
"""

from config import WIKIART_TEST_IMG_PATH, WIKIART_ARTISTS_PATH,\
    WIKIART_ARTIST_PAINTING_RAW_EMBEDDINGS_PATH, WIKIART_ARTIST_PAINTING_PCA_EMBEDDINGS_PATH,\
    WIKIART_PAINTINGS_PCA_CONTENT, WIKIART_PAINTINGS_PCA_STYLE, PCA_CONTENT_DIM, PCA_STYLE_DIM
import numpy as np
import os
import pickle
from PIL import Image
from sklearn.decomposition import PCA, IncrementalPCA

import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

from utils import get_valid_artist_names

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

    # conv4_1
    content_layer_name = 'classifier.0'        # 4096
    # content_layer_names = ['features.17']
    # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
    style_layer_name  = 'features.24'          # conv5_1
    style_layer_nfilters = 512
    # style_layer_names = ['features.0', 'features.5', 'features.10', 'features.17', 'features.24']

    def __init__(self, path):
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
        print 'Loading pretrained VGG16'
        return models.vgg16(pretrained=True)

    @staticmethod
    def get_gram_indices(N):
        """
        Return (2, N * (N+1) / 2) matrix where each column is index in upper-triangle (including the diagonal)
        of element in N x N matrix
        
        Note: not the most elegant design to have this as a static method and pass in the indices to the 
        method calls, but don't want to compute these indices every time. Not sure if there's a fast
        way of getting the indices."""
        indices = torch.zeros(2, N * (N+1) / 2).long()
        col = 0
        for i in range(N):
            for j in range(i, N):
                indices[:, col] = torch.Tensor([i, j])
                col += 1
        return indices

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
    def extract_PCA_content_style_embeddings(self, vgg16, content_PCA, style_PCA):
        """
        Use fitted PCA models to reduce dimension of raw content and style embeddings. Also L2 normalize style.
        
        Inputs:
        - content_PCA: PCA trained on raw content embeddings
        - style_PCA: PCA trained on raw style embeddings
        """
        raw_content, raw_style = self.extract_raw_content_style_embeddings(vgg16)
        content = content_PCA.transform(raw_content)

        raw_style = raw_style / np.linalg.norm(raw_style, ord=2)
        style = style_PCA.transform(raw_style)

        return content, style

    def extract_raw_content_style_embeddings(self, vgg16, gram_indices):
        """
        Content vector as FC in VGG. Style vector as Triangle of Gram matrix of conv5_1
        
        Inputs:
        - vgg16: pretrained vgg16 network
        - gram_indices: (2, ) Tensor where col is i,j indices to extract value from style Gram matrix
        """
        x = self.get_img_variable_for_vgg16()

        # Extract features from vgg16
        feature_names = set([Painting.content_layer_name]).union(set([Painting.style_layer_name]))
        extractor = VGG16Extractor(vgg16, feature_names)
        features = extractor(x)

        content_emb = features[Painting.content_layer_name]
        style_emb = self.calc_style_embedding(features, gram_indices)

        return content_emb, style_emb

    def calc_style_embedding(self, features, gram_indices):
        """
        
        Inputs:
        - features: dictionary mapping feature_name (str) to Variable; should contain name of layer for style vector
        
        Gram matrix 
        onv5_1 with signsqrt + L2norm + PCA 1024
        """
        feature = features[Painting.style_layer_name]
        a, b, c, d = feature.size()             # (batch, filters, x, y)
        feature = feature.view(a * b, c * d)    # resise F_XL into \hat F_XL
        G = torch.mm(feature, feature.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        # G =  G.div(a * b * c * d)       # TODO: Shouldn't this be c * d?
        G = G.div(c * d)

        # Extract lower triangle + diagonal elements
        V = G[gram_indices[0], gram_indices[1]]

        return V

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
        
        Returns:
        - features: dictionary from feature_name to Variable
        """
        features = {}

        batch_size = x.size(0)

        # Pass input through features part of model
        for name, module in self.pretrained.features.named_modules():
            if len(module._modules) == 0:       # first module is the entire thing, check for leaf nodes
                full_name = 'features.{}'.format(name)
                x = module(x)
                if full_name in self.feature_names:
                    features[full_name] = x

        # Reshape for linear layer
        x = x.view(batch_size, -1)

        # Pass through classifier part of model (linear layers)
        for name, module in self.pretrained.classifier.named_modules():
            if len(module._modules) == 0:  # first module is the entire thing, check for leaf nodes
                full_name = 'classifier.{}'.format(name)
                x = module(x)
                if full_name in self.feature_names:
                    features[full_name] = x

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

def test_extract_VGG16_activations():
    """Pass image through VGG16 and extract selected features"""
    vgg16 = Painting.get_vgg16()
    vgg16_extractor = VGG16Extractor(vgg16, ['features.8'])
    painting = Painting(WIKIART_TEST_IMG_PATH)
    x = painting.get_img_variable_for_vgg16()
    features = vgg16_extractor(x)
    print features
    print len(features)

def test_one_painting_emb():
    """Create embedding for one painting"""
    vgg16 = Painting.get_vgg16()
    painting = Painting(WIKIART_TEST_IMG_PATH)
    emb = painting.create_embedding(vgg16)
    #
    # print emb

########################################################################################################################
# EMBEDDINGS
# Calculate and save content and style embeddings. Two functions for two passes because need to calculate
# raw embeddings first, run PCA, then calculate reduced versions.
########################################################################################################################
def save_raw_content_and_style_embs():
    """
    Calculate content and raw style vectors for all images
    
    This will be used to fit a PCA which can reduce the dimensionality of each vector
    """

    # Get objects that will be used repeatedly
    vgg16 = Painting.get_vgg16()
    gram_indices = Painting.get_gram_indices(Painting.style_layer_nfilters)

    # Iterate over artists in main influence graph
    artist_names = get_valid_artist_names()
    n = 0
    for artist in artist_names:
        artist_path = os.path.join(WIKIART_ARTISTS_PATH, artist)
        if not os.path.exists(artist_path):         # I think I deleted one or two artists with corrupted images
            print 'Skipping {}'.format(artist_path)
            continue

        # Make directory to store artist's painting embeddings if it doesn't exist
        out_dir = os.path.join(WIKIART_ARTIST_PAINTING_RAW_EMBEDDINGS_PATH, artist)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # Iterate over artist's paintings
        imgs = os.listdir(artist_path)
        for img in imgs:
            n += 1
            path = os.path.join(artist_path, img)
            painting = Painting(path)
            print 'Processing: {}, {}, n={}'.format(artist, painting.title, n)

            # Skip if painting isn't valid for VGG16, e.g. is black and white
            if not painting.is_valid_img_for_vgg16():
                continue

            # Skip if embedding exists
            content_out_path = os.path.join(out_dir, '{}_raw-content.pkl'.format(painting.title))
            style_out_path = os.path.join(out_dir, '{}_raw-style.pkl'.format(painting.title))
            if os.path.exists(content_out_path) or os.path.exists(style_out_path):
                continue

            # Extract embedding and save as numpy matrices
            raw_content, raw_style = painting.extract_raw_content_style_embeddings(vgg16, gram_indices)
            with open(content_out_path, 'w') as f:
                pickle.dump(raw_content.data.numpy(), f, protocol=2)
            with open(style_out_path, 'w') as f:
                pickle.dump(raw_style.data.numpy(), f, protocol=2)


def fit_and_save_PCA():
    """Use all raw content and style embeddings to fit a content_PCA and style_PCA"""

    # Iterate over all saved painting embeddings
    def load_all_content_embs():
        """
        Inputs:
        - emb_type: str ('content' or 'style')
        - chunk: list of start and end percentages 
        - dropout is float in [0.0, 1.0] denoting probability of skipping image
            Used so we don't load all style embeddings
        """
        embs = []
        for artist in os.listdir(WIKIART_ARTIST_PAINTING_RAW_EMBEDDINGS_PATH):
            artist_path = os.path.join(WIKIART_ARTIST_PAINTING_RAW_EMBEDDINGS_PATH, artist)
            emb_files = os.listdir(artist_path)
            for emb_file in emb_files:
                emb_file_path = os.path.join(artist_path, emb_file)
                if emb_file_path.endswith('content.pkl'):
                    emb = pickle.load(open(emb_file_path, 'r'))
                    emb = emb.astype(np.float16)        # reduce so we can have memory to run PCA
                    emb = np.squeeze(emb)               # (1, 4096) -> (4096, )
                    embs.append(emb)

        return embs

    def get_all_style_emb_fps():
        emb_fps = []
        for artist in os.listdir(WIKIART_ARTIST_PAINTING_RAW_EMBEDDINGS_PATH):
            artist_path = os.path.join(WIKIART_ARTIST_PAINTING_RAW_EMBEDDINGS_PATH, artist)
            emb_files = os.listdir(artist_path)
            for emb_file in emb_files:
                emb_file_path = os.path.join(artist_path, emb_file)
                if emb_file_path.endswith('style.pkl'):
                    emb_fps.append(emb_file_path)
        return emb_fps

    def load_style_embs_in_chunks(emb_fps, c, num_chunks):
        """Load c-th of N chunks from emb_fps"""
        num_embs = len(emb_fps)
        chunk_size = num_embs / num_chunks
        embs = []
        emb_idx = chunk_size * c
        for i in range(chunk_size):
            emb = pickle.load(open(emb_fps[emb_idx + i], 'r'))
            emb = emb.astype(np.float16)
            emb = np.squeeze(emb)
            embs.append(emb)
        return embs
            
    def fit_PCA(list_of_embs, n_components=1024):
        X = np.array(list_of_embs)
        print X.shape
        pca = PCA(n_components=n_components)
        pca.fit(X)
        return pca

    # # Content PCA
    # print '=' * 100
    # print 'Loading content embeddings'
    # content_embs = load_all_content_embs()
    # print 'Content PCA'
    # content_PCA = fit_PCA(content_embs, PCA_CONTENT_DIM)
    # with open(WIKIART_PAINTINGS_PCA_CONTENT, 'w') as f:
    #     pickle.dump(content_PCA, f, protocol=2)

    # Style PCA
    print '=' * 100
    print 'Style PCA'
    # Raw style embeddings too large to fit in memory (each is 131328-dim), so fit
    # IncrementalPCA in chunks
    style_PCA = IncrementalPCA(n_components=PCA_STYLE_DIM)
    style_emb_fps = get_all_style_emb_fps()
    num_chunks = 20
    for c in range(num_chunks):
        print '-- chunk: {}'.format(c)
        style_embs = load_style_embs_in_chunks(style_emb_fps, c, num_chunks)
        X = np.array(style_embs)
        if c == 0:
            print X.shape
        style_PCA.partial_fit(X)
    with open(WIKIART_PAINTINGS_PCA_STYLE, 'w') as f:
        pickle.dump(style_PCA, f, protocol=2)

def save_PCA_content_and_style_embs():
    """Use saved PCA to save content and style embeddings with a reduced dimension"""

    content_PCA = pickle.load(open(WIKIART_PAINTINGS_PCA_CONTENT, 'r'))
    style_PCA = pickle.load(open(WIKIART_PAINTINGS_PCA_STYLE, 'r'))

    # Iterate over all saved raw painting embeddings
    n = 0
    for artist_name in os.listdir(WIKIART_ARTIST_PAINTING_RAW_EMBEDDINGS_PATH):
        artist_path = os.path.join(WIKIART_ARTIST_PAINTING_RAW_EMBEDDINGS_PATH, artist_name)

        # Make directory to store PCA embeddings
        out_dir = os.path.join(WIKIART_ARTIST_PAINTING_PCA_EMBEDDINGS_PATH, artist_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        for emb_fn in os.listdir(artist_path):
            n += 1
            emb_fp = os.path.join(artist_path, emb_fn)
            print 'Processing: {}, n={}'.format(emb_fp, n)

            # Dimensionality reduction using PCA
            emb = pickle.load(open(emb_fp, 'r'))
            if emb_fn.endswith('content.pkl'):
                out_fn = emb_fn.replace('raw-content', 'PCA-content')
                emb = content_PCA.transform(emb.reshape(1, -1))
            elif emb_fn.endswith('style.pkl'):
                out_fn = emb_fn.replace('raw-style', 'PCA-style')
                emb = style_PCA.transform(emb.reshape(1, -1))

            # Save
            out_path = os.path.join(out_dir, out_fn)
            with open(out_path, 'w') as f:
                pickle.dump(emb, f, protocol=2)


if __name__ == '__main__':
    # Testing
    # test_artist_title_extraction()
    # test_lazy_load()
    # test_extract_VGG16_activations()
    # test_one_painting_emb()

    # Saving painting embeddings
    # save_raw_content_and_style_embs()
    # fit_and_save_PCA()
    save_PCA_content_and_style_embs()

    #     if torch.cuda.is_available():
    #         model = model.cuda()
