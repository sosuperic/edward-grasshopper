# artist_embeddings.py

"""
Generate embedding representing artist.

Available classes:
- Artist

Available functions:
- test_avg_emb
"""
import numpy as np
import os
import pickle

from config import WIKIART_PATH, WIKIART_ARTISTS_PATH, WIKIART_ARTIST_EMBEDDINGS_PATH
from painting_embeddings import Painting

class Artist(object):
    """Represents an artist as a function of all its painting embeddings.
    
    Public attributes:
    - name: name of artist (str)
    - artist_path: path to directory containing images (str)
    """

    def __init__(self, name):
        self.name = name
        self.artist_path = os.path.join(WIKIART_ARTISTS_PATH, self.name)

    def create_avg_embedding(self):
        """
        Create embedding by taking average of all painting embeddings
        """
        def update_moving_avg(old_avg, cur_point, n):
            new_avg = (cur_point + (n * old_avg)) / float(n+1)      # n+1 is number of points received so far
            return new_avg

        avg_emb = np.zeros(Painting.emb_size)
        vgg16 = Painting.get_vgg16()
        files = os.listdir(self.artist_path)
        for i, file in enumerate(files):
            img_path = os.path.join(self.artist_path, file)
            painting = Painting(img_path)
            # emb = painting.create_embedding(vgg16)
            emb = np.random.rand(Painting.emb_size[0])  # Used as a mock until create_embedding is implemented
            avg_emb = update_moving_avg(avg_emb, emb, i)

        return avg_emb

    def create_GMM_embedding(self, k, covariance='tied'):
        pass

def test_avg_emb():
    artist = Artist('morris-louis')
    artist_emb = artist.create_avg_embedding()
    print artist_emb

if __name__ == '__main__':
    test_avg_emb()