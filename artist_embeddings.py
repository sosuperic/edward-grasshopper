# artist_embeddings.py

"""
Generate embedding representing artist.

Available classes:
- Artist

Available functions:
- test_avg_emb: Create embedding using average of all painting embeddings
- GMM_hyperpararm_search: Create embedding using GMM of all painting embeddings
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import os
from sklearn.mixture import GaussianMixture

from config import WIKIART_ARTISTS_PATH, WIKIART_ARTIST_EMBEDDINGS_PATH
from painting_embeddings import Painting


class Artist(object):
    """Represents an artist as a function of all its painting embeddings.
    
    Public attributes:
    - name: name of artist (str)
    - artist_path: path to directory containing images (str)
    - num_paintings: number of paintings artist has (int)
    """
    component_size = 4      # TODO: replace with actual size

    def __init__(self, name):
        self.name = name
        self.artist_path = os.path.join(WIKIART_ARTISTS_PATH, self.name)
        self._num_paintings = None

    @property
    def num_paintings(self):
        if self._num_paintings is None:
            self._num_paintings = len(os.listdir(self.artist_path))
        return self._num_paintings

    def __str__(self):
        return '{}: {} paintings'.format(self.name, self.num_paintings)

    ####################################################################################################################
    # Average embedding
    ####################################################################################################################
    def create_avg_embedding(self):
        """Create embedding by taking average of all painting embeddings"""
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

    ####################################################################################################################
    # Gaussian Mixture Model embedding
    ####################################################################################################################
    def create_GMM_embedding(self, k=3, covariance_type='tied'):
        """Create embedding by creating Gaussian mixture model with k modes and covariance structure.
        Fit to paintings data
        
        Inputs:
            - k: number of modes (int)
            - covariance: full, tied, diag, spherical (str) 
        
        Returns:
        """
        # TODO: replace this with array of painting embeddings
        mock_data = np.random.rand(self.num_paintings, Painting.emb_size[0])

        # Create estimator and fit to data)
        estimator = GaussianMixture(n_components=k,
                                    covariance_type=covariance_type,
                                    max_iter=100,
                                    random_state=0)
        estimator.fit(mock_data)

        # TODO: return actual embedding too, i.e. some combination of mean and variances

        return estimator

    def plot_GMMs_varying_covariance(self, estimators, k, fig_title=None):
        """Plot components of GMM estimators
        
        Inputs:
        - estimators: dictionary from covariance_type to estimator ({str: sklearn estimator})
        - k: number of components used in each estimator
        - fig_title: title of overall figure
        """

        # Color map to have different colors for each mixture component
        def get_cmap(n, name='hsv'):
            """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
            RGB color; the keyword argument name must be a standard mpl colormap name."""
            return plt.cm.get_cmap(name, n)

        cmap = get_cmap(k)

        def make_ellipses(gmm, ax):
            """Draw an ellipse per component"""
            for n in range(k):      # n is a component index
                color = cmap(n)
                if gmm.covariance_type == 'full':
                    covariances = gmm.covariances_[n][:2, :2]
                elif gmm.covariance_type == 'tied':
                    covariances = gmm.covariances_[:2, :2]
                elif gmm.covariance_type == 'diag':
                    covariances = np.diag(gmm.covariances_[n][:2])
                elif gmm.covariance_type == 'spherical':
                    covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
                v, w = np.linalg.eigh(covariances)
                u = w[0] / np.linalg.norm(w[0])
                angle = np.arctan2(u[1], u[0])
                angle = 180 * angle / np.pi  # convert to degrees
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                          180 + angle, color=color)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(0.5)
                ax.add_artist(ell)

        n_estimators = len(estimators)

        # Set up figure
        plt.figure(figsize=(3 * n_estimators / 2, 6))
        plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                            left=.01, right=.99)

        # Plot ellipses
        for index, (name, estimator) in enumerate(estimators.items()):
            h = plt.subplot(2, n_estimators // 2, index + 1)
            make_ellipses(estimator, h)

            # plt.xticks(())
            # plt.yticks(())
            plt.title(name)     # title for subfigure

        # Finish figure and show
        plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))
        plt.suptitle(fig_title)

        plt.show()

    @staticmethod
    def optimize_GMM_params(n_artists=50,
                            k_grid=[1,2,3,4,5,6,7,8,9,10]):
        """Run GMM on a number of artists with varying k and covariance_type.
        
        Purpose is to see what is a proper number of components (and covariance_type) to use for
        artist in a downstream task. That is, do most artists have two modes, three modes, one
        mode, etc. Do this by visual inspection, as we are essentially just clustering data
        without labels.
        
        Inputs:
        - n_artists: number of separate artists to try (int)
        - k_grid: number of components in GMM (list of ints)
        """
        # Run for n_artists, each time doing grid search over k_grid and covariance_types
        for i in range(n_artists):
            rand_artist_name = np.random.choice(os.listdir(WIKIART_ARTISTS_PATH), size=1, replace=False)[0]
            artist_obj = Artist(rand_artist_name)
            print artist_obj
            # For given artist, see how different k's and covariance types fit
            for k in k_grid:
                # Plot shows all covariance types together
                # It probably makes more sense to show different k's together actually...oh well
                estimators_by_covariance = {}
                for covariance_type in ['spherical', 'diag', 'tied', 'full']:
                    estimator = artist_obj.create_GMM_embedding(k, covariance_type)
                    estimators_by_covariance[covariance_type] = estimator
                artist_obj.plot_GMMs_varying_covariance(estimators_by_covariance, k,
                                                        '{}_k={}'.format(rand_artist_name, k))

####################################################################################################################
# Test functions
####################################################################################################################

def test_avg_emb():
    """Create embedding using average of all painting embeddings"""
    artist = Artist('morris-louis')
    artist_emb = artist.create_avg_embedding()
    print artist_emb

def GMM_hyperparam_search():
    """Create embedding using GMM of all painting embeddings"""
    Artist.optimize_GMM_params(n_artists=10, k_grid=[2,3,5])


if __name__ == '__main__':
    # test_avg_emb()
    GMM_hyperparam_search()