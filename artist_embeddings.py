# artist_embeddings.py

"""
Generate embedding representing artist

Available classes:
- Artist

Available functions:
- print_artist_to_num_modes():
    Print the number of GMM modes each artist has
- calculate_and_save_artist_gmm_and_embs():
    Calculate artist embeddings for all artists that have PCA painting embeddings
"""

from collections import defaultdict
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import mpld3
import numpy as np
import os
import pickle
from pprint import pprint
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

from config import WIKIART_ARTISTS_PATH, WIKIART_ARTIST_PAINTING_PCA_EMBEDDINGS_PATH, WIKIART_ARTIST_EMBEDDINGS_PATH
from painting_embeddings import Painting


class Artist(object):
    """Represents an artist as a function of all its painting embeddings.
    
    Public attributes:
    - name: name of artist (str)
    - artist_path: path to directory containing images (str)
    - embs_path: path to directory containing embeddings (str)
    - num_paintings: number of paintings artist has (int)
    - num_embs: number of painting embeddings artist has, i.e. valid paintings (int)
    """
    component_size = 4      # TODO: replace with actual size

    def __init__(self, name):
        self.name = name
        self.imgs_path = os.path.join(WIKIART_ARTISTS_PATH, self.name)
        embs_path = os.path.join(WIKIART_ARTIST_PAINTING_PCA_EMBEDDINGS_PATH, self.name)
        self.embs_path = embs_path if os.path.exists(embs_path) else None
        self._num_paintings = None
        self._num_embs = None

    @property
    def num_paintings(self):
        if self._num_paintings is None:
            self._num_paintings = len(os.listdir(self.imgs_path))
        return self._num_paintings

    @property
    def num_embs(self):
        """Some paintings don't have embeddings because they weren't valid images (e.g. black and white, size)"""
        if self._num_embs is None:
            self._num_embs = len(os.listdir(self.embs_path)) / 2      # (divide by 2 because style and content)
        return self._num_embs

    def __str__(self):
        url = 'https://www.wikiart.org/en/{}'
        return '{}: {} paintings; {}'.format(self.name, self.num_paintings, url.format(self.name))

    def has_embs(self):
        has_embs = (self.embs_path is not None) and (self.num_embs != 0)
        return has_embs

    def load_painting_embeddings(self):
        """
        Iterate through painting embeddings, combining content and style PCA embeddings
        
        Returns:
        - embs: numpy matrix -- each row represents painting embedding with dimension Painting.emb_size
        - titles: list of strings, each painting title
        """
        embs = np.zeros((self.num_embs, Painting.emb_size[0]))
        # embs = np.zeros((self.num_paintings, Painting.emb_size[0] / 2))

        embs_path = os.path.join(WIKIART_ARTIST_PAINTING_PCA_EMBEDDINGS_PATH, self.name)
        emb_fns = os.listdir(embs_path)

        # Sort so that indexing into content and style returns embedding for same painting
        content_emb_fns = sorted([fn for fn in emb_fns if fn.endswith('content.pkl')])
        style_emb_fns = sorted([fn for fn in emb_fns if fn.endswith('style.pkl')])
        titles = []
        for i in range(len(content_emb_fns)):
            # Load content and style embs, place into embs matrix
            content_emb = pickle.load(open(os.path.join(embs_path, content_emb_fns[i]), 'r'))
            style_emb = pickle.load(open(os.path.join(embs_path, style_emb_fns[i]), 'r'))
            embs[i][0: Painting.emb_size[0] / 2] = content_emb
            # embs[i][0: Painting.emb_size[0] / 2] = style_emb
            embs[i][Painting.emb_size[0] / 2:] = style_emb

            title = content_emb_fns[i].split('_PCA-content.pkl')[0]
            titles.append(title)

        return embs, titles

    def TSNE(self, painting_embs, painting_titles, show=True):
        """
        Calculate TSNE embedding on matrix of painting embeddings
        
        Returns:
        - tsne_emb: num_embs x 2 matrix (i.e. coordinates)
        """

        X = painting_embs
        model = TSNE(n_components=2, random_state=0, learning_rate=100)
        np.set_printoptions(suppress=True)
        tsne_emb = model.fit_transform(X)

        if show:    # create scatter plot
            x, y = tsne_emb[:, 0], tsne_emb[:, 1]
            fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
            scatter = ax.scatter(x, y, alpha=0.3, cmap=plt.cm.jet)
            ax.grid(color='white', linestyle='solid')
            ax.set_title('https://wikiart.org/en/{}'.format(self.name), size=20)
            tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=painting_titles)
            mpld3.plugins.connect(fig, tooltip)
            mpld3.show()

        return tsne_emb

    ####################################################################################################################
    # Gaussian Mixture Model embedding
    ####################################################################################################################
    def fit_GMM(self, painting_embs, k=3, covariance_type='tied'):
        """
        Create embedding by creating Gaussian mixture model with k modes and covariance structure.
        Fit to paintings data.
        
        Inputs:
        - painting_embs: matrix where each row is embedding for one painting
        - k: number of modes (int)
        - covariance: full, tied, diag, spherical (str)
        
        Returns:
        - estimator: fitted GMM
        """
        estimator = GaussianMixture(n_components=k,
                                    covariance_type=covariance_type,
                                    max_iter=100,
                                    random_state=0)
        estimator.fit(painting_embs)

        return estimator

    def optimize_GMM(self,
                     nruns=10,
                     k_grid=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     covariance_types=['spherical', 'diag']): # 'tied', 'full):
        """
        Run GMM on a number of artists with varying k and covariance_type.
        
        Purpose is to see how many modes an artist has.
        
        Inputs:
        - nruns: number of times to fit a GMM, on different random subsets
        - n_artists: number of separate artists to try (int)
        - k_grid: number of components in GMM (list of ints)
        """
        # Store fit results in following dictionary:
        #   - keys will be tuples, score will be list of log likelihood of fit on test set
        hyperparams2loglikelihood = defaultdict(list)

        # Load data
        painting_embs, _ = self.load_painting_embeddings()
        # painting_embs = self.TSNE(painting_embs, titles, show=False)  # was seeing how well # clusters matches with TSNE representation

        # Split into train and test
        train_test_ratio = 0.75
        ntrain = int(train_test_ratio * len(painting_embs))

        for k in k_grid:
            # Skip if number of embeddings less than number of GMM components
            if k > int(self.num_embs * train_test_ratio):   # not enough to fit
                continue
            for covariance_type in covariance_types:

                # Fit nruns time, each time shuffling data so we fit and test on different subsets
                for i in range(nruns):
                    # Shuffle data and split in train, test
                    np.random.shuffle(painting_embs)
                    painting_embs_train = painting_embs[:ntrain]
                    painting_embs_test = painting_embs[ntrain:]

                    # Fit on subset of painting_embs
                    estimator = self.fit_GMM(painting_embs_train, k, covariance_type)
                    effective_k = np.sum(estimator.weights_ > 0.01)

                    # Calculate fit on test set
                    log_likelihood = estimator.score(painting_embs_test)
                    hyperparams2loglikelihood[(k, effective_k, covariance_type)].append(log_likelihood)

        # Edge case when artist only has 1 emb (not enough to train and fit)
        if len(hyperparams2loglikelihood) == 0:
            estimator = self.fit_GMM(painting_embs, 1, 'spherical')
            effective_k = np.sum(estimator.weights_ > 0.01)
            log_likelihood = estimator.score(painting_embs)
            hyperparams2loglikelihood[(1, effective_k, 'spherical')].append(log_likelihood)

        # For each hyperparameter setting, average over all runs
        hyperparams2loglikelihood = {hp: sum(ll) / float(len(ll)) for hp, ll in hyperparams2loglikelihood.items()}

        # Get best hyperparameter setting
        best_hp, best_score = sorted(hyperparams2loglikelihood.items(), key=lambda x: x[1])[-1]
        settings = {'k': best_hp[0], 'effective_k_with_0.01': best_hp[1], 'covariance_type': best_hp[2],
                    'score': best_score}
        print 'Best setting: k={}, effective_k={}, cov={}, score={}'.format(best_hp[0], best_hp[1], best_hp[2],
                                                                            best_score)

        # Fit on all embeddings with best
        estimator = self.fit_GMM(painting_embs, best_hp[0], best_hp[2])


        return estimator, settings

####################################################################################################################
# Calculating artist embeddings
####################################################################################################################
def calculate_and_save_artist_gmm_and_embs():
    """
    Calculate artist embeddings for all artists that have PCA painting embeddings
    
    Saves 3 files:
    - estimator.pkl: 
    - settings.json: optimal settings used to fit GMM
    - embedding.pkl: matrix, each row is a mean of fitted GMM
    """
    for i, artist_name in enumerate(os.listdir(WIKIART_ARTIST_PAINTING_PCA_EMBEDDINGS_PATH)):
        artist = Artist(artist_name)
        print '=' * 100
        print artist, i

        # Get optimal estimator
        estimator, settings = artist.optimize_GMM()

        # Get embedding from estimator, i.e. the
        embedding = estimator.means_

        # Write to file
        out_dir = os.path.join(WIKIART_ARTIST_EMBEDDINGS_PATH, artist_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        estimator_fp = os.path.join(out_dir, 'estimator.pkl')
        settings_fp = os.path.join(out_dir, 'settings.json')
        embedding_fp = os.path.join(out_dir, 'embedding.pkl')

        with open(estimator_fp, 'w') as f:
            pickle.dump(estimator, f, protocol=2)
        with open(settings_fp, 'w') as f:
            json.dump(settings, f)
        with open(embedding_fp, 'w') as f:
            pickle.dump(embedding, f, protocol=2)

def print_artist_to_num_modes():
    """
    Print the number of GMM modes each artist has
    """
    artist2counts = {}
    for artist_name in os.listdir(WIKIART_ARTIST_EMBEDDINGS_PATH):
        artist = Artist(artist_name)
        fp = os.path.join(WIKIART_ARTIST_EMBEDDINGS_PATH, artist_name, 'settings.json')
        num_modes = json.load(open(fp, 'r'))['k']
        artist2counts[artist_name] = {'num_modes': num_modes, 'num_embs': artist.num_embs}

    pprint(sorted(artist2counts.items(), key=lambda x: x[1]['num_modes']))

    print 'Num artists with 1 mode: {}'.format(len([a for a, d in artist2counts.items() if d['num_modes'] == 1]))
    print 'Num artists with 2 modes: {}'.format(len([a for a, d in artist2counts.items() if d['num_modes'] == 2]))
    print 'Num artists with 3 modes: {}'.format(len([a for a, d in artist2counts.items() if d['num_modes'] == 3]))
    print 'Num artists with 4 modes: {}'.format(len([a for a, d in artist2counts.items() if d['num_modes'] == 4]))
    print 'Num artists with 5 modes: {}'.format(len([a for a, d in artist2counts.items() if d['num_modes'] == 5]))
    print 'Num artists with 6 modes: {}'.format(len([a for a, d in artist2counts.items() if d['num_modes'] == 6]))
    print 'Num artists with 7 modes: {}'.format(len([a for a, d in artist2counts.items() if d['num_modes'] == 7]))
    print 'Num artists with 8 modes: {}'.format(len([a for a, d in artist2counts.items() if d['num_modes'] == 8]))
    print 'Num artists with 9 modes: {}'.format(len([a for a, d in artist2counts.items() if d['num_modes'] == 9]))
    print 'Num artists with 10 modes: {}'.format(len([a for a, d in artist2counts.items() if d['num_modes'] == 10]))

if __name__ == '__main__':
    # Run TSNE
    # artist = Artist('pablo-picasso')
    # painting_embs, titles = artist.load_painting_embeddings()
    # artist.TSNE(painting_embs, titles, show=True)

    # Save artist embeddings
    # calculate_and_save_artist_gmm_and_embs()

    print_artist_to_num_modes()