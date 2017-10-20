# datasets.py

"""
Available classes:
- WikiartDataset: implements torch Dataset class
    Only loads images that have influencer embeddings

Available functions:
-  get_wikiart_data_loader(artist, batch_size)
    Return data loader for the wikiart Dataset. Uses CenterCrop instead of RandomCrop
- calculate_per_channel_mean_std()
    Make a pass through dataset calcuating per-channel mean and stddev so that we can normalize images
- def save_influencers_embeddings():
    For each artist, compute and save the influencers_emb, which is the embedding representing the influencers of that
    artist. Done for each artist that has an artist embedding
- def counting_stats():
    Calculate some basic stats on number of valid artists etc.
"""

import numpy as np
import os
import pickle
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import HParams,\
    WIKIART_ARTIST_EMBEDDINGS_PATH, WIKIART_ARTISTS_PATH, WIKIART_ARTIST_PAINTING_PCA_EMBEDDINGS_PATH,\
    WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH, WIKIART_ARTIST_TO_INFO_PATH

class WikiartDataset(Dataset):
    """
    Images from Wikiart Dataset that have influencer embeddings
    
    Public attributes
    - split_name?
    - transform: pytorch transforms to apply to image
    - artist_fns: list of tuple of artist_name and image_fn
    """

    def __init__(self, transform=None):
        print 'Initializing dataset'
        self.transform = transform

        self.artist_fns = []
        valid_artists = os.listdir(WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH)
        for artist in valid_artists:

            # Only add filenames for images that are valid (i.e. have embeddings)
            embs_path = os.path.join(WIKIART_ARTIST_PAINTING_PCA_EMBEDDINGS_PATH, artist)
            content_emb_fns = [fn for fn in os.listdir(embs_path) if fn.endswith('-content.pkl')]

            # Extract title so we can match with img file (some are png's, some are jpg's, so can't just append .png)
            titles = [fn.split('_PCA-content.pkl')[0] for fn in content_emb_fns]
            img_fns = os.listdir(os.path.join(WIKIART_ARTISTS_PATH, artist))

            for title in titles:
                if title.startswith('christ-taking-leave-of-his-mother-1511'):
                    print title

            # Probably a better way to match, but...
            for title in titles:
                for img_fn in img_fns:
                    if img_fn.split('.')[0] == title:
                        self.artist_fns.append((artist, img_fn))

        print 'Done initializing dataset'

    def __len__(self):
        """Return number of data points"""
        return len(self.artist_fns)

    def __getitem__(self, idx):
        """Return idx-th image and artist in dataset"""
        artist, fn = self.artist_fns[idx]


        img_fp = os.path.join(WIKIART_ARTISTS_PATH, artist, fn)
        img = Image.open(img_fp)

        # Note: image is a (3,256,256) matrix, which is what we want because
        #       pytorch conv works on (batch, channels, width, height)
        if self.transform:
            img = self.transform(img)

        return img, artist

def get_wikiart_data_loader(batch_size, img_scale_size):
    """Return data loader for the wikiart Dataset. Uses CenterCrop instead of RandomCrop"""
    normalize = transforms.Normalize(mean=[0.514754100626, 0.453982621718, 0.386429772788],
                                     std=[0.168468122614, 0.171339982617, 0.178252021991])
    transform = transforms.Compose([
        transforms.Scale(img_scale_size),
        transforms.CenterCrop(img_scale_size),
        transforms.ToTensor(),
        normalize])
    dataset = WikiartDataset(transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

########################################################################################################################
# Precompute
########################################################################################################################

def calculate_per_channel_mean_std():
    """
    Make a pass through dataset calcuating per-channel mean and stddev so that we can normalize images
    """
    hp = HParams()
    dl = get_wikiart_data_loader(1, hp.img_size)

    rgb = []
    for i, (img_batch, artist_batch) in enumerate(dl):        # batch (1), channel, x, y
        r = img_batch[:,0,:,:].mean()
        g = img_batch[:,1,:,:].mean()
        b = img_batch[:,2,:,:].mean()
        rgb.append([r,g,b])
    rgb = np.array(rgb)

    print 'R mean: {}'.format(rgb[:,0].mean())
    print 'G mean: {}'.format(rgb[:,1].mean())
    print 'B mean: {}'.format(rgb[:,2].mean())
    print 'R std: {}'.format(rgb[:,0].std())
    print 'G std: {}'.format(rgb[:,1].std())
    print 'B std: {}'.format(rgb[:,2].std())

def save_influencers_embeddings():
    """
    For each artist, compute and save the influencers_emb, which is the embedding representing the influencers of that
    artist. Done for each artist that has an artist embedding
    
    Saves:
    - influencers_emb: numpy matrix containing mode embeddings of influencers. (num_means, mean_emb_size)
    Influencers are sorted by birth year if available.
    """

    # Definitely non-robust way of extracting birth year
    def get_birth_year(artist2info, artist_name):
        info = artist2info[artist_name]
        try:
            birth_year = int(info['Born'][0][-4:])  # last 4, e.g. [u'14April1852', ...]
        except Exception:
            birth_year = None
        return birth_year

    artist2info = pickle.load(open(WIKIART_ARTIST_TO_INFO_PATH, 'r'))
    potential_artists = os.listdir(WIKIART_ARTIST_EMBEDDINGS_PATH)

    for i, artist_name in enumerate(potential_artists):
        print artist_name

        # Get influencers and sort by birth year
        influencers = artist2info[artist_name]['Influencedby']
        influencers = [(inf, get_birth_year(artist2info, inf)) for inf in influencers]
        influencers = [inf for inf, birth_year in sorted(influencers, key=lambda x: x[1])]

        # Few influencers may not have artist embeddings -- maybe images are missing, invalid, etc.
        influencers_embs_path = [os.path.join(WIKIART_ARTIST_EMBEDDINGS_PATH, influencer, 'embedding.pkl') for
                                 influencer in influencers]
        influencers_embs_path = [emb_path for emb_path in influencers_embs_path if os.path.exists(emb_path)]

        # Skip those without influencers (should mostly just be the root nodes)
        if len(influencers_embs_path) == 0:
            print 'Skipping: {}'.format(artist_name)
            continue

        # Create embedding Variable
        influencers_emb = []
        for emb_path in influencers_embs_path:
            emb = pickle.load(open(emb_path, 'r'))
            # Add each GMM mean
            for mean_idx in range(emb.shape[0]):
                influencers_emb.append(emb[mean_idx])

        influencers_emb = np.array(influencers_emb)  # (num_means, mean_emb_size)

        # Save
        out_dirpath = os.path.join(WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH, artist_name)
        if not os.path.exists(out_dirpath):
            os.mkdir(out_dirpath)
        out_path = os.path.join(out_dirpath, 'embedding.pkl')
        with open(out_path, 'w') as f:
            pickle.dump(influencers_emb, f, protocol=2)

def counting_stats():
    """
    Calculate some basic stats on number of valid artists etc.
    """
    # Final WikiartDataset used for training will only use artists that have an influencers embedding
    artists = os.listdir(WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH)
    num_modes = []
    num_paintings = []
    for artist in artists:
        influencers_emb = pickle.load(open(
            os.path.join(WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH, artist, 'embedding.pkl'), 'r'))
        num_modes.append(influencers_emb.shape[0])
        num_painting_embs = len(os.listdir(os.path.join(WIKIART_ARTIST_PAINTING_PCA_EMBEDDINGS_PATH, artist))) / 2
        num_paintings.append(num_painting_embs)
    num_modes = np.array(num_modes)
    num_paintings = np.array(num_paintings)

    print 'Number of artists: {}'.format(len(artists))
    print 'Average number of modes per artist: {}'.format(num_modes.mean())
    print 'Average number of paintings per artist: {}'.format(num_paintings.mean())
    print 'Total number of paintings: {}'.format(num_paintings.sum())

if __name__ == '__main__':
    # calculate_per_channel_mean_std()
    # save_influencers_embeddings()
    # counting_stats()

    # Test
    dl = get_wikiart_data_loader(4, 64)
    for img_batch, artist_batch in dl:
        print img_batch
        print artist_batch
        break