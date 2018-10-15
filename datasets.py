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
    WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH, WIKIART_ARTIST_TO_INFO_PATH, THIRTY_ARTISTS

class WikiartDataset(Dataset):
    """
    Images from Wikiart Dataset that have influencer embeddings
    
    Public attributes
    - split_name?
    - transform: pytorch transforms to apply to image
    - artist_fns: list of tuple of artist_name and image_fn
    """
    def __init__(self, train_on_all, transform=None, split=None):
        print 'Initializing dataset'
        self.transform = transform
        self.split = split

        self.artist_fns = []
        if train_on_all:
            valid_artists = os.listdir(WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH)    # Train on all 202 valid artists
        else:
            valid_artists = [a for n, a in THIRTY_ARTISTS]                              # Only train on top 30
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
            cur_artist_fns = []
            for title in titles:
                for img_fn in img_fns:
                    if img_fn.split('.')[0] == title:
                        cur_artist_fns.append((artist, img_fn))
            cur_artist_fns = self.extract_split(cur_artist_fns, split)      # Extract split per artist
            self.artist_fns.extend(cur_artist_fns)

        print 'Done initializing dataset'

    def extract_split(self, fns, split):
        """Return subset of fns (list of filenames) corresponding to that split"""
        n = len(fns)
        if split == 'train':
            fns = fns[:int(0.8 * n)]
        elif split == 'valid':
            fns = fns[int(0.8 * n):int(0.9 * n)]
        elif split == 'test':
            fns = fns[int(0.9 * n):]
        return fns

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

    @staticmethod
    def get_artist_one_hot_index(artist, train_on_all):
        """Return index of artist"""
        if train_on_all:
            valid_artists = os.listdir(WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH)
        else:
            valid_artists = [a for n, a in THIRTY_ARTISTS]
        valid_artists = sorted(valid_artists)
        index = valid_artists.index(artist)
        return index

    @staticmethod
    def get_num_valid_artists(train_on_all):
        """Return number of valid artists, i.e. length of one hot"""
        if train_on_all:
            n = len(os.listdir(WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH))
        else:
            n = len(THIRTY_ARTISTS)
        return n

def get_wikiart_data_loader(batch_size, img_scale_size, train_on_all, split=None):
    """Return data loader for the wikiart Dataset. Uses CenterCrop instead of RandomCrop"""

    if train_on_all:
        normalize = transforms.Normalize(mean=[0.518804936264, 0.455788331489, 0.388189828956],
                                         std=[0.175493882268, 0.177110228715, 0.185269485347])
    else:
        normalize = transforms.Normalize(mean=[0.525830027979, 0.46643356245,0.398731525078],
                                        std=[0.167530716224, 0.169428801834, 0.180189925318])
    transform = transforms.Compose([
        transforms.Scale(img_scale_size),
        transforms.RandomSizedCrop(img_scale_size),
        transforms.ToTensor(),
        normalize
    ])
    dataset = WikiartDataset(train_on_all=train_on_all, transform=transform, split=split)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

########################################################################################################################
# Precompute
########################################################################################################################

def calculate_per_channel_mean_std(train_on_all=False):
    """
    Make a pass through dataset calcuating per-channel mean and stddev so that we can normalize images
    """
    hp = HParams()
    dl = get_wikiart_data_loader(1, hp.img_size, train_on_all)

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
    calculate_per_channel_mean_std(train_on_all=False)
    # save_influencers_embeddings()
    # counting_stats()

    # Test
    # dl = get_wikiart_data_loader(4, 64, 'valid')
    # for img_batch, artist_batch in dl:
    #     print img_batch
    #     print artist_batch
    #     break