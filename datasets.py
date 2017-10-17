# datasets.py

"""
Available classes:
- WikiartDataset: implements torch Dataset class
    Only loads images that have artist embeddings

Available functions:
-  get_wikiart_data_loader(artist, batch_size):
    Return data loader for the wikiart Dataset. Uses CenterCrop instead of RandomCrop
"""

import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import WIKIART_ARTIST_EMBEDDINGS_PATH, WIKIART_ARTISTS_PATH, WIKIART_ARTIST_PAINTING_PCA_EMBEDDINGS_PATH

class WikiartDataset(Dataset):
    """
    Images from Wikiart Dataset that have artist embeddings
    
    Public attributes
    - split_name?
    - transform: pytorch transforms to apply to image
    - artist_fns: list of tuple of artist_name and image_fn
    """

    def __init__(self, transform=None):
        self.transform = transform

        self.artist_fns = []
        valid_artists = os.listdir(WIKIART_ARTIST_EMBEDDINGS_PATH)
        for artist in valid_artists:
            # Only add filenames for images that are valid (i.e. have embeddings)
            embs_path = os.path.join(WIKIART_ARTIST_PAINTING_PCA_EMBEDDINGS_PATH, artist)
            content_emb_fns = [fn for fn in os.listdir(embs_path) if fn.endswith('-content.pkl')]

            # Extract title so we can match with img file (some are png's, some are jpg's, so can't just append .png)
            titles = [fn.split('_PCA-content.pkl')[0] for fn in content_emb_fns]
            img_fns = os.listdir(os.path.join(WIKIART_ARTISTS_PATH, artist))

            # Probably a better way to match, but...
            for title in titles:
                for img_fn in img_fns:
                    if img_fn.startswith(title):
                        self.artist_fns.append((artist, img_fn))

    def __len__(self):
        """Return number of data points"""
        return len(self.artist_fns)

    def __getitem__(self, idx):
        """Return idx-th image in dataset"""
        artist, fn = self.artist_fns[idx]

        img_fp = os.path.join(WIKIART_ARTISTS_PATH, artist, fn)
        image = Image.open(img_fp)

        # Note: image is a (3,256,256) matrix, which is what we want because
        #       pytorch conv works on (batch, channels, width, height)
        if self.transform:
            image = self.transform(image)

        return image

def get_wikiart_data_loader(batch_size, img_scale_size):
    """Return data loader for the wikiart Dataset. Uses CenterCrop instead of RandomCrop"""
    transform = transforms.Compose([
        transforms.Scale(img_scale_size),
        transforms.CenterCrop(img_scale_size),
        transforms.ToTensor()])
    dataset = WikiartDataset(transform=transform)

    print dataset.__len__()

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

if __name__ == '__main__':
    # Test
    dl = get_wikiart_data_loader(4, 32)
    for img in dl:
        print img
        break