# Configuations, paths, etc.

# Default hyperparameters
class HParams():
    def __init__(self):

        # Training
        self.batch_size = 16
        self.lr = 0.001
        self.max_nepochs = 10
        self.save_every_nepochs = 1

        # Model
        self.img_size = 64
        self.lstm_emb_size = 512                # size of linear layer before LSTM
        self.lstm_hidden_size = 128
        self.d_num_filters = 64                 # number of filters for first conv in Discriminator
        self.z_size = 128                       # size of noise vector for Generator
        self.g_num_filters = 128                # number of filters for first conv in Generator

        # Other
        self.load_lstm_fp = None
        self.load_G_fp = None
        self.load_D_fp = None


# Wikiart general
WIKIART_PATH = 'data/wikiart/'
WIKIART_ARTISTS_PATH = 'data/wikiart/data'
WIKIART_ARTIST_TO_INFO_PATH = 'data/wikiart/artist_to_info.pkl'
WIKIART_TEST_IMG_PATH = 'data/wikiart/data/pablo-picasso/guernica-1937.png'

# Wikiart influence
WIKIART_INFLUENCE_GRAPH_PATH = 'data/wikiart_influence_graph.json'
# Filtered to only nodes that have an edge, used Peter's tool
WIKIART_INFLUENCE_GRAPH_FILTERED_PATH = 'data/wikiart_influence_graph_filtered.json'

# Saved Embeddings
WIKIART_ARTIST_PAINTING_RAW_EMBEDDINGS_PATH = 'data/wikiart_artist_painting_raw_embs'
WIKIART_ARTIST_PAINTING_PCA_EMBEDDINGS_PATH = 'data/wikiart_artist_painting_pca_embs'
WIKIART_PAINTINGS_PCA_CONTENT = 'data/wikiart_paintings_PCA/PCA_content.pkl'
WIKIART_PAINTINGS_PCA_STYLE = 'data/wikiart_paintings_PCA/PCA_style.pkl'
PCA_CONTENT_DIM = 1024
PCA_STYLE_DIM = 1024
WIKIART_ARTIST_EMBEDDINGS_PATH = 'data/wikiart_artist_embs'
WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH = 'data/wikiart_artist_influencers_embs'

# Network, Models
SAVED_MODELS_PATH = 'checkpoints/'