# Configuations, paths, etc.

# Default hyperparameters
class HParams():
    def __init__(self):

        # Training
        self.batch_size = 32
        self.lr_infl = 0.0002
        self.lr_G = 0.0002
        self.lr_D = 0.0002
        self.D_iters = 5                        # number of times to train Wasserstein critic for one G update
        self.max_nepochs = 50
        self.save_every_nepochs = 1

        # Model
        self.img_size = 128
        self.infl_type = 'ff'  # ff or lstm
        self.lstm_emb_size = 512                # size of linear layer before LSTM
        self.infl_hidden_size = 128
        self.d_num_filters = [32, 64, 128, 256, 512]
        self.z_size = 128                       # size of noise vector for Generator
        self.g_num_filters = [512, 384, 256, 128, 64]

        # Other
        self.load_infl_fp = None
        self.load_G_fp = None
        self.load_D_fp = None
        self.cur_epoch = None
        # self.load_infl_fp = 'checkpoints/October20_21-05-20_always-update_new-G_new-D/infl_e8.pt'
        # self.load_G_fp = 'checkpoints/October20_21-05-20_always-update_new-G_new-D/G_e8.pt'
        # self.load_D_fp = 'checkpoints/October20_21-05-20_always-update_new-G_new-D/D_e8.pt'
        self.load_infl_fp = 'checkpoints/October21_03-33-01_always-update_new-G_new-D_fuse-layer-3-G_rand-infl-emb/infl_e7.pt'
        self.load_G_fp = 'checkpoints/October21_03-33-01_always-update_new-G_new-D_fuse-layer-3-G_rand-infl-emb/G_e7.pt'
        self.load_D_fp = 'checkpoints/October21_03-33-01_always-update_new-G_new-D_fuse-layer-3-G_rand-infl-emb/D_e7.pt'
        # self.cur_epoch = 13

# Manually selected list of artists to train on (number of images is also given)
# Train on thirty instead of 202
TRAIN_ON_ALL = True
THIRTY_ARTISTS = [(251, 'gustave-courbet'),
 (258, 'lucian-freud'),
 (273, 'jean-auguste-dominique-ingres'),
 (274, 'egon-schiele'),
 (274, 'kazimir-malevich'),
 (286, 'mary-cassatt'),
 (293, 'amedeo-modigliani'),
 (316, 'rene-magritte'),
 (328, 'fernand-leger'),
 (335, 'peter-paul-rubens'),
 (337, 'maurice-prendergast'),
 (355, 'ernst-ludwig-kirchner'),
 (356, 'henri-de-toulouse-lautrec'),
 (375, 'francisco-goya'),
 (386, 'ivan-shishkin'),
 (417, 'henri-matisse'),
 (428, 'alfred-sisley'),
 (465, 'paul-cezanne'),
 (483, 'camille-corot'),
 (489, 'paul-gauguin'),
 (493, 'ilya-repin'),
 (495, 'edgar-degas'),
 (521, 'childe-hassam'),
 (715, 'rembrandt'),
 (737, 'albrecht-durer'),
 (743, 'marc-chagall'),
 (766, 'john-singer-sargent'),
 (844, 'pablo-picasso'),
 (1114, 'claude-monet'),
 (1168, 'pierre-auguste-renoir')]

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
ARTIST_EMB_DIM = 2048

# Network, Models
SAVED_MODELS_PATH = 'checkpoints/'
