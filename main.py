# main.py

"""
Main file to train or generate.
"""
import argparse

from config import HParams
from network import Network

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', default='train', help='train,generate')
    parser.add_argument('--influencers', dest='influencers', default=None,
                        help='Comma-separated list of artist-names. Used when generating')
    parser.add_argument('--rand_infl_emb', action='store_true',
                        help='Fill influencers embedding with random noise')
    parser.add_argument('--notes', dest='notes', default=None,
                        help='helpful to save some notes about this experiment (i.e. changes made)')
    cmdline = parser.parse_args()

    # Load default hyperparameters and update
    hparams = HParams()
    # hparams.lr = cmdline.lr if cmdline.lr is not None else hparams.lr
    if cmdline.rand_infl_emb:
        hparams.rand_infl_emb = True
    hparams.notes = cmdline.notes

    net = Network(hparams)
    if cmdline.mode == 'train':
        net.train()
    elif cmdline.mode == 'generate':
        net.generate(cmdline.influencers)
