# Generate networkx graph

import networkx as nx
import pickle

from config import WIKIART_ARTIST_TO_INFO_PATH

def artist_influence():
    # G = nx.DiGraph()
    G = nx.Graph()

    a2i = pickle.load(open(WIKIART_ARTIST_TO_INFO_PATH, 'rb'))
    artists = sorted(a2i.keys())
    for a in artists:
        G.add_node(a)

    #
    for artist, info in a2i.items():
        if 'Influencedby' in info:
            for other_artist in info['Influencedby']:
                if other_artist in artists: # skip schools/movements
                    G.add_edge(other_artist, artist)
                # artists.index
        if 'Influencedon' in info:
            for other_artist in info['Influencedon']:
                if other_artist in artists:
                    G.add_edge(artist, other_artist)
        # Not used right now
        # if 'FriendsandCo-workers' in a2i:

    # print G.edges()
    # Write to csv for gephi
    f = open('artist_influence.csv', 'wb')
    f.write('Source,Target\n')
    for a1, a2 in G.edges():
        f.write('{},{}\n'.format(a1, a2))

    # Draw using matplotlib and networkx
    nx.draw(G, node_size=25)
    import matplotlib.pylab as plt
    plt.show()

    # Calculate centrality
    from pprint import pprint
    # pprint(sorted(nx.betweenness_centrality(G).items(), key=lambda x: x[1]))
    pprint(sorted(nx.eigenvector_centrality(G).items(), key=lambda x: x[1]))


if __name__ == '__main__':
    # Analysis
    artist_influence()