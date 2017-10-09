# Generate json to use for Peter's graph visualization tool

import json
import pickle

from config import WIKIART_ARTIST_TO_INFO_PATH, WIKIART_INFLUENCE_GRAPH_PATH

def create_json():
    """
    Generate json to use for Peter's graph visualization tool
    """
    a2i = pickle.load(open(WIKIART_ARTIST_TO_INFO_PATH, 'rb'))
    artists = set(a2i.keys())

    # Define node attributes
    meta = {'fields': [
        {'name': 'id', 'type': 'node-id'},
        {'name': 'birth_year', 'type': 'integer'},
        {'name': 'art_movement', 'type': 'string'},
        {'name': 'genre', 'type': 'string'}
    ]}

    # Add nodes and links
    nodes = []
    links = []
    for artist, info in a2i.items():
        ### Add node
        birth_year, art_movement, genre = None, None, None

        # Probably non-robust way of extracting year, but...
        try:
            birth_year = int(info['Born'][0][-4:])  # last 4, e.g. [u'14April1852', ...]
        except Exception:
            pass

        # Note artists can have multiple art movements, not sure if it's sorted or not but...
        if 'ArtMovement' in info:
            art_movement = info['ArtMovement'][0]
        if 'Genre' in info:
            genre = info['Genre'][0]

        node = {'id': artist, 'birth_year': birth_year, 'art_movement': art_movement, 'genre': genre}
        nodes.append(node)

        ### Add edges
        if 'Influencedby' in info:
            for other_artist in info['Influencedby']:
                if other_artist in artists: # skip schools/movements
                    links.append([other_artist, artist])
        if 'Influencedon' in info:
            for other_artist in info['Influencedon']:
                if other_artist in artists:
                    links.append([artist, other_artist])

    # Create json and dump to file
    json_graph = {
        'nodes': nodes,
        'links': links,
        'meta': meta
    }

    with open(WIKIART_INFLUENCE_GRAPH_PATH, 'w') as f:
        json.dump(json_graph, f)

if __name__ == '__main__':
    create_json()