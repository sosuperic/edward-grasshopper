# utils.py

"""
Available functions:
- get_valid_artist_names: Return names of all artists that are found in main part of influence graph, as filtered
  using Peter's tool
"""
import json

from config import WIKIART_INFLUENCE_GRAPH_FILTERED_PATH

def get_valid_artist_names():
    """Return names of all artists that are found in main part of influence graph, as filtered using Peter's tool"""
    # Load filtered data
    with open(WIKIART_INFLUENCE_GRAPH_FILTERED_PATH, 'r') as f:
        data = json.load(f, encoding='latin1')  # TODO: not sure why utf8 throws error, doesn't happen for non-filtered

    # Get names
    valid = set()
    for node in data['nodes']:
        valid.add(node['id'])

    return valid
