# app.py
# Run by PYTHONPATH=. python website/app.py

from flask import Flask, request, render_template, jsonify
import os

from config import WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH

app = Flask(__name__)

artists = os.listdir(WIKIART_ARTIST_INFLUENCERS_EMBEDDINGS_PATH)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search = request.args.get('q')
    results = [artist for artist in artists if artist.startswith(str(search))]
    return jsonify(matching_results=results)

if __name__ == '__main__':
    app.run(debug=True)