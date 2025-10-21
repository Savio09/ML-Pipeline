"""
Genre Labeling Script
Fetches genres for each artist using the Spotify API and merges them into the audio features dataset.
"""

import pandas as pd
import os
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm

# Load environment variables from .env
from dotenv import load_dotenv

load_dotenv()

# Authenticate with Spotify API
sp = Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    )
)


def get_artist_genres(artist_name):
    """
    Fetch genres for a given artist using Spotify API.
    """
    try:
        result = sp.search(q="artist:" + artist_name, type="artist", limit=1)
        items = result["artists"]["items"]
        if items:
            return items[0]["genres"]
        return []
    except Exception as e:
        print(f"Error fetching genres for {artist_name}: {e}")
        return []


def main():
    # Load extracted features
    features_path = "Extracted_Features/audio_features.csv"
    df = pd.read_csv(features_path)

    # Get unique artists
    if "artist_name" in df.columns:
        artists = df["artist_name"].dropna().unique()
    else:
        print("Error: 'artist_name' column not found in features file.")
        return

    # Build artist to genre mapping
    artist_genre_map = {}
    print(f"Fetching genres for {len(artists)} artists...")
    for artist in tqdm(artists):
        genres = get_artist_genres(artist)
        artist_genre_map[artist] = genres[0] if genres else "Unknown"

    # Map genres to tracks
    df["genre"] = df["artist_name"].map(artist_genre_map)

    # Save merged dataset
    output_path = "Extracted_Features/audio_features_with_genres.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved features with genre labels to: {output_path}")


if __name__ == "__main__":
    main()
