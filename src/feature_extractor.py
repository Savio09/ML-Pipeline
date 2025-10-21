"""
This module handles downloading Spotify preview files and extracting audio features.
"""

import os
import pandas as pd
import numpy as np
import requests
import librosa
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class AudioFeatureExtractor:
    """
    A class to handle downloading and processing of Spotify preview files.

    This class provides functionality to:
    1. Download preview files from Spotify
    2. Extract audio features using librosa
    3. Save features to disk

    Attributes:
        preview_data_path (Path): Path to the CSV file containing preview URLs
        output_dir (Path): Directory to save downloaded previews
        features_dir (Path): Directory to save extracted features
    """

    def __init__(
        self,
        preview_data_path: str,
        output_dir: str = "data/audio_previews",
        features_dir: str = "data/features",
    ):
        """
        Initialize the AudioFeatureExtractor.

        Args:
            preview_data_path (str): Path to CSV file with preview URLs
            output_dir (str): Directory to save downloaded previews
            features_dir (str): Directory to save extracted features
        """
        self.preview_data_path = Path(preview_data_path)
        self.output_dir = Path(output_dir)
        self.features_dir = Path(features_dir)

        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

        # Load preview data
        self.preview_data = pd.read_csv(preview_data_path)

    def download_preview(self, track_id: str, preview_url: str) -> Optional[str]:
        """
        Download a preview file from Spotify.

        Args:
            track_id (str): Spotify track ID
            preview_url (str): URL to download the preview from

        Returns:
            Optional[str]: Path to the downloaded file, or None if download failed
        """
        output_path = self.output_dir / f"{track_id}.mp3"

        # Skip if already downloaded
        if output_path.exists():
            return str(output_path)

        try:
            response = requests.get(preview_url, timeout=10)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            return str(output_path)
        except Exception as e:
            print(f"Error downloading {track_id}: {str(e)}")
            return None

    def extract_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract audio features from a preview file using librosa.

        Args:
            audio_path (str): Path to the audio file

        Returns:
            Dict[str, float]: Dictionary containing extracted features
        """
        try:
            # Load the audio file
            y, sr = librosa.load(audio_path)

            # Extract features
            features = {
                # Spectral features
                "spectral_centroid_mean": np.mean(
                    librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                ),
                "spectral_bandwidth_mean": np.mean(
                    librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                ),
                "spectral_rolloff_mean": np.mean(
                    librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                ),
                # Rhythmic features
                "tempo": librosa.beat.tempo(y=y, sr=sr)[0],
                # Tonal features
                "zero_crossing_rate": np.mean(
                    librosa.feature.zero_crossing_rate(y=y)[0]
                ),
                # MFCC features
                **{
                    f"mfcc_{i+1}_mean": np.mean(mfcc)
                    for i, mfcc in enumerate(librosa.feature.mfcc(y=y, sr=sr)[:13])
                },
                # Chroma features
                **{
                    f"chroma_{i}_mean": np.mean(chroma)
                    for i, chroma in enumerate(librosa.feature.chroma_stft(y=y, sr=sr))
                },
            }

            return features
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {str(e)}")
            return {}

    def process_track(self, track: pd.Series) -> Optional[Dict]:
        """
        Process a single track: download preview and extract features.

        Args:
            track (pd.Series): Track information including preview URL

        Returns:
            Optional[Dict]: Dictionary with track info and features, or None if processing failed
        """
        if pd.isna(track["preview_url"]):
            return None

        # Download preview
        audio_path = self.download_preview(
            track["matched_spotify_uri"], track["preview_url"]
        )
        if not audio_path:
            return None

        # Extract features
        features = self.extract_features(audio_path)
        if not features:
            return None

        # Combine track info with features
        return {
            "track_id": track["matched_spotify_uri"],
            "track_name": track["track_name"],
            "artist_name": track["artist_name"],
            "album_name": track["album_name"],
            "spotify_popularity": track["spotify_popularity"],
            "match_confidence": track["match_confidence"],
            **features,
        }

    def process_all_tracks(self, num_workers: int = 4) -> pd.DataFrame:
        """
        Process all tracks in parallel.

        Args:
            num_workers (int): Number of parallel workers

        Returns:
            pd.DataFrame: DataFrame containing all tracks and their features
        """
        results = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.process_track, row)
                for _, row in self.preview_data.iterrows()
            ]

            for future in tqdm(futures, desc="Processing tracks"):
                result = future.result()
                if result:
                    results.append(result)

        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Save features
        output_path = self.features_dir / "audio_features.csv"
        df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")

        return df

    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names that will be extracted.

        Returns:
            List[str]: List of feature names
        """
        return [
            "spectral_centroid_mean",
            "spectral_bandwidth_mean",
            "spectral_rolloff_mean",
            "tempo",
            "zero_crossing_rate",
            *[f"mfcc_{i+1}_mean" for i in range(13)],
            *[f"chroma_{i}_mean" for i in range(12)],
        ]
