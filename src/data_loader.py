"""
This module contains the DataLoader class for processing Spotify streaming history data.
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os


class SpotifyDataLoader:
    """
    A class to load and process Spotify streaming history data from JSON files.

    This class provides functionality to load Spotify streaming history data,
    perform initial data cleaning, and prepare it for further analysis.

    Attributes:
        file_path (Path): Path to the JSON file containing Spotify streaming history
        data (pd.DataFrame): Processed DataFrame containing the streaming history
    """

    def __init__(self, file_path: str):
        """
        Initialize the SpotifyDataLoader with a path to the JSON file.

        Args:
            file_path (str): Path to the JSON file containing Spotify streaming history

        Raises:
            FileNotFoundError: If the specified file does not exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        self.file_path = Path(file_path)
        self.data = None
        self._validate_file()

    def _validate_file(self) -> None:
        """
        Validate that the JSON file exists and is properly formatted.

        Raises:
            FileNotFoundError: If the specified file does not exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Try to read the file to validate JSON format
        with open(self.file_path, "r", encoding="utf-8") as f:
            try:
                json.load(f)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"Invalid JSON file: {e.msg}", e.doc, e.pos)

    def load_data(self) -> pd.DataFrame:
        """
        Load the JSON file into a pandas DataFrame and perform initial cleaning.

        Returns:
            pd.DataFrame: Processed DataFrame containing the streaming history

        Raises:
            json.JSONDecodeError: If the file cannot be parsed as JSON
        """
        # Read the JSON file
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to DataFrame
        self.data = pd.DataFrame(data)

        # Basic cleaning steps
        self._clean_data()

        return self.data

    def _clean_data(self) -> None:
        """
        Perform initial data cleaning operations on the loaded DataFrame.

        This includes:
        - Converting timestamps to datetime
        - Removing duplicates
        - Handling missing values
        - Standardizing column names
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Convert timestamps to datetime
        if "ts" in self.data.columns:
            self.data["timestamp"] = pd.to_datetime(self.data["ts"])
            self.data.drop("ts", axis=1, inplace=True)

        # Convert duration from milliseconds to seconds
        if "ms_played" in self.data.columns:
            self.data["duration_seconds"] = self.data["ms_played"] / 1000
            self.data.drop("ms_played", axis=1, inplace=True)

        # Remove rows with null track names
        self.data = self.data.dropna(subset=["master_metadata_track_name"])

        # Remove duplicates based on track name and timestamp
        self.data = self.data.drop_duplicates(
            subset=["master_metadata_track_name", "timestamp"]
        )

        # Rename columns to be more user-friendly
        column_mapping = {
            "master_metadata_track_name": "track_name",
            "master_metadata_album_artist_name": "artist_name",
            "master_metadata_album_album_name": "album_name",
        }
        self.data.rename(columns=column_mapping, inplace=True)

    def get_unique_tracks(self) -> pd.DataFrame:
        """
        Get a DataFrame of unique tracks from the streaming history.

        Returns:
            pd.DataFrame: DataFrame containing unique tracks and their metadata

        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Group by track identifiers and get unique entries
        track_columns = [col for col in self.data.columns if "track" in col.lower()]
        if track_columns:
            return self.data[track_columns].drop_duplicates()
        return pd.DataFrame()

    def get_listening_history_stats(self) -> Dict:
        """
        Calculate basic statistics about the listening history.

        Returns:
            Dict: Dictionary containing various statistics about listening history

        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        stats = {
            "total_streams": len(self.data),
            "unique_tracks": len(self.get_unique_tracks()),
            "total_listening_time": self.data["duration_seconds"].sum()
            / 3600,  # in hours
            "date_range": (
                [self.data["timestamp"].min(), self.data["timestamp"].max()]
                if len(self.data) > 0
                else None
            ),
            "unique_artists": self.data["artist_name"].nunique(),
        }

        return stats

    def save_processed_data(self, base_path: str = "data") -> Dict[str, str]:
        """
        Save processed data into CSV files for further analysis.

        Args:
            base_path (str): Base directory to save the data files

        Returns:
            Dict[str, str]: Dictionary with descriptions and paths of saved files

        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Create directories if they don't exist
        processed_dir = Path(base_path) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Save main streaming history
        streaming_path = processed_dir / "streaming_history.csv"
        self.data.to_csv(streaming_path, index=False)

        # Save unique tracks dataset
        unique_tracks = self.get_unique_tracks()
        tracks_path = processed_dir / "unique_tracks.csv"
        unique_tracks.to_csv(tracks_path, index=False)

        # Save listening statistics
        stats = self.get_listening_history_stats()
        stats_df = pd.DataFrame([stats])
        stats_path = processed_dir / "listening_stats.csv"
        stats_df.to_csv(stats_path, index=False)

        # Create a summary of saved files
        saved_files = {
            "streaming_history": str(streaming_path),
            "unique_tracks": str(tracks_path),
            "listening_stats": str(stats_path),
        }

        return saved_files
