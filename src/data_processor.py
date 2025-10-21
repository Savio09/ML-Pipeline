"""
Data Processor Module
Handles loading, cleaning, and preprocessing of Spotify streaming history data
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


class DataProcessor:
    """Class for processing Spotify streaming history data."""

    def __init__(self, data_path):
        """
        Initialize the data processor.

        Args:
            data_path (str): Path to the Spotify streaming history JSON file
        """
        self.data_path = Path(data_path)
        self.data = None

    def load_data(self):
        """Load streaming history from JSON file."""
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            self.data = pd.DataFrame(raw_data)
            print(f"Loaded {len(self.data)} records from {self.data_path}")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def parse_timestamps(self):
        """Convert timestamp strings to datetime objects and extract time features."""
        if self.data is not None:
            self.data["timestamp"] = pd.to_datetime(self.data["ts"])
            self.data["date"] = self.data["timestamp"].dt.date
            self.data["hour"] = self.data["timestamp"].dt.hour
            self.data["day_of_week"] = self.data["timestamp"].dt.day_name()
            self.data["month"] = self.data["timestamp"].dt.month
            self.data["year"] = self.data["timestamp"].dt.year

    def convert_durations(self):
        """Convert milliseconds played to seconds and minutes."""
        if self.data is not None:
            self.data["secondsPlayed"] = self.data["ms_played"] / 1000
            self.data["minutesPlayed"] = self.data["secondsPlayed"] / 60

    def clean_names(self):
        """Clean artist and track names."""
        if self.data is not None:
            # Rename columns for consistency
            self.data["artistName"] = self.data[
                "master_metadata_album_artist_name"
            ].str.strip()
            self.data["trackName"] = self.data["master_metadata_track_name"].str.strip()
            self.data["albumName"] = self.data[
                "master_metadata_album_album_name"
            ].str.strip()

    def filter_short_plays(self, min_seconds=30):
        """Remove tracks played for less than min_seconds."""
        if self.data is not None:
            initial_count = len(self.data)
            self.data = self.data[self.data["secondsPlayed"] >= min_seconds]
            removed = initial_count - len(self.data)
            print(f"Removed {removed} short plays (<{min_seconds}s)")

    def remove_duplicates(self):
        """Remove duplicate entries based on timestamp and track info."""
        if self.data is not None:
            initial_count = len(self.data)
            self.data = self.data.drop_duplicates(
                subset=["timestamp", "artistName", "trackName"], keep="first"
            )
            removed = initial_count - len(self.data)
            print(f"Removed {removed} duplicate entries")

    def process_data(self):
        """Run all processing steps on the data."""
        if self.load_data():
            print("\nProcessing data...")
            self.parse_timestamps()
            self.convert_durations()
            self.clean_names()
            self.filter_short_plays()
            self.remove_duplicates()
            return self.data
        return None

    def save_processed_data(self, output_path):
        """Save processed data to CSV file."""
        if self.data is not None:
            self.data.to_csv(output_path, index=False)
            print(f"\nSaved processed data to {output_path}")
            print(f"Final dataset shape: {self.data.shape}")
            return True
        return False


if __name__ == "__main__":
    # Example usage
    data_path = "../data/raw/Streaming_History_Audio_2023-2025_0.json"
    output_path = "../data/processed/cleaned_streaming_history.csv"

    processor = DataProcessor(data_path)
    processed_data = processor.process_data()

    if processed_data is not None:
        processor.save_processed_data(output_path)
