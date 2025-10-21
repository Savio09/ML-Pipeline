"""
Clean and preprocess Spotify streaming history data
"""

import pandas as pd
import os
from datetime import datetime
from pathlib import Path


def load_data(file_path):
    """Load the CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def parse_timestamps(df):
    """Parse timestamp strings to datetime and extract time features."""
    df["timestamp"] = pd.to_datetime(df["ts"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year

    return df


def convert_durations(df):
    """Convert milliseconds played to minutes and seconds."""
    df["seconds_played"] = df["ms_played"] / 1000
    df["minutes_played"] = df["seconds_played"] / 60

    return df


def filter_data(df, min_seconds=30):
    """Filter out invalid and short plays."""
    initial_count = len(df)

    # Remove rows with missing track/artist info
    df = df.dropna(
        subset=["master_metadata_track_name", "master_metadata_album_artist_name"]
    )

    # Remove short plays
    df = df[df["seconds_played"] >= min_seconds]

    removed = initial_count - len(df)
    print(f"Removed {removed} invalid or short (<{min_seconds}s) plays")

    return df


def clean_names(df):
    """Clean artist and track names."""
    # Standardize names
    df["artist_name"] = df["master_metadata_album_artist_name"].str.strip()
    df["track_name"] = df["master_metadata_track_name"].str.strip()
    df["album_name"] = df["master_metadata_album_album_name"].str.strip()

    return df


def remove_duplicates(df):
    """Remove duplicate entries."""
    initial_count = len(df)

    # Remove duplicates based on timestamp and track info
    df = df.drop_duplicates(
        subset=["timestamp", "artist_name", "track_name"], keep="first"
    )

    removed = initial_count - len(df)
    print(f"Removed {removed} duplicate entries")

    return df


def save_cleaned_data(df, output_path):
    """Save the cleaned DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    print(f"Final dataset shape: {df.shape}")


def main():
    """Main execution function."""
    # File paths
    input_file = "Ingested_Data/combined_streaming_history.csv"
    output_file = "Ingested_Data/cleaned_streaming_history.csv"

    # Load data
    df = load_data(input_file)
    if df is None:
        print("Failed to load data. Please run parse.py first.")
        return

    print("\nCleaning data...")

    # Process data
    df = parse_timestamps(df)
    df = convert_durations(df)
    df = filter_data(df)
    df = clean_names(df)
    df = remove_duplicates(df)

    # Save cleaned data
    save_cleaned_data(df, output_file)

    # Print summary
    print("\nCleaning Complete!")
    print(
        "Added columns:",
        "seconds_played, minutes_played, date, hour, day_of_week, month, year",
    )
    print(f"\nSummary Statistics:")
    print(f"Total tracks: {len(df)}")
    print(f"Unique tracks: {df['track_name'].nunique()}")
    print(f"Unique artists: {df['artist_name'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Average listening time: {df['minutes_played'].mean():.2f} minutes")


if __name__ == "__main__":
    # Change to project root directory
    os.chdir(Path(__file__).parent.parent)
    main()
