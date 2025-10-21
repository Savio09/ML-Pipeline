"""
Test script for the SpotifyDataLoader class.
"""

from data_loader import SpotifyDataLoader
import pandas as pd

pd.set_option("display.max_columns", None)


def main():
    # Initialize the data loader with absolute path
    data_loader = SpotifyDataLoader(
        "/Users/fdeclan/Public/ml_pipeline/Streaming_History_Audio_2023-2025_0.json"
    )

    # Load the data
    print("Loading data...")
    df = data_loader.load_data()

    # Display basic information
    print("\nDataset Info:")
    print("-" * 50)
    print(df.info())

    print("\nSample of the data:")
    print("-" * 50)
    print(df.head())

    # Get statistics
    print("\nListening History Statistics:")
    print("-" * 50)
    stats = data_loader.get_listening_history_stats()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    # Check data quality
    print("\nMissing Values:")
    print("-" * 50)
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    print("\nUnique Values Count:")
    print("-" * 50)
    for column in df.columns:
        unique_count = df[column].nunique()
        print(f"{column}: {unique_count} unique values")

    # Save processed data
    print("\nSaving processed data...")
    print("-" * 50)
    saved_files = data_loader.save_processed_data()
    print("Files saved:")
    for desc, path in saved_files.items():
        print(f"{desc}: {path}")


if __name__ == "__main__":
    main()
