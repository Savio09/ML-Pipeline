"""
Test script for the AudioFeatureExtractor class.
"""

from feature_extractor import AudioFeatureExtractor
import pandas as pd


def main():
    """Main function to test audio feature extraction."""

    # Initialize feature extractor
    extractor = AudioFeatureExtractor(
        preview_data_path="data/processed/tracks_with_previews.csv"
    )

    # Display available features
    print("Features that will be extracted:")
    print("-" * 50)
    for feature in extractor.get_feature_names():
        print(f"- {feature}")

    # Process all tracks
    print("\nProcessing tracks...")
    features_df = extractor.process_all_tracks(num_workers=4)

    # Display summary
    print("\nFeature Extraction Summary:")
    print("-" * 50)
    print(f"Total tracks processed: {len(features_df)}")
    print("\nFeature statistics:")
    print(features_df.describe())
