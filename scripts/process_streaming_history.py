"""
Process Spotify streaming history data
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processor import DataProcessor


def main():
    # File paths
    data_path = (
        Path(__file__).parent.parent / "Streaming_History_Audio_2023-2025_0.json"
    )
    output_path = (
        Path(__file__).parent.parent / "data/processed/cleaned_streaming_history.csv"
    )

    # Create processor and process data
    processor = DataProcessor(data_path)
    processed_data = processor.process_data()

    if processed_data is not None:
        # Save processed data
        processor.save_processed_data(output_path)

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total unique tracks: {processed_data['trackName'].nunique()}")
        print(f"Total unique artists: {processed_data['artistName'].nunique()}")
        print(
            f"Date range: {processed_data['date'].min()} to {processed_data['date'].max()}"
        )
        print(
            f"Average listening time: {processed_data['minutesPlayed'].mean():.2f} minutes"
        )


if __name__ == "__main__":
    main()
