"""
Parse Spotify Streaming History
Combines multiple StreamingHistory JSON files into a single CSV dataset
"""

import pandas as pd
import glob
import os
from pathlib import Path


def create_dataset(json_pattern="StreamingHistory_*.json"):
    """
    Read and combine all streaming history JSON files.
    
    Args:
        json_pattern (str): Pattern to match JSON files
        
    Returns:
        pd.DataFrame: Combined streaming history data
    """
    # Find all matching JSON files in current directory
    # Support both StreamingHistory_*.json and Streaming_History_*.json
    all_files = glob.glob("StreamingHistory_*.json") + glob.glob("Streaming_History_*.json")
    
    if not all_files:
        print(f"No files found matching pattern: StreamingHistory_*.json or Streaming_History_*.json")
        return None
    
    # Read and combine all files
    dataframes = []
    for file in all_files:
        try:
            df = pd.read_json(file)
            print(f"Loaded {len(df)} records from {file}")
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dataframes:
        print("No data loaded from any files")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nTotal combined records: {len(combined_df)}")
    
    return combined_df


def save_to_csv(df, output_dir="Ingested_Data"):
    """
    Save DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): Data to save
        output_dir (str): Directory to save the CSV
    """
    if df is None:
        return
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, "combined_streaming_history.csv")
    df.to_csv(output_path, index=False)
    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    # Change to script's directory
    os.chdir(Path(__file__).parent.parent)
    
    # Create combined dataset
    spotify_df = create_dataset()
    
    if spotify_df is not None:
        # Save to CSV
        save_to_csv(spotify_df)
        
        # Display data info
        print("\nDataset Info:")
        print(f"Columns: {', '.join(spotify_df.columns)}")
        print("\nFirst few rows:")
        print(spotify_df.head())
