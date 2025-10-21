import pandas as pd
import os

# Paths
stream_path = 'Ingested_Data/cleaned_streaming_history.csv'
features_path = 'Extracted_Features/audio_features_with_genres.csv'
out_path = 'EDA/unified_streaming_features.csv'
os.makedirs('EDA', exist_ok=True)

# Read data
stream_df = pd.read_csv(stream_path)
features_df = pd.read_csv(features_path)

# Merge on track name and artist name
merged_df = pd.merge(
    stream_df,
    features_df,
    left_on=['track_name', 'artist_name'],
    right_on=['track_name', 'artist_name'],
    how='left',
    suffixes=('_stream', '_features')
)

# Save unified CSV
merged_df.to_csv(out_path, index=False)
print(f"Unified CSV saved to {out_path} with {len(merged_df)} rows.")
