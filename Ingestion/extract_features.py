"""
Audio Feature Extraction
Extracts features from downloaded audio samples using librosa
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm


class AudioFeatureExtractor:
    """Extract audio features from MP3 samples using librosa."""

    def __init__(self, sample_rate=22050, duration=None):
        """
        Initialize feature extractor.

        Args:
            sample_rate (int): Audio sample rate in Hz
            duration (float): Duration to load in seconds, None for full file
        """
        self.sample_rate = sample_rate
        self.duration = duration

    def load_audio(self, file_path):
        """
        Load audio file using librosa.

        Args:
            file_path (str): Path to audio file

        Returns:
            tuple: (audio_time_series, sample_rate) or (None, None) if error
        """
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            return y, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    def extract_features(self, y, sr):
        """
        Extract audio features using librosa.

        Args:
            y (np.ndarray): Audio time series
            sr (int): Sample rate

        Returns:
            dict: Dictionary of extracted features
        """
        features = {}

        try:
            # Spectrogram
            S = np.abs(librosa.stft(y))

            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Chromagram
            chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            # Rhythm features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

            # Spectral features
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

            # Zero crossing rate
            zero_crossing = librosa.feature.zero_crossing_rate(y)

            # RMS energy
            rms = librosa.feature.rms(y=y)

            # Store features
            features.update(
                {
                    # Basic features
                    "duration": float(librosa.get_duration(y=y, sr=sr)),
                    "tempo": float(tempo),
                    # Statistical summaries of features
                    "mfcc_mean": mfcc.mean(axis=1).tolist(),
                    "mfcc_std": mfcc.std(axis=1).tolist(),
                    "mfcc_delta_mean": mfcc_delta.mean(axis=1).tolist(),
                    "mfcc_delta_std": mfcc_delta.std(axis=1).tolist(),
                    "mfcc_delta2_mean": mfcc_delta2.mean(axis=1).tolist(),
                    "mfcc_delta2_std": mfcc_delta2.std(axis=1).tolist(),
                    "chroma_mean": chromagram.mean(axis=1).tolist(),
                    "chroma_std": chromagram.std(axis=1).tolist(),
                    "mel_spec_mean": mel_spec_db.mean(axis=1).tolist(),
                    "mel_spec_std": mel_spec_db.std(axis=1).tolist(),
                    "spec_centroid_mean": float(spec_centroid.mean()),
                    "spec_centroid_std": float(spec_centroid.std()),
                    "spec_bandwidth_mean": float(spec_bandwidth.mean()),
                    "spec_bandwidth_std": float(spec_bandwidth.std()),
                    "spec_contrast_mean": spec_contrast.mean(axis=1).tolist(),
                    "spec_contrast_std": spec_contrast.std(axis=1).tolist(),
                    "spec_rolloff_mean": float(spec_rolloff.mean()),
                    "spec_rolloff_std": float(spec_rolloff.std()),
                    "zero_crossing_mean": float(zero_crossing.mean()),
                    "zero_crossing_std": float(zero_crossing.std()),
                    "rms_mean": float(rms.mean()),
                    "rms_std": float(rms.std()),
                    # Number of beats detected
                    "beat_count": len(beats),
                    "beat_tempo": float(tempo),
                }
            )

            return features

        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def process_file(self, file_path):
        """
        Process a single audio file.

        Args:
            file_path (str): Path to audio file

        Returns:
            dict: Extracted features or None if error
        """
        # Load audio
        y, sr = self.load_audio(file_path)
        if y is None:
            return None

        # Extract features
        features = self.extract_features(y, sr)
        if features is None:
            return None

        # Add metadata
        features["file_path"] = str(file_path)
        features["sample_rate"] = sr

        return features

    def process_directory(self, input_dir, output_dir, download_log=None):
        """
        Process all audio files in directory.

        Args:
            input_dir (str): Directory containing audio files
            output_dir (str): Directory to save feature files
            download_log (str): Optional path to download log CSV

        Returns:
            pd.DataFrame: DataFrame of extracted features
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Get list of audio files
        audio_files = list(input_dir.glob("*.mp3"))
        print(f"Found {len(audio_files)} audio files")

        if not audio_files:
            print("No audio files found!")
            return None

        # Load download log if provided
        track_info = {}
        if download_log and os.path.exists(download_log):
            log_df = pd.read_csv(download_log)
            # Create mapping of file paths to track info
            for _, row in log_df.iterrows():
                if row["success"] and row["file_path"]:
                    track_info[row["file_path"]] = {
                        "track_name": row["track_name"],
                        "artist_name": row["artist_name"],
                        "track_id": row["track_id"],
                        "spotify_url": row["spotify_url"],
                    }

        # Process each file
        all_features = []
        failed_files = []

        print("\nExtracting features...")
        for file_path in tqdm(audio_files):
            features = self.process_file(file_path)

            # Always add metadata fields, even if missing from log
            meta = track_info.get(str(file_path), {})
            features = features or {}
            features["file_path"] = str(file_path)
            features["track_name"] = meta.get("track_name", "")
            features["artist_name"] = meta.get("artist_name", "")
            features["track_id"] = meta.get("track_id", "")
            features["spotify_url"] = meta.get("spotify_url", "")

            # Only append if features were extracted
            meta_keys = [
                "file_path",
                "track_name",
                "artist_name",
                "track_id",
                "spotify_url",
            ]
            if features and any(k for k in features if k not in meta_keys):
                all_features.append(features)
            else:
                failed_files.append(str(file_path))

        if not all_features:
            print("No features extracted successfully!")
            return None

        # Convert to DataFrame
        df = pd.json_normalize(all_features)

        # Ensure metadata columns are first and always present
        meta_cols = [
            "file_path",
            "track_name",
            "artist_name",
            "track_id",
            "spotify_url",
        ]
        for col in meta_cols:
            if col not in df.columns:
                df[col] = ""
        ordered_cols = meta_cols + [c for c in df.columns if c not in meta_cols]
        df = df[ordered_cols]

        # Save features
        features_path = output_dir / "audio_features.csv"
        df.to_csv(features_path, index=False)

        # Save failures log if any
        if failed_files:
            failures_path = output_dir / "failed_extractions.txt"
            with open(failures_path, "w") as f:
                for file_path in failed_files:
                    f.write(f"{file_path}\n")
            print(f"\n{len(failed_files)} files failed. See {failures_path}")

        print(f"\nFeatures saved to: {features_path}")
        print(f"Features extracted: {len(df)}")
        print(f"Failed files: {len(failed_files)}")

        return df


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract audio features")
    parser.add_argument(
        "--input",
        type=str,
        default="Audio_Samples",
        help="Input directory with audio files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Extracted_Features",
        help="Output directory for features",
    )
    parser.add_argument(
        "--log", type=str, default=None, help="Path to download log CSV"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Duration to extract (seconds)"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=22050, help="Audio sample rate (Hz)"
    )

    args = parser.parse_args()

    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        print("Please run 'python Ingestion/extract_audio_samples.py' first.")
        return

    # Create extractor
    extractor = AudioFeatureExtractor(
        sample_rate=args.sample_rate, duration=args.duration
    )

    # Process files
    features_df = extractor.process_directory(
        args.input, args.output, download_log=args.log
    )

    if features_df is not None:
        print("\nFeature extraction complete!")
        print(f"Shape of features DataFrame: {features_df.shape}")

        # Print feature names
        print("\nExtracted features:")
        for col in sorted(features_df.columns):
            print(f"- {col}")


if __name__ == "__main__":
    # Change to project root directory
    os.chdir(Path(__file__).parent.parent)
    main()
