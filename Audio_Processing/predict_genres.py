"""
Audio Genre Prediction
Predicts music genres using pre-trained models and audio features
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class GenreClassifier:
    """Base class for genre classifiers."""

    def __init__(self, model_path=None):
        """
        Initialize the classifier.

        Args:
            model_path (str): Path to saved model
        """
        self.model = None
        self.scaler = None
        self.genre_labels = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load trained model from file.

        Args:
            model_path (str): Path to model file
        """
        raise NotImplementedError

    def save_model(self, model_path):
        """
        Save trained model to file.

        Args:
            model_path (str): Output path
        """
        raise NotImplementedError

    def preprocess_features(self, features_df):
        """
        Preprocess features for prediction.

        Args:
            features_df (pd.DataFrame): Raw features

        Returns:
            np.ndarray: Preprocessed features
        """
        raise NotImplementedError

    def predict(self, features):
        """
        Predict genres from features.

        Args:
            features (np.ndarray): Preprocessed features

        Returns:
            tuple: (predicted_labels, prediction_probabilities)
        """
        raise NotImplementedError

    def evaluate(self, features, true_labels):
        """
        Evaluate model performance.

        Args:
            features (np.ndarray): Test features
            true_labels (np.ndarray): True genre labels

        Returns:
            dict: Classification metrics
        """
        pred_labels, _ = self.predict(features)
        report = classification_report(
            true_labels, pred_labels, target_names=self.genre_labels, output_dict=True
        )
        return report


class RandomForestGenreClassifier(GenreClassifier):
    """Random Forest based genre classifier."""

    def load_model(self, model_path):
        """Load Random Forest model and scaler."""
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.genre_labels = model_data["genres"]

    def save_model(self, model_path):
        """Save Random Forest model and scaler."""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "genres": self.genre_labels,
        }
        joblib.dump(model_data, model_path)

    def preprocess_features(self, features_df):
        """Scale features using fitted scaler."""
        feature_cols = [
            col
            for col in features_df.columns
            if any(key in col for key in ["mean", "std", "mfcc", "chroma"])
        ]
        X = features_df[feature_cols].values
        if self.scaler:
            X = self.scaler.transform(X)
        return X

    def predict(self, features):
        """Predict genres using Random Forest."""
        if not self.model:
            raise ValueError("Model not loaded!")

        # Get predictions and probabilities
        pred_probs = self.model.predict_proba(features)
        pred_labels = self.model.predict(features)

        return pred_labels, pred_probs


class DeepGenreClassifier(GenreClassifier):
    """Deep neural network based genre classifier."""

    def __init__(self, model_path=None, input_dim=None):
        """
        Initialize deep classifier.

        Args:
            model_path (str): Path to saved model
            input_dim (int): Input feature dimension
        """
        super().__init__(model_path)
        self.input_dim = input_dim

        if not model_path and input_dim:
            self._build_model(input_dim)

    def _build_model(self, input_dim):
        """Build neural network architecture."""
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.genre_labels)),
        )

    def load_model(self, model_path):
        """Load PyTorch model and preprocessing info."""
        checkpoint = torch.load(model_path)
        self.model = checkpoint["model"]
        self.scaler = checkpoint["scaler"]
        self.genre_labels = checkpoint["genres"]
        self.input_dim = checkpoint["input_dim"]

    def save_model(self, model_path):
        """Save PyTorch model and preprocessing info."""
        checkpoint = {
            "model": self.model,
            "scaler": self.scaler,
            "genres": self.genre_labels,
            "input_dim": self.input_dim,
        }
        torch.save(checkpoint, model_path)

    def preprocess_features(self, features_df):
        """Scale features and convert to PyTorch tensor."""
        X = super().preprocess_features(features_df)
        return torch.FloatTensor(X)

    def predict(self, features):
        """Predict genres using neural network."""
        if not self.model:
            raise ValueError("Model not loaded!")

        self.model.eval()
        with torch.no_grad():
            logits = self.model(features)
            pred_probs = F.softmax(logits, dim=1).numpy()
            pred_labels = np.argmax(pred_probs, axis=1)
            pred_labels = [self.genre_labels[i] for i in pred_labels]

        return pred_labels, pred_probs


def predict_genres(features_df, model_path, output_dir):
    """
    Predict genres for audio features.

    Args:
        features_df (pd.DataFrame): Audio features
        model_path (str): Path to trained model
        output_dir (Path): Output directory

    Returns:
        pd.DataFrame: Features with predicted genres
    """
    # Load model based on file extension
    model_path = Path(model_path)
    if model_path.suffix == ".joblib":
        classifier = RandomForestGenreClassifier(model_path)
    elif model_path.suffix == ".pt":
        classifier = DeepGenreClassifier(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_path.suffix}")

    # Preprocess features
    print("\nPreprocessing features...")
    X = classifier.preprocess_features(features_df)

    # Predict genres
    print("Predicting genres...")
    pred_labels, pred_probs = classifier.predict(X)

    # Add predictions to DataFrame
    df_pred = features_df.copy()
    df_pred["predicted_genre"] = pred_labels

    # Add probability for each genre
    for i, genre in enumerate(classifier.genre_labels):
        df_pred[f"prob_{genre}"] = pred_probs[:, i]

    # Save predictions
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "genre_predictions.csv"
    df_pred.to_csv(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")

    # Print genre distribution
    print("\nPredicted Genre Distribution:")
    print(df_pred["predicted_genre"].value_counts())

    return df_pred


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Predict music genres")
    parser.add_argument(
        "--features",
        type=str,
        default="Extracted_Features/audio_features.csv",
        help="Path to audio features CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Models/genre_classifier.joblib",
        help="Path to trained model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Predictions",
        help="Output directory for predictions",
    )

    args = parser.parse_args()

    # Check if features exist
    if not os.path.exists(args.features):
        print(f"Error: Features file not found: {args.features}")
        print("Please run audio feature extraction first.")
        return

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train or download a model first.")
        return

    # Load features
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)

    # Make predictions
    df_pred = predict_genres(features_df, args.model, args.output)

    print("\nPrediction complete!")


if __name__ == "__main__":
    # Change to project root directory
    os.chdir(Path(__file__).parent.parent)
    main()
