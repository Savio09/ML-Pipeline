"""
Genre Classifier Training
Trains machine learning models for music genre classification
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import our genre prediction module
from predict_genres import RandomForestGenreClassifier, DeepGenreClassifier


class AudioFeatureDataset(Dataset):
    """PyTorch dataset for audio features."""

    def __init__(self, features, labels):
        """
        Initialize dataset.

        Args:
            features (np.ndarray): Audio features
            labels (np.ndarray): Genre labels
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        """Get dataset size."""
        return len(self.features)

    def __getitem__(self, idx):
        """Get a single sample."""
        return self.features[idx], self.labels[idx]


def prepare_training_data(features_df, genre_labels):
    """
    Prepare features and labels for training.

    Args:
        features_df (pd.DataFrame): Audio features
        genre_labels (list): List of genre labels

    Returns:
        tuple: (features, labels, label_encoder, feature_cols)
    """
    print("\nPreparing training data...")

    # Get numerical feature columns
    feature_cols = [
        col
        for col in features_df.columns
        if any(key in col for key in ["mean", "std", "mfcc", "chroma"])
    ]

    X = features_df[feature_cols].values
    y = features_df["genre"].values

    # Create label mapping
    label_to_idx = {label: i for i, label in enumerate(genre_labels)}
    y = np.array([label_to_idx[label] for label in y])

    print(f"Features shape: {X.shape}")
    print(f"Number of genres: {len(genre_labels)}")

    return X, y, feature_cols


def train_random_forest(X_train, y_train, X_val, y_val, genre_labels):
    """
    Train Random Forest classifier.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation labels
        genre_labels (list): List of genre labels

    Returns:
        tuple: (trained_model, scaler)
    """
    print("\nTraining Random Forest classifier...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train model
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=10, n_jobs=-1, random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_val_scaled)
    report = classification_report(y_val, y_pred, target_names=genre_labels)
    print("\nRandom Forest Performance:")
    print(report)

    return model, scaler


def train_neural_network(
    X_train,
    y_train,
    X_val,
    y_val,
    genre_labels,
    batch_size=32,
    epochs=50,
    learning_rate=0.001,
):
    """
    Train neural network classifier.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation labels
        genre_labels (list): List of genre labels
        batch_size (int): Batch size
        epochs (int): Number of epochs
        learning_rate (float): Learning rate

    Returns:
        tuple: (trained_model, scaler)
    """
    print("\nTraining neural network...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create datasets
    train_dataset = AudioFeatureDataset(X_train_scaled, y_train)
    val_dataset = AudioFeatureDataset(X_val_scaled, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = X_train.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, len(genre_labels)),
    )

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.numpy())
                val_true.extend(labels.numpy())

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Final evaluation
    report = classification_report(val_true, val_preds, target_names=genre_labels)
    print("\nNeural Network Performance:")
    print(report)

    return model, scaler


def save_models(rf_classifier, nn_classifier, output_dir):
    """
    Save trained models.

    Args:
        rf_classifier (RandomForestGenreClassifier): Random Forest model
        nn_classifier (DeepGenreClassifier): Neural network model
        output_dir (Path): Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save Random Forest
    rf_path = output_dir / "random_forest_classifier.joblib"
    rf_classifier.save_model(rf_path)
    print(f"Saved Random Forest model to: {rf_path}")

    # Save neural network
    nn_path = output_dir / "neural_network_classifier.pt"
    nn_classifier.save_model(nn_path)
    print(f"Saved neural network model to: {nn_path}")


def train_genre_classifiers(features_df, genre_labels, output_dir):
    """
    Train both Random Forest and neural network classifiers.

    Args:
        features_df (pd.DataFrame): Audio features
        genre_labels (list): List of genre labels
        output_dir (Path): Output directory

    Returns:
        tuple: (random_forest_classifier, neural_network_classifier)
    """
    # Prepare data
    X, y, feature_cols = prepare_training_data(features_df, genre_labels)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest
    rf_model, rf_scaler = train_random_forest(
        X_train, y_train, X_val, y_val, genre_labels
    )

    rf_classifier = RandomForestGenreClassifier()
    rf_classifier.model = rf_model
    rf_classifier.scaler = rf_scaler
    rf_classifier.genre_labels = genre_labels

    # Train neural network
    nn_model, nn_scaler = train_neural_network(
        X_train, y_train, X_val, y_val, genre_labels
    )

    nn_classifier = DeepGenreClassifier()
    nn_classifier.model = nn_model
    nn_classifier.scaler = nn_scaler
    nn_classifier.genre_labels = genre_labels
    nn_classifier.input_dim = X.shape[1]

    # Save models
    save_models(rf_classifier, nn_classifier, output_dir)

    return rf_classifier, nn_classifier


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Train genre classifiers")
    parser.add_argument(
        "--features",
        type=str,
        default="Extracted_Features/labeled_features.csv",
        help="Path to labeled features CSV",
    )
    parser.add_argument(
        "--output", type=str, default="Models", help="Output directory for models"
    )

    args = parser.parse_args()

    # Check if features exist
    if not os.path.exists(args.features):
        print(f"Error: Features file not found: {args.features}")
        print("Please prepare labeled training data first.")
        return

    # Load features
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)

    if "genre" not in features_df.columns:
        print("Error: Features file must contain 'genre' column")
        return

    # Get unique genres
    genre_labels = sorted(features_df["genre"].unique())

    # Train models
    rf_classifier, nn_classifier = train_genre_classifiers(
        features_df, genre_labels, args.output
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    # Change to project root directory
    os.chdir(Path(__file__).parent.parent)
    main()
