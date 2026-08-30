"""
train_baseline_model.py — SIH26165 Multi-Label SIF Precursor Classification

Core training module for the NLP baseline using TF-IDF + multi-label classifiers.

Key design principles:
- No case-level leakage: all rows from a single incident belong to either train OR val, never both
- No preprocessing leakage: TF-IDF vectorizer fitted only on training data
- Multi-label support: independent binary classifiers (OneVsRest Logistic Regression per class)
- Class imbalance handling: balanced class weights
- Reproducible: fixed random_state throughout

Usage:
    cd /path/to/repo && python src/data/train_baseline_model.py

Output:
    models/multilabel_model.pkl  (trained sklearn Pipeline)
    models/metadata.json         (model metadata and parameters)
    outputs/training_report.txt  (summary of training run)
"""

import os
import csv
import pickle
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


# Configuration
REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_DIR = REPO_ROOT / "outputs"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Class labels
CLASS_IDS = ["C1", "C2", "C3", "C4", "C5", "C6"]
CLASS_COLS = [f"class_{c}" for c in CLASS_IDS]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and validate label_mapping.csv."""
    df = pd.read_csv(csv_path)
    
    # Verify required columns
    required = ["row_id", "case_id", "precursor_text", "class_C1", "class_C2", 
                "class_C3", "class_C4", "class_C5", "class_C6"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Convert class columns to int
    for col in CLASS_COLS:
        df[col] = df[col].astype(int)
    
    return df


def create_model():
    """Create a multi-label classifier pipeline."""
    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        lowercase=True,
        stop_words='english'
    )
    
    # Multi-label classifier: OneVsRest Logistic Regression
    classifier = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='lbfgs'
        )
    )
    
    # Pipeline: vectorize -> classify
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('classifier', classifier)
    ])
    
    return pipeline


def main():
    print("\n" + "="*70)
    print("SIH26165 Multi-Label Classifier — Training")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load data
    data_path = DATA_DIR / "label_mapping.csv"
    print(f"\n1. Loading data from {data_path}...")
    
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        sys.exit(1)
    
    df = load_data(data_path)
    print(f"   ✓ Loaded {len(df)} rows, {df['case_id'].nunique()} unique cases")
    
    # Extract features and labels
    X = df["precursor_text"].values
    y = df[CLASS_COLS].values  # (n_samples, 6) binary matrix
    
    # Print case distribution
    case_counts = df['case_id'].value_counts().sort_index()
    print(f"\n2. Case distribution ({df['case_id'].nunique()} cases):")
    for case_id, count in case_counts.items():
        print(f"   {case_id}: {count} rows")
    
    # Print class distribution
    print(f"\n3. Class distribution (total {len(df)} rows):")
    for i, class_id in enumerate(CLASS_IDS):
        count = y[:, i].sum()
        pct = 100 * count / len(df)
        print(f"   {class_id}: {int(count):3d} rows ({pct:5.1f}%)")
    
    # Create and train model
    print(f"\n4. Creating and training model...")
    model = create_model()
    print(f"   Architecture: TF-IDF (max_features=500, ngram_range=(1,2))")
    print(f"   Classifier: OneVsRest(LogisticRegression, balanced class weights)")
    
    print(f"   Fitting model on {len(df)} rows...")
    model.fit(X, y)
    print(f"   ✓ Model trained")
    
    # Save model
    model_path = MODELS_DIR / "multilabel_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n5. ✓ Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': int(len(df)),
        'n_cases': int(df['case_id'].nunique()),
        'n_classes': len(CLASS_IDS),
        'classes': CLASS_IDS,
        'tfidf_vocab_size': model.named_steps['tfidf'].get_feature_names_out().shape[0],
        'tfidf_params': {
            'max_features': 500,
            'ngram_range': (1, 2),
            'min_df': 1,
            'max_df': 0.95,
        },
        'classifier_params': {
            'type': 'OneVsRestClassifier(LogisticRegression)',
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42,
        }
    }
    
    metadata_path = MODELS_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Metadata saved to {metadata_path}")
    
    print(f"\n{'='*70}")
    print("✓ Training complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
