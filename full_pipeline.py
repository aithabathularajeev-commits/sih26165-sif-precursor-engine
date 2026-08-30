#!/usr/bin/env python3
"""
full_pipeline.py — Complete SIH26165 NLP baseline pipeline in one script

Performs:
1. Data loading and validation
2. Model training
3. Case-grouped evaluation (Leave-One-Case-Out)
4. Example predictions
5. Report generation

This script is self-contained and can be run as: python full_pipeline.py
"""

import sys
import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, hamming_loss
)


# Setup paths
REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_DIR = REPO_ROOT / "outputs"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
CLASS_IDS = ["C1", "C2", "C3", "C4", "C5", "C6"]
CLASS_COLS = [f"class_{c}" for c in CLASS_IDS]


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80)


def print_section(text):
    """Print formatted section."""
    print(f"\n{text}")
    print("-" * 80)


# ============================================================================
# STAGE 1: DATA LOADING & VALIDATION
# ============================================================================

def load_and_validate_data():
    """Load and validate the dataset."""
    print_header("STAGE 1: DATA LOADING & VALIDATION")
    
    data_path = DATA_DIR / "label_mapping.csv"
    print(f"\nLoading from: {data_path}")
    
    if not data_path.exists():
        print(f"ERROR: File not found: {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    
    # Validate
    required_cols = ["row_id", "case_id", "precursor_text"] + CLASS_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)
    
    # Convert class columns to int
    for col in CLASS_COLS:
        df[col] = df[col].astype(int)
    
    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Found {df['case_id'].nunique()} unique cases")
    
    # Print case distribution
    print_section("Case Distribution")
    case_counts = df['case_id'].value_counts().sort_index()
    for case_id, count in case_counts.items():
        print(f"  {case_id}: {count} rows")
    
    # Print class distribution
    print_section("Class Distribution")
    for i, class_id in enumerate(CLASS_IDS):
        count = df[CLASS_COLS[i]].sum()
        pct = 100 * count / len(df)
        print(f"  {class_id}: {int(count):3d} rows ({pct:5.1f}%)")
    
    # Check for multi-label rows
    y = df[CLASS_COLS].values
    multi_label_count = (y.sum(axis=1) > 1).sum()
    print(f"\n✓ Multi-label rows (>1 class): {multi_label_count}")
    
    return df


# ============================================================================
# STAGE 2: MODEL CREATION & TRAINING
# ============================================================================

def create_model():
    """Create multi-label classifier pipeline."""
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        lowercase=True,
        stop_words='english'
    )
    
    classifier = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='lbfgs'
        )
    )
    
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('classifier', classifier)
    ])
    
    return pipeline


def train_model(df):
    """Train model on full dataset."""
    print_header("STAGE 2: MODEL TRAINING")
    
    X = df["precursor_text"].values
    y = df[CLASS_COLS].values
    
    print(f"\nTraining multi-label classifier...")
    print(f"  Architecture: TF-IDF (max_features=500, ngram=(1,2))")
    print(f"  Classifier: OneVsRest(LogisticRegression, balanced)")
    print(f"  Samples: {len(X)}")
    
    model = create_model()
    model.fit(X, y)
    
    print(f"✓ Model trained successfully")
    
    # Save model
    model_path = MODELS_DIR / "multilabel_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {model_path}")
    
    # Save metadata
    vocab_size = model.named_steps['tfidf'].get_feature_names_out().shape[0]
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': int(len(df)),
        'n_cases': int(df['case_id'].nunique()),
        'n_classes': 6,
        'classes': CLASS_IDS,
        'vocab_size': int(vocab_size),
        'tfidf_params': {
            'max_features': 500,
            'ngram_range': (1, 2),
            'min_df': 1,
            'max_df': 0.95,
        },
        'classifier_type': 'OneVsRest(LogisticRegression)',
        'class_weight': 'balanced',
    }
    
    metadata_path = MODELS_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")
    
    return model


# ============================================================================
# STAGE 3: CASE-GROUPED EVALUATION (Leave-One-Case-Out)
# ============================================================================

def evaluate_case_grouped(df):
    """Perform Leave-One-Case-Out cross-validation."""
    print_header("STAGE 3: CASE-GROUPED EVALUATION (Leave-One-Case-Out CV)")
    
    print_section("Evaluation Methodology")
    print("  Leave-One-Case-Out (LOCO) Cross-Validation")
    print("  - 12 folds: each fold holds out ALL rows from ONE case")
    print("  - Prevents case-level leakage")
    print("  - TF-IDF fitted ONLY on training data (no leakage)")
    print("  - Results aggregated across all 12 folds")
    
    X = df["precursor_text"].values
    y = df[CLASS_COLS].values
    case_ids = df["case_id"].values
    
    unique_cases = sorted(df['case_id'].unique())
    all_fold_results = []
    
    print_section("Performing LOCO Evaluation")
    
    for fold_idx, held_out_case in enumerate(unique_cases, 1):
        # Split by case
        train_mask = case_ids != held_out_case
        test_mask = case_ids == held_out_case
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        print(f"\nFold {fold_idx:2d}: Held-out {held_out_case:7s} (n={len(X_test):2d})")
        
        # Train model (TF-IDF fitted only on training data)
        model = create_model()
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.decision_function(X_test)
        y_pred_proba_clamped = np.clip(y_pred_proba, 0, 1)
        
        # Evaluate
        fold_result = {
            'case': held_out_case,
            'n_test': len(X_test),
            'hamming_loss': float(hamming_loss(y_test, y_pred)),
            'subset_accuracy': float(accuracy_score(y_test, y_pred)),
            'per_class': {},
        }
        
        for i, class_id in enumerate(CLASS_IDS):
            y_true_class = y_test[:, i]
            y_pred_class = y_pred[:, i]
            
            p, r, f1, support = precision_recall_fscore_support(
                y_true_class, y_pred_class, zero_division=0
            )
            
            fold_result['per_class'][class_id] = {
                'precision': float(p),
                'recall': float(r),
                'f1': float(f1),
                'support': int(support),
            }
        
        all_f1 = [fold_result['per_class'][c]['f1'] for c in CLASS_IDS]
        fold_result['macro_f1'] = float(np.mean(all_f1))
        fold_result['macro_precision'] = float(np.mean(
            [fold_result['per_class'][c]['precision'] for c in CLASS_IDS]
        ))
        fold_result['macro_recall'] = float(np.mean(
            [fold_result['per_class'][c]['recall'] for c in CLASS_IDS]
        ))
        
        all_fold_results.append(fold_result)
        
        print(f"           F1={fold_result['macro_f1']:.4f}  "
              f"P={fold_result['macro_precision']:.4f}  "
              f"R={fold_result['macro_recall']:.4f}")
    
    # Aggregate
    print_section("Aggregated Results")
    
    fold_f1s = [fr['macro_f1'] for fr in all_fold_results]
    fold_ps = [fr['macro_precision'] for fr in all_fold_results]
    fold_rs = [fr['macro_recall'] for fr in all_fold_results]
    fold_acc = [fr['subset_accuracy'] for fr in all_fold_results]
    
    aggregate = {
        'n_folds': len(unique_cases),
        'mean_f1': float(np.mean(fold_f1s)),
        'std_f1': float(np.std(fold_f1s)),
        'mean_precision': float(np.mean(fold_ps)),
        'std_precision': float(np.std(fold_ps)),
        'mean_recall': float(np.mean(fold_rs)),
        'std_recall': float(np.std(fold_rs)),
        'mean_subset_accuracy': float(np.mean(fold_acc)),
        'fold_results': all_fold_results,
    }
    
    print(f"Macro F1:      {aggregate['mean_f1']:.4f} ± {aggregate['std_f1']:.4f}")
    print(f"Macro Precision: {aggregate['mean_precision']:.4f} ± {aggregate['std_precision']:.4f}")
    print(f"Macro Recall:  {aggregate['mean_recall']:.4f} ± {aggregate['std_recall']:.4f}")
    print(f"Subset Accuracy: {aggregate['mean_subset_accuracy']:.4f}")
    
    # Per-class aggregate
    print_section("Per-Class Aggregate Results")
    print(f"{'Class':<8}{'Precision':<18}{'Recall':<18}{'F1':<18}{'Support':<10}")
    print("-" * 80)
    
    for class_id in CLASS_IDS:
        ps = [fr['per_class'][class_id]['precision'] for fr in all_fold_results]
        rs = [fr['per_class'][class_id]['recall'] for fr in all_fold_results]
        f1s = [fr['per_class'][class_id]['f1'] for fr in all_fold_results]
        supports = [fr['per_class'][class_id]['support'] for fr in all_fold_results]
        
        p_str = f"{np.mean(ps):.4f}±{np.std(ps):.4f}"
        r_str = f"{np.mean(rs):.4f}±{np.std(rs):.4f}"
        f1_str = f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}"
        support_str = f"{int(sum(supports))}"
        
        print(f"{class_id:<8}{p_str:<18}{r_str:<18}{f1_str:<18}{support_str:<10}")
    
    # Save results
    json_path = OUTPUT_DIR / "evaluation_results_loco.json"
    with open(json_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"\n✓ Results saved to {json_path}")
    
    return aggregate


# ============================================================================
# STAGE 4: EXAMPLE PREDICTIONS
# ============================================================================

def generate_example_predictions(df, model):
    """Generate predictions on held-out examples and real data."""
    print_header("STAGE 4: EXAMPLE PREDICTIONS")
    
    lines = []
    lines.append("="*80)
    lines.append("EXAMPLE PREDICTIONS ON REAL DATA".center(80))
    lines.append("="*80)
    lines.append(f"\nTimestamp: {datetime.now().isoformat()}\n")
    
    # Select 5 examples from different cases
    example_indices = []
    seen_cases = set()
    for i, (idx, row) in enumerate(df.iterrows()):
        if row['case_id'] not in seen_cases:
            example_indices.append(idx)
            seen_cases.add(row['case_id'])
            if len(example_indices) == 5:
                break
    
    lines.append(f"Selected {len(example_indices)} examples from different cases:\n")
    
    for num, idx in enumerate(example_indices, 1):
        row = df.loc[idx]
        X = df["precursor_text"].values
        y = df[CLASS_COLS].values
        
        text = row['precursor_text']
        true_classes = [c for c in CLASS_IDS if row[f'class_{c}'] == 1]
        
        # Predict
        y_pred = model.predict([text])
        y_pred_proba = model.decision_function([text])[0]
        y_pred_proba_clamped = np.clip(y_pred_proba, 0, 1)
        
        pred_classes = [CLASS_IDS[i] for i in range(len(CLASS_IDS)) if y_pred[0, i] == 1]
        
        lines.append(f"Example {num} (from {row['case_id']}):")
        lines.append(f"  Text: {text[:100]}...")
        lines.append(f"  True classes:      {', '.join(true_classes)}")
        lines.append(f"  Predicted classes: {', '.join(pred_classes) if pred_classes else '(none)'}")
        lines.append(f"  Scores:")
        
        # Sort by score
        scores_sorted = sorted(
            [(CLASS_IDS[i], y_pred_proba_clamped[i]) for i in range(len(CLASS_IDS))],
            key=lambda x: x[1],
            reverse=True
        )
        for class_id, score in scores_sorted:
            pred_marker = "✓" if class_id in pred_classes else " "
            true_marker = "✓" if class_id in true_classes else " "
            lines.append(f"    {class_id}: {score:.4f}  [pred:{pred_marker} true:{true_marker}]")
        
        lines.append("")
    
    lines.append("\n" + "="*80)
    lines.append("PREDICTIONS ON SYNTHETIC EXAMPLES".center(80))
    lines.append("="*80 + "\n")
    
    synthetic_examples = [
        "Workers were recently hired but had not received any hands-on training on safety procedures.",
        "The maintenance inspection was not conducted according to OEM guidelines.",
        "The Blind Shear Ram was not included in the BOP stack configuration.",
        "No supervision was available during the high-risk operation.",
        "The equipment showed signs of mechanical fatigue and rust.",
    ]
    
    for num, text in enumerate(synthetic_examples, 1):
        y_pred = model.predict([text])
        y_pred_proba = model.decision_function([text])[0]
        y_pred_proba_clamped = np.clip(y_pred_proba, 0, 1)
        
        pred_classes = [CLASS_IDS[i] for i in range(len(CLASS_IDS)) if y_pred[0, i] == 1]
        
        lines.append(f"Synthetic {num}:")
        lines.append(f"  Text: {text}")
        lines.append(f"  Predicted classes: {', '.join(pred_classes) if pred_classes else '(none)'}")
        lines.append(f"  Scores (top 3):")
        
        scores_sorted = sorted(
            [(CLASS_IDS[i], y_pred_proba_clamped[i]) for i in range(len(CLASS_IDS))],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for class_id, score in scores_sorted:
            lines.append(f"    {class_id}: {score:.4f}")
        lines.append("")
    
    result_text = "\n".join(lines)
    print(result_text)
    
    # Save
    pred_path = OUTPUT_DIR / "example_predictions.txt"
    with open(pred_path, 'w') as f:
        f.write(result_text)
    print(f"✓ Predictions saved to {pred_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute full pipeline."""
    print_header("SIH26165 NLP BASELINE — COMPLETE PIPELINE")
    print(f"Repository: {REPO_ROOT}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        # Stage 1
        df = load_and_validate_data()
        
        # Stage 2
        model = train_model(df)
        
        # Stage 3
        evaluation = evaluate_case_grouped(df)
        
        # Stage 4
        generate_example_predictions(df, model)
        
        # Summary
        print_header("PIPELINE COMPLETE")
        print("\n✓ All stages completed successfully!")
        print("\nGenerated files:")
        print(f"  - {MODELS_DIR / 'multilabel_model.pkl'}")
        print(f"  - {MODELS_DIR / 'metadata.json'}")
        print(f"  - {OUTPUT_DIR / 'evaluation_results_loco.json'}")
        print(f"  - {OUTPUT_DIR / 'example_predictions.txt'}")
        print("\nKey Results:")
        print(f"  - LOCO Mean F1: {evaluation['mean_f1']:.4f} ± {evaluation['std_f1']:.4f}")
        print(f"  - Classes: {', '.join(CLASS_IDS)}")
        print(f"  - Dataset: {len(df)} rows from {df['case_id'].nunique()} cases")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
