"""
evaluate_case_grouped.py — Case-Grouped Evaluation for SIH26165

Implements Leave-One-Case-Out (LOCO) cross-validation to prevent case-level leakage.

CRITICAL: The random 60/15 split in src/data/processed/ is subject to case-level leakage
because rows from the same incident can appear in both train and validation. This module
implements proper case-grouped evaluation where all rows from a single incident are kept
together (either fully in training or fully in validation).

Evaluation Strategy:
- Leave-One-Case-Out (LOCO): 12 folds, each fold holds out all rows from one case
- For each fold:
  1. Remove all rows belonging to one case
  2. Fit TF-IDF vectorizer ONLY on training rows (no leakage)
  3. Train model on training data
  4. Evaluate on held-out case
- Aggregate results across all 12 folds

This is appropriate for:
- Small datasets (75 rows / 12 cases)
- Realistic scenario: predict performance on unseen future cases
- Prevents case-level data leakage

Usage:
    python src/data/evaluate_case_grouped.py

Output:
    outputs/evaluation_report_loco.txt (detailed per-fold and aggregate results)
    outputs/evaluation_results_loco.json (structured results for programmatic access)
"""

import json
import pickle
import sys
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


# Configuration
REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_DIR = REPO_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_IDS = ["C1", "C2", "C3", "C4", "C5", "C6"]
CLASS_COLS = [f"class_{c}" for c in CLASS_IDS]


def load_data(csv_path):
    """Load and validate label_mapping.csv."""
    df = pd.read_csv(csv_path)
    
    required = ["row_id", "case_id", "precursor_text", "class_C1", "class_C2",
                "class_C3", "class_C4", "class_C5", "class_C6"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    for col in CLASS_COLS:
        df[col] = df[col].astype(int)
    
    return df


def create_model():
    """Create a multi-label classifier pipeline."""
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


def evaluate_on_fold(model, X_test, y_test, case_id_test):
    """Evaluate model on a single held-out case."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.decision_function(X_test)
    y_pred_proba_clamped = np.clip(y_pred_proba, 0, 1)
    
    fold_results = {
        'held_out_case': case_id_test,
        'n_test_samples': len(X_test),
        'hamming_loss': float(hamming_loss(y_test, y_pred)),
        'subset_accuracy': float(accuracy_score(y_test, y_pred)),
        'per_class': {},
    }
    
    # Per-class metrics
    for i, class_id in enumerate(CLASS_IDS):
        y_true_class = y_test[:, i]
        y_pred_class = y_pred[:, i]
        
        p, r, f1, support = precision_recall_fscore_support(
            y_true_class, y_pred_class, zero_division=0
        )
        
        fold_results['per_class'][class_id] = {
            'precision': float(p),
            'recall': float(r),
            'f1': float(f1),
            'support': int(support),
        }
    
    # Macro-averaged metrics
    all_f1 = [fold_results['per_class'][c]['f1'] for c in CLASS_IDS]
    all_p = [fold_results['per_class'][c]['precision'] for c in CLASS_IDS]
    all_r = [fold_results['per_class'][c]['recall'] for c in CLASS_IDS]
    
    fold_results['macro_precision'] = float(np.mean(all_p))
    fold_results['macro_recall'] = float(np.mean(all_r))
    fold_results['macro_f1'] = float(np.mean(all_f1))
    
    return fold_results


def aggregate_fold_results(all_fold_results):
    """Aggregate results across all LOCO folds."""
    n_folds = len(all_fold_results)
    
    # Initialize aggregates
    aggregate = {
        'n_folds': n_folds,
        'fold_results': all_fold_results,
    }
    
    # Aggregate metrics
    fold_f1_scores = [fr['macro_f1'] for fr in all_fold_results]
    fold_precision = [fr['macro_precision'] for fr in all_fold_results]
    fold_recall = [fr['macro_recall'] for fr in all_fold_results]
    fold_subset_acc = [fr['subset_accuracy'] for fr in all_fold_results]
    fold_hamming = [fr['hamming_loss'] for fr in all_fold_results]
    
    aggregate['overall'] = {
        'mean_macro_f1': float(np.mean(fold_f1_scores)),
        'std_macro_f1': float(np.std(fold_f1_scores)),
        'mean_macro_precision': float(np.mean(fold_precision)),
        'std_macro_precision': float(np.std(fold_precision)),
        'mean_macro_recall': float(np.mean(fold_recall)),
        'std_macro_recall': float(np.std(fold_recall)),
        'mean_subset_accuracy': float(np.mean(fold_subset_acc)),
        'std_subset_accuracy': float(np.std(fold_subset_acc)),
        'mean_hamming_loss': float(np.mean(fold_hamming)),
        'std_hamming_loss': float(np.std(fold_hamming)),
    }
    
    # Per-class aggregates
    aggregate['per_class_aggregate'] = {}
    for class_id in CLASS_IDS:
        class_f1s = [fr['per_class'][class_id]['f1'] for fr in all_fold_results]
        class_ps = [fr['per_class'][class_id]['precision'] for fr in all_fold_results]
        class_rs = [fr['per_class'][class_id]['recall'] for fr in all_fold_results]
        class_supports = [fr['per_class'][class_id]['support'] for fr in all_fold_results]
        
        aggregate['per_class_aggregate'][class_id] = {
            'mean_precision': float(np.mean(class_ps)),
            'std_precision': float(np.std(class_ps)),
            'mean_recall': float(np.mean(class_rs)),
            'std_recall': float(np.std(class_rs)),
            'mean_f1': float(np.mean(class_f1s)),
            'std_f1': float(np.std(class_f1s)),
            'total_support': int(sum(class_supports)),
        }
    
    return aggregate


def format_results_text(aggregate):
    """Format evaluation results as human-readable text."""
    lines = []
    lines.append("\n" + "="*80)
    lines.append("SIH26165 CASE-GROUPED EVALUATION (Leave-One-Case-Out Cross-Validation)".center(80))
    lines.append("="*80)
    lines.append(f"Timestamp: {datetime.now().isoformat()}\n")
    
    lines.append("EVALUATION METHODOLOGY:")
    lines.append("-" * 80)
    lines.append("  Leave-One-Case-Out (LOCO) Cross-Validation")
    lines.append("  - 12 folds: each fold holds out ALL rows from ONE incident/case")
    lines.append("  - Prevents case-level leakage: same case never in both train and test")
    lines.append("  - For each fold:")
    lines.append("      1. Remove all rows belonging to one case")
    lines.append("      2. Fit TF-IDF vectorizer ONLY on training rows")
    lines.append("      3. Train multi-label classifier on training data")
    lines.append("      4. Evaluate on held-out case")
    lines.append("  - Aggregate results across all 12 folds")
    lines.append("")
    
    lines.append("OVERALL RESULTS (across all 12 folds):")
    lines.append("-" * 80)
    overall = aggregate['overall']
    lines.append(f"  Mean Macro F1:          {overall['mean_macro_f1']:.4f} ± {overall['std_macro_f1']:.4f}")
    lines.append(f"  Mean Macro Precision:   {overall['mean_macro_precision']:.4f} ± {overall['std_macro_precision']:.4f}")
    lines.append(f"  Mean Macro Recall:      {overall['mean_macro_recall']:.4f} ± {overall['std_macro_recall']:.4f}")
    lines.append(f"  Mean Subset Accuracy:   {overall['mean_subset_accuracy']:.4f} ± {overall['std_subset_accuracy']:.4f}")
    lines.append(f"  Mean Hamming Loss:      {overall['mean_hamming_loss']:.4f} ± {overall['std_hamming_loss']:.4f}")
    lines.append("")
    
    lines.append("PER-CLASS AGGREGATE RESULTS:")
    lines.append("-" * 80)
    lines.append(f"{'Class':<8}{'Precision':<18}{'Recall':<18}{'F1':<18}{'Support':<10}")
    lines.append("-" * 80)
    for class_id in CLASS_IDS:
        pc = aggregate['per_class_aggregate'][class_id]
        p_str = f"{pc['mean_precision']:.4f}±{pc['std_precision']:.4f}"
        r_str = f"{pc['mean_recall']:.4f}±{pc['std_recall']:.4f}"
        f1_str = f"{pc['mean_f1']:.4f}±{pc['std_f1']:.4f}"
        lines.append(
            f"{class_id:<8}{p_str:<18}{r_str:<18}{f1_str:<18}{pc['total_support']:<10}"
        )
    lines.append("")
    
    lines.append("PER-FOLD RESULTS:")
    lines.append("-" * 80)
    for i, fold_result in enumerate(aggregate['fold_results'], 1):
        lines.append(f"\nFold {i}: Held-out case = {fold_result['held_out_case']} (n={fold_result['n_test_samples']})")
        lines.append(f"  Macro F1:       {fold_result['macro_f1']:.4f}")
        lines.append(f"  Macro Precision: {fold_result['macro_precision']:.4f}")
        lines.append(f"  Macro Recall:    {fold_result['macro_recall']:.4f}")
        lines.append(f"  Subset Accuracy: {fold_result['subset_accuracy']:.4f}")
        lines.append(f"  Hamming Loss:    {fold_result['hamming_loss']:.4f}")
    
    lines.append("\n" + "="*80)
    lines.append("IMPORTANT LIMITATIONS:")
    lines.append("="*80)
    lines.append("  1. Small dataset (75 rows / 12 cases): high variance in per-fold results")
    lines.append("  2. Class imbalance (C4=30 rows vs C6=7 rows): some classes have few examples")
    lines.append("  3. Some held-out cases have only 4-7 rows: class predictions may be sparse")
    lines.append("  4. Per-class metrics for C5/C6 may be unreliable due to small support")
    lines.append("  5. No ablation or hyperparameter tuning: this is a baseline model")
    lines.append("")
    
    return "\n".join(lines)


def main():
    print("\n" + "="*80)
    print("SIH26165 Case-Grouped Evaluation (Leave-One-Case-Out CV)")
    print("="*80)
    
    # Load data
    data_path = DATA_DIR / "label_mapping.csv"
    print(f"\n1. Loading data from {data_path}...")
    
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        sys.exit(1)
    
    df = load_data(data_path)
    X = df["precursor_text"].values
    y = df[CLASS_COLS].values
    case_ids = df["case_id"].values
    
    print(f"   ✓ Loaded {len(df)} rows, {df['case_id'].nunique()} unique cases")
    
    # Get unique cases
    unique_cases = sorted(df['case_id'].unique())
    print(f"   ✓ Cases: {', '.join(unique_cases)}")
    
    # Leave-One-Case-Out cross-validation
    print(f"\n2. Performing Leave-One-Case-Out (LOCO) cross-validation...")
    all_fold_results = []
    
    for fold_idx, held_out_case in enumerate(unique_cases, 1):
        print(f"\n   Fold {fold_idx}/12: Holding out {held_out_case}...")
        
        # Split by case
        train_mask = case_ids != held_out_case
        test_mask = case_ids == held_out_case
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        print(f"      Train: {len(X_train)} rows from {len(np.unique(case_ids[train_mask]))} cases")
        print(f"      Test:  {len(X_test)} rows from {held_out_case}")
        
        # Create and train model (TF-IDF fitted ONLY on training data)
        model = create_model()
        model.fit(X_train, y_train)
        
        # Evaluate on held-out case
        fold_results = evaluate_on_fold(model, X_test, y_test, held_out_case)
        all_fold_results.append(fold_results)
        
        print(f"      Results: F1={fold_results['macro_f1']:.4f}, "
              f"Precision={fold_results['macro_precision']:.4f}, "
              f"Recall={fold_results['macro_recall']:.4f}")
    
    # Aggregate results
    print(f"\n3. Aggregating results across {len(unique_cases)} folds...")
    aggregate = aggregate_fold_results(all_fold_results)
    
    # Display results
    results_text = format_results_text(aggregate)
    print(results_text)
    
    # Save results
    report_path = OUTPUT_DIR / "evaluation_report_loco.txt"
    with open(report_path, 'w') as f:
        f.write(results_text)
    print(f"\n4. ✓ Report saved to {report_path}")
    
    json_path = OUTPUT_DIR / "evaluation_results_loco.json"
    with open(json_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"   ✓ Structured results saved to {json_path}")
    
    print(f"\n{'='*80}")
    print("✓ Evaluation complete")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
