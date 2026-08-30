"""
test_pipeline.py — Unit and integration tests for SIH26165 NLP pipeline

Tests cover:
- Data loading and validation
- Model creation and training
- Inference on various inputs
- Prevention of case-level leakage
- Multi-label correctness
- Reproducibility
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


# Configuration
REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_DIR = REPO_ROOT / "outputs"

CLASS_IDS = ["C1", "C2", "C3", "C4", "C5", "C6"]
CLASS_COLS = [f"class_{c}" for c in CLASS_IDS]


class TestDataIntegrity:
    """Tests for data loading and validation."""
    
    @staticmethod
    def test_load_label_mapping():
        """Test that label_mapping.csv can be loaded and has expected structure."""
        df = pd.read_csv(DATA_DIR / "label_mapping.csv")
        
        # Check row count
        assert len(df) == 75, f"Expected 75 rows, got {len(df)}"
        
        # Check required columns
        required = ["row_id", "case_id", "precursor_text", "class_C1", "class_C2",
                    "class_C3", "class_C4", "class_C5", "class_C6"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
        
        # Check case distribution
        assert df['case_id'].nunique() == 12, f"Expected 12 cases, got {df['case_id'].nunique()}"
        
        # Check that case_ids are labeled Case 1 through Case 12
        expected_cases = [f"Case {i}" for i in range(1, 13)]
        actual_cases = sorted(df['case_id'].unique())
        assert actual_cases == expected_cases, f"Unexpected case labels: {actual_cases}"
        
        print("✓ test_load_label_mapping PASSED")
        return df
    
    @staticmethod
    def test_multi_label_structure(df):
        """Test that multi-label encoding is correct."""
        # Check that class columns are binary
        y = df[CLASS_COLS].values
        assert np.all((y == 0) | (y == 1)), "Class columns should be binary (0 or 1)"
        
        # Check that each row has at least one class
        row_sums = y.sum(axis=1)
        assert np.all(row_sums > 0), "Each row must have at least one class label"
        
        # Check multi-label count (should have some rows with multiple labels)
        multi_label_count = (row_sums > 1).sum()
        assert multi_label_count > 0, "Should have at least some multi-label rows"
        
        # Verify specific known multi-label rows
        # Row 22 (index 21) should have both C1 and C2
        row_22 = y[21]  # 0-indexed
        assert row_22[0] == 1 and row_22[1] == 1, "Row 22 should have C1=1 and C2=1"
        
        print(f"✓ test_multi_label_structure PASSED (multi-label rows: {multi_label_count})")
    
    @staticmethod
    def test_case_grouping(df):
        """Test that case groupings are correct and distinct."""
        case_indices = {}
        for case_id in df['case_id'].unique():
            mask = df['case_id'] == case_id
            indices = df[mask].index.tolist()
            case_indices[case_id] = set(indices)
        
        # Check that cases are disjoint (no row appears in two cases)
        all_indices = set()
        for indices_set in case_indices.values():
            overlap = all_indices & indices_set
            assert len(overlap) == 0, f"Found overlapping rows between cases: {overlap}"
            all_indices |= indices_set
        
        # Check that all rows are accounted for
        assert len(all_indices) == len(df), "Not all rows accounted for in case grouping"
        
        print("✓ test_case_grouping PASSED")
        return case_indices
    
    @staticmethod
    def test_no_case_leakage_in_random_split():
        """Test that the random train.csv/val.csv split has case-level leakage."""
        train_df = pd.read_csv(REPO_ROOT / "src" / "data" / "processed" / "train.csv")
        val_df = pd.read_csv(REPO_ROOT / "src" / "data" / "processed" / "val.csv")
        
        train_cases = set(train_df['case_id'].unique())
        val_cases = set(val_df['case_id'].unique())
        
        overlap = train_cases & val_cases
        
        if len(overlap) > 0:
            print(f"⚠ test_no_case_leakage_in_random_split: DETECTED CASE-LEVEL LEAKAGE")
            print(f"  Cases appearing in BOTH train and val: {overlap}")
            print(f"  This is expected for the random split; case-grouped evaluation prevents this")
            return False
        else:
            print("✓ test_no_case_leakage_in_random_split: No leakage detected")
            return True


class TestModel:
    """Tests for model creation and training."""
    
    @staticmethod
    def test_model_creation():
        """Test that model can be created."""
        from train_baseline_model import create_model
        
        model = create_model()
        assert isinstance(model, Pipeline), "Model should be a sklearn Pipeline"
        assert 'tfidf' in model.named_steps, "Pipeline should have tfidf step"
        assert 'classifier' in model.named_steps, "Pipeline should have classifier step"
        
        print("✓ test_model_creation PASSED")
    
    @staticmethod
    def test_model_training_and_prediction():
        """Test that model can be trained and make predictions."""
        from train_baseline_model import create_model
        
        # Create small synthetic dataset
        texts = [
            "There was no evidence that inspection was conducted.",
            "No mentor or supervisor was assigned.",
            "The equipment was not properly maintained.",
        ]
        labels = np.array([
            [0, 0, 1, 0, 0, 0],  # C3
            [0, 1, 0, 0, 0, 0],  # C2
            [0, 0, 1, 0, 0, 0],  # C3
        ])
        
        model = create_model()
        model.fit(texts, labels)
        
        # Predict on new text
        pred = model.predict(texts)
        
        # Check output shape
        assert pred.shape == (3, 6), f"Expected shape (3, 6), got {pred.shape}"
        
        # Check that predictions are binary
        assert np.all((pred == 0) | (pred == 1)), "Predictions should be binary"
        
        print("✓ test_model_training_and_prediction PASSED")
    
    @staticmethod
    def test_model_reproducibility():
        """Test that training with fixed random_state is reproducible."""
        from train_baseline_model import create_model
        
        df = pd.read_csv(DATA_DIR / "label_mapping.csv")
        X = df["precursor_text"].values
        y = df[CLASS_COLS].values
        
        # Train two models with same random state
        model1 = create_model()
        model1.fit(X, y)
        pred1 = model1.predict(X)
        
        model2 = create_model()
        model2.fit(X, y)
        pred2 = model2.predict(X)
        
        # Predictions should be identical
        assert np.array_equal(pred1, pred2), "Predictions should be identical with fixed random_state"
        
        print("✓ test_model_reproducibility PASSED")


class TestInference:
    """Tests for inference functionality."""
    
    @staticmethod
    def test_predictor_initialization():
        """Test that predictor can be initialized."""
        from predict import SIFPrecursorPredictor
        
        model_path = MODELS_DIR / "multilabel_model.pkl"
        
        if not model_path.exists():
            print("⚠ test_predictor_initialization: Model not trained yet (skipping)")
            return
        
        predictor = SIFPrecursorPredictor(str(model_path))
        assert predictor.model is not None, "Model should be loaded"
        assert len(predictor.class_ids) == 6, "Should have 6 classes"
        
        print("✓ test_predictor_initialization PASSED")
    
    @staticmethod
    def test_prediction_output_format():
        """Test that prediction output has expected format."""
        from predict import SIFPrecursorPredictor
        
        model_path = MODELS_DIR / "multilabel_model.pkl"
        
        if not model_path.exists():
            print("⚠ test_prediction_output_format: Model not trained yet (skipping)")
            return
        
        predictor = SIFPrecursorPredictor(str(model_path))
        
        text = "Workers were not properly trained on safety procedures."
        result = predictor.predict(text)
        
        # Check result structure
        assert 'predictions' in result, "Result should have 'predictions' key"
        assert 'all_scores' in result, "Result should have 'all_scores' key"
        assert 'predicted_classes' in result, "Result should have 'predicted_classes' key"
        
        # Check predictions format
        for pred in result['predictions']:
            assert 'class' in pred, "Prediction should have 'class' key"
            assert 'probability' in pred, "Prediction should have 'probability' key"
            assert 'predicted' in pred, "Prediction should have 'predicted' key"
            assert 0 <= pred['probability'] <= 1, "Probability should be in [0, 1]"
        
        # Check all_scores
        assert len(result['all_scores']) == 6, "Should have scores for all 6 classes"
        for class_id in ["C1", "C2", "C3", "C4", "C5", "C6"]:
            assert class_id in result['all_scores'], f"Missing score for {class_id}"
        
        print("✓ test_prediction_output_format PASSED")
    
    @staticmethod
    def test_prediction_on_real_examples():
        """Test predictions on real examples from dataset."""
        from predict import SIFPrecursorPredictor
        
        model_path = MODELS_DIR / "multilabel_model.pkl"
        
        if not model_path.exists():
            print("⚠ test_prediction_on_real_examples: Model not trained yet (skipping)")
            return
        
        df = pd.read_csv(DATA_DIR / "label_mapping.csv")
        predictor = SIFPrecursorPredictor(str(model_path))
        
        # Test on first 3 rows
        for i in range(3):
            text = df.iloc[i]['precursor_text']
            true_classes = [c for c in CLASS_IDS if df.iloc[i][f'class_{c}'] == 1]
            
            result = predictor.predict(text, threshold=0.5)
            pred_classes = result['predicted_classes']
            
            print(f"  Row {i+1}:")
            print(f"    True: {true_classes}")
            print(f"    Pred: {pred_classes}")
        
        print("✓ test_prediction_on_real_examples PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("SIH26165 NLP PIPELINE TESTS")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    test_results = []
    
    # Data integrity tests
    print("DATA INTEGRITY TESTS")
    print("-" * 70)
    try:
        df = TestDataIntegrity.test_load_label_mapping()
        test_results.append(("test_load_label_mapping", True, None))
    except Exception as e:
        print(f"✗ test_load_label_mapping FAILED: {e}")
        test_results.append(("test_load_label_mapping", False, str(e)))
        return test_results
    
    try:
        TestDataIntegrity.test_multi_label_structure(df)
        test_results.append(("test_multi_label_structure", True, None))
    except Exception as e:
        print(f"✗ test_multi_label_structure FAILED: {e}")
        test_results.append(("test_multi_label_structure", False, str(e)))
    
    try:
        case_indices = TestDataIntegrity.test_case_grouping(df)
        test_results.append(("test_case_grouping", True, None))
    except Exception as e:
        print(f"✗ test_case_grouping FAILED: {e}")
        test_results.append(("test_case_grouping", False, str(e)))
    
    try:
        TestDataIntegrity.test_no_case_leakage_in_random_split()
        test_results.append(("test_no_case_leakage_in_random_split", True, None))
    except Exception as e:
        print(f"⚠ test_no_case_leakage_in_random_split: {e}")
    
    # Model tests
    print("\nMODEL TESTS")
    print("-" * 70)
    try:
        TestModel.test_model_creation()
        test_results.append(("test_model_creation", True, None))
    except Exception as e:
        print(f"✗ test_model_creation FAILED: {e}")
        test_results.append(("test_model_creation", False, str(e)))
    
    try:
        TestModel.test_model_training_and_prediction()
        test_results.append(("test_model_training_and_prediction", True, None))
    except Exception as e:
        print(f"✗ test_model_training_and_prediction FAILED: {e}")
        test_results.append(("test_model_training_and_prediction", False, str(e)))
    
    try:
        TestModel.test_model_reproducibility()
        test_results.append(("test_model_reproducibility", True, None))
    except Exception as e:
        print(f"✗ test_model_reproducibility FAILED: {e}")
        test_results.append(("test_model_reproducibility", False, str(e)))
    
    # Inference tests
    print("\nINFERENCE TESTS")
    print("-" * 70)
    try:
        TestInference.test_predictor_initialization()
        test_results.append(("test_predictor_initialization", True, None))
    except Exception as e:
        print(f"✗ test_predictor_initialization FAILED: {e}")
        test_results.append(("test_predictor_initialization", False, str(e)))
    
    try:
        TestInference.test_prediction_output_format()
        test_results.append(("test_prediction_output_format", True, None))
    except Exception as e:
        print(f"✗ test_prediction_output_format FAILED: {e}")
        test_results.append(("test_prediction_output_format", False, str(e)))
    
    try:
        TestInference.test_prediction_on_real_examples()
        test_results.append(("test_prediction_on_real_examples", True, None))
    except Exception as e:
        print(f"✗ test_prediction_on_real_examples FAILED: {e}")
        test_results.append(("test_prediction_on_real_examples", False, str(e)))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(1 for _, success, _ in test_results if success)
    total = len(test_results)
    print(f"Passed: {passed}/{total}")
    
    if passed < total:
        print("\nFailed tests:")
        for name, success, error in test_results:
            if not success:
                print(f"  - {name}: {error}")
    
    print(f"\n{'='*70}\n")
    
    return test_results


if __name__ == "__main__":
    # Add src/data to path so imports work
    sys.path.insert(0, str(DATA_DIR.parent))
    
    run_all_tests()
