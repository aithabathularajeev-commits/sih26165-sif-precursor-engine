#!/usr/bin/env python3
"""
run_pipeline.py — Complete SIH26165 NLP baseline pipeline

Executes the full pipeline:
1. Tests
2. Train model
3. Case-grouped evaluation
4. Example predictions

Usage:
    python run_pipeline.py

Output:
    - models/multilabel_model.pkl
    - models/metadata.json
    - outputs/evaluation_report_loco.txt
    - outputs/evaluation_results_loco.json
    - outputs/example_predictions.txt
"""

import sys
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parent
SCRIPTS = [
    ("Tests", REPO_ROOT / "src" / "data" / "test_pipeline.py"),
    ("Training", REPO_ROOT / "src" / "data" / "train_baseline_model.py"),
    ("Case-Grouped Evaluation", REPO_ROOT / "src" / "data" / "evaluate_case_grouped.py"),
]


def run_script(name, script_path):
    """Run a Python script and capture output."""
    print("\n" + "="*80)
    print(f"▶ {name.upper()}")
    print("="*80)
    
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(REPO_ROOT),
            capture_output=False,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"\n✓ {name} completed successfully")
            return True
        else:
            print(f"\n✗ {name} failed with return code {result.returncode}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"\n✗ {name} timed out")
        return False
    except Exception as e:
        print(f"\n✗ {name} failed: {e}")
        return False


def main():
    print("\n" + "="*80)
    print("SIH26165 NLP BASELINE PIPELINE")
    print("="*80)
    print(f"Repository: {REPO_ROOT}")
    
    results = {}
    for name, script_path in SCRIPTS:
        results[name] = run_script(name, script_path)
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL STAGES COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nGenerated outputs:")
        print("  - models/multilabel_model.pkl")
        print("  - models/metadata.json")
        print("  - outputs/evaluation_report_loco.txt")
        print("  - outputs/evaluation_results_loco.json")
        print("\nNext steps:")
        print("  1. Review evaluation_report_loco.txt for detailed results")
        print("  2. Try predictions: python src/data/predict.py \"<safety report text>\"")
        print("  3. Check outputs/example_predictions.txt for examples")
    else:
        print("✗ PIPELINE FAILED")
        print("="*80)
        sys.exit(1)


if __name__ == "__main__":
    main()
