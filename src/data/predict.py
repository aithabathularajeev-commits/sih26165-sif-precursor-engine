"""
predict.py — Inference module for SIH26165 multi-label classifier

Loads a trained model and provides inference on new precursor text.

Usage (as module):
    from predict import SIFPrecursorPredictor
    
    predictor = SIFPrecursorPredictor('models/multilabel_model.pkl', 'data/labels_schema.json')
    result = predictor.predict("Some safety report text...")
    print(result)

Usage (command-line):
    python src/data/predict.py "Text to classify"

Output format:
    {
        'text': '...',
        'predictions': [
            {
                'class': 'C4',
                'probability': 0.78,
                'label_name': 'Procedure, Permit & Risk-Assessment Bypass',
            },
            ...
        ],
        'all_scores': {'C1': 0.15, 'C2': 0.22, ...},
    }
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class SIFPrecursorPredictor:
    """Multi-label precursor classifier inference engine."""
    
    def __init__(self, model_path: str, schema_path: str = None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained sklearn Pipeline (pickle file)
            schema_path: Optional path to labels_schema.json for label metadata
        """
        self.model_path = Path(model_path)
        self.schema_path = Path(schema_path) if schema_path else None
        
        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load schema if provided
        self.schema = None
        if self.schema_path and self.schema_path.exists():
            with open(self.schema_path, 'r') as f:
                self.schema = json.load(f)
        
        # Class information
        self.class_ids = ["C1", "C2", "C3", "C4", "C5", "C6"]
        self.class_names = {
            "C1": "Training & Competency Gaps",
            "C2": "Supervision & Communication Breakdown",
            "C3": "Maintenance & Inspection Failures",
            "C4": "Procedure, Permit & Risk-Assessment Bypass",
            "C5": "Missing/Defunct Critical Safety Barriers",
            "C6": "Environmental & Physical Workspace Hazards",
        }
    
    def predict(self, text: str, threshold: float = 0.5) -> Dict:
        """
        Predict precursor classes for input text.
        
        Args:
            text: Safety report or precursor statement (raw text)
            threshold: Confidence threshold for binary prediction (default 0.5)
        
        Returns:
            Dictionary with predictions and scores
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {
                'text': text,
                'error': 'Empty or invalid input text',
                'predictions': [],
                'all_scores': {},
            }
        
        # Get raw decision function scores
        scores_raw = self.model.decision_function([text])[0]  # (n_classes,)
        
        # Clamp scores to [0, 1] for probability interpretation
        scores_proba = np.clip(scores_raw, 0, 1)
        
        # Apply threshold for binary predictions
        predictions_binary = (scores_proba >= threshold).astype(int)
        
        # Build predictions list (sorted by score, descending)
        predictions = []
        score_dict = {}
        
        for i, class_id in enumerate(self.class_ids):
            score = float(scores_proba[i])
            score_dict[class_id] = score
            
            predicted = bool(predictions_binary[i])
            
            predictions.append({
                'class': class_id,
                'label_name': self.class_names.get(class_id, ''),
                'probability': score,
                'predicted': predicted,
            })
        
        # Sort by probability (descending)
        predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)
        
        # Get predicted classes (those above threshold)
        predicted_classes = [p['class'] for p in predictions if p['predicted']]
        
        # Build result
        result = {
            'text': text[:200] + ('...' if len(text) > 200 else ''),  # Truncate for display
            'text_length': len(text),
            'predicted_classes': predicted_classes,
            'predictions': predictions,
            'all_scores': score_dict,
            'threshold_used': threshold,
        }
        
        return result
    
    def predict_batch(self, texts: List[str], threshold: float = 0.5) -> List[Dict]:
        """
        Predict on multiple texts.
        
        Args:
            texts: List of text strings
            threshold: Confidence threshold
        
        Returns:
            List of prediction results
        """
        return [self.predict(text, threshold) for text in texts]


def format_prediction_output(result: Dict) -> str:
    """Format prediction result for console output."""
    lines = []
    lines.append("\n" + "="*70)
    lines.append("SIF PRECURSOR PREDICTION RESULT")
    lines.append("="*70)
    
    if 'error' in result:
        lines.append(f"ERROR: {result['error']}")
        return "\n".join(lines)
    
    lines.append(f"Input text ({result['text_length']} chars): {result['text']}")
    lines.append(f"Threshold: {result['threshold_used']:.2f}")
    lines.append("")
    
    lines.append(f"PREDICTED CLASSES: {', '.join(result['predicted_classes']) if result['predicted_classes'] else '(none above threshold)'}")
    lines.append("")
    
    lines.append("SCORES FOR ALL CLASSES:")
    lines.append(f"{'Class':<8}{'Label':<45}{'Score':<10}{'Pred':<6}")
    lines.append("-"*70)
    
    for pred in result['predictions']:
        class_id = pred['class']
        label = pred['label_name'][:42]  # Truncate
        score = pred['probability']
        pred_str = "✓ YES" if pred['predicted'] else "  no"
        
        lines.append(f"{class_id:<8}{label:<45}{score:<10.4f}{pred_str:<6}")
    
    lines.append("="*70)
    
    return "\n".join(lines)


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <text_to_classify>")
        print("\nExample:")
        print("  python predict.py \"Workers were recently hired but had no hands-on training\"")
        sys.exit(1)
    
    text = " ".join(sys.argv[1:])
    
    # Initialize predictor
    model_path = Path(__file__).parent.parent.parent / "models" / "multilabel_model.pkl"
    schema_path = Path(__file__).parent.parent.parent / "data" / "labels_schema.json"
    
    try:
        predictor = SIFPrecursorPredictor(str(model_path), str(schema_path))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure the model is trained. Run:")
        print(f"  python src/data/train_baseline_model.py")
        sys.exit(1)
    
    # Predict
    result = predictor.predict(text, threshold=0.5)
    
    # Display result
    output = format_prediction_output(result)
    print(output)
    
    # Also save as JSON
    json_output = Path(__file__).parent.parent.parent / "outputs" / "last_prediction.json"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    with open(json_output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nDetailed result saved to: {json_output}")


if __name__ == "__main__":
    main()
