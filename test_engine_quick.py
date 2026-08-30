#!/usr/bin/env python3
"""
Quick syntax check and basic test of analysis_engine.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    # Test import
    from analysis_engine import SIH26165AnalysisEngine
    print("✓ Import successful")
    
    # Test initialization
    engine = SIH26165AnalysisEngine(repo_root=".")
    print("✓ Engine initialized")
    
    # Test analysis
    test_text = (
        "JSA was not conducted. The blind shear ram was not included in the BOP stack "
        "because it was assumed to be an oil well. Trip tank was isolated during perforation."
    )
    
    result = engine.analyze(test_text)
    print("✓ Analysis completed")
    
    print(f"\nDetected classes: {len(result.detected_classes)}")
    for cls in result.detected_classes:
        print(f"  - {cls.class_id}: {cls.class_name} (Provenance: {cls.provenance})")
    
    print(f"\nSIF Assessment: {result.sif_assessment.level} (Provenance: {result.sif_assessment.provenance})")
    
    print("\n✓ All basic tests passed")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
