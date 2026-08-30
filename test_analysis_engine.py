"""
test_analysis_engine.py — Unit tests for SIH26165 Analysis Engine

Tests each component independently:
- RootCauseAnalyzer
- SIFAssessor
- EvidenceExtractor

To run tests:
    cd sih26165-sif-precursor-engine/mikara-heart-scaling-fishstick
    python test_analysis_engine.py -v
"""

import unittest
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from analysis_engine import (
        SIH26165AnalysisEngine,
        RootCauseAnalyzer,
        SIFAssessor,
        EvidenceExtractor,
        load_schema,
        load_training_data,
        load_label_mapping,
        load_recurring_patterns,
    )
    IMPORT_OK = True
except ImportError as e:
    print(f"⚠️ Import failed: {e}")
    IMPORT_OK = False


class TestDataLoading(unittest.TestCase):
    """Test data loading functions"""
    
    @unittest.skipIf(not IMPORT_OK, "Import failed")
    def test_load_schema(self):
        """Test labels_schema.json loading"""
        schema = load_schema(".")
        self.assertIsNotNone(schema)
        self.assertIn("case_reference", schema)
        self.assertIn("top_level_classes", schema)
        
        # Verify 12 cases
        cases = schema["case_reference"]
        self.assertEqual(len(cases), 12)
        
        # Verify 6 classes
        classes = schema["top_level_classes"]
        self.assertEqual(len(classes), 6)
        
        class_ids = [c["class_id"] for c in classes]
        self.assertEqual(set(class_ids), {"C1", "C2", "C3", "C4", "C5", "C6"})
    
    @unittest.skipIf(not IMPORT_OK, "Import failed")
    def test_load_training_data(self):
        """Test training_data.csv loading"""
        data = load_training_data(".")
        self.assertEqual(len(data), 75)
        
        # Verify required columns
        first_row = data[0]
        self.assertIn("case_id", first_row)
        self.assertIn("precursor_text", first_row)
        self.assertIn("severity", first_row)
    
    @unittest.skipIf(not IMPORT_OK, "Import failed")
    def test_load_label_mapping(self):
        """Test label_mapping.csv loading"""
        mapping = load_label_mapping(".")
        self.assertEqual(len(mapping), 75)
        
        # Verify required columns
        first_row = mapping[0]
        self.assertIn("class_ids", first_row)
        self.assertIn("sub_tag_ids", first_row)
        for c in ["class_C1", "class_C2", "class_C3", "class_C4", "class_C5", "class_C6"]:
            self.assertIn(c, first_row)


class TestRootCauseAnalyzer(unittest.TestCase):
    """Test RootCauseAnalyzer component"""
    
    @unittest.skipIf(not IMPORT_OK, "Import failed")
    def setUp(self):
        self.schema = load_schema(".")
        self.training_data = load_training_data(".")
        self.label_mapping = load_label_mapping(".")
        self.analyzer = RootCauseAnalyzer(self.schema, self.training_data, self.label_mapping)
    
    @unittest.skipIf(not IMPORT_OK, "Import failed")
    def test_analyze_procedure_bypass(self):
        """Test detection of C4 (Procedure Bypass)"""
        text = "JSA was not conducted. No work permit was issued. Risk assessment was skipped."
        classes, provenance = self.analyzer.analyze(text)
        
        class_ids = [c.class_id for c in classes]
        self.assertIn("C4", class_ids)
        self.assertEqual(provenance, "RULE_BASED")
    
    @unittest.skipIf(not IMPORT_OK, "Import failed")
    def test_analyze_missing_barriers(self):
        """Test detection of C5 (Missing Safety Barriers)"""
        text = "Blind shear ram was not installed on the BOP stack. Trip tank was isolated."
        classes, provenance = self.analyzer.analyze(text)
        
        class_ids = [c.class_id for c in classes]
        self.assertIn("C5", class_ids)
        self.assertEqual(provenance, "RULE_BASED")


class TestSIFAssessor(unittest.TestCase):
    """Test SIFAssessor component"""
    
    @unittest.skipIf(not IMPORT_OK, "Import failed")
    def setUp(self):
        self.schema = load_schema(".")
        self.label_mapping = load_label_mapping(".")
        self.assessor = SIFAssessor(self.schema, self.label_mapping)
    
    @unittest.skipIf(not IMPORT_OK, "Import failed")
    def test_assess_high_risk(self):
        """Test SIF-HIGH assessment"""
        assessment = self.assessor.assess(["C4", "C5"])
        self.assertEqual(assessment.level, "HIGH")
        self.assertEqual(assessment.provenance, "RULE_BASED")
    
    @unittest.skipIf(not IMPORT_OK, "Import failed")
    def test_assess_medium_risk(self):
        """Test SIF-MEDIUM assessment"""
        assessment = self.assessor.assess(["C2", "C3"])
        self.assertEqual(assessment.level, "MEDIUM")


class TestFullAnalysis(unittest.TestCase):
    """Integration tests on full analysis pipeline"""
    
    @unittest.skipIf(not IMPORT_OK, "Import failed")
    def setUp(self):
        self.engine = SIH26165AnalysisEngine(repo_root=".")
    
    @unittest.skipIf(not IMPORT_OK, "Import failed")
    def test_analyze_real_case_5(self):
        """Test analysis on real Case 5 text"""
        text = (
            "Blind shear ram was not included in the BOP stack configuration "
            "because well was assumed to be oil well, not gas. "
            "Trip tank was isolated during perforation."
        )
        result = self.engine.analyze(text)
        
        class_ids = [c.class_id for c in result.detected_classes]
        self.assertIn("C4", class_ids)
        self.assertIn("C5", class_ids)
        self.assertEqual(result.sif_assessment.level, "HIGH")
    
    @unittest.skipIf(not IMPORT_OK, "Import failed")
    def test_provenance_labeled(self):
        """Test that all results have provenance labels"""
        text = "Training gaps and missing barriers."
        result = self.engine.analyze(text)
        
        for cls in result.detected_classes:
            self.assertIsNotNone(cls.provenance)
        
        self.assertEqual(result.sif_assessment.provenance, "RULE_BASED")
        self.assertEqual(result.lsr_mapping.status, "NOT_YET_IMPLEMENTED")


if __name__ == "__main__":
    if not IMPORT_OK:
        print("\n⚠️ Could not import analysis_engine. Verify src/analysis_engine.py exists.")
        sys.exit(1)
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
