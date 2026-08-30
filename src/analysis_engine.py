"""
analysis_engine.py — SIH26165 SIF Precursor Analysis Engine

Core pipeline that orchestrates:
1. Root-cause detection (C1-C6 classes)
2. SIF-potential assessment
3. Evidence linkage
4. IOGP LSR mapping (deferred)

All results include provenance labels:
- TRAINED_ML: From a trained model (only if model validation F1 >= 0.5)
- RULE_BASED: From deterministic keyword matching
- DETERMINISTIC_EXTRACTION: From direct lookups (case/evidence/pattern mappings)
"""

import json
import os
import re
import csv
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


# ============================================================================
# DATA LOADING
# ============================================================================

def load_schema(repo_root: str = ".") -> Dict:
    """Load labels_schema.json"""
    path = Path(repo_root) / "data" / "labels_schema.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_training_data(repo_root: str = ".") -> List[Dict]:
    """Load training_data.csv"""
    path = Path(repo_root) / "data" / "training_data.csv"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_label_mapping(repo_root: str = ".") -> List[Dict]:
    """Load label_mapping.csv"""
    path = Path(repo_root) / "data" / "label_mapping.csv"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_recurring_patterns(repo_root: str = ".") -> Dict[str, List[str]]:
    """
    Extract recurring patterns from raw_research/2.Recurring Patterns...
    
    Returns: {
        "pattern_name": ["Case X", "Case Y", ...],
        ...
    }
    """
    patterns = {
        "Blind Shear Ram Omission": ["Case 5", "Case 9"],
        "Trip Tank Isolation During Perforation": ["Case 5", "Case 9"],
        "Exceeding Trip Speeds": ["Case 5", "Case 9"],
        "Absence of Supervision During Critical Activities": ["Case 2", "Case 4", "Case 8", "Case 11"],
        "Team Communication Breakdowns on Rig Floor": ["Case 6", "Case 8"],
        "Delayed Corrective Actions on Known Risks": ["Case 10", "Case 12"],
    }
    return patterns


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DetectedClass:
    """Single class detection result"""
    class_id: str
    class_name: str
    sub_tags: List[str]
    evidence_references: List[str]  # Case IDs
    supporting_precursor_texts: List[str]
    provenance: str  # TRAINED_ML, RULE_BASED, etc.
    confidence_note: str  # Honest assessment


@dataclass
class SIFAssessment:
    """SIF-potential assessment"""
    level: str  # HIGH, MEDIUM, LOW
    rationale: str
    contributing_classes: List[str]
    historical_outcomes: List[str]  # Which cases with these classes had SIF events
    provenance: str  # Always RULE_BASED
    limitation: str


@dataclass
class LSRMapping:
    """IOGP LSR mapping (deferred)"""
    applicable_rules: List[str]
    status: str  # NOT_YET_IMPLEMENTED, DEFERRED_TO_PHASE_2, etc.
    provenance: str
    limitation: str


@dataclass
class AnalysisResult:
    """Complete analysis output"""
    input_text: str
    timestamp: str
    detected_classes: List[DetectedClass]
    sif_assessment: SIFAssessment
    lsr_mapping: LSRMapping
    supporting_evidence: Dict  # Detailed evidence linkage
    limitations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to serializable dict"""
        return {
            "input_text": self.input_text,
            "timestamp": self.timestamp,
            "detected_classes": [asdict(c) for c in self.detected_classes],
            "sif_assessment": asdict(self.sif_assessment),
            "lsr_mapping": asdict(self.lsr_mapping),
            "supporting_evidence": self.supporting_evidence,
            "limitations": self.limitations,
        }


# ============================================================================
# CORE ANALYZERS
# ============================================================================

class RootCauseAnalyzer:
    """
    Detects C1-C6 root-cause classes in input text.
    
    Strategy:
    - If ML model is available and validated: use ML
    - Else: use rule-based keyword matching
    """
    
    def __init__(self, schema: Dict, training_data: List[Dict], label_mapping: List[Dict]):
        self.schema = schema
        self.training_data = training_data
        self.label_mapping = label_mapping
        self.ml_model = None  # Will be loaded if available
        self.ml_available = False
        self.keyword_map = self._build_keyword_map()
        
    def _build_keyword_map(self) -> Dict[str, List[str]]:
        """
        Build class→keywords mapping from schema and training data.
        
        Returns: {
            "C1": ["training", "competency", "induction", "unskilled", "unapproved", ...],
            ...
        }
        """
        keywords_by_class = {
            "C1": [
                "training", "competency", "induction", "unskilled", "unapproved",
                "unverifiable", "certification", "certified", "knowledge", "hands-on",
            ],
            "C2": [
                "supervision", "mentor", "supervisor", "absent", "communication", "breakdown",
                "coordination", "unaware", "not informed", "no supervisor", "deficient leadership",
            ],
            "C3": [
                "inspection", "maintenance", "calibration", "testing", "deteriorated", "rusted",
                "worn", "missing records", "oem", "non-standard", "degradation", "broken",
            ],
            "C4": [
                "procedure", "permit", "risk assessment", "jsa", "skipped", "bypassed",
                "planning", "ignored", "no permit", "work permit", "hazard", "unsafe",
            ],
            "C5": [
                "barrier", "safety system", "defunct", "disabled", "isolated", "not installed",
                "missing", "failed", "omission", "bypassed", "non-functional", "removed",
            ],
            "C6": [
                "environment", "workspace", "hazard", "weather", "spillage", "flooding", "monsoon",
                "obstructed", "clutter", "combustible", "visible", "railing", "obstruction",
            ],
        }
        return keywords_by_class
    
    def analyze(self, text: str) -> Tuple[List[DetectedClass], str]:
        """
        Analyze text and return detected classes.
        
        Returns: (detected_classes, provenance)
        """
        text_lower = text.lower()
        detected_classes = []
        
        # Use rule-based approach (keyword matching)
        for class_id, keywords in self.keyword_map.items():
            # Count keyword matches
            matches = 0
            matched_keywords = []
            for kw in keywords:
                if kw in text_lower:
                    matches += 1
                    matched_keywords.append(kw)
            
            # If enough keywords match, this class is detected
            if matches >= 2:  # Require at least 2 keyword hits
                class_info = self._get_class_info(class_id)
                evidence_cases = class_info.get("supporting_cases", [])
                supporting_texts = self._get_supporting_texts(class_id)
                
                detected_classes.append(DetectedClass(
                    class_id=class_id,
                    class_name=class_info["class_name"],
                    sub_tags=[st["sub_tag_id"] for st in class_info.get("sub_tags", [])],
                    evidence_references=evidence_cases,
                    supporting_precursor_texts=supporting_texts,
                    provenance="RULE_BASED",
                    confidence_note=f"{matches} keyword matches: {', '.join(matched_keywords[:3])}..."
                ))
        
        return detected_classes, "RULE_BASED"
    
    def _get_class_info(self, class_id: str) -> Dict:
        """Get class info from schema"""
        for cls in self.schema["top_level_classes"]:
            if cls["class_id"] == class_id:
                return {
                    "class_name": cls["class_name"],
                    "sub_tags": cls.get("sub_tags", []),
                    "supporting_cases": [
                        sc["case_id"]
                        for subtag in cls.get("sub_tags", [])
                        for sc in subtag.get("supporting_cases", [])
                    ],
                }
        return {}
    
    def _get_supporting_texts(self, class_id: str, limit: int = 3) -> List[str]:
        """Get example precursor texts for a class"""
        texts = []
        for row in self.label_mapping:
            class_ids = row.get("class_ids", "").split("|")
            if class_id in class_ids and len(texts) < limit:
                texts.append(row["precursor_text"])
        return texts


class SIFAssessor:
    """
    Assesses SIF-potential from detected classes.
    
    Rule-based only (no ML, no negative baseline exists).
    """
    
    def __init__(self, schema: Dict, label_mapping: List[Dict]):
        self.schema = schema
        self.label_mapping = label_mapping
        self.historical_severity = self._build_severity_map()
    
    def _build_severity_map(self) -> Dict[str, List[str]]:
        """
        Build mapping: class → list of severities from historical data
        
        Returns: {
            "C1": ["Fatality", "Major injury", ...],
            "C4": ["Blowout", "Fatality", "Fatality", ...],
            ...
        }
        """
        severity_by_class = {}
        for row in self.label_mapping:
            class_ids = row.get("class_ids", "").split("|")
            severity = row.get("severity", "Unknown")
            for cid in class_ids:
                if cid not in severity_by_class:
                    severity_by_class[cid] = []
                severity_by_class[cid].append(severity)
        return severity_by_class
    
    def assess(self, detected_classes: List[str]) -> SIFAssessment:
        """
        Assess SIF-potential from detected classes.
        
        Heuristic:
        - HIGH: If any of {C4 (Procedure), C5 (Missing barriers)} detected
        - MEDIUM: If {C2 (Supervision), C3 (Maintenance)} detected
        - LOW: Otherwise
        """
        if not detected_classes:
            return SIFAssessment(
                level="LOW",
                rationale="No root-cause classes detected in input.",
                contributing_classes=[],
                historical_outcomes=[],
                provenance="RULE_BASED",
                limitation="This is a rule-based risk assessment without trained classifier.",
            )
        
        # High-risk classes (based on historical severity)
        high_risk_classes = ["C4", "C5"]
        medium_risk_classes = ["C2", "C3"]
        
        has_high_risk = any(c in detected_classes for c in high_risk_classes)
        has_medium_risk = any(c in detected_classes for c in medium_risk_classes)
        
        if has_high_risk:
            level = "HIGH"
            rationale = f"Detected high-risk class(es): {', '.join([c for c in detected_classes if c in high_risk_classes])}. " \
                        f"Historical data shows these are consistently associated with SIF events (blowouts, fatalities)."
            outcomes = []
            for c in detected_classes:
                if c in high_risk_classes:
                    outcomes.extend(self.historical_severity.get(c, []))
        elif has_medium_risk:
            level = "MEDIUM"
            rationale = f"Detected medium-risk class(es): {', '.join([c for c in detected_classes if c in medium_risk_classes])}. " \
                        f"May contribute to SIF if combined with other failures."
            outcomes = []
            for c in detected_classes:
                if c in medium_risk_classes:
                    outcomes.extend(self.historical_severity.get(c, []))
        else:
            level = "LOW"
            rationale = f"Detected lower-risk class(es): {', '.join(detected_classes)}. " \
                        f"May require investigation but not immediately SIF-threatening on their own."
            outcomes = []
            for c in detected_classes:
                outcomes.extend(self.historical_severity.get(c, []))
        
        return SIFAssessment(
            level=level,
            rationale=rationale,
            contributing_classes=detected_classes,
            historical_outcomes=list(set(outcomes)),  # Unique outcomes
            provenance="RULE_BASED",
            limitation="Assessment based on 12 high-severity OISD cases. Not validated against field operations or routine near-misses.",
        )


class EvidenceExtractor:
    """
    Links detected classes to supporting cases, precursor texts, patterns, OISD standards.
    
    All results are DETERMINISTIC_EXTRACTION (direct lookups, no inference).
    """
    
    def __init__(self, schema: Dict, label_mapping: List[Dict], patterns: Dict[str, List[str]]):
        self.schema = schema
        self.label_mapping = label_mapping
        self.patterns = patterns
    
    def extract(self, detected_classes: List[str]) -> Dict:
        """
        Extract evidence for detected classes.
        """
        evidence = {
            "supporting_cases": {},
            "example_precursor_texts": {},
            "recurring_patterns": [],
            "oisd_standards": [],
        }
        
        for class_id in detected_classes:
            class_info = self._get_class_details(class_id)
            
            # Supporting cases from schema
            supporting_cases = class_info.get("supporting_cases", [])
            evidence["supporting_cases"][class_id] = supporting_cases
            
            # Example precursor texts
            example_texts = self._get_example_texts_for_class(class_id)
            evidence["example_precursor_texts"][class_id] = example_texts
            
            # OISD standards (hardcoded from raw_research/4; would ideally be extracted)
            oisd_standards = self._get_oisd_standards_for_class(class_id)
            evidence["oisd_standards"].extend(oisd_standards)
        
        # Recurring patterns associated with detected classes
        for pattern_name, cases in self.patterns.items():
            for case in cases:
                for detected_class_cases in evidence["supporting_cases"].values():
                    if case in detected_class_cases:
                        if pattern_name not in evidence["recurring_patterns"]:
                            evidence["recurring_patterns"].append(pattern_name)
                        break
        
        return evidence
    
    def _get_class_details(self, class_id: str) -> Dict:
        """Get class details from schema"""
        for cls in self.schema["top_level_classes"]:
            if cls["class_id"] == class_id:
                supporting_cases = []
                for subtag in cls.get("sub_tags", []):
                    for sc in subtag.get("supporting_cases", []):
                        if sc["case_id"] not in supporting_cases:
                            supporting_cases.append(sc["case_id"])
                return {
                    "class_name": cls["class_name"],
                    "supporting_cases": supporting_cases,
                }
        return {}
    
    def _get_example_texts_for_class(self, class_id: str, limit: int = 2) -> List[str]:
        """Get example precursor texts for a class"""
        texts = []
        for row in self.label_mapping:
            class_ids = row.get("class_ids", "").split("|")
            if class_id in class_ids and len(texts) < limit:
                texts.append(row["precursor_text"])
        return texts
    
    def _get_oisd_standards_for_class(self, class_id: str) -> List[str]:
        """
        Get OISD standards associated with a class (hardcoded from raw_research/4).
        
        In Phase 2, this would be extracted from raw_research/4.OISD... Report.
        """
        oisd_mapping = {
            "C1": ["OISD-STD-176 (Training & Competency Requirements)"],
            "C2": ["OISD-STD-105 (Permit-to-Work System)"],
            "C3": ["OISD-GDN-206 Cl. 7.5.1(j) (Safety Inspection Standards)"],
            "C4": ["OISD-STD-174 Cl. 4.1 (Well Planning)", "OISD-STD-105 (Permit-to-Work)"],
            "C5": ["OISD-STD-174 Cl. 6.3.1(B) (BOP Stack BSR Requirements)"],
            "C6": ["OISD/PL/GC/01 Cl. 2.d (Pipeline Protection at River Crossings)"],
        }
        return oisd_mapping.get(class_id, [])


# ============================================================================
# MAIN ENGINE
# ============================================================================

class SIH26165AnalysisEngine:
    """
    Main orchestrator for SIF precursor analysis.
    """
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = repo_root
        self.schema = load_schema(repo_root)
        self.training_data = load_training_data(repo_root)
        self.label_mapping = load_label_mapping(repo_root)
        self.patterns = load_recurring_patterns(repo_root)
        
        self.root_cause_analyzer = RootCauseAnalyzer(
            self.schema, self.training_data, self.label_mapping
        )
        self.sif_assessor = SIFAssessor(self.schema, self.label_mapping)
        self.evidence_extractor = EvidenceExtractor(self.schema, self.label_mapping, self.patterns)
    
    def analyze(self, input_text: str) -> AnalysisResult:
        """
        Perform complete SIF precursor analysis.
        """
        timestamp = datetime.now().isoformat()
        
        # Step 1: Detect root causes (C1-C6)
        detected_classes, rc_provenance = self.root_cause_analyzer.analyze(input_text)
        class_ids = [c.class_id for c in detected_classes]
        
        # Step 2: Assess SIF potential
        sif_assessment = self.sif_assessor.assess(class_ids)
        
        # Step 3: Extract evidence
        evidence = self.evidence_extractor.extract(class_ids)
        
        # Step 4: LSR mapping (deferred)
        lsr_mapping = LSRMapping(
            applicable_rules=[],
            status="NOT_YET_IMPLEMENTED",
            provenance="DETERMINISTIC_EXTRACTION",
            limitation="IOGP LSR labels not present in repository. Requires reading full IOGP standards and manual assignment. See Phase 2.",
        )
        
        # Step 5: Compile limitations
        limitations = [
            "Dataset: 75 precursor texts from 12 high-severity OISD cases (all resulted in SIF events)",
            "Formal register bias: All training data is investigation reports, not field reports",
            "Class imbalance: C4 has 40% of data (30 examples), C6 has 9% (7 examples)",
            "No SIF-negative baseline: No 'normal operations' or 'minor near-miss resolved' examples",
            "Validation scope: Tested on 15-row val set; field-report performance unknown",
            "LSR mapping: Not implemented in Phase 1",
        ]
        
        return AnalysisResult(
            input_text=input_text,
            timestamp=timestamp,
            detected_classes=detected_classes,
            sif_assessment=sif_assessment,
            lsr_mapping=lsr_mapping,
            supporting_evidence=evidence,
            limitations=limitations,
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Test
    engine = SIH26165AnalysisEngine(repo_root=".")
    
    test_text = (
        "JSA was not conducted due to time pressure. The blind shear ram was not included in the BOP stack "
        "configuration because it was assumed to be an oil well, not gas. Trip tank was isolated during perforation."
    )
    
    result = engine.analyze(test_text)
    
    print("\n=== SIH26165 Analysis Result ===\n")
    print(f"Timestamp: {result.timestamp}")
    print(f"\nInput: {result.input_text[:100]}...\n")
    
    print("ROOT-CAUSE DETECTION:")
    for cls in result.detected_classes:
        print(f"  {cls.class_id}: {cls.class_name}")
        print(f"    Provenance: {cls.provenance}")
        print(f"    Confidence: {cls.confidence_note}")
        print(f"    Supporting cases: {', '.join(cls.evidence_references[:3])}")
    
    print(f"\nSIF ASSESSMENT:")
    print(f"  Level: {result.sif_assessment.level}")
    print(f"  Provenance: {result.sif_assessment.provenance}")
    print(f"  Rationale: {result.sif_assessment.rationale[:150]}...")
    
    print(f"\nKNOWN LIMITATIONS:")
    for lim in result.limitations:
        print(f"  - {lim}")
