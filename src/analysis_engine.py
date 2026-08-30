"""
analysis_engine.py — SIH26165 SIF Precursor Analysis Engine

Core pipeline:
1. Root-cause detection (C1-C6)
2. SIF-potential assessment
3. Evidence linkage
4. IOGP LSR mapping (deferred)

Provenance:
- TRAINED_ML
- RULE_BASED
- DETERMINISTIC_EXTRACTION

Detection design:
- Strong deficiency phrases have highest priority.
- Sentence-level detection supports multiple simultaneous classes.
- Generic keyword detection requires multiple relevant signals.
- Compliant statements are protected from false-positive detection.
- C1 + C4 can be detected simultaneously from the same report.
- Existing C1-C6 architecture is preserved.
"""

import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


# ============================================================================
# DATA LOADING
# ============================================================================

def load_schema(repo_root: str = ".") -> Dict:
    """Load labels_schema.json."""
    path = Path(repo_root) / "data" / "labels_schema.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_training_data(repo_root: str = ".") -> List[Dict]:
    """Load training_data.csv."""
    path = Path(repo_root) / "data" / "training_data.csv"

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def load_label_mapping(repo_root: str = ".") -> List[Dict]:
    """Load label_mapping.csv."""
    path = Path(repo_root) / "data" / "label_mapping.csv"

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def load_recurring_patterns(repo_root: str = ".") -> Dict[str, List[str]]:
    """
    Curated recurring-pattern -> case-id mapping.

    This is a deterministic hand-maintained lookup.
    It is not dynamically extracted from research documents.
    """

    return {
        "Blind Shear Ram Omission": [
            "Case 5",
            "Case 9",
        ],

        "Trip Tank Isolation During Perforation": [
            "Case 5",
            "Case 9",
        ],

        "Exceeding Trip Speeds": [
            "Case 5",
            "Case 9",
        ],

        "Absence of Supervision During Critical Activities": [
            "Case 2",
            "Case 4",
            "Case 8",
            "Case 11",
        ],

        "Team Communication Breakdowns on Rig Floor": [
            "Case 6",
            "Case 8",
        ],

        "Delayed Corrective Actions on Known Risks": [
            "Case 10",
            "Case 12",
        ],
    }


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DetectedClass:
    """Single class detection result."""

    class_id: str
    class_name: str
    sub_tags: List[str]
    evidence_references: List[str]
    supporting_precursor_texts: List[str]
    provenance: str
    confidence_note: str


@dataclass
class SIFAssessment:
    """SIF-potential assessment."""

    level: str
    rationale: str
    contributing_classes: List[str]
    historical_outcomes: List[str]
    provenance: str
    limitation: str


@dataclass
class LSRMapping:
    """IOGP LSR mapping."""

    applicable_rules: List[str]
    status: str
    provenance: str
    limitation: str


@dataclass
class AnalysisResult:
    """Complete analysis output."""

    input_text: str
    timestamp: str
    detected_classes: List[DetectedClass]
    sif_assessment: SIFAssessment
    lsr_mapping: LSRMapping
    supporting_evidence: Dict
    limitations: List[str]

    def to_dict(self) -> Dict:
        return {
            "input_text": self.input_text,
            "timestamp": self.timestamp,
            "detected_classes": [
                asdict(c)
                for c in self.detected_classes
            ],
            "sif_assessment": asdict(self.sif_assessment),
            "lsr_mapping": asdict(self.lsr_mapping),
            "supporting_evidence": self.supporting_evidence,
            "limitations": self.limitations,
        }


# ============================================================================
# ROOT CAUSE ANALYZER
# ============================================================================

class RootCauseAnalyzer:
    """
    Detects C1-C6 root-cause classes.

    Detection priority:

    1. Strong deficiency phrase
       Example:
           "no proper training"
           "JSA was skipped"

    2. Sentence-level deficiency detection
       Example:
           "The worker had inadequate training."
           "The approved procedure was not followed."

    3. Generic keyword detection
       Requires multiple relevant keywords plus a deficiency signal.

    Every C1-C6 class is evaluated independently, so multiple
    classes can be detected from the same input.
    """

    def __init__(
        self,
        schema: Dict,
        training_data: List[Dict],
        label_mapping: List[Dict],
    ):
        self.schema = schema
        self.training_data = training_data
        self.label_mapping = label_mapping

        # Reserved for future validated ML model.
        self.ml_model = None
        self.ml_available = False

        self.keyword_map = self._build_keyword_map()
        self.phrase_map = self._build_phrase_map()

        self._keyword_patterns = {
            class_id: [
                self._compile_pattern(keyword)
                for keyword in keywords
            ]
            for class_id, keywords in self.keyword_map.items()
        }

        self._phrase_patterns = {
            class_id: [
                self._compile_pattern(phrase)
                for phrase in phrases
            ]
            for class_id, phrases in self.phrase_map.items()
        }

        # Deficiency signals used by generic detection.
        self._deficiency_patterns = [
            self._compile_pattern(word)
            for word in [
                "no",
                "not",
                "without",
                "lack",
                "lacked",
                "failure",
                "fail",
                "failed",
                "missing",
                "bypass",
                "bypassed",
                "skip",
                "skipped",
                "ignore",
                "ignored",
                "absent",
                "defunct",
                "disabled",
                "non-functional",
                "nonfunctional",
                "non-standard",
                "unsafe",
                "inadequate",
                "insufficient",
                "untrained",
                "unskilled",
                "unaware",
                "deteriorated",
                "worn",
                "rusted",
                "broken",
                "defective",
                "omission",
                "omitted",
                "poor",
            ]
        ]

        # Words indicating that a safety practice was completed correctly.
        self._completion_patterns = [
            self._compile_pattern(word)
            for word in [
                "completed",
                "conducted",
                "provided",
                "verified",
                "followed",
                "obtained",
                "issued",
                "prepared",
                "performed",
                "installed",
                "functional",
                "operational",
                "adequate",
                "properly",
                "successfully",
                "valid",
                "approved",
                "as scheduled",
                "on schedule",
            ]
        ]

    # ------------------------------------------------------------------------
    # PATTERN COMPILATION
    # ------------------------------------------------------------------------

    @staticmethod
    def _compile_pattern(text: str) -> re.Pattern:
        """
        Compile literal text with safe word boundaries.

        Prevents accidental substring matches.
        """

        return re.compile(
            r"(?<!\w)" +
            re.escape(text.lower()) +
            r"(?!\w)",
            re.IGNORECASE
        )

    # ------------------------------------------------------------------------
    # KEYWORD MAP
    # ------------------------------------------------------------------------

    def _build_keyword_map(self) -> Dict[str, List[str]]:
        """Generic C1-C6 keyword map."""

        return {
            "C1": [
                "training",
                "competency",
                "induction",
                "unskilled",
                "untrained",
                "unapproved",
                "unverifiable",
                "certification",
                "certified",
                "knowledge",
                "hands-on",
                "hands on",
                "qualification",
                "qualified",
                "skill",
                "skills",
            ],

            "C2": [
                "supervision",
                "mentor",
                "supervisor",
                "communication",
                "breakdown",
                "coordination",
                "unaware",
                "not informed",
                "no supervisor",
                "deficient leadership",
                "leadership",
            ],

            "C3": [
                "inspection",
                "maintenance",
                "calibration",
                "testing",
                "deteriorated",
                "rusted",
                "worn",
                "missing records",
                "oem",
                "non-standard",
                "degradation",
                "broken",
                "defective",
                "repair",
            ],

            "C4": [
                "procedure",
                "permit",
                "risk assessment",
                "jsa",
                "job safety analysis",
                "skipped",
                "bypassed",
                "planning",
                "ignored",
                "no permit",
                "work permit",
                "permit to work",
                "hazard",
                "unsafe",
                "not followed",
                "not conducted",
            ],

            "C5": [
                "barrier",
                "safety system",
                "defunct",
                "disabled",
                "isolated",
                "not installed",
                "missing",
                "failed",
                "omission",
                "bypassed",
                "non-functional",
                "removed",
                "protective system",
                "safety barrier",
            ],

            "C6": [
                "environment",
                "workspace",
                "hazard",
                "weather",
                "spillage",
                "flooding",
                "monsoon",
                "obstructed",
                "clutter",
                "combustible",
                "visible",
                "railing",
                "obstruction",
                "housekeeping",
                "working area",
            ],
        }

    # ------------------------------------------------------------------------
    # STRONG PHRASE MAP
    # ------------------------------------------------------------------------

    def _build_phrase_map(self) -> Dict[str, List[str]]:
        """
        Strong phrases representing explicit deficiencies.

        A single strong phrase is sufficient to detect the class.
        """

        return {

            # =================================================================
            # C1 — TRAINING / COMPETENCY
            # =================================================================

            "C1": [
                "no proper training",
                "no proper training was provided",
                "no proper training was given",
                "no proper training was received",

                "no training",
                "no training was provided",
                "no training was given",
                "no training was received",

                "not trained",
                "was not trained",
                "were not trained",
                "was untrained",
                "were untrained",

                "worker was not trained",
                "workers were not trained",
                "personnel were not trained",
                "employee was not trained",

                "not properly trained",
                "was not properly trained",
                "were not properly trained",

                "not adequately trained",
                "was not adequately trained",
                "were not adequately trained",

                "insufficient training",
                "inadequate training",

                "lack of training",
                "lack of proper training",

                "lack of competency",
                "lack of competence",

                "insufficient competency",
                "inadequate competency",

                "without proper training",
                "without training",

                "no hands-on training",
                "no hands on training",

                "untrained worker",
                "untrained personnel",
                "untrained employee",

                "unskilled worker",
                "unskilled personnel",
                "unskilled employee",

                "worker lacked training",
                "worker lacked competency",
                "personnel lacked training",
                "personnel lacked competency",

                "training was inadequate",
                "training was insufficient",
                "training was not adequate",
                "training was not sufficient",

                "competency was inadequate",
                "competency was insufficient",

                "qualification was inadequate",
                "qualification was insufficient",
            ],

            # =================================================================
            # C2 — SUPERVISION / COMMUNICATION
            # =================================================================

            "C2": [
                "no supervision",
                "no supervisor",
                "without supervision",
                "lack of supervision",
                "inadequate supervision",
                "insufficient supervision",
                "supervision was absent",
                "supervisor was absent",
                "supervisor was not present",
                "no competent supervisor",

                "poor communication",
                "communication breakdown",
                "communication broke down",
                "lack of communication",
                "not informed",
                "was not informed",
                "were not informed",

                "poor coordination",
                "lack of coordination",
                "coordination breakdown",
                "inadequate coordination",

                "deficient leadership",
                "poor leadership",
                "lack of leadership",
            ],

            # =================================================================
            # C3 — INSPECTION / MAINTENANCE
            # =================================================================

            "C3": [
                "no inspection",
                "inspection was not conducted",
                "inspection was not carried out",
                "inspection was not completed",

                "not inspected",
                "was not inspected",
                "were not inspected",

                "inspection was missed",

                "maintenance was not performed",
                "maintenance was not carried out",
                "maintenance was not completed",

                "no maintenance",
                "lack of maintenance",
                "poor maintenance",
                "inadequate maintenance",

                "calibration was not performed",
                "calibration was not completed",
                "not calibrated",
                "was not calibrated",
                "were not calibrated",

                "failed inspection",
                "inspection failed",

                "equipment was not maintained",
                "equipment was poorly maintained",
                "equipment maintenance was inadequate",

                "equipment was deteriorated",
                "equipment was worn",
                "equipment was broken",
                "equipment was defective",
            ],

            # =================================================================
            # C4 — PROCEDURE / JSA / PERMIT / PLANNING
            # =================================================================

            "C4": [
                "jsa was skipped",
                "jsa was not conducted",
                "jsa was not completed",
                "jsa was not prepared",
                "jsa was not done",

                "job safety analysis was skipped",
                "job safety analysis was not conducted",
                "job safety analysis was not completed",
                "job safety analysis was not prepared",
                "job safety analysis was not done",

                "risk assessment was skipped",
                "risk assessment was not conducted",
                "risk assessment was not completed",
                "risk assessment was not prepared",
                "risk assessment was not done",

                "permit was not obtained",
                "permit was not issued",
                "permit was missing",

                "no permit was obtained",
                "no permit was issued",
                "no work permit",
                "no permit to work",

                "work was performed without a permit",
                "work was carried out without a permit",
                "work was conducted without a permit",
                "work was performed without permit",

                "procedure was bypassed",
                "procedure was not followed",
                "procedure was ignored",
                "procedure was skipped",

                "safety procedure was not followed",
                "safety procedure was bypassed",
                "safety procedure was ignored",
                "safety procedure was skipped",

                "permit to work was not obtained",
                "permit to work was not issued",
                "permit to work was bypassed",

                "planned procedure was not followed",
                "approved procedure was not followed",

                "work procedure was not followed",
                "work procedure was bypassed",

                "planning was inadequate",
                "planning was insufficient",
                "poor planning",
                "inadequate planning",

                "risk assessment was inadequate",
                "risk assessment was insufficient",

                "hazard was not identified",
                "hazards were not identified",
                "hazard identification was inadequate",
            ],

            # =================================================================
            # C5 — BARRIERS / SAFETY SYSTEMS
            # =================================================================

            "C5": [
                "safety barrier was missing",
                "safety barrier was removed",
                "safety barrier was bypassed",
                "safety barrier was disabled",

                "barrier was missing",
                "barrier was removed",
                "barrier was bypassed",
                "barrier was disabled",
                "barrier was not installed",

                "safety system was disabled",
                "safety system was bypassed",
                "safety system was not functional",
                "safety system was not functioning",

                "protective system was disabled",
                "protective system was bypassed",
                "protective system was not functional",

                "safety system failed",
                "safety barrier failed",

                "barrier was omitted",
                "barrier omission",
                "missing safety barrier",

                "not installed safety barrier",
            ],

            # =================================================================
            # C6 — ENVIRONMENT / WORKSPACE
            # =================================================================

            "C6": [
                "unsafe environment",
                "unsafe working environment",
                "unsafe workplace",
                "unsafe work environment",

                "oil spillage",
                "oil spill",
                "water flooding",

                "work area was obstructed",
                "workspace was obstructed",
                "working area was obstructed",

                "poor housekeeping",
                "poor workplace housekeeping",
                "cluttered work area",
                "cluttered workspace",

                "blocked walkway",
                "obstructed walkway",
                "obstructed access",

                "unsafe weather",
                "adverse weather conditions",

                "combustible material was present",
                "combustible materials were present",
            ],
        }

    # ------------------------------------------------------------------------
    # SENTENCE SPLITTING
    # ------------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        Split report text into individual sentences.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            text
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # ------------------------------------------------------------------------
    # ADEQUATE PRACTICE CHECK
    # ------------------------------------------------------------------------

    def _looks_compliant(self, sentence: str) -> bool:
        """
        Detect language indicating that a safety activity was completed.

        This is only used when there is no explicit strong deficiency phrase.
        """

        has_completion = any(
            pattern.search(sentence)
            for pattern in self._completion_patterns
        )

        has_deficiency = any(
            pattern.search(sentence)
            for pattern in self._deficiency_patterns
        )

        return has_completion and not has_deficiency

    # ------------------------------------------------------------------------
    # ANALYZE
    # ------------------------------------------------------------------------

    def analyze(
        self,
        text: str
    ) -> Tuple[List[DetectedClass], str]:
        """
        Analyze input and detect all applicable C1-C6 classes.

        Important:
        Each class is evaluated independently.

        Therefore:

            "JSA was skipped and the worker had no proper training."

        produces:

            C1
            C4
        """

        if not text or not text.strip():
            return [], "RULE_BASED"

        normalized_text = re.sub(
            r"\s+",
            " ",
            text.lower()
        ).strip()

        sentences = self._split_sentences(
            normalized_text
        )

        detected_classes = []

        # ====================================================================
        # CHECK EACH CLASS INDEPENDENTLY
        # ====================================================================

        for class_id in self.keyword_map.keys():

            keywords = self.keyword_map[class_id]
            keyword_patterns = self._keyword_patterns[class_id]

            strong_phrases = self.phrase_map.get(
                class_id,
                []
            )

            phrase_patterns = self._phrase_patterns.get(
                class_id,
                []
            )

            # ----------------------------------------------------------------
            # 1. GLOBAL STRONG PHRASE MATCH
            # ----------------------------------------------------------------

            matched_phrases = []

            for phrase, pattern in zip(
                strong_phrases,
                phrase_patterns
            ):

                if pattern.search(normalized_text):
                    matched_phrases.append(
                        phrase
                    )

            detected_by_phrase = bool(
                matched_phrases
            )

            # ----------------------------------------------------------------
            # 2. SENTENCE-LEVEL DETECTION
            # ----------------------------------------------------------------

            sentence_match_details = []

            for sentence in sentences:

                # Strong phrase in this sentence.
                sentence_phrases = []

                for phrase, pattern in zip(
                    strong_phrases,
                    phrase_patterns
                ):

                    if pattern.search(sentence):
                        sentence_phrases.append(
                            phrase
                        )

                if sentence_phrases:
                    sentence_match_details.extend(
                        sentence_phrases
                    )
                    continue

                # Generic keywords in this sentence.
                sentence_keywords = []

                for keyword, pattern in zip(
                    keywords,
                    keyword_patterns
                ):

                    if pattern.search(sentence):
                        sentence_keywords.append(
                            keyword
                        )

                # Need at least 2 related keywords.
                if len(sentence_keywords) < 2:
                    continue

                # Do not classify clearly compliant statements.
                if self._looks_compliant(sentence):
                    continue

                # Need an actual deficiency signal.
                sentence_has_deficiency = any(
                    pattern.search(sentence)
                    for pattern in self._deficiency_patterns
                )

                if sentence_has_deficiency:
                    sentence_match_details.extend(
                        sentence_keywords
                    )

            # ----------------------------------------------------------------
            # 3. WHOLE-TEXT GENERIC KEYWORD DETECTION
            # ----------------------------------------------------------------

            matched_keywords = []

            for keyword, pattern in zip(
                keywords,
                keyword_patterns
            ):

                if pattern.search(normalized_text):
                    matched_keywords.append(
                        keyword
                    )

            keyword_matches = len(
                matched_keywords
            )

            detected_by_keywords = False

            if keyword_matches >= 2:

                whole_text_has_deficiency = any(
                    pattern.search(normalized_text)
                    for pattern in self._deficiency_patterns
                )

                # If the complete input is clearly compliant,
                # do not allow generic keywords to create a false positive.
                whole_text_is_compliant = (
                    self._looks_compliant(normalized_text)
                )

                if (
                    whole_text_has_deficiency
                    and not whole_text_is_compliant
                ):
                    detected_by_keywords = True

            # ----------------------------------------------------------------
            # 4. FINAL CLASS DECISION
            # ----------------------------------------------------------------

            detected = (
                detected_by_phrase
                or bool(sentence_match_details)
                or detected_by_keywords
            )

            if not detected:
                continue

            # ----------------------------------------------------------------
            # CLASS INFORMATION
            # ----------------------------------------------------------------

            class_info = self._get_class_info(
                class_id
            )

            evidence_cases = class_info.get(
                "supporting_cases",
                []
            )

            supporting_texts = (
                self._get_supporting_texts(
                    class_id
                )
            )

            # ----------------------------------------------------------------
            # CONFIDENCE / EXPLANATION
            # ----------------------------------------------------------------

            evidence_parts = []

            unique_phrases = list(
                dict.fromkeys(
                    matched_phrases
                )
            )

            if unique_phrases:

                evidence_parts.append(
                    f"{len(unique_phrases)} strong phrase match"
                    f"{'es' if len(unique_phrases) != 1 else ''}: "
                    + ", ".join(
                        unique_phrases[:5]
                    )
                )

            unique_sentence_matches = list(
                dict.fromkeys(
                    sentence_match_details
                )
            )

            if unique_sentence_matches:

                evidence_parts.append(
                    "sentence-level deficiency match: "
                    + ", ".join(
                        unique_sentence_matches[:5]
                    )
                )

            if detected_by_keywords:

                evidence_parts.append(
                    f"{keyword_matches} keyword matches: "
                    + ", ".join(
                        matched_keywords[:5]
                    )
                )

            confidence_note = "; ".join(
                evidence_parts
            )

            # ----------------------------------------------------------------
            # APPEND DETECTION
            # ----------------------------------------------------------------

            detected_classes.append(
                DetectedClass(
                    class_id=class_id,
                    class_name=class_info.get(
                        "class_name",
                        class_id
                    ),
                    sub_tags=[
                        st["sub_tag_id"]
                        for st in class_info.get(
                            "sub_tags",
                            []
                        )
                    ],
                    evidence_references=evidence_cases,
                    supporting_precursor_texts=supporting_texts,
                    provenance="RULE_BASED",
                    confidence_note=confidence_note,
                )
            )

        return detected_classes, "RULE_BASED"

    # ------------------------------------------------------------------------
    # GET CLASS INFO
    # ------------------------------------------------------------------------

    def _get_class_info(
        self,
        class_id: str
    ) -> Dict:
        """
        Get class information from labels_schema.json.

        Supporting case IDs are deduplicated.
        """

        for cls in self.schema.get(
            "top_level_classes",
            []
        ):

            if cls.get("class_id") != class_id:
                continue

            supporting_cases = []

            for subtag in cls.get(
                "sub_tags",
                []
            ):

                for supporting_case in subtag.get(
                    "supporting_cases",
                    []
                ):

                    case_id = supporting_case.get(
                        "case_id"
                    )

                    if (
                        case_id
                        and case_id not in supporting_cases
                    ):
                        supporting_cases.append(
                            case_id
                        )

            return {
                "class_name": cls.get(
                    "class_name",
                    class_id
                ),
                "sub_tags": cls.get(
                    "sub_tags",
                    []
                ),
                "supporting_cases": supporting_cases,
            }

        return {}

    # ------------------------------------------------------------------------
    # SUPPORTING TEXTS
    # ------------------------------------------------------------------------

    def _get_supporting_texts(
        self,
        class_id: str,
        limit: int = 3
    ) -> List[str]:
        """Get example precursor texts for a class."""

        texts = []

        for row in self.label_mapping:

            class_ids = (
                row.get(
                    "class_ids",
                    ""
                )
                .split("|")
            )

            if (
                class_id in class_ids
                and len(texts) < limit
            ):

                precursor_text = row.get(
                    "precursor_text",
                    ""
                )

                if precursor_text:
                    texts.append(
                        precursor_text
                    )

        return texts


# ============================================================================
# SIF ASSESSOR
# ============================================================================

class SIFAssessor:
    """
    Rule-based SIF-potential assessment.

    HIGH:
        C4, C5

    MEDIUM:
        C2, C3

    LOW:
        C1, C6 or nothing detected
    """

    def __init__(
        self,
        schema: Dict,
        label_mapping: List[Dict]
    ):
        self.schema = schema
        self.label_mapping = label_mapping
        self.historical_severity = (
            self._build_severity_map()
        )

    def _build_severity_map(
        self
    ) -> Dict[str, List[str]]:

        severity_by_class = {}

        for row in self.label_mapping:

            class_ids = (
                row.get(
                    "class_ids",
                    ""
                )
                .split("|")
            )

            severity = row.get(
                "severity",
                "Unknown"
            )

            for class_id in class_ids:

                if not class_id:
                    continue

                severity_by_class.setdefault(
                    class_id,
                    []
                ).append(
                    severity
                )

        return severity_by_class

    def assess(
        self,
        detected_classes: List[str]
    ) -> SIFAssessment:

        if not detected_classes:

            return SIFAssessment(
                level="LOW",
                rationale=(
                    "No root-cause classes detected in input."
                ),
                contributing_classes=[],
                historical_outcomes=[],
                provenance="RULE_BASED",
                limitation=(
                    "This is a rule-based risk assessment "
                    "without a trained classifier."
                ),
            )

        high_risk_classes = [
            "C4",
            "C5",
        ]

        medium_risk_classes = [
            "C2",
            "C3",
        ]

        has_high_risk = any(
            class_id in detected_classes
            for class_id in high_risk_classes
        )

        has_medium_risk = any(
            class_id in detected_classes
            for class_id in medium_risk_classes
        )

        if has_high_risk:

            contributing = [
                class_id
                for class_id in detected_classes
                if class_id in high_risk_classes
            ]

            level = "HIGH"

            rationale = (
                f"Detected high-risk class(es): "
                f"{', '.join(contributing)}. "
                "Historical data associates these classes "
                "with serious incidents including blowouts "
                "and fatalities."
            )

        elif has_medium_risk:

            contributing = [
                class_id
                for class_id in detected_classes
                if class_id in medium_risk_classes
            ]

            level = "MEDIUM"

            rationale = (
                f"Detected medium-risk class(es): "
                f"{', '.join(contributing)}. "
                "These may contribute to SIF when combined "
                "with other failures."
            )

        else:

            contributing = detected_classes

            level = "LOW"

            rationale = (
                f"Detected lower-risk class(es): "
                f"{', '.join(contributing)}. "
                "These may require investigation but are "
                "not classified as high SIF potential by "
                "the current rule set."
            )

        outcomes = []

        for class_id in contributing:

            outcomes.extend(
                self.historical_severity.get(
                    class_id,
                    []
                )
            )

        return SIFAssessment(
            level=level,
            rationale=rationale,
            contributing_classes=detected_classes,
            historical_outcomes=sorted(
                set(outcomes)
            ),
            provenance="RULE_BASED",
            limitation=(
                "Assessment based on 12 high-severity OISD cases. "
                "Not validated against field operations or routine "
                "near-misses."
            ),
        )


# ============================================================================
# EVIDENCE EXTRACTOR
# ============================================================================

class EvidenceExtractor:
    """
    Links detected classes to:

    - supporting cases
    - example precursor texts
    - recurring patterns
    - OISD standards
    """

    def __init__(
        self,
        schema: Dict,
        label_mapping: List[Dict],
        patterns: Dict[str, List[str]]
    ):
        self.schema = schema
        self.label_mapping = label_mapping
        self.patterns = patterns

    def extract(
        self,
        detected_classes: List[str]
    ) -> Dict:

        evidence = {
            "supporting_cases": {},
            "example_precursor_texts": {},
            "recurring_patterns": [],
            "oisd_standards": [],
        }

        for class_id in detected_classes:

            class_info = self._get_class_details(
                class_id
            )

            supporting_cases = class_info.get(
                "supporting_cases",
                []
            )

            evidence[
                "supporting_cases"
            ][class_id] = supporting_cases

            evidence[
                "example_precursor_texts"
            ][class_id] = (
                self._get_example_texts_for_class(
                    class_id
                )
            )

            for standard in (
                self._get_oisd_standards_for_class(
                    class_id
                )
            ):

                if standard not in evidence[
                    "oisd_standards"
                ]:

                    evidence[
                        "oisd_standards"
                    ].append(
                        standard
                    )

        # ------------------------------------------------------------
        # Recurring patterns
        # ------------------------------------------------------------

        for pattern_name, cases in self.patterns.items():

            for case in cases:

                for detected_class_cases in evidence[
                    "supporting_cases"
                ].values():

                    if case in detected_class_cases:

                        if pattern_name not in evidence[
                            "recurring_patterns"
                        ]:

                            evidence[
                                "recurring_patterns"
                            ].append(
                                pattern_name
                            )

                        break

        return evidence

    def _get_class_details(
        self,
        class_id: str
    ) -> Dict:

        for cls in self.schema.get(
            "top_level_classes",
            []
        ):

            if cls.get("class_id") != class_id:
                continue

            supporting_cases = []

            for subtag in cls.get(
                "sub_tags",
                []
            ):

                for supporting_case in subtag.get(
                    "supporting_cases",
                    []
                ):

                    case_id = supporting_case.get(
                        "case_id"
                    )

                    if (
                        case_id
                        and case_id not in supporting_cases
                    ):

                        supporting_cases.append(
                            case_id
                        )

            return {
                "class_name": cls.get(
                    "class_name",
                    class_id
                ),
                "supporting_cases": supporting_cases,
            }

        return {}

    def _get_example_texts_for_class(
        self,
        class_id: str,
        limit: int = 2
    ) -> List[str]:

        texts = []

        for row in self.label_mapping:

            class_ids = (
                row.get(
                    "class_ids",
                    ""
                )
                .split("|")
            )

            if (
                class_id in class_ids
                and len(texts) < limit
            ):

                precursor_text = row.get(
                    "precursor_text",
                    ""
                )

                if precursor_text:
                    texts.append(
                        precursor_text
                    )

        return texts

    def _get_oisd_standards_for_class(
        self,
        class_id: str
    ) -> List[str]:

        oisd_mapping = {

            "C1": [
                "OISD-STD-176 "
                "(Training & Competency Requirements)"
            ],

            "C2": [
                "OISD-STD-105 "
                "(Permit-to-Work System)"
            ],

            "C3": [
                "OISD-GDN-206 Cl. 7.5.1(j) "
                "(Safety Inspection Standards)"
            ],

            "C4": [
                "OISD-STD-174 Cl. 4.1 "
                "(Well Planning)",
                "OISD-STD-105 "
                "(Permit-to-Work)"
            ],

            "C5": [
                "OISD-STD-174 Cl. 6.3.1(B) "
                "(BOP Stack BSR Requirements)"
            ],

            "C6": [
                "OISD/PL/GC/01 Cl. 2.d "
                "(Pipeline Protection at River Crossings)"
            ],
        }

        return oisd_mapping.get(
            class_id,
            []
        )


# ============================================================================
# MAIN ENGINE
# ============================================================================

class SIH26165AnalysisEngine:
    """Main orchestrator for SIH26165 SIF precursor analysis."""

    def __init__(
        self,
        repo_root: str = "."
    ):
        self.repo_root = repo_root

        self.schema = load_schema(
            repo_root
        )

        self.training_data = load_training_data(
            repo_root
        )

        self.label_mapping = load_label_mapping(
            repo_root
        )

        self.patterns = load_recurring_patterns(
            repo_root
        )

        self.root_cause_analyzer = RootCauseAnalyzer(
            self.schema,
            self.training_data,
            self.label_mapping,
        )

        self.sif_assessor = SIFAssessor(
            self.schema,
            self.label_mapping,
        )

        self.evidence_extractor = EvidenceExtractor(
            self.schema,
            self.label_mapping,
            self.patterns,
        )

    def analyze(
        self,
        input_text: str
    ) -> AnalysisResult:

        timestamp = datetime.now().isoformat()

        # ================================================================
        # STEP 1 — ROOT CAUSE DETECTION
        # ================================================================

        detected_classes, _ = (
            self.root_cause_analyzer.analyze(
                input_text
            )
        )

        class_ids = [
            detected_class.class_id
            for detected_class in detected_classes
        ]

        # ================================================================
        # STEP 2 — SIF ASSESSMENT
        # ================================================================

        sif_assessment = (
            self.sif_assessor.assess(
                class_ids
            )
        )

        # ================================================================
        # STEP 3 — EVIDENCE EXTRACTION
        # ================================================================

        evidence = (
            self.evidence_extractor.extract(
                class_ids
            )
        )

        # ================================================================
        # STEP 4 — IOGP LSR
        # ================================================================

        lsr_mapping = LSRMapping(
            applicable_rules=[],
            status="NOT_YET_IMPLEMENTED",
            provenance="DETERMINISTIC_EXTRACTION",
            limitation=(
                "IOGP LSR labels are not present in the repository. "
                "Requires reading the full IOGP standards and manual "
                "assignment. See Phase 2."
            ),
        )

        # ================================================================
        # STEP 5 — LIMITATIONS
        # ================================================================

        limitations = [

            "Dataset: 75 precursor texts from 12 high-severity "
            "OISD cases (all resulted in SIF events)",

            "Formal register bias: All training data is investigation "
            "reports, not field reports",

            "Class imbalance: C4 has 40% of data (30 examples), "
            "C6 has 9% (7 examples)",

            "No SIF-negative baseline: No normal-operation or "
            "minor-near-miss baseline examples",

            "Validation scope: Tested on 15-row validation set; "
            "field-report performance is unknown",

            "Detection is rule-based keyword/phrase matching "
            "rather than a validated ML/NLP classifier",

            "Strong deficiency phrases and sentence-level matching "
            "allow simultaneous C1-C6 multi-label detection",

            "Compliant-practice protection is heuristic and may "
            "not perfectly handle very long documents containing "
            "both compliant and deficient statements",

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
# TEST / ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    engine = SIH26165AnalysisEngine(
        repo_root="."
    )

    test_cases = [

        # ------------------------------------------------------------
        # CORE C1 + C4 TEST
        # ------------------------------------------------------------

        (
            "JSA was skipped and the worker had no proper training."
        ),

        # ------------------------------------------------------------
        # C1
        # ------------------------------------------------------------

        (
            "The worker was not properly trained before performing "
            "the critical task."
        ),

        # ------------------------------------------------------------
        # C4
        # ------------------------------------------------------------

        (
            "The JSA was not conducted before the job started."
        ),

        # ------------------------------------------------------------
        # C1 + C4
        # ------------------------------------------------------------

        (
            "The employee had insufficient training and the approved "
            "work procedure was not followed."
        ),

        (
            "Work was carried out without a valid permit and the risk "
            "assessment was not completed."
        ),

        # ------------------------------------------------------------
        # COMPLIANT C1 — MUST NOT DETECT C1
        # ------------------------------------------------------------

        (
            "All crew training and certification records were completed "
            "and verified before the shift began."
        ),

        # ------------------------------------------------------------
        # COMPLIANT C4 — MUST NOT DETECT C4
        # ------------------------------------------------------------

        (
            "The JSA was completed and the approved procedure was "
            "followed before work started."
        ),

        # ------------------------------------------------------------
        # MULTI-CLASS REGRESSION
        # ------------------------------------------------------------

        (
            "The worker was untrained, supervision was absent, the JSA "
            "was skipped, and the safety barrier was bypassed."
        ),
    ]

    print()
    print("=" * 80)
    print("SIH26165 ANALYSIS ENGINE — REGRESSION TEST")
    print("=" * 80)

    for index, test_text in enumerate(
        test_cases,
        start=1
    ):

        result = engine.analyze(
            test_text
        )

        print()
        print("-" * 80)
        print(f"TEST {index}")
        print("-" * 80)

        print(
            f"Input: {result.input_text}"
        )

        print()

        print("Detected classes:")

        if not result.detected_classes:

            print("  NONE")

        else:

            for cls in result.detected_classes:

                print(
                    f"  {cls.class_id}: "
                    f"{cls.class_name}"
                )

                print(
                    f"      Provenance: "
                    f"{cls.provenance}"
                )

                print(
                    f"      Confidence: "
                    f"{cls.confidence_note}"
                )

        print()

        print(
            f"SIF level: "
            f"{result.sif_assessment.level}"
        )

        print(
            f"SIF provenance: "
            f"{result.sif_assessment.provenance}"
        )

    print()
    print("=" * 80)
    print("REGRESSION TEST COMPLETE")
    print("=" * 80)