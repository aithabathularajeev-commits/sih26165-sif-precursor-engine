# SIH26165 P0 BUILD SPECIFICATION
**Status:** P0 (3-day demo vertical slice)  
**Date:** 2026-08-30  
**User:** HSE Supervisor/Auditor  
**Principle:** Small, polished, explainable, technically honest

---

## 1. FINAL ARCHITECTURE

```
User Input
    ↓
[Streamlit UI]
    ↓
[SIH26165AnalysisEngine]
    ├─ Parse user precursor text
    ├─ Run [C1-C6 Root-Cause Analyzer]
    │   ├─ ML-based (if trained model exists)
    │   └─ Rule-based fallback (deterministic keyword matching)
    ├─ Run [SIF-Potential Assessor]
    │   └─ Rule-based (no negative labels exist; transparent)
    ├─ Run [IOGP LSR Mapper]
    │   └─ Deterministic extraction from repository evidence
    └─ Run [Evidence Linker]
        └─ Cross-reference to case_id, precursor_text, labels_schema.json
    ↓
Output: {
  "classes_detected": [{"class": "C1", "sub_tags": [...], "evidence": [...], "provenance": "RULE_BASED"}],
  "sif_potential": {"level": "HIGH|MEDIUM|LOW", "rationale": "...", "provenance": "RULE_BASED"},
  "lsr_applicability": {"applicable_rules": [...], "provenance": "DETERMINISTIC_EXTRACTION"},
  "supporting_cases": [{"case_id": "Case 1", "oisd_id": "...", "precursor_match": "..."}],
  "limitations": ["Dataset trained on formal investigation reports", "No field-report validation yet", ...]
}
```

---

## 2. COMPONENTS

### **2.1 C1-C6 Root-Cause Analyzer**

**Purpose:** Detect which of the 6 SIF-precursor classes are present in user input  

**Data Sources:**
- `data/labels_schema.json` — class definitions, sub-tags, supporting evidence, examples
- `data/label_mapping.csv` — 75 labeled examples (precursor text + class assignments)
- `data/training_data.csv` — raw precursor text (used for fallback keyword matching)

**Implementation Strategy:**

1. **Try: ML-based (if model can be trained honestly in <4 hours)**
   - Load train.csv (60 rows)
   - Train a simple scikit-learn classifier (TF-IDF + Logistic Regression or Naive Bayes)
   - Use class_weight='balanced' to handle imbalance (C4: 30 examples, C6: 7 examples)
   - Validate on val.csv (15 rows)
   - Report per-class metrics: precision, recall, F1 (not macro-accuracy, which hides rare-class failures)
   - **IF** validation metrics on real data look reasonable (F1 > 0.6 for C4, >0.4 for rare classes): keep ML
   - **ELSE:** Fall back to rule-based

2. **Fallback: Rule-based (deterministic keyword matching from labels_schema.json + evidence examples)**
   - Extract representative keywords from each class's supporting cases
   - Build keyword→class mapping: `C1: ["inadequate", "no training", "unapproved", "induction", ...]`
   - For each user input, find exact/fuzzy keyword matches
   - Return matched classes with evidence references
   - Mark as `RULE_BASED` provenance

**Provenance Labels:**
- `TRAINED_ML` — if a trained model is used and validates with F1 ≥ 0.5
- `RULE_BASED` — if keyword matching is used
- `RULE_BASED_FALLBACK` — if ML model failed validation; fell back to rules

**Constraints:**
- Do NOT claim accuracy/confidence unless validated on test data
- Report F1, precision, recall per class (not macro-accuracy)
- Flag rare classes (C5, C6) as having high uncertainty
- Do NOT return a single binary confidence score; return per-class explainability

---

### **2.2 SIF-Potential Assessor**

**Purpose:** Assess whether detected precursors constitute a SIF (Safety Integrity Function) threat  

**Data Sources:**
- `data/labels_schema.json` — class severity mappings (which classes are highest-risk)
- `data/raw_research/1.Safety Analysis...` — RCA categories + incident severity mappings
- `data/label_mapping.csv` — historical severity outcomes (fatality, blowout, major injury)

**Implementation Strategy (RULE-BASED ONLY):**

Reason: Dataset contains only SIF-positive cases (all 12 incidents resulted in fatality/major injury/blowout). No SIF-negative baseline exists (no "normal operation" or "minor near-miss resolved" data). Cannot build a supervised binary classifier without negative examples.

Instead, build a **transparent risk-ranking system**:

```python
def assess_sif_potential(detected_classes: List[str]) -> Dict:
    """
    Map detected root-cause classes to historical severity outcomes.
    Return: {"level": "HIGH|MEDIUM|LOW", "rationale": "...", "provenance": "RULE_BASED"}
    """
    # SIF-High classes (always associated with catastrophic outcomes in data):
    # C4 (Procedure/Permit bypass) → 30/75 examples, all led to major incidents
    # C5 (Missing safety barriers) → 10/75 examples, all led to blowouts/fatalities
    
    # SIF-Medium classes:
    # C2 (Supervision failure) → 9/75 examples, led to fatalities
    # C3 (Maintenance failure) → 13/75 examples, led to losses
    
    # SIF-Lower classes:
    # C1 (Training gaps) → 9/75 examples, varies
    # C6 (Environmental) → 7/75 examples, varies
    
    # Heuristic: If ANY of {C4, C5, C2} detected → SIF-HIGH
    #            Else if any of {C3} + C1 → SIF-MEDIUM
    #            Else → SIF-LOW
```

**Explicitly NOT to do:**
- Do NOT train a binary SIF classifier (no negative data)
- Do NOT return a false "probability" or "confidence" score
- Do NOT claim this is ML-based
- Do NOT extrapolate beyond the 12-case dataset

**Provenance Label:** `RULE_BASED` (always, for this component)

**Known Limitation:** "This assessment is based on 12 high-severity OISD cases (all resulted in SIF events). It has NOT been validated against Oil India's broader incident database or routine near-misses. Risk ranking may not generalize to field operations."

---

### **2.3 IOGP Life-Saving Rule (LSR) Mapper**

**Purpose:** Map detected precursors to applicable IOGP Life-Saving Rules  

**Data Sources:**
- `data/raw_research/4.OISD Safety Standards...` — OISD standard violations frequency (but NOT IOGP LSR mappings)
- `data/raw_research/5.Bridging Safety Standards...` — mentions IOGP LSR but provides NO explicit labels
- `data/labels_schema.json` — case evidence (but no LSR column)

**Implementation Strategy (DETERMINISTIC EXTRACTION ONLY):**

Reason: No LSR labels exist in the repository. The repo acknowledges IOGP LSR exists but does not map it to any cases or precursors.

**DECISION: Do NOT fabricate LSR mappings.** Instead:

```python
def get_lsr_applicability(detected_classes: List[str]) -> Dict:
    """
    Return: {
      "applicable_rules": [],
      "limitation": "IOGP LSR mappings not present in current dataset. See raw_research/5 for engineering requirements gap analysis.",
      "provenance": "DETERMINISTIC_EXTRACTION",
      "status": "NOT_YET_IMPLEMENTED"
    }
    """
    return {
        "applicable_rules": [],
        "note": "LSR mapping requires reading full IOGP standards and manual assignment to case data. This was not present in the repository and is out of scope for P0 demo.",
        "provenance": "DETERMINISTIC_EXTRACTION",
        "status": "DEFERRED_TO_PHASE_2"
    }
```

**Provenance Label:** `DETERMINISTIC_EXTRACTION` (informational; not performing extraction, just documenting absence)

**Known Limitation:** "IOGP LSR mappings are not present in the current dataset. Phase 2 implementation requires reading IOGP standards documentation and manually assigning labels to each case."

---

### **2.4 Evidence Linker**

**Purpose:** Link detected classes to supporting cases, precursor text, OISD standards  

**Data Sources:**
- `data/labels_schema.json` — case_reference, evidence_linking per class
- `data/training_data.csv` — all 75 precursor text examples
- `data/label_mapping.csv` — row↔class mappings
- `data/raw_research/2.Recurring Patterns...` — 6 documented patterns

**Implementation Strategy (DETERMINISTIC_EXTRACTION):**

```python
def link_evidence(detected_classes: List[str]) -> Dict:
    """
    For each detected class:
    - Return supporting cases (from labels_schema.json)
    - Return precursor text examples (from training_data.csv)
    - Return recurring patterns (from raw_research/2)
    - Return OISD standards (from raw_research/4)
    
    Provenance: DETERMINISTIC_EXTRACTION (direct lookups, no inference)
    """
    evidence = {}
    for cls in detected_classes:
        cases = schema["top_level_classes"][cls]["supporting_cases"]
        patterns = lookup_patterns_by_class(cls)  # from raw_research/2
        oisd_standards = lookup_oisd_by_class(cls)  # from raw_research/4
        
        evidence[cls] = {
            "supporting_cases": cases,
            "example_precursor_texts": [row["precursor_text"] for row in training_data if row["class_ids"].startswith(cls)],
            "recurring_patterns": patterns,
            "violated_oisd_standards": oisd_standards,
            "provenance": "DETERMINISTIC_EXTRACTION"
        }
    return evidence
```

**Provenance Label:** `DETERMINISTIC_EXTRACTION` (always; direct lookups from source files)

---

## 3. DATA SOURCES (By Component)

| Component | Data File | Usage | Modification |
|---|---|---|---|
| **All** | labels_schema.json | Class definitions, evidence, case refs | None (read-only) |
| **All** | training_data.csv | Raw precursor text | None (read-only) |
| **All** | label_mapping.csv | Class→row mappings | None (read-only) |
| **C1-C6 Analyzer** | train.csv, val.csv | ML training/validation (if ML approach) | Generate if needed; else use rules only |
| **SIF-Potential** | label_mapping.csv | Historical severity per class | None (read-only) |
| **Evidence Linker** | raw_research/* | Case details, patterns, standards | None (read-only) |
| **Synthetic demo** | synthetic_data/oil-india-near-miss-mockups.md | Clearly labeled demo examples only | Mark as SYNTHETIC; never train on |

---

## 4. IMPLEMENTATION APPROACH

### **What IS ML (if trained):**
- C1-C6 root-cause classifier: TF-IDF + Logistic Regression (if honest validation metrics ≥ 0.5 F1)
- **ONLY IF:** validation shows >50% F1 on real test data
- **MUST report:** per-class precision, recall, F1 (not macro-accuracy)
- **MUST label:** `TRAINED_ML` provenance
- **MUST warn:** "Dataset small (75 examples); validated on 15-row test set; may not generalize to field operations"

### **What IS Rule-Based:**
- Keyword matching for C1-C6 (fallback if ML fails, or as primary if ML is not pursued)
- SIF-potential assessment (heuristic risk-ranking, no ML alternative due to no negative labels)
- Evidence extraction (deterministic lookups)

### **What IS Deterministic:**
- Evidence linkage (case→precursor→standard mappings)
- OISD standard extraction
- Recurring pattern detection (keyword lookups)

### **What IS SKIPPED (honestly):**
- Binary SIF classifier (no negative labels to train on)
- IOGP LSR mapping (not present in repository; would require external IOGP standard reading)
- Field-register adaptation (no field-report samples to learn from)
- Confidence scoring (too small dataset; will not be reliable)

---

## 5. DEMO WORKFLOW

### **User Perspective:**

1. Open Streamlit app
2. Paste or upload a precursor incident report (text)
3. Click "Analyze"
4. Receive output:
   ```
   === SIH26165 Safety Precursor Analyzer ===
   
   INPUT REPORT:
   [user's text]
   
   ─── ROOT-CAUSE ANALYSIS (C1-C6) ───
   ✓ C4: Procedure/Permit Bypass (RULE_BASED)
     Sub-tags: C4-S1 (Skipped Risk Assessment), C4-S2 (No Work Permit)
     Supporting cases: Case 1, Case 5, Case 9
     Example precursor: "JSA was skipped due to time pressure"
   
   ✓ C5: Missing Safety Barriers (RULE_BASED)
     Sub-tags: C5-S1 (Defunct/Disabled Barrier)
     Supporting cases: Case 5, Case 9
     Example precursor: "Blind shear ram was not installed on BOP stack"
   
   ─── SIF-POTENTIAL ASSESSMENT ───
   Level: HIGH (RULE_BASED)
   Rationale: C4 + C5 detected. Historical data shows C4 + C5 together led to blowouts in Cases 5, 9.
   
   ⚠ LIMITATION: This assessment is based on 12 high-severity OISD cases. Not validated against field operations.
   
   ─── EVIDENCE LINEAGE ───
   Detected classes source: RULE_BASED keyword matching from labels_schema.json
   OISD standards violated: OISD-STD-174 Cl. 6.3.1(B), OISD-STD-174 Cl. 6.8(V)
   Recurring patterns: "Omission of Blind Shear Ram" (Cases 5, 9), "Operating with Isolated Trip Tank" (Cases 5, 9)
   
   ─── IOGP LSR MAPPING ───
   Status: NOT YET IMPLEMENTED (see Phase 2 plan)
   
   ─── SYNTHETIC DEMO MARKER ───
   [Only if synthetic near-miss is analyzed]
   ⚠ THIS IS A SYNTHETIC DEMO EXAMPLE (not real incident data)
   ```

5. User can:
   - Click "Show Raw Case Evidence" → displays full case details from labels_schema.json
   - Click "Show Example Precursor Texts" → displays all training examples for detected class
   - Download analysis as JSON
   - Provide feedback (logged to session, not sent externally)

### **Demo Test Cases:**

**Case A: Real precursor from training data (Case 5 blowout)**
- Input: "Blind shear ram was not included in the BOP stack configuration due to assumed oil well, not gas."
- Expected: C5 (Missing Barriers) + C4 (Procedure) → SIF-HIGH
- Provenance: RULE_BASED or TRAINED_ML (if model validation succeeds)

**Case B: Synthetic near-miss (from oil-india-near-miss-mockups.md)**
- Input: "Morning shift started with minor spillage in storage tank; promptly cleaned and reported."
- Expected: No major class detected OR low-confidence C6 (Environmental)
- Provenance: RULE_BASED + SYNTHETIC marker

**Case C: Real precursor (Case 2 fatality)**
- Input: "Worker recently promoted to Topman position but received no formal training on emergency procedures."
- Expected: C1 (Training Gap) + possible C2 (Supervision)
- Provenance: RULE_BASED or TRAINED_ML

---

## 6. FILES TO CREATE

| File | Purpose | Lines | Owner |
|---|---|---|---|
| `src/analysis_engine.py` | Core pipeline: C1-C6 analyzer + SIF assessor + evidence linker | ~400 | NLP/Backend Eng |
| `src/ml_classifier.py` | TF-IDF + Logistic Regression ML trainer & predictor (ONLY if validation succeeds) | ~150 | ML Eng |
| `src/rule_based_detector.py` | Fallback keyword matching + class detection | ~200 | NLP Eng |
| `src/evidence_extractor.py` | Evidence linker: case lookup, pattern detection, OISD mapping | ~150 | Backend Eng |
| `src/streamlit_app.py` | Streamlit UI frontend | ~300 | Frontend Eng |
| `tests/test_analysis_engine.py` | Unit tests for analysis pipeline | ~150 | QA |
| `tests/test_real_cases.py` | Integration tests: run analyzer on real training examples | ~100 | QA |
| `tests/test_synthetic_demo.py` | Verify synthetic examples are correctly marked | ~50 | QA |
| `docs/DEMO_USAGE.md` | How to run the demo, what to expect, known limitations | ~100 | Tech Writer |
| `logs/demo_run.log` | Session log of all analyses run during demo | Dynamic | Runtime |

**Total:** ~1,650 lines of code + documentation

---

## 7. FILES TO REUSE (DO NOT DUPLICATE/RENAME)

- `data/labels_schema.json` — read-only
- `data/training_data.csv` — read-only
- `data/label_mapping.csv` — read-only
- `data/processed/train.csv` — read-only (generated by prepare_dataset.py if not present)
- `data/processed/val.csv` — read-only (generated by prepare_dataset.py if not present)
- `data/raw_research/1.Safety Analysis...` — read-only
- `data/raw_research/2.Recurring Patterns...` — read-only
- `data/raw_research/3.Oil and Gas...` — read-only
- `data/raw_research/4.OISD Safety Standards...` — read-only
- `data/raw_research/5.Bridging Safety Standards...` — read-only
- `synthetic_data/oil-india-near-miss-mockups.md` — read-only (marked SYNTHETIC in output)
- `src/data/prepare_dataset.py` — use to generate train/val if needed
- `src/data/build_label_mapping.py` — use if label_mapping.csv needs regeneration
- `README.md` — read-only; may add note pointing to demo

---

## 8. FILES EXPLICITLY NOT TO CREATE

| File | Reason |
|---|---|
| `data/sif_negatives.csv` | No negative data exists; cannot build without fabricating data |
| `data/lsr_labels.csv` | No LSR mappings in repository; would require external IOGP standard reading |
| `data/master_case_index.csv` | Can be generated on-the-fly from labels_schema.json; no need to persist |
| `src/models/deep_learning_classifier.py` | Dataset too small (75 examples); deep learning will overfit; simple TF-IDF classifier sufficient |
| `src/api/flask_rest_api.py` | Not needed; Streamlit UI is sufficient for demo |
| `src/vectordb/chromadb_store.py` | Not needed; direct JSON/CSV lookups are faster and more transparent |
| `src/llm/prompt_engineering.py` | Not needed; demo must be rule-based/ML-based, not LLM-based (no LLM calls) |
| `frontend/react_app/` | Use Streamlit instead (simpler, no Node/npm overhead) |
| `docker-compose.yml` | Not needed; demo runs locally in Streamlit |

---

## 9. KNOWN LIMITATIONS (TO DOCUMENT IN UI + README)

1. **Dataset size:** 75 precursor texts from 12 high-severity cases. Model trained on this data will not generalize to Oil India's broader incident database.
2. **Formal register bias:** All training data is formal, polished investigation reports. Real field reports (shorthand, typos, colloquialisms) may be misclassified.
3. **Class imbalance:** C4 has 30 examples (40%), C6 has 7 (9%). Rare classes have high uncertainty.
4. **No SIF-negative baseline:** All 12 cases resulted in SIF events. No "normal operations" data exists. SIF assessment is rule-based risk-ranking only, not a trained classifier.
5. **No LSR mapping:** IOGP LSR labels not present in repository. This component is deferred to Phase 2.
6. **Validation scope:** Tested on 15-row validation set (20% of data). Performance on truly new field reports is unknown.
7. **Synthetic examples marked separately:** Demo includes 3 synthetic near-miss mockups. These are clearly labeled as SYNTHETIC and are never used in training or quantitative evaluation.

---

## 10. DEMO EXECUTION CHECKLIST

- [ ] **P0-1:** Core analysis pipeline (analysis_engine.py) ← calls C1-C6, SIF, evidence, LSR components
- [ ] **P0-2:** Evidence extraction (evidence_extractor.py) ← deterministic lookups from labels_schema.json + raw_research
- [ ] **P0-3:** SIF-potential assessment (in analysis_engine.py) ← rule-based risk-ranking, no ML
- [ ] **P0-4:** C1-C6 detection ← EITHER trained ML (if validation F1 ≥ 0.5) OR rule-based fallback
- [ ] **P0-5:** IOGP LSR mapping (in analysis_engine.py) ← returns "NOT YET IMPLEMENTED" with flag
- [ ] **P0-6:** Streamlit UI (streamlit_app.py) ← calls analysis_engine.py, displays results with provenance labels
- [ ] **P0-7:** Unit tests (test_analysis_engine.py) ← verify each component independently
- [ ] **P0-8:** Integration tests (test_real_cases.py) ← run on 3 real training examples
- [ ] **P0-8b:** Synthetic test (test_synthetic_demo.py) ← verify synthetic examples are correctly marked
- [ ] **P0-9:** Documentation (DEMO_USAGE.md) ← how to run, what to expect, limitations
- [ ] **P0-10:** Full demo run ← test UI with all 3 test cases, verify output quality

---

## 11. SUCCESS CRITERIA (HONESTY FIRST)

✅ **PASS:** Demo successfully detects C1-C6 classes in real training examples with correctly attributed provenance  
✅ **PASS:** SIF assessment is transparent (rule-based, not ML); does not claim false confidence  
✅ **PASS:** Evidence linkage correctly references cases, precursor text, OISD standards  
✅ **PASS:** LSR component honestly states "not yet implemented" instead of fabricating mappings  
✅ **PASS:** Synthetic examples are clearly marked and never used in quantitative evaluation  
✅ **PASS:** All limitations documented in UI + README  
✅ **PASS:** Can run end-to-end in <5 seconds per analysis  
✅ **PASS:** All tests pass; no silent failures  

❌ **FAIL:** If ML model cannot be trained with honest F1 ≥ 0.5 on real test data → fall back to rule-based  
❌ **FAIL:** If any component claims confidence/probability without validation  
❌ **FAIL:** If any component is called "ML" when it's actually rule-based  
❌ **FAIL:** If fabricated LSR mappings appear in output  
❌ **FAIL:** If synthetic examples are silently included in training/evaluation  

---

## 12. PHASE BOUNDARIES

**P0 (this spec):**
- Rule-based precursor detection (C1-C6) or simple ML if validation succeeds
- Transparent SIF-potential assessment (rule-based only)
- Evidence extraction (deterministic)
- Streamlit demo UI
- Unit + integration tests
- NO: LSR mapping, NO: field-report adaptation, NO: confidence scores, NO: LLM, NO: FastAPI

**P1 (Phase 2 future):**
- Collect real field-report samples; fine-tune ML on 200+ examples/class
- Read IOGP LSR standard; manually assign labels to cases
- Implement field-register normalization
- Confidence calibration + uncertainty quantification
- Production monitoring pipeline

**P2 (Phase 2+ future):**
- Validate model on Oil India's full incident database
- Integrate with real field-report UX
- Deploy to field supervisors/auditors

---

END OF BUILD SPEC
