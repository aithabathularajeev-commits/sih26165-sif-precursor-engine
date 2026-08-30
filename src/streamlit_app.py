"""
streamlit_app.py — SIH26165 Safety Precursor Analyzer Demo

A Streamlit-based UI for the SIF precursor analysis engine.
Intended for HSE supervisor/auditor users.

This demo provides:
1. Real-time precursor text analysis
2. Root-cause detection (C1-C6 classes) with explainability
3. SIF-potential assessment with transparent rule-based rationale
4. Evidence linkage to case studies and OISD standards
5. Clearly labeled synthetic examples (when testing with demo data)
"""

import streamlit as st
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from analysis_engine import SIH26165AnalysisEngine


# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="SIH26165 Safety Precursor Analyzer",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚠️ SIH26165 Safety Precursor Analyzer")
st.markdown("""
**Purpose:** Detect safety-critical root-cause precursors in incident reports and field narratives.

**User:** HSE Supervisors, Safety Auditors  
**Scope:** Phase 1 (Demo) — Rule-based detection on investigation report register

---
""")


# ============================================================================
# SIDEBAR: HELP & INFO
# ============================================================================

with st.sidebar:
    st.header("ℹ️ About This Demo")
    
    with st.expander("How it Works", expanded=False):
        st.markdown("""
        **Analysis Pipeline:**
        1. **Root-Cause Detection**: Identifies which of the 6 SIF-precursor classes (C1–C6) are present
        2. **SIF-Potential Assessment**: Evaluates risk level (HIGH/MEDIUM/LOW) based on detected classes
        3. **Evidence Linkage**: Shows supporting cases, example precursor texts, and violated OISD standards
        4. **LSR Mapping**: (Not yet implemented; see Phase 2)
        
        **Key Principle:** Every result is labeled with its source:
        - **RULE_BASED**: Keyword matching (deterministic)
        - **TRAINED_ML**: From a trained model (if validation F1 ≥ 0.5)
        - **DETERMINISTIC_EXTRACTION**: Direct lookups from case database
        """)
    
    with st.expander("Precursor Classes (C1–C6)", expanded=False):
        classes_info = {
            "C1": "Training & Competency Gaps — Missing hands-on training, invalid certifications, no induction",
            "C2": "Supervision & Communication Breakdown — Absent mentors, failed coordination, unaware crew",
            "C3": "Maintenance & Inspection Failures — Skipped inspections, missing calibrations, degradation undetected",
            "C4": "Procedure & Permit Bypass — Skipped JSA, no work permit, ignored risk assessment, unsafe planning",
            "C5": "Missing/Defunct Safety Barriers — Not installed equipment, disabled systems, isolated trip tanks",
            "C6": "Environmental & Physical Workspace Hazards — Spillage, weather, obstruction, combustible materials",
        }
        for cls_id, description in classes_info.items():
            st.markdown(f"**{cls_id}**: {description}")
    
    with st.expander("Known Limitations", expanded=False):
        st.markdown("""
        1. **Dataset Size**: 75 precursor texts from 12 OISD cases (all catastrophic outcomes)
        2. **Register Mismatch**: Trained on formal investigation reports, not field-report shorthand
        3. **Class Imbalance**: C4 has 40% of data; C6 has 9%; rare classes have high uncertainty
        4. **No SIF-Negative Baseline**: All training cases resulted in SIF events; no "normal ops" data
        5. **Validation Scope**: Tested on 15-row validation set; field-report accuracy unknown
        6. **LSR Not Implemented**: IOGP Life-Saving Rule mappings deferred to Phase 2
        """)
    
    st.markdown("---")
    st.markdown("""
    **Phase 1 Status:** Demo  
    **Build Date:** 2026-08-30  
    **Provenance Required:** YES (every output labeled with source)  
    **Confidence Scores:** NO (too small dataset; using transparent rule-based approach)
    """)


# ============================================================================
# MAIN CONTENT: ANALYSIS INPUT & OUTPUT
# ============================================================================

# Initialize engine once (cached)
@st.cache_resource
def load_engine():
    return SIH26165AnalysisEngine(repo_root=".")

engine = load_engine()


# --- TAB 1: ANALYSIS ---
tab_analysis, tab_test_cases, tab_evidence, tab_about = st.tabs([
    "📋 Analysis", "🧪 Test Cases", "📚 Evidence Base", "ℹ️ About"
])


with tab_analysis:
    st.header("Precursor Text Analysis")
    
    # Input method selector
    col1, col2 = st.columns(2)
    with col1:
        input_method = st.radio("Input method:", ["Paste text", "Use test case"])
    
    # Input
    if input_method == "Paste text":
        user_text = st.text_area(
            "Enter precursor incident report or field narrative:",
            height=150,
            placeholder="Example: 'JSA was skipped due to time pressure. The well had no blind shear ram installed...'",
            key="user_input"
        )
    else:
        # Load test cases
        test_cases = {
            "Case 5 Blowout (BSR Omission)": (
                "Blind shear ram was not included in the BOP stack configuration "
                "because well was assumed to be oil, not gas. "
                "Trip tank was isolated during perforation. "
                "Key personnel certificates did not match demonstrated knowledge."
            ),
            "Case 2 Fatality (Training Gap)": (
                "Victim recently approved as Topman but had received no hands-on training "
                "on safety gear, hazard spotting, or emergency response procedures. "
                "No mentor was assigned to provide on-the-job guidance."
            ),
            "Case 10 Pipeline Loss (Procedure Bypass)": (
                "Deep excavation in pipeline right-of-way was carried out as unsupervised activity. "
                "Excavation was authorized but administrative corrective action was delayed. "
                "Maintenance inspection was not conducted before excavation."
            ),
        }
        selected_test_case = st.selectbox("Select test case:", list(test_cases.keys()))
        user_text = test_cases[selected_test_case]
        st.info(f"**Selected:** {selected_test_case}")
    
    # Mark synthetic if user indicates
    is_synthetic = st.checkbox("☑️ This is a SYNTHETIC demo example (not real incident data)")
    
    # Analyze button
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not user_text.strip():
            st.error("Please enter precursor text to analyze.")
        else:
            with st.spinner("Analyzing precursor text..."):
                result = engine.analyze(user_text)
            
            # --- SYNTHETIC MARKER ---
            if is_synthetic:
                st.warning(
                    "⚠️ **SYNTHETIC DEMO EXAMPLE** — This is not real incident data. "
                    "Used for demonstration purposes only. Results should not be used for quantitative evaluation.",
                    icon="🎭"
                )
            
            # --- ROOT-CAUSE DETECTION ---
            st.subheader("🎯 Root-Cause Detection (C1–C6)")
            
            if result.detected_classes:
                for cls in result.detected_classes:
                    with st.expander(f"✓ {cls.class_id}: {cls.class_name}", expanded=True):
                        col_1, col_2 = st.columns([2, 1])
                        with col_1:
                            st.markdown(f"**Sub-tags:** {', '.join(cls.sub_tags)}")
                            st.markdown(f"**Confidence Note:** {cls.confidence_note}")
                            st.markdown(f"**Supporting Cases:** {', '.join(cls.evidence_references)}")
                            if cls.supporting_precursor_texts:
                                st.markdown(f"**Example Precursor Texts:**")
                                for ex_text in cls.supporting_precursor_texts:
                                    st.markdown(f"  - _{ex_text}_")
                        with col_2:
                            st.metric("Provenance", cls.provenance)
            else:
                st.info("No major precursor classes detected in this text. " 
                       "(This does not necessarily mean the report is safe; lower-confidence matches may still warrant review.)")
            
            # --- SIF ASSESSMENT ---
            st.subheader("⚡ SIF-Potential Assessment")
            
            sif_colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            sif_symbol = sif_colors.get(result.sif_assessment.level, "")
            
            col_sif_1, col_sif_2 = st.columns([2, 1])
            with col_sif_1:
                st.markdown(f"### {sif_symbol} Level: **{result.sif_assessment.level}**")
                st.markdown(f"**Rationale:** {result.sif_assessment.rationale}")
                st.markdown(f"**Contributing Classes:** {', '.join(result.sif_assessment.contributing_classes)}")
            with col_sif_2:
                st.metric("Provenance", result.sif_assessment.provenance)
            
            st.info(f"**Limitation:** {result.sif_assessment.limitation}")
            
            # --- EVIDENCE LINKAGE ---
            st.subheader("📚 Evidence Linkage")
            
            if result.supporting_evidence["recurring_patterns"]:
                st.markdown("**Recurring Patterns (from case database):**")
                for pattern in result.supporting_evidence["recurring_patterns"]:
                    st.markdown(f"  - {pattern}")
            
            if result.supporting_evidence["oisd_standards"]:
                st.markdown("**Violated OISD Standards:**")
                for standard in result.supporting_evidence["oisd_standards"]:
                    st.markdown(f"  - {standard}")
            
            # --- LSR MAPPING ---
            st.subheader("🏢 IOGP Life-Saving Rule Mapping")
            st.warning(
                f"**Status:** {result.lsr_mapping.status}\n\n"
                f"{result.lsr_mapping.limitation}",
                icon="⏸️"
            )
            
            # --- LIMITATIONS ---
            st.subheader("⚠️ Known Limitations of This Analysis")
            for lim in result.limitations:
                st.markdown(f"  • {lim}")
            
            # --- DOWNLOAD ---
            st.divider()
            result_json = json.dumps(result.to_dict(), indent=2, default=str)
            st.download_button(
                label="📥 Download Analysis as JSON",
                data=result_json,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


# --- TAB 2: TEST CASES ---
with tab_test_cases:
    st.header("Predefined Test Cases")
    
    st.markdown("""
    These test cases are based on **real precursor text from the training dataset**.
    They demonstrate how the analyzer handles different precursor scenarios.
    """)
    
    test_cases_detailed = [
        {
            "name": "Case 5 — Blowout (BSR Omission + Procedure Bypass)",
            "text": "Blind shear ram was not included in the BOP stack configuration because the well was assumed to be oil well, not gas. Trip tank was isolated during perforation.",
            "expected_classes": ["C4", "C5"],
            "expected_sif": "HIGH",
            "real": True,
        },
        {
            "name": "Case 2 — Fatality (Training + Supervision Gap)",
            "text": "Victim recently approved as Topman but had received no hands-on training on safety gear, hazard spotting. No mentor was assigned.",
            "expected_classes": ["C1", "C2"],
            "expected_sif": "MEDIUM",
            "real": True,
        },
        {
            "name": "Case 3 — Maintenance Failure",
            "text": "No OEM inspection conducted. No calibration records for pressure switches. Non-standard components used.",
            "expected_classes": ["C3"],
            "expected_sif": "MEDIUM",
            "real": True,
        },
    ]
    
    for i, tc in enumerate(test_cases_detailed):
        with st.expander(f"{'✓' if tc['real'] else '🎭'} {tc['name']}", expanded=(i == 0)):
            st.markdown(f"**Text:** _{tc['text']}_")
            col_tc1, col_tc2, col_tc3 = st.columns(3)
            with col_tc1:
                st.markdown(f"**Expected Classes:** {', '.join(tc['expected_classes'])}")
            with col_tc2:
                st.markdown(f"**Expected SIF Level:** {tc['expected_sif']}")
            with col_tc3:
                st.markdown(f"**Data Source:** {'Real (Training Data)' if tc['real'] else 'Synthetic (Demo)'}")
            
            if st.button(f"Run Test: {tc['name']}", key=f"test_{i}"):
                result = engine.analyze(tc["text"])
                detected_classes = [c.class_id for c in result.detected_classes]
                
                st.success("✓ Analysis completed")
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown(f"**Detected Classes:** {', '.join(detected_classes) if detected_classes else 'None'}")
                    match_classes = "✓" if set(detected_classes) == set(tc['expected_classes']) else "⚠️"
                    st.markdown(f"{match_classes} Expected: {', '.join(tc['expected_classes'])}")
                with col_res2:
                    st.markdown(f"**SIF Assessment:** {result.sif_assessment.level}")
                    match_sif = "✓" if result.sif_assessment.level == tc['expected_sif'] else "⚠️"
                    st.markdown(f"{match_sif} Expected: {tc['expected_sif']}")


# --- TAB 3: EVIDENCE BASE ---
with tab_evidence:
    st.header("Evidence Base & Case Reference")
    
    st.markdown("""
    The analyzer references a curated evidence base of 12 OISD case studies.
    This tab provides access to the underlying data.
    """)
    
    # Load and display case reference
    schema = engine.schema
    
    case_ref = schema.get("case_reference", {})
    
    for case_id in sorted(case_ref.keys(), key=lambda x: int(x.split()[-1])):
        case_info = case_ref[case_id]
        with st.expander(f"{case_id}: {case_info.get('title', 'N/A')}", expanded=False):
            col_cr1, col_cr2 = st.columns(2)
            with col_cr1:
                st.markdown(f"**OISD ID:** {case_info.get('oisd_id', 'N/A')}")
            with col_cr2:
                st.markdown(f"**Severity:** {case_info.get('severity', 'N/A')}")


# --- TAB 4: ABOUT ---
with tab_about:
    st.header("About SIH26165 P0 Demo")
    
    st.markdown("""
    ### Project Overview
    
    **SIH26165 Safety Integrity Function (SIF) Precursor Engine**
    
    An NLP-based system for detecting safety-critical precursor indicators in oil & gas incident reports.
    
    ### Phase 1 (This Demo)
    
    - **Scope:** Rule-based precursor detection on investigation-report register
    - **Dataset:** 75 labeled precursor texts from 12 OISD case studies
    - **User:** HSE Supervisors, Safety Auditors
    - **Output:** Transparent, explainable root-cause detection with evidence linkage
    - **Key Principle:** Every result labeled with source (RULE_BASED, TRAINED_ML, or DETERMINISTIC_EXTRACTION)
    
    ### Phase 2 (Future)
    
    - Collect real field-report samples
    - Fine-tune ML models on 200+ examples per class
    - Implement IOGP LSR mappings
    - Deploy to field supervisors with real-time integration
    
    ### Technical Details
    
    **Architecture:**
    - Core pipeline: `src/analysis_engine.py`
    - Streamlit UI: `src/streamlit_app.py`
    - Data: `data/labels_schema.json`, `data/label_mapping.csv`, `data/training_data.csv`
    - Evidence: `data/raw_research/` (5 analysis documents)
    
    **Detection Approach:**
    - **C1–C6 Classes:** Keyword matching + rule-based deterministic extraction
    - **SIF Assessment:** Rule-based risk-ranking (no binary ML classifier due to no negative data)
    - **Evidence Linkage:** Direct case/precursor/standard lookups
    
    ### Data Provenance
    
    All source data is read-only and preserved:
    - `labels_schema.json` — 6-class taxonomy with evidence backing
    - `label_mapping.csv` — 75 labeled examples
    - `raw_research/` — Original case analyses, patterns, OISD mappings
    
    Synthetic demo examples (if used) are clearly marked as SYNTHETIC and never mixed into training/evaluation.
    
    ### Known Limitations
    
    1. **Dataset size:** 75 examples from 12 high-severity cases
    2. **Formal register bias:** All training data is investigation reports
    3. **Class imbalance:** C4 (40%), C6 (9%)
    4. **No SIF-negative baseline:** No "normal operations" data
    5. **Validation scope:** 15-row validation set
    6. **LSR not implemented:** Requires reading IOGP standards
    
    ### Next Steps (Phase 2)
    
    - [ ] Collect real field-report samples from Oil India
    - [ ] Validate precursor detector on field register
    - [ ] Train binary SIF classifier on negative examples
    - [ ] Implement IOGP LSR mappings
    - [ ] Deploy to field with monitoring pipeline
    
    ---
    
    **Build Date:** 2026-08-30  
    **Repo:** `sih26165-sif-precursor-engine`  
    **Branch:** `inspection-audit-complete` → P0 build  
    **Status:** Demo (P0) — Ready for HSE supervisor evaluation
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
---
**Disclaimer:** This is a Phase 1 (demo) safety precursor detection system. It is **NOT** 
a substitute for expert safety engineering review. All results must be independently verified 
by qualified HSE professionals. See "Known Limitations" for scope boundaries.

**Questions or Feedback:** Contact the SIH26165 team.
""")
