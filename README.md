# SIH26165 — OISD Safety Research Data Package

**AI/NLP Engine to Detect Serious Injury & Fatality (SIF) Precursors in Unsafe-Act/Unsafe-Condition and Near-Miss Reports (Oil India Limited)**

This README ties together the two data artifacts in this package and explains exactly how to use them to build the model.

---

## 1. Files in this package

| File | What it is | Rows/Entries |
|---|---|---|
| `training_data.csv` | 75 individual precursor-text statements, extracted verbatim from 12 real OISD case studies, each tagged with source case, exact severity outcome, and OISD case ID | 75 rows |
| `labels_schema.json` | The classification taxonomy: 6 top-level classes (your model's actual output labels) with 11 nested sub-tags (for explainability), each backed by real supporting cases and quoted evidence | 6 classes / 11 sub-tags |
| `oisd_corpus.json` *(from earlier)* | 6 fully structured case studies with full field breakdown (narrative, observations, root cause, recommendations) | 6 cases |
| `oisd_index.csv` *(from earlier)* | Index of all ~65 official OISD case studies with direct download links | 65 rows |

**Source:** All real data is drawn from official OISD (Oil Industry Safety Directorate) case study publications — oisd.gov.in/en-in/CaseStudies. Oil India Limited reports into this same national incident framework, so this data generalizes directly to OIL's operational context.

---

## 2. Why two separate files (CSV vs JSON)?

- `training_data.csv` is your **raw labeled text** — the (input, metadata) side. This is what gets vectorized/embedded and fed into the model.
- `labels_schema.json` is your **label ontology** — the (class definitions, evidence, hierarchy) side. This is what your loss function and evaluation reporting are built against.

They connect through the **row-to-label mapping** in Section 4 below — this is the missing link that turns raw text into a trainable supervised dataset.

---

## 3. The 6-class taxonomy at a glance

| Class ID | Class Name | Case Coverage | Sub-tags |
|---|---|---|---|
| **C1** | Training & Competency Gaps | 6 cases | Inadequate Training and Competency |
| **C2** | Supervision & Communication Breakdown | 7 cases | Deficient Supervision; Communication Breakdowns |
| **C3** | Maintenance & Inspection Failures | 7 cases | Substandard Maintenance/Inspection; Mechanical Failure & Structural Defects |
| **C4** | Procedure, Permit & Risk-Assessment Bypass | 10 cases | Flawed Risk Assessment/Planning; Weak Safety Culture; Delayed Corrective Action |
| **C5** | Missing/Defunct Critical Safety Barriers | 5 cases | Defunct or Absent Critical Safety Devices |
| **C6** | Environmental & Physical Workspace Hazards | 5 cases | Severe Environmental/Natural Hazards; Hazardous Workspace Obstructions |

Full definitions, descriptions, and quoted evidence for every sub-tag are in `labels_schema.json`.

---

## 4. Row-level multi-label mapping (training_data.csv ↔ labels_schema.json)

**This is the critical artifact** — it assigns each of the 75 precursor-text rows to one or more class/sub-tag IDs from the schema, making the dataset immediately trainable.

**Important: this is a multi-label problem, not single-label.** Several rows carry two labels (e.g., row 22 is both a training gap *and* a supervision gap — "workers were unskilled, no supervision was available" is genuinely both). Your model must use **independent binary classifiers per class** (sigmoid + binary cross-entropy per class), not a single softmax — a softmax would force the model to pick only one label per example, which misrepresents the real, multi-causal nature of these incidents.

| # | Case | Class(es) | Sub-tag(s) | Precursor Text (truncated) |
|---|------|-----------|------------|------------------------------|
| 1 | Case 1 | C3 | C3-S1 | There was no evidence that Category II inspection was conducted, which was als… |
| 2 | Case 1 | C3 | C3-S1 | The Category III inspection reports had several critical deficiencies: Failed … |
| 3 | Case 1 | C4 | C4-S2 | A system of reporting and documenting unsafe activities or conditions was avai… |
| 4 | Case 1 | C6 | C6-S2 | The visibility of the tugger was found to be obstructed due to the placement o… |
| 5 | Case 2 | C1 | C1-S1 | The victim was recently approved for the role of Topman on February 27, 2025, … |
| 6 | Case 2 | C2 | C2-S1 | No mentor or supervisor was assigned to provide on-the-job guidance, leaving t… |
| 7 | Case 2 | C4 | C4-S2 | CCTV footage reveals that the victim relied solely on the Fall Prevention Devi… |
| 8 | Case 2 | C6 | C6-S2 | The gooseneck with Kelly hose of the standpipe was positioned halfway across t… |
| 9 | Case 2 | C6 | C6-S2 | Crude oil spillage due to wet pull-out was observed on the derrick floor and m… |
| 10 | Case 2 | C4 | C4-S1 | Despite the presence of clearly unsafe condition of oil-contaminated surfaces,… |
| 11 | Case 2 | C4 | C4-S2 | Climbing the ladder and pulling out the tubing were carried out at the same ti… |
| 12 | Case 3 | C3 | C3-S2 | Two out of the three live terminals of the HVAC compressor were found blown of… |
| 13 | Case 3 | C3 | C3-S1 | Parameters like temperature and oil pressures were not monitored, except sucti… |
| 14 | Case 3 | C3 | C3-S1 | There were no written OEM guidelines available for what to cover during monthl… |
| 15 | Case 3 | C3 | C3-S1 | Records for arc Chute test of terminal were not available. No record was avail… |
| 16 | Case 3 | C6 | C6-S2 | The duct insulation near the HVAC unit was found burnt after the incident indi… |
| 17 | Case 3 | C6 | C6-S2 | Also, there was a platform built above the affected room which was being utili… |
| 18 | Case 3 | C5 | C5-S1 | There was no provision of fixed fire suppression system in the workshop area a… |
| 19 | Case 3 | C1 | C1-S1 | There was no structured training module for electrical crew members which woul… |
| 20 | Case 4 | C4 | C4-S1 | Provision to ensure zero pressure after draining was not available between NRV… |
| 21 | Case 4 | C4 | C4-S2 | It was evident from the position of the IP during work that he was not followi… |
| 22 | Case 4 | C1, C2 | C1-S1, C2-S1 | All the workers involved in the job were in the "unskilled" category. No super… |
| 23 | Case 4 | C4 | C4-S2 | Appropriate PPE, such as safety goggles and safety helmets, was not used while… |
| 24 | Case 4 | C4 | C4-S1 | For this job of gasket replacement, Work Permit, Job Safety Analysis (JSA), or… |
| 25 | Case 5 | C4 | C4-S1 | The well was believed to be oil bearing as per well plan, however after perfor… |
| 26 | Case 5 | C5 | C5-S1 | a Blind Shear Ram was not included in the BOP stack configuration. |
| 27 | Case 5 | C4 | C4-S1 | the data from previous workover jobs in the same well in different zone and te… |
| 28 | Case 5 | C5 | C5-S1 | The trip tank was not in use during the incident, and the level alarm system o… |
| 29 | Case 5 | C5 | C5-S1 | There was no well monitoring during the well perforation job. |
| 30 | Case 5 | C1 | C1-S1 | All the key personnel had a valid well control training certificates. However,… |
| 31 | Case 5 | C1 | C1-S1 | The installation manager, Mines Safety Officer and Area manager who had all jo… |
| 32 | Case 5 | C1 | C1-S1 | Essential training as per OISD-STD-176 were not provided to the crew and were … |
| 33 | Case 5 | C5 | C5-S1 | It seems that the blind ram was closed on wireline, which is against the condi… |
| 34 | Case 6 | C2 | C2-S2 | During the incident, the information regarding the placement of the pin end on… |
| 35 | Case 6 | C4 | C4-S2 | The topman used a rope to pull the tubing double which normally precedes the u… |
| 36 | Case 6 | C4 | C4-S2 | Rig crew members were using mobile phones during the workover operation on job… |
| 37 | Case 6 | C5 | C5-S1 | Twin stop device failed to function during the incident. |
| 38 | Case 6 | C3 | C3-S1 | The catwalk showed multiple impact marks and was misaligned with the substruct… |
| 39 | Case 7 | C1 | C1-S1 | All contractual employees are required to undergo five specified trainings. Ho… |
| 40 | Case 7 | C4 | C4-S2 | It was observed that personnel did not use work vests during their stay at the… |
| 41 | Case 7 | C3 | C3-S1 | The railing at the southeast (SE) location was missing and was later barricade… |
| 42 | Case 7 | C3 | C3-S2 | It was observed that the north side stair railing, made of composite material,… |
| 43 | Case 7 | C6 | C6-S2 | It was observed that no washroom facility was available at the unmanned platfo… |
| 44 | Case 8 | C4 | C4-S1 | SOP for mast lowering didn't address securing or un-securing the travelling bl… |
| 45 | Case 8 | C4 | C4-S1 | The Job Safety Analysis (JSA) did not include any steps for un-securing the tr… |
| 46 | Case 8 | C2 | C2-S1 | Installation Manager (IM - Company), HSE Engineer (Company), Rig Superintenden… |
| 47 | Case 8 | C4 | C4-S2 | Safety Harness of IP was anchored at incorrect location and harness lanyard wa… |
| 48 | Case 8 | C2 | C2-S2 | Communication gap was observed between signalman and Crane operator. |
| 49 | Case 8 | C3 | C3-S1 | No periodic inspection report was available. |
| 50 | Case 8 | C1 | C1-S1 | crew members other than Driller had no prior experience on dismantling procedu… |
| 51 | Case 8 | C1 | C1-S1 | Periodic training and awareness sessions on hazard identification, risk assess… |
| 52 | Case 9 | C4 | C4-S1 | Geo Technical Order (GTO) indicates oil throughout the depth, whereas as per t… |
| 53 | Case 9 | C4 | C4-S1 | The perforation plan was not reviewed on the basis of drilling difficulties at… |
| 54 | Case 9 | C4 | C4-S1 | CBL-VDL does not suggest good bond throughout the depth, however it is not cle… |
| 55 | Case 9 | C4 | C4-S1 | The mud density planned in the final phase of drilling was lower than those ac… |
| 56 | Case 9 | C5 | C5-S1 | BOP stack had no shear ram, violating Clause 6.3.1, making it impossible to sh… |
| 57 | Case 9 | C5 | C5-S1 | The trip tank was kept isolated during perforation, which violated Clause 6.8(… |
| 58 | Case 9 | C4 | C4-S2 | During wireline tool recovery, pull out was done too quickly, which may have f… |
| 59 | Case 9 | C5 | C5-S1 | Fire water pump was not available at site. |
| 60 | Case 10 | C3 | C3-S1 | Due to unavailability of crude oil in required quantity post commissioning and… |
| 61 | Case 10 | C4 | C4-S3 | Magnetic Tomography (MTM) survey was carried out in 2020 which had concluded t… |
| 62 | Case 10 | C4 | C4-S1, C4-S3 | Another MTM survey was conducted in 2023. Report mentioned that pipeline segme… |
| 63 | Case 10 | C5 | C5-S1 | Safety instrumentation system and respective interlocks to trip the crude oil … |
| 64 | Case 10 | C3 | C3-S1 | Corrosion Inhibitor (CI) dosing rate was not adequate to ensure that corrosion… |
| 65 | Case 11 | C2 | C2-S2 | Mainline maintenance contractor was carrying out excavation work for TLP cable… |
| 66 | Case 11 | C2 | C2-S2 | in the existing permit system, permits were being received by the Mainline off… |
| 67 | Case 11 | C2 | C2-S2 | Several system failures were observed like not getting information of unauthor… |
| 68 | Case 11 | C2, C4 | C2-S1, C4-S1 | Incident happened because an unsupervised activity was being carried out witho… |
| 69 | Case 12 | C3 | C3-S1 | Incident-1: A Leak alarm in the Leak Detection System was logged, however the … |
| 70 | Case 12 | C4 | C4-S3 | Incident-2: Longer length HDD work to protect the pipeline could not be comple… |
| 71 | Case 12 | C4 | C4-S3 | Incident-3: Previously reduced soil cover was reported on this side of the riv… |
| 72 | Case 12 | C4 | C4-S1 | Incident-4: Multiproduct pipeline has crossed the seasonal river through open … |
| 73 | Case 12 | C4 | C4-S1 | Incident-5: The pipeline was laid via open cut method across the river crossin… |
| 74 | Case 12 | C4 | C4-S3 | In some of the incidents, although river bank erosion was noticed and fresh HD… |
| 75 | Case 12 | C4, C6 | C4-S1, C6-S1 | In one of the incidents, river crossing was carried out by open-cut method sin… |

---

## 5. How to use this for training (practical steps)

1. **Load `training_data.csv`** — each row's `precursor_text` is your input `X`.
2. **Multi-hot encode the label column** using the mapping table above — for each row, create a 6-length binary vector (or 11-length if you want to train on sub-tags directly) marking which class(es)/sub-tag(s) apply.
3. **Baseline model:** TF-IDF + One-vs-Rest Logistic Regression across the 6 classes — fast, interpretable, good first benchmark given the small dataset size (75 examples, 12 source incidents).
4. **Stronger model:** Fine-tune a pretrained transformer (e.g., DistilBERT or a domain-adapted model) with a multi-label sigmoid output head. With only 75 examples this will overfit fast — use this only after augmenting with public near-miss/injury narrative data (OSHA/SafeOCS) for pretraining, then fine-tune on this OISD set for domain adaptation.
5. **Evaluate per-class**, not just overall accuracy — with this few examples per class (some classes have only 5 supporting cases), report precision/recall per class so weak classes are visible rather than averaged away.
6. **Cross-reference clause citations:** when your model flags a class, have it also surface the relevant OISD clause(s) from `labels_schema.json`'s evidence — this is a strong demo differentiator (see earlier OISD clause-frequency research).

---

## 6. Known limitations (be upfront about these with judges)

- **Small dataset (75 rows / 12 incidents).** Real, but not large — treat this as a domain-adaptation fine-tuning set, not a from-scratch training set. State this explicitly rather than overclaiming model accuracy.
- **All source cases are retrospective, post-incident investigation reports** — formal, passive, committee-written prose. The PS asks for detection in raw, informal, worker-written UA/UC field reports, which is a different linguistic register this data does not represent. Frame your build as "Phase 1: mine historical investigation reports to build the precursor taxonomy and risk model → Phase 2: apply the trained model to score incoming live UA/UC submissions" — honest and still answers the PS's real-time framing as a roadmap item.
- **No true low-severity "near-miss" baseline.** Every source case resulted in a fatality, blowout, major loss, or similar — there's no "normal operations" or truly minor near-miss control group. Supplement with the synthetic near-miss mockups (`oil-india-near-miss-mockups.md`, clearly labeled synthetic) for UI/demo purposes only — never present these as real training accuracy evidence.
- **Class imbalance.** C4 (10 cases) is far more represented than C5/C6 (5 cases each) — expect weaker recall on the smaller classes unless you apply class weighting or augmentation.

---

## 7. Quick reference — file relationships

```
training_data.csv  ──┐
                      ├──► (Section 4 mapping) ──► trainable (X, multi-hot Y) dataset
labels_schema.json ──┘

oisd_corpus.json   ──► additional structured case detail (narrative/root-cause/recommendations)
                        for building richer features or a retrieval/explanation layer

oisd_index.csv     ──► source of ~59 additional case studies to expand the dataset further
```
