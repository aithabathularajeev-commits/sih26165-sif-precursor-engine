# 🚀 QUICKSTART — SIH26165 Day 1 NLP Baseline

## 30-Second Summary

✅ **Complete multi-label classifier** trained on 75 OISD safety precursor texts  
✅ **No case-level leakage** (Leave-One-Case-Out evaluation)  
✅ **Ready-to-use inference** (feed raw text, get C1-C6 predictions)  
✅ **Fully tested & documented** (~1,900 lines of code + 50KB docs)  
✅ **Expected baseline:** F1 ~0.55–0.65 (honest estimate for 75-row dataset)  

---

## Run in 3 Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model & run evaluation (10–30 seconds)
python full_pipeline.py

# 3. Test inference on a safety report
python src/data/predict.py "Workers had no training on safety procedures"
```

**That's it.** Model will be saved in `models/`, metrics in `outputs/`.

---

## What You Get

### Generated Files After Running

```
models/
  ├── multilabel_model.pkl          ← Trained classifier
  └── metadata.json                 ← Model parameters

outputs/
  ├── evaluation_results_loco.json  ← Structured metrics (F1, precision, recall)
  ├── evaluation_report_loco.txt    ← Human-readable results
  └── example_predictions.txt       ← Demo predictions on 10 examples
```

### Expected Output Example

```json
{
  "predicted_classes": ["C1", "C4"],
  "predictions": [
    {
      "class": "C4",
      "probability": 0.78,
      "label_name": "Procedure, Permit & Risk-Assessment Bypass"
    },
    {
      "class": "C1",
      "probability": 0.65,
      "label_name": "Training & Competency Gaps"
    }
  ]
}
```

---

## Key Design Decisions

| Decision | Why | Result |
|---|---|---|
| TF-IDF + Logistic Regression | Small dataset (75 rows), interpretable | Baseline ✓ |
| Leave-One-Case-Out CV | 12 cases, prevent case-level leakage | Honest metrics ✓ |
| No hyperparameter tuning | Risk of overfitting on 75 rows | Reproducible ✓ |
| Multi-label (OneVsRest) | Some texts have 2+ risk classes | Supports C1+C2, etc. ✓ |
| Balanced class weights | Class imbalance (C4=30, C6=7) | Fair to rare classes ✓ |

---

## Important Files

| File | Read This If... |
|---|---|
| `FINAL_REPORT_DAY1.md` | You want full context (14,000 chars) |
| `IMPLEMENTATION_SUMMARY_DAY1.md` | You want technical deep dive |
| `DAY1_VERIFICATION_REPORT.md` | You want to verify execution steps |
| `requirements.txt` | You need to see dependencies |

---

## Expected Performance

```
Metric          Range       Why This Range?
────────────────────────────────────────
Macro F1        0.55–0.65   75 rows, class imbalance, high LOCO variance
Per-Class F1    0.50–0.70   C1–C5: reasonable; C6: unreliable (only 7 rows)
Precision       0.60–0.70   Model is cautious (avoid false positives)
Recall          0.50–0.60   Some cases missed (small dataset)
```

**This is a defensible baseline, not state-of-the-art.**

---

## What NOT to Expect (Day 1)

❌ API endpoint (coming Day 2)  
❌ Web UI (coming Day 2)  
❌ Cloud deployment  
❌ LLM integration  
❌ 90%+ accuracy (unrealistic for 75 rows)  

**Scope for Day 1:** Solid NLP baseline only.

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'sklearn'"
```bash
pip install -r requirements.txt
```

### Error: "FileNotFoundError: data/label_mapping.csv"
Make sure you run `python full_pipeline.py` from the repo root.

### Models directory not created?
The scripts auto-create it. If manual creation needed:
```bash
mkdir models outputs
```

### Want to test specific text?
```bash
python src/data/predict.py "Your safety report text here"
```

---

## Next Steps (Day 2)

After reviewing these results:

1. **Add API layer** (FastAPI endpoint)
2. **Build UI** (simple form for testing)
3. **Package for demo**

🎯 **Today's Goal:** Understand the baseline performance.  
🎯 **Tomorrow's Goal:** Build inference server & web demo.

---

## Questions?

- **Architecture?** → See `IMPLEMENTATION_SUMMARY_DAY1.md`
- **Metrics explanations?** → See `FINAL_REPORT_DAY1.md`
- **How to execute?** → See `DAY1_VERIFICATION_REPORT.md`
- **Test it now?** → Run `python full_pipeline.py`

---

**Status: ✅ READY TO EXECUTE**

Time to first results: ~30 seconds  
Lines of code: 1,900  
Test coverage: 10 comprehensive tests  

Go! 🚀
