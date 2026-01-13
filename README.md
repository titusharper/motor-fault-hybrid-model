# motor-fault-hybrid-model
# Motor Fault Diagnosis with Hybrid (1D + 2D) Fusion
Hybrid fault classification using **1D time-series statistical features** together with **high-resolution reassigned spectrogram** images extracted from **startup stator current** signals.

> **Note on data:** Raw current measurements were obtained from a university laboratory dataset. Sharing raw data publicly may be unethical / restricted. Therefore, this repository contains **code + pipeline** only (no raw data).

---

## What this repository contains
- **2D branch:** Reassigned spectrogram generation from segmented startup-current signals.
- **1D branch:** Statistical feature extraction per segment (≈19 features; excluding label/split columns).
- **Fusion model:** CNN + MLP fusion with **FiLM-style feature-wise affine modulation**.
- **Evaluation:** classification report, confusion matrix, and summary metrics.

---

## Results (Test Set)
- **Accuracy:** 80.79%
- **Macro F1-score:** 0.81  
- **Per-class F1:** Healthy 0.83 | Imbalance 0.78 | Loose 0.86 | Misalignment 0.78  
(Test set: 380 samples; 95 per class)

---

