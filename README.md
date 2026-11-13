# Explainable Neural Networks for Anisotropy Detection in Electrical Impedance Tomography (EIT)

An end-to-end, **explainable** deep-learning pipeline that detects **anisotropy** directly from **Dirichlet-to-Neumann (D–N) matrices** in EIT. A compact ANN is trained on synthetic D–N data and evaluated on real-time tank measurements; post-hoc **SHAP** and **gradient saliency** validate that the model focuses on physically meaningful electrode interactions.

> **Headline results**  
> • **Synthetic test set:** 92.0% accuracy, balanced accuracy 92.1%, F1(positive) 0.917, MCC 0.840.  
> • **Real-time data:** 72.3% accuracy at τ = 0.485, with **high anisotropy recall** (0.960) and lower specificity; confusion matrix  
> \[\[73 77\]; \[6 144\]\] for 300 samples.

---

## Why this matters

- **Direct decision on boundary data:** Frame anisotropy detection as binary classification on flattened 16×16 D–N matrices (R²⁵⁶), avoiding intermediate image reconstruction.  
- **Physics-aligned explanations:** SHAP and saliency consistently emphasize **drive–measure corridors** that straddle the anisotropy pathway—exactly where EIT sensitivity theory predicts signal.

---

## Repository contents

- `code/ANN model training.ipynb` — trains the compact MLP; exports feature stats and weights.  
- `code/preprocess_real_time_data_csv.ipynb` — converts raw complex CSVs to R²⁵⁶ feature vectors; consistent scaling with train-time stats when available.  
- `code/Real_time_data_analysis.ipynb` — evaluates the trained model on experimental data; threshold selection and metrics.  
- `code/Shap and saliency analysis of synthetic data.ipynb` / `SHAP_analysis_of_realtime_data.ipynb` — post-hoc XAI analyses (beeswarms, heatmaps, saliency maps).  
- `ann_anisotropy_classifier.pth` — trained ANN weights (single hidden layer).  
- `processed data/` — real-time features (`X_real_256.csv`) and labels (`y_real_256.csv`).  
- `Outputs/` — confusion matrices, SHAP and saliency figures used in the thesis.  
- [`EIT_Anisotropy_XAI_Thesis.pdf`](EIT_Anisotropy_XAI_Thesis.pdf) — full MSc thesis.

> See Appendix A (Code & Reproducibility) in the [thesis](EIT_Anisotropy_XAI_Thesis.pdf) for script variants and environment notes.

---

## Method at a glance

**Input features (R²⁵⁶):** Flattened 16×16 D–N matrices captured with **16 boundary electrodes** on a **28 mm radius** circular tank; boundary excitations use trigonometric patterns up to the 8th harmonic.

**Synthetic data generator:**  
Perturbations emulate hardware artefacts:
- Electrode angular **jitter** σθ = 0.05 rad,  
- **Additive Gaussian noise** with σ ∈ [0.05, 0.10],  
- **Boundary deformation**: 0.03-scale Gaussian with p=0.10,  
- **Electrode dropout** (single row/col) with p=0.15,  
- Dataset size **Nₛ = 1000**.

**Model:**  
- MLP, **256 → 16 → 2**, ReLU, **Dropout p=0.5**, softmax head; Glorot init.  
- Cross-entropy, **Adam(1e-3)**, batch 32, **20 epochs**; feature z-scoring saved to `train_stats_256.npz`.

**Explainability:**  
- **SHAP** (DeepExplainer) + **gradient saliency**; both localize importance to compact drive–measure bands; top-k features dominate attribution mass.

---

## Results

### Synthetic (held-out)
- Confusion: \[\[240 25\]; \[15 220\]\], **N=500**  
- **Acc 0.920**, **BalAcc 0.921**, **Precision(1) 0.898**, **Recall(1) 0.936**, **F1(1) 0.917**, **Spec 0.906**, **MCC 0.840**.

### Real-time (no retraining)
- Threshold **τ = 0.485**, Confusion: \[\[73 77\]; \[6 144\]\], **N=300**  
- **Acc 0.723**, **BalAcc 0.723**, **Recall(1) 0.960**, **Precision(1) 0.652**, **MCC 0.507**.  
- Operating-point trade-off: high sensitivity to anisotropy with reduced specificity; adjust τ or calibrate scores depending on application.

---

## Quickstart

### 1) Environment
- Python 3.10+, PyTorch, scikit-learn, NumPy/Pandas, Captum, SHAP. (MATLAB R2023b+ for synthetic generation, optional.)

### 2) Train (or skip if you use the provided weights)
Open **`ANN model training.ipynb`** and run all cells. This saves `ann_anisotropy_classifier.pth` and `train_stats_256.npz`.

### 3) Preprocess experimental CSVs
Use **`preprocess_real_time_data_csv.ipynb`** to parse complex CSV matrices into `X_real_256_bulk.csv` and `y_real_256_bulk.csv` (magnitude features, robust parsing). If stats are present, scaling is consistent with training.

### 4) Evaluate + threshold
Run **`Real_time_data_analysis.ipynb`** to compute metrics and select a threshold (or fix τ=0.485 to reproduce reported numbers).

### 5) Explainability
Open **`SHAP_analysis_of_realtime_data.ipynb`** (and the synthetic counterpart) to generate beeswarms and 16×16 attribution heatmaps. Expect compact lobes along anisotropy-sensitive corridors.

---

## Design choices & insights

- **Feature choice (D–N entries):** Preserves reciprocity structure and avoids reconstruction instabilities; vectorization to R²⁵⁶ yields well-conditioned inputs for supervised learning.  
- **Shallow MLP:** Adequate capacity without imposing spatial inductive biases alien to electrode geometry; also easier to interpret post-hoc.  
- **Explainability agreement:** SHAP (signed contributions) and saliency (gradient sensitivity) co-localize on the same electrode bands across both domains, boosting trust.

---

## Limitations

1) **Label surrogacy** in real-time data; 2) **2D modelling** (no out-of-plane effects); 3) **Single-frequency** regime; 4) **Score calibration** not applied prior to evaluation; 5) **Attribution uncertainty** not quantified.

---

## Roadmap

- **Calibration & operating-point selection** on held-out real-time splits (e.g., temperature scaling + τ tuned to app constraints).  
- **Simulation realism upgrades** (contact-impedance drift, correlated band noise, mild boundary ellipticity, session offsets).  
- **Lightweight domain adaptation** (last-layer tuning with MMD regularization).  
- **Multifrequency & temporal context** (stack {D(ωₖ)} or add sequence models to stabilise decisions).

---

## Citation

If you use this repository, please cite:

> **R. Suresh Raj**. *Explainable Neural Networks for Anisotropy Detection in Electrical Impedance Tomography (MSc Thesis)*, University of Limerick, **Aug 15, 2025**.

```bibtex
@mastersthesis{SureshRaj2025EITXAI,
  title  = {Explainable Neural Networks for Anisotropy Detection in Electrical Impedance Tomography},
  author = {Rohith Suresh Raj},
  school = {University of Limerick},
  year   = {2025},
  month  = aug
}
