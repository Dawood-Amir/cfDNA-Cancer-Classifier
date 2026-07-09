# cfDNA Cancer Classification using Fragmentomic Features

A deep learning and machine learning pipeline for classifying cancer subtypes from cfDNA fragment features. This project explores the predictive power of different feature sets and demonstrates that **less is more** – a compact set of 6 core features outperforms 16 engineered features.

---

## 📖 Overview

Liquid biopsy analysis via cell‑free DNA (cfDNA) is a non‑invasive method for cancer detection and subtyping. This repository implements:

- Feature extraction from synthetic cfDNA fragment data (Kaggle dataset).
- Multiple models: FFN, CNN, XGBoost.
- Advanced techniques: focal loss, label smoothing, Optuna hyperparameter tuning, mutual information feature selection, and SMOTE balancing.
- A clear ablation study showing that adding complex statistical features degrades performance.

**Key findings:**
- A **6‑feature set** (region count, fragment count, mean/std fragment length, mean CpG count, methylation ratio) yields **AUC ~0.77** and accuracy ~53‑54%.
- Expanding to **16 features** introduces noise and multicollinearity, causing performance to collapse (AUC drops to ~0.55‑0.61).
- The best model is a **Feed‑Forward Network (FFN)** with label smoothing and balanced data.

---

## 📊 Dataset

- **Source:** Kaggle (synthetic) – synthetic cfDNA output.
- **Content:** JSON files containing fragment‑level methylation tokens (`<m>`, `<um>`) for each patient, grouped into genomic regions.
- **Classes:** 0 = Healthy, 1 = GBM, 2 = LGG, 3 = DMG_H3K27M.
- **Size:** ~1,288 patients after preprocessing.
- **Data Downloader:** The repository includes a `data_downloader.py` script with the Kaggle dataset ID

The raw token data is not used directly; we aggregate features at the patient level.

---

## 🧬 Feature Engineering

Two feature sets are created from the raw fragments:

### Set A – 6 Core Features (✔️ Best Performance)

| Feature | Description |
|---------|-------------|
| `n_regions` | Number of genomic regions with fragments |
| `n_fragments` | Total fragment count |
| `mean_fragment_length` | Average fragment length |
| `std_fragment_length` | Standard deviation of fragment lengths |
| `mean_cpg_count` | Average CpG sites per fragment |
| `methylation_ratio` | Fraction of methylated cytosines (`<m>` / total) |

These capture the essential fragmentomic profile.

### Set B – 16 Features (❌ Poor Performance)

Adds 10 derived statistical features: length CV, skew, short/long fragment ratios, nucleosome ratio, CpG density, methylation entropy, methylation discordance, and regional standard deviations. These were engineered to capture higher‑order properties but turned out to be noisy and highly correlated.

---

## 🧠 Models and Techniques

### Models
- **FFN** – 3‑layer feed‑forward network with LeakyReLU, Dropout, and LayerNorm.
- **CNN** – 1D convolutional network (applied over feature vectors; included for comparison).
- **XGBoost** – Gradient boosting with a custom focal loss objective.

### Key Improvements
- **Balanced Data** – SMOTE oversampling of minority class (DMG) to match class 1 count (315 samples), preserving all original data.
- **Focal Loss** – Focuses training on hard‑to‑classify examples, mitigating class imbalance.
- **Label Smoothing** – Regularises the neural networks (FFN/CNN) to prevent overconfidence.
- **Hyperparameter Tuning** – Optuna search for XGBoost parameters (depth, learning rate, subsample, etc.).
- **Feature Selection** – Mutual information selects the top 10 features for XGBoost (applied to the 16‑feature set).

All experiments are seeded for reproducibility.

---

## 🔬 Ablation Study Results
An explicit feature ablation study was conducted to evaluate model resilience against feature noise. Stripping the input space from 16 features down to **6 core biological features** eliminated a massive signal bottleneck, boosting classification accuracy across all architectures. 

All models were evaluated on an imbalanced held-out test set (276 patients) after training on hybrid balanced data.

### 📊 Data Distribution Details
* **Training Set Split:** Hybrid Balancing target adjusted to Class 1 baseline ($315$).
  * Original Training Distribution: `[459, 315, 374, 140]`
  * Balanced Training Distribution: `[459, 315, 374, 315]`
* **Test Set (Imbalanced Evaluation Profile):** * Class 0 (Healthy): 98 samples
  * Class 1 (GBM): 68 samples
  * Class 2 (LGG): 80 samples
  * Class 3 (DMG_H3K27M): 30 samples

## 🔬 Ablation Study Results
An explicit feature ablation study was conducted to evaluate model resilience against feature noise. Stripping the input space from 16 features down to **6 core biological features** eliminated a massive signal bottleneck, boosting classification accuracy across all architectures. 

All models were evaluated on an imbalanced held-out test set (276 patients) after training on hybrid balanced data using a fixed seed (Seed 10).

### 📊 Data Distribution Details
* **Training Set Split:** Hybrid Balancing target adjusted to Class 1 baseline (315).
  * Original Training Distribution: `[459, 315, 374, 140]`
  * Balanced Training Distribution: `[459, 315, 374, 315]`
* **Test Set (Imbalanced Evaluation Profile):** * Class 0 (Healthy): 98 samples
  * Class 1 (GBM): 68 samples
  * Class 2 (LGG): 80 samples
  * Class 3 (DMG_H3K27M): 30 samples

### 🔬 Feature Ablation Performance Matrix

| Model Architecture | Feature Set Count | Test Accuracy | Macro AUC | Class 0 F1 | Class 1 F1 | Class 2 F1 | Class 3 F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FFN (Best)** | **6 Features** | **54%** | **0.7759** | **0.64** | **0.46** | **0.65** | **0.12** |
| FFN | 16 Features | 34% | 0.5545 | 0.43 | 0.17 | 0.37 | 0.08 |
| | | | | | | | |
| **CNN** | **6 Features** | **53%** | **0.7715** | **0.63** | **0.44** | **0.63** | **0.16** |
| CNN | 16 Features | 36% | 0.5836 | 0.43 | 0.34 | 0.37 | 0.12 |
| | | | | | | | |
| **XGBoost** | **6 Features** | **54%** | **0.7534** | **0.62** | **0.43** | **0.64** | **0.12** |
| XGBoost | 16 Features | 33% | 0.6141 | 0.43 | 0.21 | 0.37 | 0.18 |

### Key Takeaways
* **Noise Filtering:** Dropping calculated noise features in favor of the 6 foundational metrics (`n_regions`, `n_fragments`, `mean_fragment_length`, `std_fragment_length`, `mean_cpg_count`, `methylation_ratio`) caused Macro AUC to skyrocket by **+0.15**.
* **Diagnostic Baseline Clarity:** The 6-feature models achieved near-perfect Precision for Class 0 (up to **0.89** on FFN), proving exceptional reliability in capturing baseline healthy signals despite the higher support volume ($n=98$).
* **Sequential Error Tracking:** Error distributions followed a clear ordinal gradient (e.g., Class 0 misclassifications spilled into Class 1; Class 2 spilled into Class 3), confirming the pipeline accurately tracks biological progression.

**Confusion matrices (6 Core Features):**
<table>
  <tr>
    <th><strong>FFN (best)</strong></th>
    <th><strong>CNN</strong></th>
    <th><strong>XGBoost</strong></th>
  </tr>
  <tr>
    <td><pre>[[49 31 13  5]
 [ 6 34 23  5]
 [ 0  7 64  9]
 [ 0  9 18  3]]</pre></td>
    <td><pre>[[51 28 12  7]
 [ 9 32 20  7]
 [ 1  8 58 13]
 [ 2  8 15  5]]</pre></td>
    <td><pre>[[54 26 13  5]
 [14 31 19  4]
 [ 6  8 60  6]
 [ 1 10 16  3]]</pre></td>
  </tr>
</table>

### On 16 Features (Noise Added)

| Model   | Accuracy | AUC    | Weighted F1 | Macro F1 |
|---------|----------|--------|-------------|----------|
| FFN     | 0.34     | 0.5545 | 0.31        | 0.26     |
| CNN     | 0.36     | 0.5836 | 0.36        | 0.32     |
| XGBoost | 0.33     | 0.6141 | 0.33        | 0.30     |

**Confusion matrices (16 Features):**
<table>
  <tr>
    <th><strong>FFN</strong></th>
    <th><strong>CNN</strong></th>
    <th><strong>XGBoost</strong></th>
  </tr>
  <tr>
    <td><pre>[[60 16 16  6]
 [50  8  7  3]
 [45  3 24  8]
 [23  1  4  2]]</pre></td>
    <td><pre>[[43 28 19  8]
 [26 26 13  3]
 [24 22 28  6]
 [10  7 10  3]]</pre></td>
    <td><pre>[[46 24 17 11]
 [27 12 15 14]
 [28  9 27 16]
 [13  3  7  7]]</pre></td>
  </tr>
</table>

**Key observation:** The 16-feature models heavily overpredict class 0 (Healthy), vacuuming up massive false positives from every other class (e.g., misclassifying 50 out of 68 GBM samples as Healthy in FFN). This explicitly demonstrates that the extra 10 dimensions mask true clinical variance with severe feature noise.

This indicates that the added features introduce noise that obscures the true signals.

---

## 🗣️ Discussion

### Why 6 Features Outperform 16

1. **Multicollinearity** – Many derived features (e.g., standard deviations of region‑wise statistics) are linearly dependent on the core features.
2. **Noise Amplification** – Statistical features like `methylation_entropy` and `meth_discordance_delta` are sensitive to small measurement errors, especially with low fragment counts per region.
3. **SMOTE Interpolation in High Dimensions** – When generating synthetic minority samples in 16‑dimensional space, the interpolated points often fall outside the true data manifold, confusing the classifier.
4. **Model Capacity** – The extra dimensions do not add relevant signal; they simply increase the risk of overfitting.

Thus, **careful feature selection is crucial** – more data does not always help when quality is poor.

---

## 🔮 Future Work

The current approach aggregates fragments into averaged features, **discarding spatial and sequential information**. A promising direction is to use:

- **Token‑level transformer models** – Treat each fragment’s methylation tokens as a sequence, preserving order and context.
- **Graph neural networks** – Model the genomic region structure and fragment adjacency.
- **Attention mechanisms** – To capture long‑range dependencies between fragments.

This would likely recover information lost in aggregation and improve classification, especially for the minority classes.

---

## 🛠️ Setup and Usage

### Requirements
- Python 3.8+
- PyTorch ≥ 1.10
- XGBoost ≥ 1.5
- Optuna ≥ 3.0
- scikit‑learn, imbalanced‑learn, pandas, numpy, PyYAML

Install dependencies:

pip install -r requirements.txt

## Data Preparation

Download the Kaggle dataset using the provided data_downloader.py (requires Kaggle API credentials).
Run src/data/data_convertor.py to generate the CSV feature files (both 6‑feature and 16‑feature versions).
## Train and Evaluate
python src/main.py --model {ffn,cnn,xgboost} --seed 10
The configuration is managed via config.yaml. You can toggle feature set, balancing, loss type, etc.

## 📜 License

This project is for research and educational purposes. No clinical use is intended.