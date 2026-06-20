# 🔬 Diagnostic Pattern Recognition via K-Nearest Neighbors (KNN) Classifier

## 📌 Project Overview
This repository contains an end-to-end instance-based Machine Learning pipeline designed to classify diagnostic profiles using the **Wisconsin Breast Cancer Dataset**. By implementing a **K-Nearest Neighbors (KNN) Classification Engine**, this project explores feature engineering standardization frameworks, parameter space optimization via cross-validation tuning curves, and the deep asymmetric business/clinical risks associated with predictive diagnostic models.

---

## 📊 Dataset Specifications & Preparation
The dataset consists of clinical parameters evaluated from digitized images of fine needle aspirates (FNA) of breast masses.

### Data Engineering Protocol:
* **Null Verification:** Confirmed 100% data integrity with zero missing structural values across the dataset.
* **Target Classification Normalization:** Mapped target properties directly into explicit diagnostic vectors (`0` for Malignant, `1` for Benign).
* **Standard Scaling Normalization:** Because KNN leverages spatial geometry (Euclidean Distance Metric), features with larger raw magnitudes (e.g., `mean area` ranging over 2000) would completely overwhelm features with small values (e.g., `smoothness` under 0.1). A `StandardScaler` was applied to ensure equal feature weight distribution.
* **Validation Strategy:** Implemented a stratified 80/20 train-test split to preserve identical target proportions across partitions.

---

## ⚙️ Hyperparameter Optimization & Tuning
To identify the optimal number of voting neighbors ($k$), 5-Fold Cross-Validation was conducted across odd interval spaces to prevent geometric assignment conflicts:

* **Evaluated Hyperparameter Range:** $k \in [1, 3, 5, 7, 9, 11, 13, 15]$
* **Metric Driver:** Cross-Validated Classification Accuracy.
* **Distance Architecture:** Standard Euclidean Distance.

---

## 📈 Classification Performance Metrics

The optimized classification engine demonstrates exceptional baseline discrimination capacity across all validation profiles:

* **Overall Classification Accuracy:** `96.49%`
* **Precision Profile (Positive Predictive Value):** `95.95%`
* **Recall / Sensitivity Matrix:** `98.61%` (Successfully captures the highest proportion of target records).
* **Balanced F1-Score Profile:** `97.26%`

### 🧮 Model Confusion Matrix:
| | Predicted Malignant | Predicted Benign |
|---|---|---|
| **Actual Malignant** | **39** (True Negatives) | **3** (False Positives) |
| **Actual Benign** | **1** (False Negatives) | **71** (True Positives) |

---

## ⚖️ Domain Cost & Asymmetric Risk Analysis

Predictive model trade-offs carry heavily skewed real-world cost penalties in medical diagnostics:

1. **The Critical Penalty of False Negatives (Recall Failure):** Misclassifying a malignant case as benign delays lifesaving medical interventions. Minimizing this risk is paramount, which our pipeline successfully achieves by securing a **Recall of 98.61%** (only 1 case misclassified).
2. **The Buffer Cost of False Positives (Precision Failure):** Misclassifying a benign case as malignant triggers secondary screening, laboratory validations, and clinical anxiety. This functions as an operational and financial resource sunk cost but avoids immediate patient risk.
