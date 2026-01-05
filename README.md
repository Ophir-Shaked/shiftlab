# ShiftLab — Distribution Shift Laboratory
Author: Ophir Shaked


ShiftLab is a lightweight, reproducible experimentation framework for studying **the robustness of classical machine-learning models under controlled distribution shifts**.

The project is designed for clean and transparent experimentation: strict separation between training, validation, and testing, deterministic execution, explicit generation of distribution shifts, and consistent evaluation using standard metrics.



---

## Core Goal

The central question addressed by ShiftLab is:

**How do common machine-learning models behave when the data distribution changes?**

Rather than treating distribution shift as noise or an unavoidable artifact, ShiftLab treats it as a **first-class experimental variable**.  
The framework enables systematic comparison across datasets, models, shift mechanisms, shift intensities, and random seeds.

---

## Key Principles

- **Strict train / validation / test separation**  
  Training, diagnostics, and final evaluation are clearly separated and never mixed.

- **Deterministic execution**  
  All sources of randomness are explicitly seeded, enabling reproducibility.

- **Explicit distribution shift**  
  Shifts are generated in a controlled and parameterized manner.

- **Score-based evaluation**  
  Models are primarily analyzed using probability scores and curve-based metrics, without additional tuning.

---

## What the System Actually Does


It is an experimental pipeline that:

1. Loads and validates a tabular classification dataset  
2. Splits the data into training, validation, and test subsets  
3. Trains classical machine-learning models on the training set  
4. Applies synthetic distribution shifts of varying intensity  
5. Evaluates how model performance degrades under shift  
6. Produces plots and structured artifacts for analysis  

The experimental protocol is fixed, explicit, and reproducible.

---

## Data and Splits

The dataset is partitioned into three **mutually exclusive subsets**:

**TRAIN**  
Used exclusively for model fitting, including preprocessing and parameter estimation.

**VAL**  
Used for diagnostics and intermediate analysis, such as inspecting performance outside the training data and generating evaluation plots.

**TEST**  
Used only once for the final evaluation of the trained model.

This enforces a clear separation between training, validation, and final evaluation stages.

---

## Supported Datasets

ShiftLab operates on tabular binary-classification datasets.

Example datasets included in the project:

- Adult income prediction  
- Credit default classification  

Before use, datasets undergo column normalization, feature typing, and validation of the target variable.

---

## Distribution Shift Generators

ShiftLab supports **explicit and controllable mechanisms** for generating distribution shift.  
Shift intensity is controlled via a continuous ratio between zero and one.

Implemented shift types include:

- **Bootstrap Noise** – resampling feature values with replacement  
- **Gaussian Noise** – additive noise on numeric features  
- **Gaussian Copula-Based Shift** – altering feature dependencies while preserving marginals  
- **Conditional Feature Shift** – perturbations conditioned on class or feature subsets  

---

## Models

ShiftLab evaluates a focused set of **classical machine-learning models commonly used for tabular data**:

- **Logistic Regression** – linear probabilistic baseline  
- **HistGradientBoostingClassifier** – non-linear tree boosting model  
- **ExtraTreesClassifier** – randomized ensemble of decision trees  
- **XGBoost** (optional) – included when available  

All models are trained on the training set only.

### Logistic Regression

The model computes a linear score:

z = w^T*x + b

and converts it to a probability using the sigmoid function:

σ(z) = 1 / (1 + e^(−z))

yielding:

P(y = 1 | x) = σ(w^T*x + b)

---

## Probability Outputs and Decision Rules

Models that support probability estimation are evaluated primarily in a **score-based manner** using predicted class probabilities.

When binary predictions are required, the default decision rule provided by the underlying library is used. No explicit threshold optimization or calibration procedure is performed.

---

## Metrics

Let:
TP = true positives  
TN = true negatives  
FP = false positives  
FN = false negatives  

**Accuracy**  
Accuracy = (TP + TN) / (TP + TN + FP + FN)

**ROC Curve**  
TPR = TP / (TP + FN)  
FPR = FP / (FP + TN)

**ROC-AUC**  
Area under the ROC curve, measuring ranking quality independent of a fixed threshold.

**Precision–Recall Curve**  
Precision = TP / (TP + FP)  
Recall = TP / (TP + FN)

**Confusion Matrix**  
[[TN, FP],  
 [FN, TP]]

---

## Artifacts

Each experiment produces a structured set of artifacts, including:

- ROC curves  
- Precision–Recall curves  
- Confusion matrices  
- Metric summaries  

Artifacts are stored in a deterministic directory structure, enabling reproducibility and comparison across experiments.

---

## Project Structure

shiftlab/
  main.py
  src/
    data/
    models/
    shifts/
    evaluation/
    plots/
    utils/
  results/

---

## Running the Project

Experiments are launched from the main entry point.

Configuration parameters are centralized and controlled via constants or flags, with no hidden defaults or inline magic numbers.

---

## Reproducibility Checklist

- No fitting on validation or test data  
- Deterministic execution via seeded randomness  
- Explicit and parameterized shift generation  
- Per-run artifacts stored for inspection  

---

