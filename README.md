# Breast Cancer Tumor Classification

## Overview

This project applies classical machine learning methods to classify breast tumors as malignant or benign using the Breast Cancer Wisconsin (Diagnostic) dataset. We compare four models and evaluate performance using cross-validation and hyperparameter tuning, with a focus on maximizing recall for malignant cases given the clinical stakes of a false negative.

---

## Clinical Motivation

Early detection of malignant breast tumors is critical for improving patient survival rates. Missing a malignant tumor is far more dangerous than a false alarm, so we prioritized recall as our primary evaluation metric throughout this project.

---

## Dataset

- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset (UCI / OpenML)
- **Samples:** 569
- **Features:** 30 numerical features derived from imaging data
- **Target:** 0 = Malignant, 1 = Benign
- No missing values were observed

---

## Models Evaluated

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

All models were tuned using GridSearchCV with 5-fold cross-validation, optimizing for macro recall.

---

## Results

| Model | Accuracy | Malignant Recall | Macro F1 | Best CV Recall |
|-------|----------|-----------------|----------|----------------|
| Logistic Regression | 96% | 0.98 | 0.96 | 0.9748 |
| Random Forest | 96% | 0.93 | 0.95 | 0.9607 |
| Gradient Boosting | 96% | 0.93 | 0.95 | 0.9724 |
| SVM | 96% | 0.98 | 0.95 | 0.9800 |

**Selected Model: Logistic Regression.** It matched SVM on malignant recall, achieved the highest macro F1, and is the most interpretable of the four which matters in a clinical context.

---

## Project Structure
```
PROJECT1_BREASTCANCER_ML/
│
├── models/
│   └── log_reg_breast_cancer.pkl
│
├── notebooks/
│   ├── Breast_Cancer_Tumor_Classification.ipynb
│   └── Demo.ipynb
│
├── src/
│   └── train.py
│
├── requirements.txt
└── README.md
```

---

## Setup

### (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Run Instructions

To retrain the model and save it:
```bash
python src/train.py
```

To see the model in action, open `notebooks/Model Demo.ipynb` and run all cells.

For the full analysis, open `notebooks/Breast Cancer Tumor Classification.ipynb`.