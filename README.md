# Breast Cancer Classification using Classical Machine Learning

## Project Overview

This project applies classical machine learning methods to classify breast tumors as malignant or benign using the Breast Cancer Wisconsin (Diagnostic) dataset.

The objective is to compare different models and evaluate their performance using cross-validation and hyperparameter tuning.

---

## Clinical Motivation

Early detection of malignant breast tumors is critical for improving patient survival rates. Machine learning models can assist clinicians by providing decision support based on structured medical features extracted from imaging data.

This project explores whether simple linear models are sufficient for this classification task.

---

## Dataset

- Source: Breast Cancer Wisconsin (Diagnostic) dataset (UCI / OpenML)
- Samples: 569
- Features: 30 numerical features
- Target:
  - 0 = Malignant
  - 1 = Benign

All features are continuous and no missing values were observed.

---

## Methods

The following models were evaluated:

- Logistic Regression (baseline)
- Random Forest (nonlinear comparison)

Validation strategy:
- 80/20 Train-Test split
- 5-fold Cross-Validation
- Hyperparameter tuning using GridSearchCV

---

## Results Summary

| Model | Cross-Validation Accuracy | Test Accuracy |
|--------|---------------------------|---------------|
| Logistic Regression | ~98.1% | ~97% |
| Random Forest | ~95.6% | ~96% |

Logistic regression demonstrated higher stability and slightly better performance.

---

## Project Structure
PROJECT1_BREASTCANCER_ML/
│
├── notebooks/
│ ├── EDA.ipynb
│ └── Demo.ipynb
│
├── src/
│ └── train.py
│
├── requirements.txt
└── README.md

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt

Run Training Script ：python src/train.py