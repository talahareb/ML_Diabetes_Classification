# Diabetes Classification | Machine Learning & Data Analytics

**An end-to-end supervised learning project for classifying diabetes status using demographic information and symptom indicators.**

**Project area:** Machine Learning · Data Analytics · Classification · Model Evaluation  
**Technologies:** Python · pandas · scikit-learn · imbalanced-learn · Matplotlib · Flask  
**Academic context:** Lebanese American University (LAU) · Consolidated ML and Business Data Analytics group coursework

![Interactive classification web app, displaying a model prediction](docs/images/app-preview.png)

## Overview

This repository combines the most useful work from two related LAU course projects into one reproducible machine learning workflow. It covers exploratory data analysis, duplicate detection, feature encoding, multiple classification algorithms, class-imbalance handling, model evaluation, explainability plots, and an interactive Flask demonstration.

The problem is to **classify the positive/negative diabetes label in a symptom-based dataset**. The dataset does not contain future diabetes-onset outcomes, so the model is **not a predictor of future disease or a clinical diagnostic system**.

## Key features

- **Data preparation:** inspect missing values, remove exact duplicate rows before the train/test split, encode categorical attributes, and analyze class balance.
- **Model comparison:** Logistic Regression, Decision Tree, Random Forest, and Soft Voting Ensemble.
- **Imbalance experiment:** Random Forest with SMOTENC, which resamples categorical and numeric training features without touching the test set.
- **Evaluation:** stratified five-fold cross-validation, accuracy, balanced accuracy, precision, recall, F1-score, ROC–AUC, and confusion matrix.
- **Interpretation:** permutation feature importance, exploratory correlation heatmap, and a sample tree from the forest.
- **Web demo:** an interactive Flask form for all 16 features with the predicted class and corresponding positive-class probability.

## Dataset and evaluation

The supplied dataset contains **520 rows and 17 columns** (16 predictors plus one label). Removing **269 exact duplicate rows** leaves **251 distinct records** (173 positive, 78 negative). The consolidated evaluation uses an 80/20 stratified split: **200 training records, 51 held-out records**. The same split is used for all approaches.

Model selection is based on **mean balanced accuracy across five stratified folds of the training set**, not on choosing whichever model scores highest on the holdout set.

| Model | Training CV balanced accuracy | Holdout accuracy | Holdout balanced accuracy |
|---|---:|---:|---:|
| Logistic Regression | 88.1% | 80.4% | 82.3% |
| Decision Tree | 85.0% | 76.5% | 81.2% |
| **Random Forest (selected)** | **89.8%** | **90.2%** | **89.5%** |
| Soft Voting Ensemble | 88.2% | 86.3% | 88.3% |
| Random Forest + SMOTENC | 89.4% | 92.2% | 92.6% |

**Selected model:** Random Forest. On the holdout set it achieved **90.2% accuracy**, **92.8% positive-class F1**, and **96.3% ROC–AUC** (rounded). The SMOTENC variant happened to obtain a higher holdout accuracy, but was *not* selected because its training cross-validation balanced accuracy was lower.

The holdout set is small and results should be interpreted cautiously. The experiments are **educational**, with no independent external or clinical validation.

### Model comparison

![Comparison of cross-validation and holdout balanced accuracy](docs/images/model-comparison.png)

### Confusion matrix and ROC curve

| Holdout confusion matrix | ROC curve |
|---|---|
| ![Confusion matrix](docs/images/confusion-matrix.png) | ![ROC curve](docs/images/roc-curve.png) |

### Feature importance

![Feature permutation importance](docs/images/feature-importance.png)

Feature importance describes predictive associations in this dataset, **not causal effects**. Additional visualizations, including a correlation heatmap, decision-tree preview, class distribution, and screenshots of the application, are available in [`docs/images/`](docs/images/).

## Run locally

**Requirements:** Python 3.10 or newer and an internet connection to install Python packages.

**1. Create and activate an environment**

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Verify and (optionally) regenerate models and figures**

```bash
python tests.py
python train.py
```

The ZIP already contains the trained model and plots; `python train.py` regenerates them with your installed package versions. Models saved with `joblib` may require retraining when scikit-learn versions differ. Only load artifacts from sources you trust.

**3. Launch the Flask web application**

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser and fill in the form. A sample output is shown below.

![Full-page application screenshot with sample classification](docs/images/app-prediction.png)

**4. Explore the notebook (optional)**

```bash
jupyter notebook notebooks/Consolidated_Diabetes_Classification.ipynb
```

The notebook is stored with executed outputs and can be opened directly in GitHub.

## Repository structure

```text
Diabetes-Classification-ML/
├── README.md                         Project documentation
├── ACKNOWLEDGMENTS.md               Original academic teams and dataset attribution
├── train.py                         Reproducible comparison and figures
├── app.py                           Flask form and inference
├── tests.py                         Basic data/model checks
├── requirements.txt                 Python dependencies
├── data/
│   └── diabetes_risk.csv            Original supplied dataset
├── src/
│   ├── modeling.py                  Data preparation + model definitions
│   └── predict.py                   Minimal inference example
├── models/
│   ├── diabetes_model.joblib        Fitted selected model
│   ├── metadata.json                Evaluation and data profile
│   └── model_comparison.csv         Full metric comparison
├── notebooks/
│   └── Consolidated_Diabetes_Classification.ipynb
├── templates/                       Web-app HTML
├── static/                          CSS + original project background image
├── docs/images/                     Actual UI screenshots and evaluated plots
└── archive/                          Original ML/BDA notebooks, reports, presentation
```



**Dataset source, as cited in the reports:** [Diabetes Risk Prediction on Kaggle](https://www.kaggle.com/datasets/rcratos/diabetes-risk-prediction/data). 
