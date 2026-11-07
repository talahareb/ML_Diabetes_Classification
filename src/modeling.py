"""Data preparation, baselines, and reproducible models.

Consolidates the original LAU ML and BDA notebooks; all modeling choices
here are documented in README.md.
"""
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "diabetes_risk.csv"
FEATURES = [
    'Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
    'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
    'Itching', 'Irritability', 'delayed healing', 'partial paresis',
    'muscle stiffness', 'Alopecia', 'Obesity',
]
TARGET = 'class'
YES_NO = {"Yes": 1, "No": 0}
GENDER = {"Male": 1, "Female": 0}


def load_data(path=DATA_PATH):
    """Deduplicate before splitting, then encode explicit binary categories.

    Assumes source rows have a trusted consistent schema. Invalid entries are
    rejected, not guessed or silently repaired.
    """
    raw = pd.read_csv(path)
    if set(raw.columns) != set(FEATURES + [TARGET]):
        raise ValueError("Unexpected dataset columns")
    if raw.isna().any().any():
        raise ValueError("Source dataset contains missing values; review manually")
    duplicates = int(raw.duplicated().sum())
    data = raw.drop_duplicates().reset_index(drop=True).copy()
    data['Age'] = pd.to_numeric(data['Age'], errors='raise').astype(int)
    data['Gender'] = data['Gender'].map(GENDER)
    for name in FEATURES[2:]:
        data[name] = data[name].map(YES_NO)
    data[TARGET] = data[TARGET].map({"Positive": 1, "Negative": 0})
    if data.isna().any().any():
        raise ValueError("Unrecognized category; inspect source CSV")
    for name in FEATURES[1:] + [TARGET]:
        data[name] = data[name].astype(int)
    return data, {"raw_rows": len(raw), "duplicate_rows": duplicates, "unique_rows": len(data)}


def candidate_models(seed=42):
    """Baseline comparisons from both student notebooks + categorical-aware SMOTE.

    SMOTENC is appropriate because only Age is continuous. It runs strictly
    inside each CV/training fold, never on the test set.
    """
    logistic = SkPipeline([
        ('scale', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000, random_state=seed)),
    ])
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, random_state=seed)
    forest = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1,
    )
    voting = VotingClassifier(
        estimators=[
            ('logistic', SkPipeline([
                ('scale', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=2000, random_state=seed)),
            ])),
            ('tree', DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, random_state=seed)),
            ('forest', RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1)),
        ],
        voting='soft',
    )
    smote_forest = ImbPipeline([
        ('smotenc', SMOTENC(categorical_features=list(range(1, len(FEATURES))),
                             random_state=seed, k_neighbors=3)),
        ('classifier', RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                              random_state=seed, n_jobs=-1)),
    ])
    return {
        'Logistic Regression': logistic,
        'Decision Tree': tree,
        'Random Forest': forest,
        'Soft Voting Ensemble': voting,
        'Random Forest + SMOTENC': smote_forest,
    }


def validate_form(form):
    """Convert web-form values to the same ordered features used for training."""
    age = int(form['age'])
    if age < 1 or age > 120:
        raise ValueError('Age must be between 1 and 120')
    gender = str(form['gender'])
    if gender not in GENDER:
        raise ValueError('Gender must be Male or Female')
    names = [
        'polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness',
        'polyphagia', 'genital_thrush', 'visual_blurring', 'itching',
        'irritability', 'delayed_healing', 'partial_paresis',
        'muscle_stiffness', 'alopecia', 'obesity',
    ]
    vals = []
    for name in names:
        text = str(form[name])
        if text not in {'0', '1'}:
            raise ValueError('Please choose Yes/No for each symptom')
        vals.append(int(text))
    values = [age, GENDER[gender], *vals]
    return pd.DataFrame([values], columns=FEATURES)
