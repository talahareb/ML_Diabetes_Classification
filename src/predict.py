"""Simple verified inference example without Flask."""
from pathlib import Path
import joblib
from src.modeling import FEATURES
import pandas as pd


def predict_sample():
    data = joblib.load(Path(__file__).resolve().parents[1] / 'models' / 'diabetes_model.joblib')
    model = data['model']
    # Example is synthetic for demonstration, not a patient's record.
    row = pd.DataFrame([[45, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], columns=FEATURES)
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0, list(model.classes_).index(1)])
    return {'predicted_class': 'Positive' if pred == 1 else 'Negative',
            'probability_positive': prob}


if __name__ == '__main__':
    print(predict_sample())
