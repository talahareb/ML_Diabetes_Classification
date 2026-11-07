"""Quick checks: python tests.py"""
from pathlib import Path
import joblib
from src.modeling import FEATURES, load_data, validate_form


def run():
    data, counts = load_data()
    assert counts == {'raw_rows': 520, 'duplicate_rows': 269, 'unique_rows': 251}, counts
    assert data.shape == (251, 17)
    assert data['class'].isin([0, 1]).all()
    assert not data.duplicated().any()
    params = {'age': '45', 'gender': 'Female'}
    for k in ('polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness',
              'polyphagia', 'genital_thrush', 'visual_blurring', 'itching',
              'irritability', 'delayed_healing', 'partial_paresis',
              'muscle_stiffness', 'alopecia', 'obesity'):
        params[k] = '0'
    features = validate_form(params)
    assert list(features.columns) == FEATURES
    assert features.iloc[0]['Gender'] == 0
    artifact = joblib.load(Path(__file__).parent / 'models' / 'diabetes_model.joblib')
    prediction = artifact['model'].predict(features)
    proba = artifact['model'].predict_proba(features)
    assert prediction[0] in (0, 1) and proba.shape == (1, 2)
    assert abs(proba.sum() - 1) < 1e-7
    print('PASS: source data, preprocessing, form encoding, saved model inference')


if __name__ == '__main__':
    run()
