"""Flask demonstration UI for the consolidated undergraduate ML project.

Run: python app.py    (requires pip install -r requirements.txt)
The app is a teaching demo, not a clinical decision support tool.
"""
from pathlib import Path
import json

import joblib
from flask import Flask, render_template, request

from src.modeling import validate_form

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / 'models' / 'diabetes_model.joblib'
app = Flask(__name__)


def template_context(**kwargs):
    metadata = json.loads((ROOT / 'models' / 'metadata.json').read_text(encoding='utf-8'))
    chosen = next(r for r in metadata['model_results'] if r['model'] == metadata['selected_model'])
    return dict(model_name=metadata['selected_model'],
                record_count=metadata['unique_rows'],
                holdout_accuracy=round(chosen['accuracy'] * 100, 1), **kwargs)


def load_model():
    if not ARTIFACT.exists():
        raise RuntimeError('No trained model found. Run `python train.py` first.')
    # Never load model files from untrusted sources: joblib uses pickle.
    return joblib.load(ARTIFACT)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', **template_context(prediction=None, probability=None, values=None))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        row = validate_form(request.form)
        trained = load_model()['model']
        predicted = int(trained.predict(row)[0])
        classes = list(trained.classes_)
        positive_probability = float(trained.predict_proba(row)[0, classes.index(1)])
        return render_template('index.html',
                               **template_context(prediction='Positive' if predicted else 'Negative',
                               probability=round(100 * positive_probability, 1),
                               values=dict(request.form)))
    except (ValueError, KeyError, TypeError) as error:
        return render_template('index.html', **template_context(error=str(error),
                               prediction=None, probability=None,
                               values=dict(request.form))), 400


if __name__ == '__main__':
    if not ARTIFACT.exists():
        raise SystemExit('Please run: python train.py')
    app.run(host='127.0.0.1', port=5000, debug=False)
