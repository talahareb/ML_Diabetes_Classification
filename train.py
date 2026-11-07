"""Train models, compare clean holdout results, save charts + app model.

Usage: python train.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from src.modeling import FEATURES, TARGET, candidate_models, load_data

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'docs' / 'images'
MODELS = ROOT / 'models'
SEED = 42


def save_figure(name):
    plt.savefig(OUT / name, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    data, profile = load_data()
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y,
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = candidate_models(SEED)
    rows = []
    trained = {}
    print(f"Source: {profile['raw_rows']} rows, {profile['duplicate_rows']} duplicates, {profile['unique_rows']} unique")
    for name, candidate in candidates.items():
        cv = cross_val_score(candidate, X_train, y_train, cv=folds,
                             scoring='balanced_accuracy', n_jobs=1)
        fitted = clone(candidate).fit(X_train, y_train)
        prediction = fitted.predict(X_test)
        probs = fitted.predict_proba(X_test)[:, list(fitted.classes_).index(1)]
        row = {
            'model': name,
            'cv_balanced_accuracy_mean': round(float(cv.mean()), 4),
            'cv_balanced_accuracy_std': round(float(cv.std()), 4),
            'accuracy': round(float(accuracy_score(y_test, prediction)), 4),
            'balanced_accuracy': round(float(balanced_accuracy_score(y_test, prediction)), 4),
            'precision_positive': round(float(precision_score(y_test, prediction)), 4),
            'recall_positive': round(float(recall_score(y_test, prediction)), 4),
            'f1_positive': round(float(f1_score(y_test, prediction)), 4),
            'roc_auc': round(float(roc_auc_score(y_test, probs)), 4),
        }
        rows.append(row)
        trained[name] = fitted
        print(f"{name:26s} train CV balanced accuracy={cv.mean():.3f}, holdout accuracy={row['accuracy']:.3f}")

    report = pd.DataFrame(rows)
    report.to_csv(MODELS / 'model_comparison.csv', index=False)
    # Select using TRAINING CV only; do not choose based on test results.
    best_index = int(report['cv_balanced_accuracy_mean'].idxmax())
    selected_name = str(report.loc[best_index, 'model'])
    model = trained[selected_name]
    test_probs = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    preds = model.predict(X_test)
    matrix = confusion_matrix(y_test, preds, labels=[0, 1])
    joblib.dump({'model': model, 'features': FEATURES, 'chosen_model': selected_name},
                MODELS / 'diabetes_model.joblib')
    profile.update({
        'unique_class_counts': {'Negative': int((y == 0).sum()), 'Positive': int((y == 1).sum())},
        'train_rows': int(len(X_train)), 'test_rows': int(len(X_test)),
        'random_seed': SEED, 'selected_model': selected_name,
        'selection_method': 'highest 5-fold stratified training CV balanced accuracy',
        'holdout_confusion_matrix_negative_positive': matrix.tolist(),
        'model_results': rows,
        'evaluation_note': 'Classify diabetes status (not onset); results are illustrative and not medically validated.'
    })
    (MODELS / 'metadata.json').write_text(json.dumps(profile, indent=2), encoding='utf-8')

    # Chart: feature relevance (permutation works for every estimator).
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(model, X_test, y_test, scoring='balanced_accuracy',
                                  n_repeats=20, random_state=SEED, n_jobs=1)
    importance = pd.Series(perm.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(10.5, 7.6))
    bars = ax.barh(importance.index, importance.values, color='#2767ae')
    ax.axvline(0, color='#64748b', linewidth=0.8)
    ax.set(xlabel='Change in balanced accuracy after feature permutation',
           title=f'Feature importance · {selected_name}')
    ax.tick_params(axis='y', labelsize=9)
    for bar in bars:
        if bar.get_width() > 0.015:
            ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                    f'{bar.get_width():.3f}', va='center', fontsize=8)
    ax.grid(axis='x', alpha=0.15)
    fig.tight_layout()
    save_figure('feature-importance.png')

    # Chart: confusion matrix on untouched holdout.
    fig, ax = plt.subplots(figsize=(6, 5))
    heat = ax.imshow(matrix, cmap='Blues', vmin=0)
    fig.colorbar(heat, ax=ax, fraction=0.045)
    ax.set_xticks([0, 1], ['Negative', 'Positive'])
    ax.set_yticks([0, 1], ['Negative', 'Positive'])
    ax.set(xlabel='Predicted', ylabel='Actual', title='Holdout confusion matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i,j] > matrix.max()/2 else '#13334b',
                    fontsize=19, fontweight='bold')
    fig.tight_layout()
    save_figure('confusion-matrix.png')

    # Chart: holdout ROC.
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='#1c65b8', linewidth=2.6,
            label=f'AUC = {roc_auc_score(y_test, test_probs):.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='#7f8a99', linewidth=1)
    ax.set(xlabel='False positive rate', ylabel='True positive rate',
           title='ROC curve · holdout set', xlim=(0,1), ylim=(0,1.02))
    ax.legend(loc='lower right'); ax.grid(alpha=.15)
    fig.tight_layout()
    save_figure('roc-curve.png')

    # Chart: all approaches (sort by independent train-only CV).
    display_df = report.sort_values('cv_balanced_accuracy_mean', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.7))
    yy = np.arange(len(display_df))
    ax.barh(yy, display_df['cv_balanced_accuracy_mean'], color='#2767ae',
            height=0.38, label='5-fold CV balanced accuracy (training)')
    ax.barh(yy+0.4, display_df['balanced_accuracy'], color='#8cb9d8',
            height=0.33, label='Holdout balanced accuracy')
    ax.set_yticks(yy+0.20, display_df['model'])
    ax.set_xlim(0, 1.0)
    ax.set(xlabel='Score (0–1)', title='Model comparison')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=9)
    ax.grid(axis='x', alpha=.13)
    fig.tight_layout()
    save_figure('model-comparison.png')

    # Chart: dataset profile.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    axes[0].bar(['Original rows', 'Unique rows'],
                [profile['raw_rows'], profile['unique_rows']], color=['#9caabf', '#2767ae'])
    for i, val in enumerate([profile['raw_rows'], profile['unique_rows']]):
        axes[0].text(i, val+7, str(val), ha='center', fontweight='bold')
    axes[0].set(ylim=(0, 580), title='Duplicates removed before splitting')
    axes[1].bar(['Negative', 'Positive'],
                [profile['unique_class_counts']['Negative'], profile['unique_class_counts']['Positive']],
                color=['#8cb9d8','#2767ae'])
    for i, val in enumerate([profile['unique_class_counts']['Negative'], profile['unique_class_counts']['Positive']]):
        axes[1].text(i, val+4, str(val), ha='center', fontweight='bold')
    axes[1].set(ylim=(0, 210), title='Target distribution · unique records')
    for ax in axes:
        ax.grid(axis='y', alpha=.12)
    fig.tight_layout()
    save_figure('dataset-profile.png')

    # Chart: compact correlation map (also included in original ML report).
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=78, ha='right', fontsize=8)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=8)
    ax.set(title='Exploratory feature correlations · deduplicated data')
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label='Pearson correlation')
    fig.tight_layout()
    save_figure('correlation-heatmap.png')

    # Chart: one tree from the selected Random Forest (limited depth for readability).
    from sklearn.tree import plot_tree
    forest = model.named_steps['classifier'] if hasattr(model, 'named_steps') else model
    if hasattr(forest, 'estimators_') and hasattr(forest.estimators_[0], 'tree_'):
        fig, ax = plt.subplots(figsize=(16, 6.7))
        plot_tree(forest.estimators_[0], max_depth=3, feature_names=FEATURES,
                  class_names=['Negative', 'Positive'], fontsize=8,
                  filled=True, rounded=True, impurity=False, ax=ax)
        ax.set_title('Illustrative decision tree · selected forest (first 4 levels)', pad=20)
        fig.tight_layout()
        save_figure('decision-tree-preview.png')

    print(f"Selected using training CV: {selected_name}")
    print(f"Holdout confusion matrix: {matrix.tolist()}")
    print('Saved model, metadata, CSV table, and five evaluation figures.')
    return profile


if __name__ == '__main__':
    main()
