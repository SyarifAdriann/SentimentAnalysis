import json
from pathlib import Path
import sys

import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.models import training

DATA_PATH = ROOT / 'data' / 'processed' / 'tweets_normalized.parquet'
OUTPUT_PATH = ROOT / 'reports' / 'extra_model_comparison.json'

if not DATA_PATH.exists():
    raise FileNotFoundError('Processed dataset not found. Run phase1 first.')

print('Loading dataset...')
df = pd.read_parquet(DATA_PATH)
X, y = training.prepare_dataset(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

def result_dict(name, y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'per_class': {label: metrics for label, metrics in report.items() if label in {'negative', 'neutral', 'positive'}},
    }

models = {
    'sgd_classifier': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=6000)),
        ('clf', SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, class_weight='balanced', max_iter=2000, tol=1e-4, random_state=42)),
    ]),
    'ridge_classifier': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=6000)),
        ('clf', RidgeClassifier(class_weight='balanced')),
    ]),
    'multinomial_nb': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=6000)),
        ('clf', MultinomialNB(alpha=0.5)),
    ]),
    'random_forest': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=4000)),
        ('todense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1))
    ])
}

results = {}

for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    stats = result_dict(name, y_test, preds)
    results[name] = stats
    print(f"{name} accuracy: {stats['accuracy']:.4f}, macro F1: {stats['macro_f1']:.4f}")

OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding='utf-8')
print(f'Saved detailed metrics to {OUTPUT_PATH}')




