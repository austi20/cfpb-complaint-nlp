"""Predict a complaint's product category from its narrative.

Splits chronologically, scores a majority-class baseline, then compares
TF-IDF into LogisticRegression and LinearSVC on validation macro-F1.
The test slice is not touched here. That is evaluate.py.
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

HERE = os.path.dirname(__file__)
PARQUET_PATH = os.path.join(HERE, "..", "data", "raw", "complaints.parquet")

# CFPB renamed several products over the years. Merge the aliases.
PRODUCT_MAP = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit reporting": "Credit reporting",
    "Credit card or prepaid card": "Credit card",
    "Credit card": "Credit card",
    "Prepaid card": "Credit card",
    "Checking or savings account": "Bank account",
    "Bank account or service": "Bank account",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Money transfers": "Money transfer",
    "Virtual currency": "Money transfer",
    "Payday loan, title loan, or personal loan": "Payday or personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday or personal loan",
    "Payday loan": "Payday or personal loan",
}

MIN_CLASS_SIZE = 500
TRAIN_END = 0.70
VAL_END = 0.85
N_BOOTSTRAP = 1000


def build_labels(products):
    """Merge the renamed aliases, then bin anything too rare to predict."""
    labels = products.map(PRODUCT_MAP).fillna(products)
    counts = labels.value_counts()
    rare = counts[counts < MIN_CLASS_SIZE].index
    return labels.where(~labels.isin(rare), "Other")


def strip_redaction(narratives):
    """XXXX is redaction, not signal."""
    return narratives.str.replace(r"X{2,}", " ", regex=True)


def load_data():
    df = pd.read_parquet(PARQUET_PATH)
    df["date_received"] = pd.to_datetime(df["date_received"])
    df["label"] = build_labels(df["product"])
    df["text"] = strip_redaction(df["narrative"])
    return df.sort_values("date_received")


def split(df):
    """Oldest complaints train, newest test. Random would leak vocabulary."""
    n = len(df)
    train = df.iloc[: int(n * TRAIN_END)]
    val = df.iloc[int(n * TRAIN_END) : int(n * VAL_END)]
    test = df.iloc[int(n * VAL_END) :]
    return train, val, test


def score(true_labels, predicted):
    return {
        "accuracy": accuracy_score(true_labels, predicted),
        "macro_f1": f1_score(true_labels, predicted, average="macro", zero_division=0),
    }


def bootstrap_gap(true_labels, first, second):
    """95% interval on macro F1 of first minus second, same rows resampled."""
    true_labels = np.asarray(true_labels)
    first = np.asarray(first)
    second = np.asarray(second)
    rng = np.random.default_rng(42)
    gaps = []
    for _ in range(N_BOOTSTRAP):
        rows = rng.integers(0, len(true_labels), len(true_labels))
        gap = (f1_score(true_labels[rows], first[rows], average="macro", zero_division=0)
               - f1_score(true_labels[rows], second[rows], average="macro", zero_division=0))
        gaps.append(gap)
    return np.percentile(gaps, [2.5, 97.5])


def main():
    df = load_data()
    train, val, test = split(df)

    print(f"train {len(train)} rows, through {train['date_received'].max().date()}")
    print(f"val   {len(val)} rows, {val['date_received'].min().date()} to {val['date_received'].max().date()}")
    print(f"test  {len(test)} rows, {test['date_received'].min().date()} to {test['date_received'].max().date()}")
    print(f"{df['label'].nunique()} classes after merging aliases\n")

    # baseline first, so the models have something to beat
    majority = train["label"].value_counts().idxmax()
    baseline = score(val["label"], [majority] * len(val))
    print(f"baseline (always '{majority}')  acc {baseline['accuracy']:.4f}  macro-F1 {baseline['macro_f1']:.4f}")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    x_train = vectorizer.fit_transform(train["text"])
    x_val = vectorizer.transform(val["text"])
    print(f"{x_train.shape[1]} tf-idf features\n")

    # balanced weights because Credit reporting is 60% of the training rows
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "LinearSVC": LinearSVC(class_weight="balanced", random_state=42),
    }

    results = {}
    predictions = {}
    for name, model in models.items():
        model.fit(x_train, train["label"])
        predictions[name] = model.predict(x_val)
        results[name] = score(val["label"], predictions[name])
        print(f"{name:20} acc {results[name]['accuracy']:.4f}  macro-F1 {results[name]['macro_f1']:.4f}")

    best = max(results, key=lambda name: results[name]["macro_f1"])
    lift = results[best]["macro_f1"] - baseline["macro_f1"]
    print(f"\nbest on validation: {best}, macro-F1 {results[best]['macro_f1']:.4f}, "
          f"{lift:.4f} above the baseline")

    low, high = bootstrap_gap(val["label"], predictions["LinearSVC"], predictions["LogisticRegression"])
    print(f"LinearSVC minus LogisticRegression, 95% bootstrap interval {low:.4f} to {high:.4f}")


if __name__ == "__main__":
    main()
