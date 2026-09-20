"""Predict a complaint's product category from its narrative.

Splits chronologically, scores a majority-class baseline, then compares
TF-IDF into LogisticRegression and LinearSVC on validation macro-F1.
The test slice is not touched here.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.svm import LinearSVC

HERE = os.path.dirname(__file__)
PARQUET_PATH = os.path.join(HERE, "..", "data", "raw", "complaints.parquet")
FIGURE_PATH = os.path.join(HERE, "..", "figures", "confusion_matrix.png")

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


def load_data():
    df = pd.read_parquet(PARQUET_PATH)
    df["date_received"] = pd.to_datetime(df["date_received"])

    df["label"] = df["product"].map(PRODUCT_MAP).fillna(df["product"])
    counts = df["label"].value_counts()
    rare = counts[counts < MIN_CLASS_SIZE].index
    df.loc[df["label"].isin(rare), "label"] = "Other"

    # XXXX is redaction, not signal
    df["text"] = df["narrative"].str.replace(r"X{2,}", " ", regex=True)

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
    for name, model in models.items():
        model.fit(x_train, train["label"])
        predicted = model.predict(x_val)
        results[name] = score(val["label"], predicted)
        print(f"{name:20} acc {results[name]['accuracy']:.4f}  macro-F1 {results[name]['macro_f1']:.4f}")

    best = max(results, key=lambda name: results[name]["macro_f1"])
    lift = results[best]["macro_f1"] - baseline["macro_f1"]
    print(f"\nbest on validation: {best}, macro-F1 {results[best]['macro_f1']:.4f}, "
          f"{lift:.4f} above the baseline")

    save_confusion_matrix(models[best], x_val, val["label"], best)


def save_confusion_matrix(model, x_val, true_labels, model_name):
    """Normalized by true class, so small classes are still readable."""
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_estimator(
        model, x_val, true_labels, normalize="true", values_format=".2f",
        xticks_rotation=45, cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(f"{model_name} on validation, normalized by true class")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    print(f"wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
