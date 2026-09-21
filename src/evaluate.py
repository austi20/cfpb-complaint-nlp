"""Score the locked model on the held-out test slice, once.

LinearSVC won on validation in train.py, so that is the model. It is refit on
everything before the test cutoff (train plus validation), then scored on the
test slice a single time. Nothing is tuned after this runs.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.svm import LinearSVC

from train import load_data, score, split

HERE = os.path.dirname(__file__)
CONFUSION_PATH = os.path.join(HERE, "..", "figures", "confusion_matrix.png")
TERMS_PATH = os.path.join(HERE, "..", "figures", "top_terms.png")

TOP_N = 15


def main():
    df = load_data()
    train, val, test = split(df)

    # validation is spent, fold it back in so the model sees up to the cutoff
    fit_on = pd.concat([train, val])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    x_fit = vectorizer.fit_transform(fit_on["text"])
    x_test = vectorizer.transform(test["text"])

    model = LinearSVC(class_weight="balanced", random_state=42)
    model.fit(x_fit, fit_on["label"])
    predicted = model.predict(x_test)

    majority = fit_on["label"].value_counts().idxmax()
    baseline = score(test["label"], [majority] * len(test))
    final = score(test["label"], predicted)

    print(f"fit on {len(fit_on)} rows through {fit_on['date_received'].max().date()}")
    print(f"test   {len(test)} rows, {test['date_received'].min().date()} "
          f"to {test['date_received'].max().date()}\n")
    print(f"baseline (always '{majority}')  acc {baseline['accuracy']:.4f}  "
          f"macro-F1 {baseline['macro_f1']:.4f}")
    print(f"LinearSVC                       acc {final['accuracy']:.4f}  macro-F1 {final['macro_f1']:.4f}")
    print(f"lift {final['macro_f1'] - baseline['macro_f1']:.4f} macro-F1\n")

    print(classification_report(test["label"], predicted, zero_division=0, digits=3))

    save_confusion_matrix(model, x_test, test["label"])
    save_top_terms(model, vectorizer)


def save_confusion_matrix(model, x_test, true_labels):
    """Normalized by true class, so small classes are still readable."""
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_estimator(
        model, x_test, true_labels, normalize="true", values_format=".2f",
        xticks_rotation=45, cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title("LinearSVC on the held-out test slice, normalized by true class")
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=150)
    plt.close(fig)
    print(f"wrote {CONFUSION_PATH}")


def save_top_terms(model, vectorizer):
    """The 15 heaviest-weighted terms the classifier uses for each class."""
    terms = vectorizer.get_feature_names_out()
    classes = model.classes_

    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    for ax, class_index in zip(axes.flat, range(len(classes))):
        weights = model.coef_[class_index]
        top = weights.argsort()[-TOP_N:]
        ax.barh([terms[i] for i in top], weights[top], color="steelblue")
        ax.set_title(classes[class_index], fontsize=11)
        ax.tick_params(labelsize=9)

    fig.suptitle(f"Top {TOP_N} weighted terms per class, LinearSVC", fontsize=14)
    plt.tight_layout()
    plt.savefig(TERMS_PATH, dpi=150)
    plt.close(fig)
    print(f"wrote {TERMS_PATH}")


if __name__ == "__main__":
    main()
