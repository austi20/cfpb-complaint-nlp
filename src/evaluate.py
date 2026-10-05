"""Score the locked model on the held-out test slice, once.

LinearSVC won on validation in train.py, so that is the model. It is refit on
everything before the test cutoff (train plus validation), then scored on the
test slice a single time. Nothing is tuned after this runs.

LogisticRegression is scored on the same slice as a check on that choice, not
as a second chance to pick. The script also counts how many debt collection
errors cite the credit reporting statute, and how much of the test slice the
model could route on its own at 95% precision.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.svm import LinearSVC

from train import bootstrap_gap, load_data, score, split

HERE = os.path.dirname(__file__)
CONFUSION_PATH = os.path.join(HERE, "..", "figures", "confusion_matrix.png")
TERMS_PATH = os.path.join(HERE, "..", "figures", "top_terms.png")

TOP_N = 15
TARGET_PRECISION = 0.95

# credit repair letters cite the FCRA, real collection complaints rarely do
FCRA_CITATIONS = ["fcra", "fair credit reporting act", "1681", "605b"]


def fit_tfidf(fit_on, score_on):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    x_fit = vectorizer.fit_transform(fit_on["text"])
    x_score = vectorizer.transform(score_on["text"])
    return vectorizer, x_fit, x_score


def cites_fcra(text):
    lowered = text.lower()
    for phrase in FCRA_CITATIONS:
        if phrase in lowered:
            return True
    return False


def pick_cutoff(confidence, correct):
    """Lowest confidence cutoff where the routed rows hit the target precision."""
    for cutoff in np.arange(-1.0, 2.0, 0.01):
        routed = confidence >= cutoff
        if routed.sum() > 0 and correct[routed].mean() >= TARGET_PRECISION:
            return cutoff
    raise ValueError("no cutoff reaches the target precision")


def main():
    df = load_data()
    train, val, test = split(df)

    # validation is spent, fold it back in so the model sees up to the cutoff
    fit_on = pd.concat([train, val])
    vectorizer, x_fit, x_test = fit_tfidf(fit_on, test)

    svc = LinearSVC(class_weight="balanced", random_state=42)
    svc.fit(x_fit, fit_on["label"])
    svc_predicted = svc.predict(x_test)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    logreg.fit(x_fit, fit_on["label"])
    logreg_predicted = logreg.predict(x_test)

    majority = fit_on["label"].value_counts().idxmax()
    baseline = score(test["label"], [majority] * len(test))
    final = score(test["label"], svc_predicted)
    check = score(test["label"], logreg_predicted)

    print(f"fit on {len(fit_on)} rows through {fit_on['date_received'].max().date()}")
    print(f"test   {len(test)} rows, {test['date_received'].min().date()} "
          f"to {test['date_received'].max().date()}\n")
    print(f"baseline (always '{majority}')  acc {baseline['accuracy']:.4f}  "
          f"macro-F1 {baseline['macro_f1']:.4f}")
    print(f"LinearSVC                       acc {final['accuracy']:.4f}  macro-F1 {final['macro_f1']:.4f}")
    print(f"LogisticRegression              acc {check['accuracy']:.4f}  macro-F1 {check['macro_f1']:.4f}")
    print(f"lift {final['macro_f1'] - baseline['macro_f1']:.4f} macro-F1")

    low, high = bootstrap_gap(test["label"], svc_predicted, logreg_predicted)
    print(f"LinearSVC minus LogisticRegression, 95% bootstrap interval {low:.4f} to {high:.4f}\n")

    print(classification_report(test["label"], svc_predicted, zero_division=0, digits=3))
    print_per_class(test["label"], svc_predicted, logreg_predicted)
    print_fcra_errors(val, test, svc_predicted)
    print_routing(train, val, test, svc.decision_function(x_test).max(axis=1), svc_predicted)

    save_confusion_matrix(svc, x_test, test["label"])
    save_top_terms(svc, vectorizer)


def print_per_class(true_labels, svc_predicted, logreg_predicted):
    classes = sorted(true_labels.unique())
    svc_f1 = f1_score(true_labels, svc_predicted, labels=classes, average=None, zero_division=0)
    logreg_f1 = f1_score(true_labels, logreg_predicted, labels=classes, average=None, zero_division=0)
    print(f"{'class':25} {'LinearSVC F1':>13} {'LogReg F1':>10} {'rows':>6}")
    for i, name in enumerate(classes):
        rows = (true_labels == name).sum()
        print(f"{name:25} {svc_f1[i]:13.3f} {logreg_f1[i]:10.3f} {rows:6}")
    print()


def print_fcra_errors(val, test, svc_predicted):
    """Do the debt collection misses look like credit repair letters?"""
    debt = test["label"] == "Debt collection"
    called_reporting = debt & (svc_predicted == "Credit reporting")
    called_debt = debt & (svc_predicted == "Debt collection")
    cites = test["text"].apply(cites_fcra)

    print(f"debt collection test rows {debt.sum()}, called credit reporting {called_reporting.sum()} "
          f"({called_reporting.sum() / debt.sum():.3f})")
    print(f"  cite the FCRA: {cites[called_reporting].sum()} of those errors "
          f"({cites[called_reporting].mean():.3f}), against {cites[called_debt].mean():.3f} "
          f"of the ones it got right")
    print(f"  error rate when the complaint cites the FCRA {called_reporting[debt & cites].mean():.3f} "
          f"(n {(debt & cites).sum()}), when it does not {called_reporting[debt & ~cites].mean():.3f} "
          f"(n {(debt & ~cites).sum()})")

    val_debt = val[val["label"] == "Debt collection"]
    print(f"  share of debt collection citing the FCRA, validation "
          f"{val_debt['text'].apply(cites_fcra).mean():.3f}, test {cites[debt].mean():.3f}\n")


def print_routing(train, val, test, test_confidence, svc_predicted):
    """Cutoff set on validation by a model fit before it, then applied to test."""
    _, x_train, x_val = fit_tfidf(train, val)
    model = LinearSVC(class_weight="balanced", random_state=42)
    model.fit(x_train, train["label"])
    val_confidence = model.decision_function(x_val).max(axis=1)
    val_correct = model.predict(x_val) == val["label"].to_numpy()
    cutoff = pick_cutoff(val_confidence, val_correct)

    routed = test_confidence >= cutoff
    correct = svc_predicted == test["label"].to_numpy()
    not_reporting = test["label"].to_numpy() != "Credit reporting"
    print(f"routing cutoff {cutoff:.2f}, set for {TARGET_PRECISION:.0%} precision on validation")
    print(f"  test: routes {routed.sum()} of {len(test)} ({routed.mean():.3f}), "
          f"precision {correct[routed].mean():.3f}, human queue {(~routed).sum()}")
    print(f"  of the complaints that are not credit reporting, routes {routed[not_reporting].mean():.3f}\n")


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
