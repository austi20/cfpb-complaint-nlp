"""NMF topic modeling inside each product, and the CFPB issue labels to compare against.

This is description, not prediction, so it runs on every row rather than on the
training slice. Prints the top terms per topic; the topic names in the README are
written by hand from these terms.
"""
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from train import load_data

N_TOPICS = 6
N_TERMS = 12
N_ISSUES = 6


def print_topics(narratives, product):
    vectorizer = TfidfVectorizer(
        stop_words="english", min_df=10, max_df=0.5, max_features=3000
    )
    x = vectorizer.fit_transform(narratives)
    terms = vectorizer.get_feature_names_out()

    # the default coordinate-descent solver crashes on the smaller products, mu does not
    model = NMF(
        n_components=N_TOPICS, init="nndsvda", solver="mu", max_iter=400, random_state=42
    )
    model.fit(x)

    print(f"\n### {product}  ({len(narratives)} complaints)")
    for index, weights in enumerate(model.components_):
        top = weights.argsort()[-N_TERMS:][::-1]
        print(f"  topic {index}: {', '.join(terms[i] for i in top)}")


def print_issues(frame, product):
    counts = frame["issue"].value_counts(normalize=True).head(N_ISSUES)
    print(f"  CFPB issue labels for {product}:")
    for issue, share in counts.items():
        print(f"    {share:5.1%}  {issue}")


def main():
    df = load_data()

    # Other is a bag of unrelated leftovers, so its topics would mean nothing
    products = [p for p in sorted(df["label"].unique()) if p != "Other"]

    for product in products:
        frame = df[df["label"] == product]
        print_topics(frame["text"], product)
        print_issues(frame, product)


if __name__ == "__main__":
    main()
