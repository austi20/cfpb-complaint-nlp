"""Tests for the label building, redaction stripping and chronological split."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import MIN_CLASS_SIZE, build_labels, split, strip_redaction


def test_build_labels_merges_renamed_products():
    products = pd.Series(
        ["Credit reporting"] * MIN_CLASS_SIZE
        + ["Credit reporting or other personal consumer reports"] * MIN_CLASS_SIZE
    )

    labels = build_labels(products)

    assert set(labels) == {"Credit reporting"}


def test_build_labels_bins_a_class_that_is_too_small():
    products = pd.Series(["Mortgage"] * MIN_CLASS_SIZE + ["Debt or credit management"] * 3)

    labels = build_labels(products)

    assert labels.value_counts()["Other"] == 3
    assert "Debt or credit management" not in set(labels)


def test_strip_redaction_drops_the_xxxx_spans():
    narratives = pd.Series(["they charged me XXXX on XX/XX/2020 for nothing"])

    cleaned = strip_redaction(narratives).iloc[0]

    assert "XXXX" not in cleaned
    assert "charged" in cleaned


def test_split_keeps_the_slices_in_date_order():
    df = pd.DataFrame({"date_received": pd.date_range("2020-01-01", periods=100)})

    train, val, test = split(df)

    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert train["date_received"].max() < val["date_received"].min()
    assert val["date_received"].max() < test["date_received"].min()
