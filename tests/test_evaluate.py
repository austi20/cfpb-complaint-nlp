"""Tests for the FCRA citation check, the routing cutoff and the bootstrap."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import TARGET_PRECISION, cites_fcra, pick_cutoff
from train import bootstrap_gap


def test_cites_fcra_finds_a_statute_citation():
    assert cites_fcra("This is a violation of 15 USC 1681s-2 and my rights")
    assert cites_fcra("Please block these accounts under FCRA 605B")


def test_cites_fcra_ignores_a_plain_complaint():
    assert not cites_fcra("The collector called me six times a day about a debt I paid")


def test_pick_cutoff_drops_the_low_confidence_mistakes():
    confidence = np.array([0.1, 0.2, 0.9, 1.0, 1.1, 1.2])
    correct = np.array([False, False, True, True, True, True])

    cutoff = pick_cutoff(confidence, correct)

    assert correct[confidence >= cutoff].mean() >= TARGET_PRECISION
    assert (confidence >= cutoff).sum() == 4


def test_bootstrap_gap_is_zero_for_identical_predictions():
    labels = ["a", "b", "a", "c", "b"] * 20

    low, high = bootstrap_gap(labels, labels, labels)

    assert low == 0 and high == 0
