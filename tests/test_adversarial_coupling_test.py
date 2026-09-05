"""Regression tests for adversarial coupling diagnostics."""

from scripts.adversarial_coupling_test import _pct_below_threshold


def test_pct_below_threshold_counts_all_matching_values() -> None:
    """Percentage should reflect count/total, not a constant 1/N artifact."""
    ratios = [0.2, 0.8, 1.1, 1.4]
    assert _pct_below_threshold(ratios, 1.0) == 50.0


def test_pct_below_threshold_handles_empty_input() -> None:
    assert _pct_below_threshold([], 1.0) == 0.0
