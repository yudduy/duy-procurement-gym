# PLAN: Tests for the text verifier compiler — serializer, parser, reward.
# SOURCE: Plan at .claude/plans/imperative-finding-music.md
"""Tests for the text ↔ partition compiler."""

from __future__ import annotations

import pytest

from procurement_gym.baselines import category_partition, geographic_partition
from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.core.partition import LotPartition
from procurement_gym.instances.transport import TransportInstanceGenerator
from procurement_gym.verifier.parser import parse_partition, validate_partition
from procurement_gym.verifier.reward import ProcurementVerifier
from procurement_gym.verifier.serializer import serialize_instance, serialize_partition


@pytest.fixture()
def instance():
    ic = InstanceConfig(n_items=10, n_suppliers=5, n_categories=2, capacity_scarcity=1.0)
    sc = SupplierModelConfig(coupling_strength=0.0)
    gen = TransportInstanceGenerator(ic, sc)
    return gen.generate(seed=42)


@pytest.fixture()
def large_instance():
    ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
    sc = SupplierModelConfig(coupling_strength=1.0)
    gen = TransportInstanceGenerator(ic, sc)
    return gen.generate(seed=42)


# === Serializer ===


class TestSerializer:
    def test_serialize_instance_contains_items(self, instance) -> None:
        text = serialize_instance(instance)
        assert "ITEMS (10 total)" in text
        assert "| ID |" in text
        assert "cat_0" in text or "cat_1" in text

    def test_serialize_instance_contains_suppliers(self, instance) -> None:
        text = serialize_instance(instance)
        assert "SUPPLIERS (5 total)" in text
        assert "Capacity" in text

    def test_serialize_instance_contains_rules(self, instance) -> None:
        text = serialize_instance(instance)
        assert "RULES:" in text
        assert "exactly" in text  # partition length instruction

    def test_serialize_partition_tagged(self, instance) -> None:
        part = category_partition(instance)
        text = serialize_partition(part)
        assert text.startswith("<partition>[")
        assert text.endswith("]</partition>")
        assert str(len(instance.items) - 1) not in text or "," in text

    def test_round_trip(self, instance) -> None:
        """serialize → parse → same partition."""
        part = category_partition(instance)
        text = serialize_partition(part)
        parsed = parse_partition(text, len(instance.items))
        assert parsed is not None
        assert parsed.assignments == part.assignments


# === Parser ===


class TestParser:
    def test_parse_tagged_format(self) -> None:
        text = "Some reasoning...\n<partition>[0, 1, 0, 2, 1]</partition>\nMore text"
        result = parse_partition(text, n_items=5)
        assert result is not None
        assert result.assignments == (0, 1, 0, 2, 1)

    def test_parse_json_format(self) -> None:
        text = '{"assignments": [0, 1, 0, 2, 1]}'
        result = parse_partition(text, n_items=5)
        assert result is not None
        assert result.assignments == (0, 1, 0, 2, 1)

    def test_parse_bare_brackets(self) -> None:
        text = "The partition is [0, 1, 0, 2, 1] which groups items..."
        result = parse_partition(text, n_items=5)
        assert result is not None
        assert result.assignments == (0, 1, 0, 2, 1)

    def test_parse_wrong_length_returns_none(self) -> None:
        text = "<partition>[0, 1, 0]</partition>"
        result = parse_partition(text, n_items=5)
        assert result is None

    def test_parse_negative_values_returns_none(self) -> None:
        text = "<partition>[0, -1, 0, 2, 1]</partition>"
        # Regex won't match negative numbers, so this falls through
        result = parse_partition(text, n_items=5)
        assert result is None

    def test_parse_empty_string_returns_none(self) -> None:
        result = parse_partition("", n_items=5)
        assert result is None

    def test_parse_garbage_returns_none(self) -> None:
        result = parse_partition("I don't know how to do this task.", n_items=5)
        assert result is None

    def test_parse_multiple_brackets_picks_correct_length(self) -> None:
        text = "First [1, 2, 3] then the real answer [0, 1, 0, 2, 1]"
        result = parse_partition(text, n_items=5)
        assert result is not None
        assert result.assignments == (0, 1, 0, 2, 1)

    def test_parse_case_insensitive_tags(self) -> None:
        text = "<PARTITION>[0, 1, 2, 0, 1]</PARTITION>"
        result = parse_partition(text, n_items=5)
        assert result is not None

    def test_validate_valid_partition(self, instance) -> None:
        part = category_partition(instance)
        errors = validate_partition(part, instance)
        assert errors == []

    def test_validate_incomplete_partition(self, instance) -> None:
        part = LotPartition.empty(len(instance.items))
        errors = validate_partition(part, instance)
        assert len(errors) > 0
        assert any("unassigned" in e.lower() or "negative" in e.lower() for e in errors)


# === Reward Function ===


class TestRewardFunction:
    def test_reward_fn_signature(self, instance) -> None:
        """Must match TRL GRPOTrainer: completions -> list[float]."""
        sc = SupplierModelConfig(coupling_strength=0.0)
        verifier = ProcurementVerifier(instance, sc)

        part = category_partition(instance)
        text = serialize_partition(part)
        completions = [[{"role": "assistant", "content": text}]]

        rewards = verifier.reward_fn(completions)
        assert isinstance(rewards, list)
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)

    def test_reward_penalty_on_invalid(self, instance) -> None:
        sc = SupplierModelConfig(coupling_strength=0.0)
        verifier = ProcurementVerifier(instance, sc)

        completions = [[{"role": "assistant", "content": "I can't do this."}]]
        rewards = verifier.reward_fn(completions)
        assert rewards[0] < 0

    def test_reward_deterministic(self, instance) -> None:
        """Same text → same reward."""
        sc = SupplierModelConfig(coupling_strength=0.0)
        verifier = ProcurementVerifier(instance, sc)

        part = category_partition(instance)
        text = serialize_partition(part)
        completions = [[{"role": "assistant", "content": text}]]

        r1 = verifier.reward_fn(completions)
        r2 = verifier.reward_fn(completions)
        assert r1[0] == r2[0]

    def test_reward_multiple_completions(self, instance) -> None:
        """Batch of completions returns batch of rewards."""
        sc = SupplierModelConfig(coupling_strength=0.0)
        verifier = ProcurementVerifier(instance, sc)

        import numpy as np

        rng = np.random.default_rng(42)
        cat = category_partition(instance)
        geo = geographic_partition(instance, n_lots=2, rng=rng)

        completions = [
            [{"role": "assistant", "content": serialize_partition(cat)}],
            [{"role": "assistant", "content": serialize_partition(geo)}],
            [{"role": "assistant", "content": "garbage"}],
        ]
        rewards = verifier.reward_fn(completions)
        assert len(rewards) == 3
        assert rewards[2] < 0  # garbage gets penalty

    def test_reward_with_reasoning_prefix(self, instance) -> None:
        """Parser should handle reasoning text before the partition tag."""
        sc = SupplierModelConfig(coupling_strength=0.0)
        verifier = ProcurementVerifier(instance, sc)

        part = category_partition(instance)
        text = (
            "Let me analyze the items. Items 0-4 are cat_0, items 5-9 are cat_1.\n"
            "I should group by category for maximum supplier eligibility.\n\n"
            + serialize_partition(part)
        )
        completions = [[{"role": "assistant", "content": text}]]
        rewards = verifier.reward_fn(completions)
        # Valid partition parsed despite reasoning prefix — reward is not the penalty
        penalty = verifier._penalty
        assert rewards[0] > penalty, "Should not be penalty — partition was valid"
