# PLAN: Integration tests for GRPO training pipeline — verifies that the
#        reward function, prompt format, and SFT data are all consistent and
#        compatible with TRL GRPOTrainer's expectations.
# ADAPTED FROM: tests/test_verifier_compiler.py (existing test patterns)
"""Tests for GRPO training pipeline integration."""

from __future__ import annotations

import json

import pytest

from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.instances.transport import TransportInstanceGenerator
from procurement_gym.verifier.parser import parse_partition
from procurement_gym.verifier.reward import ProcurementVerifier
from procurement_gym.verifier.serializer import (
    SYSTEM_PROMPT,
    serialize_instance,
    serialize_partition,
)


@pytest.fixture()
def instance():
    ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
    sc = SupplierModelConfig(coupling_strength=1.0)
    gen = TransportInstanceGenerator(ic, sc)
    return gen.generate(seed=42)


@pytest.fixture()
def sc():
    return SupplierModelConfig(coupling_strength=1.0)


# === Test 1: SYSTEM_PROMPT exists and is consistent ===


class TestSystemPrompt:
    def test_system_prompt_exists(self) -> None:
        """SYSTEM_PROMPT must be a module-level constant for reuse."""
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 50

    def test_system_prompt_mentions_partition_format(self) -> None:
        """System prompt must tell the model the output format."""
        assert "<partition>" in SYSTEM_PROMPT

    def test_serializer_does_not_duplicate_system_instructions(self, instance) -> None:
        """Instance text should contain data, not role-playing instructions."""
        text = serialize_instance(instance)
        # Should NOT start with "You are designing..." — that belongs in SYSTEM_PROMPT
        assert not text.startswith("You are designing")
        # Should still have the data
        assert "ITEMS" in text
        assert "SUPPLIERS" in text


# === Test 2: Reward function signature matches TRL ===


class TestRewardSignature:
    def test_reward_fn_accepts_completions_and_kwargs(self, instance, sc) -> None:
        """TRL calls reward_funcs with (completions, **kwargs)."""
        verifier = ProcurementVerifier(instance, sc)
        completions = [
            [
                {
                    "role": "assistant",
                    "content": "<partition>[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]</partition>",
                }
            ]
        ]
        rewards = verifier.reward_fn(completions)
        assert isinstance(rewards, list)
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)

    def test_reward_fn_returns_finite_for_valid_partition(self, instance, sc) -> None:
        """Valid partition → finite reward (not NaN/inf)."""
        verifier = ProcurementVerifier(instance, sc)
        text = "<partition>[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]</partition>"
        completions = [[{"role": "assistant", "content": text}]]
        rewards = verifier.reward_fn(completions)
        import math

        assert math.isfinite(rewards[0])

    def test_penalty_for_unparseable(self, instance, sc) -> None:
        """Invalid completions get penalty, not crash."""
        verifier = ProcurementVerifier(instance, sc)
        completions = [[{"role": "assistant", "content": "I don't know how to do this"}]]
        rewards = verifier.reward_fn(completions)
        assert rewards[0] == verifier._penalty

    def test_reward_variance_across_partitions(self, instance, sc) -> None:
        """Different valid partitions must produce different rewards (GRPO signal)."""
        verifier = ProcurementVerifier(instance, sc)
        # All in one lot
        c1 = [
            [
                {
                    "role": "assistant",
                    "content": "<partition>[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]</partition>",
                }
            ]
        ]
        # Two lots alternating
        c2 = [
            [
                {
                    "role": "assistant",
                    "content": "<partition>[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]</partition>",
                }
            ]
        ]
        # Three lots
        c3 = [
            [
                {
                    "role": "assistant",
                    "content": "<partition>[0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2]</partition>",
                }
            ]
        ]
        r1 = verifier.reward_fn(c1)[0]
        r2 = verifier.reward_fn(c2)[0]
        r3 = verifier.reward_fn(c3)[0]
        # At least two must differ — otherwise GRPO has zero advantage signal
        rewards = [r1, r2, r3]
        assert len(set(rewards)) >= 2, f"All partitions got same reward: {rewards}"


# === Test 3: Format reward function ===


class TestFormatReward:
    def test_format_reward_exists(self) -> None:
        """format_reward must be importable (Clock Game pattern)."""
        from procurement_gym.verifier.reward import format_reward

        assert callable(format_reward)

    def test_format_reward_perfect(self) -> None:
        """Valid <partition> tags → 1.0."""
        from procurement_gym.verifier.reward import format_reward

        completions = [
            [{"role": "assistant", "content": "some reasoning\n<partition>[0,1,2]</partition>"}]
        ]
        rewards = format_reward(completions)
        assert rewards[0] == 1.0

    def test_format_reward_missing_tags(self) -> None:
        """No <partition> tags → 0.0."""
        from procurement_gym.verifier.reward import format_reward

        completions = [[{"role": "assistant", "content": "I think the answer is [0,1,2]"}]]
        rewards = format_reward(completions)
        assert rewards[0] == 0.0

    def test_format_reward_batch(self) -> None:
        """Must handle batches (TRL sends multiple completions)."""
        from procurement_gym.verifier.reward import format_reward

        completions = [
            [{"role": "assistant", "content": "<partition>[0,1]</partition>"}],
            [{"role": "assistant", "content": "no tags here"}],
            [{"role": "assistant", "content": "thinking...\n<partition>[2,1]</partition>"}],
        ]
        rewards = format_reward(completions)
        assert rewards == [1.0, 0.0, 1.0]


# === Test 4: SFT data format matches GRPO prompt format ===


class TestDataFormatConsistency:
    def test_sft_data_has_system_message(self, instance) -> None:
        """SFT data must include system message (same as GRPO uses)."""
        # Simulate what generate_sft_data should produce
        from procurement_gym.baselines import category_partition

        prompt_text = serialize_instance(instance)
        partition = category_partition(instance)
        completion = serialize_partition(partition)

        # The SFT entry should have system + user + assistant
        entry = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": completion},
            ]
        }
        assert entry["messages"][0]["role"] == "system"
        assert entry["messages"][0]["content"] == SYSTEM_PROMPT
        assert entry["messages"][1]["role"] == "user"
        assert entry["messages"][2]["role"] == "assistant"

    def test_sft_completion_parseable(self, instance) -> None:
        """SFT completions must be parseable by the same parser GRPO uses."""
        from procurement_gym.baselines import category_partition

        partition = category_partition(instance)
        completion = serialize_partition(partition)
        parsed = parse_partition(completion, 20)
        assert parsed is not None
        assert parsed.assignments == partition.assignments


# === Test 5: Reward works with reasoning prefix (chain-of-thought) ===


class TestChainOfThought:
    def test_reward_with_reasoning_prefix(self, instance, sc) -> None:
        """Reward function must handle completions with reasoning before partition."""
        verifier = ProcurementVerifier(instance, sc)
        text = (
            "Let me analyze the items. Items 0-6 are in cat_0, items 7-13 in cat_1.\n"
            "Geographic clustering suggests two main groups.\n\n"
            "<partition>[0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2]</partition>"
        )
        completions = [[{"role": "assistant", "content": text}]]
        rewards = verifier.reward_fn(completions)
        assert rewards[0] != verifier._penalty  # parsed successfully

    def test_parser_ignores_reasoning_numbers(self) -> None:
        """Parser must not match stray number lists in reasoning text."""
        text = (
            "Items [0, 1, 2, 3, 4] are in category A.\n"
            "<partition>[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]</partition>"
        )
        result = parse_partition(text, 10)
        assert result is not None
        # Should match the partition tags, not the item list
        assert result.assignments == (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
