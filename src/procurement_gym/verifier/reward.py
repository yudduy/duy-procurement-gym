# PLAN: TRL-compatible reward function — bridges LLM completions to
#        deterministic evaluate_partition(). REQ-4.
# SOURCE: Plan at .claude/plans/imperative-finding-music.md
"""ProcurementVerifier: TRL GRPOTrainer-compatible reward function."""

from __future__ import annotations

from typing import Any

from procurement_gym.config import SupplierModelConfig
from procurement_gym.core.types import ProcurementInstance
from procurement_gym.evaluation import evaluate_partition
from procurement_gym.verifier.parser import parse_partition, validate_partition


def _extract_text(completion: list[dict[str, str]] | str) -> str:
    """Extract text from a completion (handles both string and message-dict formats).

    ADAPTED FROM: clock-game-rl/rewards/clock_game_rewards.py
    """
    if isinstance(completion, str):
        return completion
    # Message dict format — get last assistant message
    for msg in reversed(completion):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    # Fallback: last message content
    return completion[-1].get("content", "") if completion else ""


def format_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """Reward for correct output format: <partition>[...]</partition> tags present.

    Returns 1.0 if tags found, 0.0 otherwise.
    ADAPTED FROM: clock-game-rl format_reward (0.5 for structure + 0.5 for action)
    """
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        has_tags = "<partition>" in text and "</partition>" in text
        rewards.append(1.0 if has_tags else 0.0)
    return rewards


class ProcurementVerifier:
    """Deterministic verifier that scores LLM-generated lot partitions.

    Implements the TRL GRPOTrainer reward function signature:
        reward_fn(completions, **kwargs) -> list[float]
    """

    def __init__(
        self,
        instance: ProcurementInstance,
        config: SupplierModelConfig,
        area_size: float = 100.0,
        penalty_per_item: float = -10.0,
    ) -> None:
        self._instance = instance
        self._config = config
        self._area_size = area_size
        self._n_items = len(instance.items)
        self._mean_value = sum(item.base_value for item in instance.items) / self._n_items
        self._penalty = penalty_per_item * self._mean_value

    def reward_fn(self, completions: list[list[dict[str, str]]], **kwargs: Any) -> list[float]:
        """Score LLM completions. TRL GRPOTrainer-compatible signature."""
        return [self._score_one(c) for c in completions]

    def _score_one(self, completion: list[dict[str, str]]) -> float:
        """Score a single completion."""
        text = _extract_text(completion)

        # Parse
        partition = parse_partition(text, self._n_items)
        if partition is None:
            return float(self._penalty)

        # Validate
        errors = validate_partition(partition, self._instance)
        if errors:
            return float(self._penalty)

        # Evaluate (deterministic: seed=0)
        result = evaluate_partition(
            partition,
            self._instance,
            self._config,
            seed=0,
            area_size=self._area_size,
        )
        return result.reward
