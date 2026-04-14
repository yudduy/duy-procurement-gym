# PLAN: Text verifier package — the "compiler" between LLM text and the procurement gym.
# SOURCE: Plan at .claude/plans/imperative-finding-music.md (REQ-1 through REQ-6)
"""Text ↔ Partition compiler for RLVR training."""

from procurement_gym.verifier.parser import parse_partition, validate_partition
from procurement_gym.verifier.reward import ProcurementVerifier
from procurement_gym.verifier.serializer import (
    SYSTEM_PROMPT,
    serialize_instance,
    serialize_partition,
)

__all__ = [
    "ProcurementVerifier",
    "SYSTEM_PROMPT",
    "parse_partition",
    "serialize_instance",
    "serialize_partition",
    "validate_partition",
]
