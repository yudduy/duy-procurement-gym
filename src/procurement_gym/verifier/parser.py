# PLAN: Partition parser — extracts LotPartition from LLM text output.
#        Three-stage fallback: tags → JSON → regex. REQ-2, REQ-3.
# SOURCE: Plan at .claude/plans/imperative-finding-music.md
"""Parse LLM text output into LotPartition (the compiler front-end)."""

from __future__ import annotations

import json
import re

from procurement_gym.core.partition import LotPartition
from procurement_gym.core.types import ProcurementInstance

# Primary: <partition>[0, 1, 2, ...]</partition>
_TAG_RE = re.compile(r"<partition>\s*\[([^\]]+)\]\s*</partition>", re.IGNORECASE)

# Fallback: any bracketed int list
_BRACKET_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_partition(text: str, n_items: int) -> LotPartition | None:
    """Extract a LotPartition from LLM text output.

    Returns None if parsing fails. Never raises.
    """
    # Stage 1: tagged format <partition>[...]</partition>
    m = _TAG_RE.search(text)
    if m:
        result = _parse_int_list(m.group(1), n_items)
        if result is not None:
            return result

    # Stage 2: JSON {"assignments": [...]}
    try:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{") and "assignments" in line.lower():
                obj = json.loads(line)
                key = "assignments"
                if key not in obj:
                    # case-insensitive key search
                    for k in obj:
                        if k.lower() == "assignments":
                            key = k
                            break
                if key in obj and isinstance(obj[key], list):
                    assignments = [int(x) for x in obj[key]]
                    if len(assignments) == n_items and all(a >= 0 for a in assignments):
                        return LotPartition(assignments=tuple(assignments), n_items=n_items)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Stage 3: any bracketed int list of correct length
    for m in _BRACKET_RE.finditer(text):
        result = _parse_int_list(m.group(1), n_items)
        if result is not None:
            return result

    return None


def _parse_int_list(raw: str, n_items: int) -> LotPartition | None:
    """Parse a comma-separated int string into a LotPartition."""
    try:
        values = [int(x.strip()) for x in raw.split(",")]
        if len(values) == n_items and all(v >= 0 for v in values):
            return LotPartition(assignments=tuple(values), n_items=n_items)
    except (ValueError, TypeError):
        pass
    return None


def validate_partition(partition: LotPartition, instance: ProcurementInstance) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors: list[str] = []
    if len(partition.assignments) != len(instance.items):
        errors.append(
            f"Length mismatch: {len(partition.assignments)} assignments for {len(instance.items)} items"
        )
    if any(a < 0 for a in partition.assignments):
        errors.append("Negative lot IDs found (unassigned items)")
    if not partition.is_complete():
        errors.append("Partition is not complete (unassigned items)")
    return errors
