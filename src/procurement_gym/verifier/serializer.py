# PLAN: Instance serializer — converts ProcurementInstance to LLM prompt text
#        and LotPartition to tagged output text. REQ-1.
# SOURCE: Plan at .claude/plans/imperative-finding-music.md
"""Serialize procurement instances and partitions to/from text."""

from __future__ import annotations

from procurement_gym.core.partition import LotPartition
from procurement_gym.core.types import ProcurementInstance


def serialize_instance(instance: ProcurementInstance) -> str:
    """Convert a ProcurementInstance to an LLM prompt string."""
    n_items = len(instance.items)
    n_suppliers = len(instance.suppliers)

    lines = [
        "You are designing lot structures for a procurement auction to maximize buyer welfare.",
        "",
        f"ITEMS ({n_items} total):",
        "| ID | Category | X     | Y     | Volume | Qualified Suppliers |",
        "|----|----------|-------|-------|--------|---------------------|",
    ]
    for item in instance.items:
        sups = "{" + ", ".join(str(s) for s in sorted(item.qualified_supplier_ids)) + "}"
        lines.append(
            f"| {item.id:>2d} | {item.category:<8s} | {item.x:>5.1f} | {item.y:>5.1f} | {item.volume:>6.1f} | {sups:<19s} |"
        )

    lines.append("")
    lines.append(f"SUPPLIERS ({n_suppliers} total):")
    lines.append("| ID | Categories     | Capacity | Base X | Base Y |")
    lines.append("|----|----------------|----------|--------|--------|")
    for sup in instance.suppliers:
        cats = "{" + ", ".join(sorted(sup.categories)) + "}"
        lines.append(
            f"| {sup.id:>2d} | {cats:<14s} | {sup.capacity:>8.1f} | {sup.base_x:>6.1f} | {sup.base_y:>6.1f} |"
        )

    lines.extend(
        [
            "",
            "RULES:",
            "- Each item must be assigned to exactly one lot.",
            "- Lots are numbered starting from 0.",
            "- Buyer welfare = total item value - supplier costs - lot overhead.",
            "- More lots = higher overhead. Fewer lots = less supplier competition.",
            "- Items near each other geographically are cheaper to serve together.",
            "- Suppliers can only bid on lots containing categories they cover.",
            "",
            "Output your partition as:",
            f"<partition>[lot_id_for_item_0, lot_id_for_item_1, ..., lot_id_for_item_{n_items - 1}]</partition>",
            f"where the list has exactly {n_items} integers.",
        ]
    )

    return "\n".join(lines)


def serialize_partition(partition: LotPartition) -> str:
    """Convert a LotPartition to tagged text (for SFT training data)."""
    assignments = ", ".join(str(a) for a in partition.assignments)
    return f"<partition>[{assignments}]</partition>"
