# PLAN: Package exports and Gymnasium registration for procurement_gym.
# SOURCE: Plan "Package Exports" and "Gymnasium Registration" sections.
"""ProcurementGym — RL environment for procurement lot structure optimization."""

from procurement_gym.config import EnvConfig, InstanceConfig, SupplierModelConfig
from procurement_gym.core.partition import LotPartition
from procurement_gym.core.types import Item, ProcurementInstance, Supplier
from procurement_gym.env import ProcurementEnv

__all__ = [
    "ProcurementEnv",
    "EnvConfig",
    "InstanceConfig",
    "SupplierModelConfig",
    "Item",
    "Supplier",
    "ProcurementInstance",
    "LotPartition",
]

import gymnasium

gymnasium.register(
    id="Procurement-v0",
    entry_point="procurement_gym.env:ProcurementEnv",
)
