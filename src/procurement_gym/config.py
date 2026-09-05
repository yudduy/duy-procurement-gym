# PLAN: Pydantic config schema for env, instance generation, and supplier model.
#        Every parameter has a default. Agents never guess values.
# SOURCE: docs/research/procurement-rl/PRODUCT_INTENT.md lines 314-347
"""Configuration schema for ProcurementGym."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SupplierModelConfig(BaseModel):
    """Configuration for the three-layer supplier simulation model."""

    synergy_weight: float = Field(0.1, description="Weight for geographic synergy savings")
    participation_w_cap: float = Field(2.0, description="Logistic weight for capability match")
    participation_w_size: float = Field(0.5, description="Logistic weight for log(lot_size)")
    participation_w_comp: float = Field(
        -0.3, description="Logistic weight for expected competition"
    )
    participation_w_geo: float = Field(
        1.5, description="Logistic weight for geographic proximity (scaled by coupling_strength)"
    )
    participation_bias_mean: float = Field(-1.0, description="Mean per-supplier participation bias")
    participation_bias_std: float = Field(0.5, description="Std of per-supplier participation bias")
    markup_alpha_mean: float = Field(0.15, description="Mean markup intercept")
    markup_alpha_std: float = Field(0.05, description="Std of markup intercept")
    markup_beta_mean: float = Field(0.5, description="Mean competition elasticity")
    markup_beta_std: float = Field(0.1, description="Std of competition elasticity")
    cost_noise_std: float = Field(
        0.05, description="Std of idiosyncratic cost noise (fraction of base cost)"
    )

    # Noise reduction (gated by coupling_strength)
    participation_coupling_boost: float = Field(
        2.0,
        ge=0.0,
        description="Additive bias boost for participation at high coupling (realistic: 80%+ bid rate)",
    )
    noise_reduction: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="At 1.0 with coupling=1.0: fully deterministic verification (like AlphaZero/DeepSeek)",
    )
    lot_overhead_base: float = Field(
        100.0,
        ge=0.0,
        description="Fixed administrative cost per lot (scaled by coupling_strength)",
    )
    economy_of_scale: float = Field(
        0.15,
        ge=0.0,
        le=0.5,
        description="Bulk cost reduction for larger lots (scaled by coupling_strength)",
    )

    # Cross-lot coupling parameters
    coupling_strength: float = Field(
        0.0, ge=0.0, le=1.0, description="Cross-lot coupling intensity [0=none, 1=full]"
    )
    max_package_size: int = Field(3, ge=2, le=5, description="Max lots per package bid")
    package_discount_min: float = Field(0.05, description="Min discount rate for packages")
    package_discount_max: float = Field(0.25, description="Max discount rate for packages")
    package_synergy_threshold: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Min synergy for package bid (softened by coupling_strength)",
    )
    max_packages_per_supplier: int = Field(
        64,
        ge=1,
        description="Cap emitted package bids per supplier to bound verifier runtime",
    )
    volume_tier_thresholds: tuple[float, ...] = Field(
        (), description="Volume thresholds for discount tiers (empty=disabled)"
    )
    volume_tier_discounts: tuple[float, ...] = Field(
        (),
        description="Discount rates at each volume tier (empty=disabled, solver rebate bug at high lot counts)",
    )


class InstanceConfig(BaseModel):
    """Configuration for procurement instance generation."""

    n_items: int = Field(20, ge=2, le=500, description="Number of procurement items")
    n_suppliers: int = Field(10, ge=2, le=100, description="Number of suppliers")
    n_categories: int = Field(3, ge=1, le=20, description="Number of item categories")
    area_size: float = Field(100.0, description="Geographic area side length (km)")
    min_supplier_categories: int = Field(1, description="Min categories per supplier")
    max_supplier_categories: int = Field(2, description="Max categories per supplier")
    base_value_mean: float = Field(1000.0, description="Mean item value")
    base_value_std: float = Field(200.0, description="Std item value")
    capacity_scarcity: float = Field(
        1.5,
        ge=0.0,
        le=10.0,
        description="Capacity tightening factor (applied proportional to coupling_strength)",
    )

    @model_validator(mode="after")
    def _validate_categories(self) -> "InstanceConfig":
        if self.max_supplier_categories > self.n_categories:
            object.__setattr__(self, "max_supplier_categories", self.n_categories)
        if self.min_supplier_categories > self.max_supplier_categories:
            object.__setattr__(self, "min_supplier_categories", self.max_supplier_categories)
        return self


class EnvConfig(BaseModel):
    """Top-level configuration for ProcurementEnv."""

    instance: InstanceConfig = InstanceConfig()
    supplier_model: SupplierModelConfig = SupplierModelConfig()
    k_max: int = Field(20, description="Maximum number of lots")
    solver_timeout_s: float = Field(60.0, description="Solver timeout in seconds")
    item_order: str = Field(
        "ascending_qualified",
        description="Item presentation order: ascending_qualified | random | by_category",
    )
    seed: int = Field(42, description="Random seed for reproducibility")
