# ProcurementGym

Gymnasium environment for procurement lot structure optimization using RL with Verifiable Rewards (RLVR).

An agent partitions procurement items into auction lots. OR-Tools CP-SAT solves the Winner Determination Problem exactly per episode. A three-layer supplier model (cost + participation + markup + package bids) produces calibratable reward signals.

## Install

```bash
uv sync
```

## Test

```bash
uv run pytest tests/ -v --tb=short
uv run ruff check src/
uv run mypy src/ --strict
```

## Architecture

```
Agent proposes lot partition (sequential, one item per step)
  -> Supplier Simulator (cost + participation + markup + packages)
    -> OR-Tools CP-SAT (exact WDP solver)
      -> Reward = buyer welfare (value - cost)
```

See `CLAUDE.md` for full details.
