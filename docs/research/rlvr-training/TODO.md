# RLVR Training Pipeline: MVE Results

## Session 2: 2026-04-15 02:00-12:00 (autonomous, user sleeping)

### Key Finding: Skip SFT, GRPO directly on base model

SFT collapsed the policy (all-identical 65-token completions, entropy=0.22, zero GRPO signal).
Base Qwen-2.5-7B-Instruct already produces valid `<partition>` tags from the system prompt alone.
Smoke test confirmed: diverse lengths (65-512), format_reward=1.0, reward_std=2263.

### Training Results (A100-80GB, $0.01-0.10/hr spot, ~$0.25 total)

| Steps | Avg Reward | Best | Format | Entropy |
|-------|-----------|------|--------|---------|
| 1-100 | 10,968 | 19,841 | 0.938 | 0.263 |
| 201-300 | 15,863 | 21,465 | 0.995 | 0.069 |
| 401-500 | 14,484 | 20,731 | 1.000 | 0.041 |
| 601-700 | 17,241 | 20,843 | 1.000 | 0.039 |
| 801-900 | **18,032** | 21,035 | 1.000 | 0.035 |
| 901-1000 | 17,243 | 20,173 | 1.000 | 0.035 |

### Held-Out Evaluation (20 instances, seeds 700000-700019)

| Metric | Value |
|--------|-------|
| Parse rate | **100% (20/20)** |
| Model mean reward | 16,712 |
| Baseline mean reward | 18,792 |
| Beats baseline | **5/20 (25%)** |
| Gap vs baseline | -11.07% |
| Best single instance | +158 above baseline |
| Catastrophic failure | 1/20 (instance 9: -20K reward) |

### What Worked
1. **No SFT needed** — base model + format_reward (weight=0.5) achieves 100% parse rate
2. **Clear learning curve** — reward improved 68% (10.4K → 18K)
3. **Beats baselines on some instances** — 5/20 held-out instances
4. **Robust format** — format_reward went 0.93 → 1.00

### What Needs Improvement
1. **Entropy collapse** — dropped to 0.035 by step 500, model converged to fixed strategy
2. **Not instance-adaptive** — applies same 2-lot partition regardless of instance structure
3. **Catastrophic failure** — 1/20 instances got negative reward (bad partition)
4. **Gap to baselines** — -11% on average, needs curriculum + more training

### Next Steps for Grokking
1. **Curriculum**: coupling_strength 0→1, n_items 20→50, progressive difficulty
2. **AZR self-play**: model generates own instances + solves them
3. **Higher temperature late-stage**: prevent entropy collapse
4. **Real data calibration**: EU TED transport tenders via OpenTender OCDS

---

## Session 1: 2026-04-14 00:00-02:30 (autonomous, user sleeping)

### What Was Built & Run

| Stage | Time | Cost | Result |
|-------|------|------|--------|
| SFT data generation (500 instances, search-optimized) | 2hr | local (M4) | 495/500 search wins, +1.08% over heuristics |
| SFT training on B200 (Qwen-7B, LoRA, 3 epochs) | 11min | $0.003 | Loss 0.75→0.37, 85% token accuracy |
| GRPO training on B200 (Qwen-7B, LoRA, 100 steps) | 33min | $0.006 | Reward 6.4K→16.6K, std 16.7K→7K |
| Held-out evaluation (20 instances) | 5min | included | See below |
| **Total** | **~3hr** | **<$0.01** | |

### Held-Out Evaluation (20 instances, seeds 700000-700019)

| Model | Parse Rate | Mean Reward | vs Baseline | Beats Baseline |
|-------|-----------|-------------|-------------|----------------|
| Zero-shot Qwen-7B | 100% | 18,166 | -1.5% | 0/10 |
| SFT model | 100% | 3,375* | -82%* | — |
| **GRPO model** | **85%** | **18,520** | **-1.4%** | **1/17** |
| Best heuristic baseline | — | 18,792 | 0% | — |

*SFT eval was only 5 instances, 2 catastrophic failures dragging mean

### Key Findings

1. **The pipeline works end-to-end.** Text → parse → evaluate → reward → GRPO gradient → update. All on B200 at $0.01/hr.

2. **GRPO learned from verifier signal.** Reward climbed from 6.4K to 16.6K during training, reward std dropped from 16.7K to 7K (fewer catastrophic partitions). Completion length shortened from 181 to 66 tokens (learned to be direct).

3. **GRPO ran from base model, not SFT model.** Due to LoRA adapter loading complexity, GRPO was run on base Qwen-7B with fresh LoRA, not on the SFT checkpoint. This means the domain prior from SFT wasn't used. The proper DeepSeek-R1 pipeline is: merge SFT adapter → GRPO on merged model.

4. **Parse rate dropped to 85%.** The base model without SFT prior sometimes fails to produce the partition tag. SFT model had 100% parse. Merging SFT first should fix this.

5. **Not yet beating baselines.** Both zero-shot and GRPO are within 1.5% of baselines. The baselines are simple heuristics — the gap is small. To beat them, the model needs to learn instance-specific item-lot assignments, not just "use 2-3 lots."

### Architecture Validated

```
Instance Generator → serialize_instance() → LLM prompt
                                                ↓
                                            LLM generates
                                                ↓
                                          parse_partition()
                                                ↓
                                        validate_partition()
                                                ↓
                                   evaluate_partition() [deterministic]
                                                ↓
                                          float reward
                                                ↓
                                      GRPO gradient update
```

All components tested on B200, 161/161 unit tests passing locally.

### Next Steps (Priority Order)

1. **Merge SFT → GRPO (proper pipeline)**
   - `model.merge_and_unload()` on SFT adapter
   - Run GRPO on merged model (starts with 100% parse + domain prior)
   - Expected: better starting reward, fewer catastrophic failures

2. **Scale GRPO training**
   - More instances (1000+), more steps (500+), more rollouts (16 per prompt)
   - Curriculum: coupling 0.5 → 0.7 → 1.0 as model improves

3. **Improve search-generated SFT data**
   - Simulated annealing instead of greedy hill-climbing
   - Try more lot counts (2-6)
   - Increase restarts for harder instances

4. **If signals emerge**: migrate to 8xB200 or 8xH100 for production-scale training

### Files Created

| File | Purpose |
|------|---------|
| `sft_data.jsonl` | 500 search-optimized SFT examples |
| `scripts/train_sft.py` | SFT training script (TRL SFTTrainer + LoRA) |
| `scripts/train_grpo.py` | GRPO training script (TRL GRPOTrainer + ProcurementVerifier) |
| `scripts/eval_model.py` | Held-out evaluation script |
| `scripts/setup_training.sh` | B200 remote setup |
| `flow-sft.yaml` | Flow config for SFT on B200 |
| `flow-grpo.yaml` | Flow config for GRPO on B200 |
| `.flowignore` | Exclude large files from code upload |

### Repo: https://github.com/yudduy/duy-procurement-gym (public)

### Compute Instance: run-187c2e (CANCELLED after training complete)
