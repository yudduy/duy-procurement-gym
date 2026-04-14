# PLAN: B200 startup script for GRPO training with SFT adapter merge.
#        Installs deps, loads SFT checkpoint from volume, merges LoRA, runs GRPO, evals.
#        Combines setup_training.sh + train_grpo.py + eval_model.py into one flow command.
# ADAPTED FROM: scripts/setup_training.sh (existing B200 setup pattern)
#!/bin/bash
set -euo pipefail
unset http_proxy https_proxy
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

cd /home/ubuntu/procurement-gym

# Install deps (never pip install torch on B200)
pip install --no-deps trl peft
pip install datasets accelerate bitsandbytes safetensors
pip install ortools pydantic numpy
pip install fa4 2>/dev/null || true
pip install -e . --no-deps

# Verify
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
python -c "from peft import AutoPeftModelForCausalLM; print('PEFT OK')"

# Check SFT checkpoint
ls -la /volumes/proc/sft/final/ || echo "WARNING: No SFT checkpoint"

# GRPO with SFT merge
python scripts/train_grpo.py /volumes/proc/sft/final /volumes/proc/grpo 1000

# Eval
python scripts/eval_model.py /volumes/proc/grpo/final 20 700000
