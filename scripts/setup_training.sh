#!/bin/bash
# PLAN: Remote setup script for B200 training environment.
# SOURCE: CLAUDE.md B200 gotchas section
# ADAPTED FROM: scripts/setup_remote.sh pattern from CLAUDE.md
set -euo pipefail

echo "=== Setup: ProcurementGym RLVR Training ==="

# B200 Blackwell gotchas: never pip install torch (destroys sm_100 build)
unset http_proxy https_proxy
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Install dependencies WITHOUT touching torch
pip install --no-deps trl peft
pip install datasets accelerate bitsandbytes safetensors
pip install ortools pydantic numpy

# Install flash attention 4 for Blackwell
pip install fa4 2>/dev/null || echo "fa4 not available, will use eager attention"

# Install the procurement_gym package
cd /home/ubuntu/procurement-gym
pip install -e . --no-deps 2>/dev/null || pip install -e .

echo "=== Verifying setup ==="
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"
python -c "from procurement_gym.verifier.reward import ProcurementVerifier; print('ProcurementVerifier: OK')"
python -c "from trl import SFTTrainer, GRPOTrainer; print('TRL: OK')"
python -c "from peft import LoraConfig; print('PEFT: OK')"

echo "=== Setup complete ==="
