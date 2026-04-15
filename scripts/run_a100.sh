# PLAN: A100-compatible version of run_grpo.sh. Runs full SFT→merge→GRPO→eval
#        pipeline without B200-specific env vars or volume mounts. Clones from git.
# ADAPTED FROM: scripts/run_grpo.sh (B200 version) + scripts/setup_training.sh
#!/bin/bash
set -euo pipefail
unset http_proxy https_proxy

cd /home/ubuntu/procurement-gym

echo "=== Setting up venv ==="
python3 -m venv --system-site-packages /home/ubuntu/venv
source /home/ubuntu/venv/bin/activate

echo "=== Installing deps ==="
pip install --no-deps trl peft
pip install datasets accelerate bitsandbytes safetensors transformers
pip install ortools pydantic numpy scipy gymnasium
touch README.md
pip install -e . --no-deps

echo "=== Verifying ==="
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0)}')"
python -c "from trl import SFTTrainer, GRPOTrainer; print('TRL OK')"
python -c "from procurement_gym.verifier.reward import ProcurementVerifier; print('Verifier OK')"

echo "=== Smoke Test: 3-step GRPO on base model ==="
python scripts/smoke_test_gpu.py Qwen/Qwen2.5-7B-Instruct 1.0
echo "=== Smoke test passed — proceeding to training ==="

echo "=== Stage 1: SFT ==="
python scripts/train_sft.py Qwen/Qwen2.5-7B-Instruct sft_data.jsonl checkpoints/sft

echo "=== Stage 2: GRPO (merges SFT LoRA automatically) ==="
python scripts/train_grpo.py checkpoints/sft/final checkpoints/grpo 500

echo "=== Stage 3: Eval ==="
python scripts/eval_model.py checkpoints/grpo/final 20 700000

echo "=== DONE ==="
