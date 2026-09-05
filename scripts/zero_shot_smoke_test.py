# PLAN: Zero-shot LLM smoke test — prompts a model with procurement instances
#        and measures parse success rate + reward vs baselines. REQ-6.
# SOURCE: Plan at .claude/plans/imperative-finding-music.md
"""Zero-shot smoke test: can an LLM produce valid lot partitions?"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project to path for script usage
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from procurement_gym.baselines import (
    category_partition,
    equal_size_partition,
    geographic_partition,
)
from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.instances.transport import TransportInstanceGenerator
from procurement_gym.verifier.parser import parse_partition
from procurement_gym.verifier.reward import ProcurementVerifier
from procurement_gym.verifier.serializer import serialize_instance, serialize_partition

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load_model_transformers(model_name: str) -> tuple:
    """Load model via transformers (for local/Colab GPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading %s on %s...", model_name, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)

    return model, tokenizer, device


def generate_transformers(
    model, tokenizer, device: str, prompt: str, max_new_tokens: int = 512
) -> str:
    """Generate completion using transformers."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response


def generate_api(endpoint: str, api_key: str, model_name: str, prompt: str) -> str:
    """Generate completion using OpenAI-compatible API."""
    import urllib.request

    body = json.dumps(
        {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.1,
        }
    ).encode()

    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    return data["choices"][0]["message"]["content"]


def run_smoke_test(
    n_instances: int = 10,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    backend: str = "transformers",
    api_endpoint: str = "",
    api_key: str = "",
    output_path: str = "smoke_test_results.json",
) -> dict:
    """Run zero-shot smoke test."""
    sc = SupplierModelConfig(coupling_strength=1.0)
    ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
    gen = TransportInstanceGenerator(ic, sc)

    # Load model if using transformers
    model, tokenizer, device = None, None, None
    if backend == "transformers":
        model, tokenizer, device = load_model_transformers(model_name)

    results: list[dict] = []
    parse_successes = 0
    valid_partitions = 0
    t0 = time.monotonic()

    for i in range(n_instances):
        seed = 900000 + i
        instance = gen.generate(seed=seed)
        n_items = len(instance.items)
        n_cats = len({item.category for item in instance.items})
        rng = np.random.default_rng(seed)

        prompt = serialize_instance(instance)

        # Generate LLM completion
        log.info("Instance %d/%d (seed=%d)...", i + 1, n_instances, seed)
        try:
            if backend == "transformers":
                assert model is not None and tokenizer is not None and device is not None
                completion = generate_transformers(model, tokenizer, device, prompt)
            elif backend == "api":
                completion = generate_api(api_endpoint, api_key, model_name, prompt)
            else:
                raise ValueError(f"Unknown backend: {backend}")
        except Exception as e:
            log.error("Generation failed for instance %d: %s", i, e)
            results.append(
                {
                    "seed": seed,
                    "status": "GENERATION_FAILED",
                    "error": str(e),
                }
            )
            continue

        # Parse
        partition = parse_partition(completion, n_items)
        parsed = partition is not None
        if parsed:
            parse_successes += 1

        # Score
        reward = None
        if parsed:
            verifier = ProcurementVerifier(instance, sc)
            rewards = verifier.reward_fn([[{"role": "assistant", "content": completion}]])
            reward = rewards[0]
            if reward > verifier._penalty:
                valid_partitions += 1

        # Baselines
        cat_r = ProcurementVerifier(instance, sc).reward_fn(
            [[{"role": "assistant", "content": serialize_partition(category_partition(instance))}]]
        )[0]
        geo_r = ProcurementVerifier(instance, sc).reward_fn(
            [
                [
                    {
                        "role": "assistant",
                        "content": serialize_partition(
                            geographic_partition(instance, n_lots=n_cats, rng=rng)
                        ),
                    }
                ]
            ]
        )[0]
        eq_r = ProcurementVerifier(instance, sc).reward_fn(
            [
                [
                    {
                        "role": "assistant",
                        "content": serialize_partition(
                            equal_size_partition(instance, n_lots=n_cats)
                        ),
                    }
                ]
            ]
        )[0]
        best_baseline = max(cat_r, geo_r, eq_r)

        result = {
            "seed": seed,
            "parsed": parsed,
            "reward": reward,
            "completion_preview": completion[:200],
            "baselines": {"category": cat_r, "geographic": geo_r, "equal_size": eq_r},
            "best_baseline": best_baseline,
            "beats_baseline": reward is not None and reward >= best_baseline,
        }
        results.append(result)
        status = "VALID" if parsed else "PARSE_FAIL"
        log.info(
            "  %s | reward=%.1f | best_baseline=%.1f | beats=%s",
            status,
            reward if reward is not None else float("nan"),
            best_baseline,
            result["beats_baseline"],
        )

    elapsed = time.monotonic() - t0

    summary = {
        "model": model_name,
        "backend": backend,
        "n_instances": n_instances,
        "parse_success_rate": parse_successes / n_instances if n_instances > 0 else 0,
        "valid_partition_rate": valid_partitions / n_instances if n_instances > 0 else 0,
        "parse_successes": parse_successes,
        "valid_partitions": valid_partitions,
        "elapsed_s": elapsed,
        "results": results,
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"SMOKE TEST RESULTS: {model_name}")
    print("=" * 60)
    print(f"Instances:          {n_instances}")
    print(
        f"Parse success rate: {summary['parse_success_rate']:.0%} ({parse_successes}/{n_instances})"
    )
    print(
        f"Valid partitions:   {summary['valid_partition_rate']:.0%} ({valid_partitions}/{n_instances})"
    )
    print(f"Time:               {elapsed:.1f}s")

    if valid_partitions > 0:
        valid_rewards = [
            r["reward"] for r in results if r.get("reward") is not None and r["reward"] > -10000
        ]
        baseline_rewards = [
            r["best_baseline"]
            for r in results
            if r.get("reward") is not None and r["reward"] > -10000
        ]
        beats = sum(1 for r in results if r.get("beats_baseline", False))
        print(
            f"\nValid partition rewards: mean={np.mean(valid_rewards):.1f}, std={np.std(valid_rewards):.1f}"
        )
        print(f"Best baseline rewards:  mean={np.mean(baseline_rewards):.1f}")
        print(f"Beats baseline:         {beats}/{valid_partitions}")
    print("=" * 60)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Results saved to %s", output_path)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot LLM smoke test")
    parser.add_argument("-n", type=int, default=10, help="Number of instances")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--backend", choices=["transformers", "api"], default="transformers")
    parser.add_argument("--api-endpoint", default="", help="OpenAI-compatible API endpoint")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--output", default="smoke_test_results.json", help="Output path")
    args = parser.parse_args()

    run_smoke_test(
        n_instances=args.n,
        model_name=args.model,
        backend=args.backend,
        api_endpoint=args.api_endpoint,
        api_key=args.api_key,
        output_path=args.output,
    )
