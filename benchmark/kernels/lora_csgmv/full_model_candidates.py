"""Full Qwen3.5-9B comparison: early-return baseline versus compact expand + split-K shrink.

Eight synthetic nonzero rank-32 adapters, 32 concurrent requests, 512 prompt
 tokens, exactly 4096/8192 output tokens. No changes to deployed Lilo services.
Run: modal run benchmark/kernels/lora_csgmv/full_model_candidates.py --output <json>
"""

import asyncio
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import modal

BENCH = Path(__file__).resolve().parent
# Match the serving runtime used for the original full-model comparison.
RUNTIME_REVISION = "d050d06437d96196fc68d5b4e5c246408790d537"
image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.17")
    .apt_install("git")
    .run_commands(
        "git clone --filter=blob:none --no-checkout https://github.com/modal-projects/sglang.git /opt/runtime-source"
        f" && git -C /opt/runtime-source checkout --detach {RUNTIME_REVISION}"
        " && mkdir -p /opt/full-lora/python"
        " && cp -a /opt/runtime-source/python/sglang /opt/full-lora/python/"
        " && rm -rf /opt/runtime-source"
    )
    .env({"PYTHONPATH": "/opt/full-lora/python:/opt/full-lora/candidates"})
)
for helper in (
    "experimental_kernels.py",
    "serving_candidates.py",
    "validate_candidates.py",
):
    image = image.add_local_file(BENCH / helper, "/opt/full-lora/candidates/" + helper)
assets = modal.Volume.from_name("lilo-model-assets")
results_volume = modal.Volume.from_name(
    "lilo-sglang-kernel-bench", create_if_missing=True
)
app = modal.App("lilo-lora-full-model-candidates")

with image.imports():
    import httpx
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from validate_candidates import validate

MODEL = "/assets/Qwen3.5-9B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
URL = "http://127.0.0.1:30000"


def create_adapters(root):
    config = json.loads((Path(MODEL) / "config.json").read_text())["text_config"]
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    rank = 32
    adapters = []
    for adapter in range(8):
        generator = torch.Generator().manual_seed(1000 + adapter)
        weights = {}
        for layer in range(config["num_hidden_layers"]):
            modules = [
                ("mlp.gate_proj", hidden, intermediate),
                ("mlp.up_proj", hidden, intermediate),
                ("mlp.down_proj", intermediate, hidden),
            ]
            if config["layer_types"][layer] == "full_attention":
                q = config["num_attention_heads"] * config["head_dim"] * 2
                kv = config["num_key_value_heads"] * config["head_dim"]
                modules += [
                    ("self_attn.q_proj", hidden, q),
                    ("self_attn.k_proj", hidden, kv),
                    ("self_attn.v_proj", hidden, kv),
                    ("self_attn.o_proj", q // 2, hidden),
                ]
            for module, k, n in modules:
                key = f"base_model.model.model.layers.{layer}.{module}"
                weights[f"{key}.lora_A.weight"] = (
                    torch.randn(rank, k, generator=generator) / math.sqrt(k)
                ).bfloat16()
                weights[f"{key}.lora_B.weight"] = (
                    torch.randn(n, rank, generator=generator) * 0.001
                ).bfloat16()
        path = root / f"adapter{adapter}"
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "Qwen/Qwen3.5-9B",
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "r": rank,
                    "lora_alpha": rank,
                    "lora_dropout": 0.0,
                    "bias": "none",
                    "target_modules": TARGETS,
                }
            )
        )
        save_file(weights, str(path / "adapter_model.safetensors"))
        adapters.append(path)
    return adapters, config


async def run_batch(prompts, length):
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=64)
    async with httpx.AsyncClient(timeout=1800, limits=limits) as client:

        async def request(i, prompt):
            start = time.perf_counter()
            response = await client.post(
                URL + "/generate",
                json={
                    "input_ids": prompt,
                    "lora_path": f"adapter{i % 8}",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": length,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": len(prompt),
                    "stream": False,
                },
            )
            response.raise_for_status()
            body = response.json()
            elapsed = time.perf_counter() - start
            meta = body["meta_info"]
            assert meta["completion_tokens"] == length, (i, meta)
            tokens = body["output_ids"]
            assert len(tokens) == length
            return {
                "request": i,
                "adapter": i % 8,
                "latency_s": elapsed,
                "tokens_sha256": hashlib.sha256(
                    json.dumps(tokens).encode()
                ).hexdigest(),
                "logprobs_sha256": hashlib.sha256(
                    json.dumps(meta.get("output_token_logprobs", [])).encode()
                ).hexdigest(),
                "meta": {k: v for k, v in meta.items() if "logprob" not in k},
            }

        start = time.perf_counter()
        records = await asyncio.gather(
            *(request(i, prompt) for i, prompt in enumerate(prompts))
        )
        elapsed = time.perf_counter() - start
    latencies = sorted(r["latency_s"] for r in records)
    return {
        "output_tokens_per_request": length,
        "concurrency": len(prompts),
        "total_output_tokens": length * len(prompts),
        "batch_wall_s": elapsed,
        "output_tokens_per_second": length * len(prompts) / elapsed,
        "request_latency_median_s": statistics.median(latencies),
        "request_latency_p95_s": latencies[math.ceil(0.95 * len(latencies)) - 1],
        "requests": records,
    }


def wait_ready(process, logfile):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited: " + logfile.read_text()[-16000:])
        try:
            response = httpx.get(URL + "/health", timeout=2)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError("Server startup timeout: " + logfile.read_text()[-16000:])


@app.function(
    image=image,
    gpu="H200",
    cpu=16,
    memory=65536,
    timeout=7200,
    volumes={"/assets": assets, "/results": results_volume},
)
def compare(validate_only: bool = False):
    torch.set_num_threads(4)
    validation = validate()
    print(json.dumps({"event": "validation_passed", "checks": validation}), flush=True)
    if validate_only:
        return json.dumps(
            {"runtime_revision": RUNTIME_REVISION, "validation": validation}
        )
    run = Path("/results") / ("qwen35-9b-candidates-" + time.strftime("%Y%m%d-%H%M%S"))
    run.mkdir(parents=True)
    code = Path("/tmp/full-lora-source")
    shutil.copytree("/opt/full-lora/python/sglang", code / "sglang", dirs_exist_ok=True)
    kernel = code / "sglang/kernels/ops/gemm/chunked_sgmv_expand.py"
    # Apply the same early-return baseline used in the original experiment.
    baseline_source = kernel.read_text()
    marker = "    slice_end = tl.load(slice_offsets + slice_id + 1)\n"
    assert baseline_source.count(marker) == 1
    baseline_source = baseline_source.replace(
        marker,
        marker
        + "    if tl.program_id(axis=0) * BLOCK_N >= slice_end - slice_start:\n        return\n",
    )
    kernel.write_text(baseline_source)
    original = kernel.read_bytes()
    backend = code / "sglang/srt/lora/backend/chunked_backend.py"
    backend_original = backend.read_bytes()
    patched = (
        backend_original
        + b"\nfrom serving_candidates import install\ninstall(globals())\n"
    )
    adapters, config = create_adapters(Path("/tmp/full-lora-adapters"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prose = (
        "Explain how to implement an efficient external merge sort for a large collection of records. "
        "Discuss memory limits, stable ordering, disk access, and correctness. Work through concrete examples. "
    )
    prompts = []
    for i in range(32):
        text = f"Problem number {i}. " + prose * 80
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        prompts.append(ids[:480] + ids[-32:])
    assert all(len(prompt) == 512 for prompt in prompts)
    json.dumps(prompts)
    result = {
        "model": "Qwen/Qwen3.5-9B",
        "runtime_revision": RUNTIME_REVISION,
        "model_config_sha256": hashlib.sha256(
            (Path(MODEL) / "config.json").read_bytes()
        ).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "rank": 32,
        "adapters": 8,
        "concurrency": 32,
        "prompt_tokens": 512,
        "adapter_type": "synthetic nonzero adapters; performance benchmark only",
        "target_modules": TARGETS,
        "full_attention_layers": config["layer_types"].count("full_attention"),
        "total_layers": config["num_hidden_layers"],
        "baseline_expand_sha256": hashlib.sha256(original).hexdigest(),
        "candidate_kernels_sha256": hashlib.sha256(
            Path("/opt/full-lora/candidates/experimental_kernels.py").read_bytes()
        ).hexdigest(),
        "candidate_backend_sha256": hashlib.sha256(patched).hexdigest(),
        "baseline": "existing early-return expand, original shrink",
        "candidate": "compact expand; split-K shrink only for <=128 rows",
        "run_directory": str(run),
        "phases": [],
        "validation": validation,
    }
    print(
        json.dumps(
            {
                "event": "prepared",
                "run_directory": str(run),
                "full_attention_layers": result["full_attention_layers"],
            }
        ),
        flush=True,
    )
    for phase, variant in enumerate(["baseline", "candidate", "candidate", "baseline"]):
        backend.write_bytes(backend_original if variant == "baseline" else patched)
        shutil.rmtree(backend.parent / "__pycache__", ignore_errors=True)
        shutil.rmtree(kernel.parent / "__pycache__", ignore_errors=True)
        log = run / f"{phase}-{variant}.log"
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            "30000",
            "--dtype",
            "bfloat16",
            "--tp-size",
            "1",
            "--context-length",
            "16384",
            "--mem-fraction-static",
            "0.80",
            "--max-running-requests",
            "32",
            "--chunked-prefill-size",
            "2048",
            "--enable-metrics",
            "--enable-lora",
            "--lora-backend",
            "csgmv",
            "--max-lora-rank",
            "32",
            "--max-loras-per-batch",
            "8",
            "--max-loaded-loras",
            "8",
            "--lora-target-modules",
            *TARGETS,
            "--lora-paths",
            *[f"adapter{i}={path}" for i, path in enumerate(adapters)],
            "--cuda-graph-bs-decode",
            "1",
            "2",
            "4",
            "8",
            "16",
            "24",
            "32",
            "--random-seed",
            "1234",
        ]
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(code) + ":/opt/full-lora/candidates",
            SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1",
            SGLANG_DISABLE_CUDNN_CHECK="1",
        )
        print(
            json.dumps({"event": "starting", "phase": phase, "variant": variant}),
            flush=True,
        )
        start = time.perf_counter()
        with log.open("w") as output:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                wait_ready(process, log)
                startup = time.perf_counter() - start
                info = httpx.get(URL + "/get_server_info", timeout=30).json()
                (run / f"{phase}-server-info.json").write_text(
                    json.dumps(info, indent=2)
                )
                asyncio.run(run_batch(prompts, 128))
                phase_result = {
                    "phase": phase,
                    "variant": variant,
                    "startup_s": startup,
                    "command": command,
                    "measurements": [],
                }
                for length in [4096, 8192] if phase % 2 == 0 else [8192, 4096]:
                    flush = httpx.post(URL + "/flush_cache", timeout=60)
                    flush.raise_for_status()
                    measurement = asyncio.run(run_batch(prompts, length))
                    phase_result["measurements"].append(measurement)
                    print(
                        json.dumps(
                            {
                                "event": "measurement",
                                "phase": phase,
                                "variant": variant,
                                **{
                                    k: v
                                    for k, v in measurement.items()
                                    if k != "requests"
                                },
                            }
                        ),
                        flush=True,
                    )
                    (run / f"{phase}-{length}.json").write_text(
                        json.dumps(measurement, indent=2)
                    )
                    results_volume.commit()
                result["phases"].append(phase_result)
                (run / "result.json").write_text(json.dumps(result, indent=2))
                results_volume.commit()
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                results_volume.commit()
        time.sleep(3)
    return json.dumps(result)


@app.local_entrypoint()
def main(output: str, validate_only: bool = False):
    result = json.loads(compare.remote(validate_only))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {path}")
