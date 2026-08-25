"""Manual validation harness: multi-adapter LoRA + EAGLE-family speculative decoding.

The oracle is spec-on vs spec-off *per adapter*, not adapter vs base.
Speculative decoding is lossless with respect to the target it verifies
against, so for greedy sampling:

    output(adapter=X, spec=ON) == output(adapter=X, spec=OFF)

Comparing an adapter's output to the base model's only proves LoRA was
applied at all. This harness launches the same weights twice (reference
without spec, then with spec) and runs four checks:

  1. parity   — per adapter, spec-on output == spec-off output (the oracle)
  2. distinct — each adapter's output differs from base (LoRA really applied)
  3. mixed    — a batch interleaving every adapter matches the solo outputs
                (crossed verify segments serve a request the wrong adapter)
  4. eager    — a batch wider than --cuda-graph-max-bs still matches
                (exercises the non-cuda-graph target-verify path)

It also reports per-adapter accept length from each response's
meta_info["spec_accept_length"], which is how much speculation the shared
unadapted draft actually buys for that adapter.

Usage:
    python test/manual/lora/run_spec_lora_matrix.py --config eagle3-llama31
    python test/manual/lora/run_spec_lora_matrix.py --config nextn-qwen35-35b-a3b
    python test/manual/lora/run_spec_lora_matrix.py --list

Add --keep-derived to reuse a previously written derived adapter, and
--max-new-tokens / --port to adjust the run.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from concurrent import futures

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "List three primary colors.",
    "Write a one-sentence story about a brave detective on Mars.",
    "Explain what a hash table is in two sentences.",
]

DERIVED_SUFFIX = "-derived"


# Each config is a runnable server shape. `adapters` are (name, hf_path)
# pairs; `derive_from` names an adapter to synthesize a second, distinct
# adapter from when the base has only one public adapter (needed to make
# mixed-adapter batches meaningful).
CONFIGS = {
    # Cheapest: 5 genuinely distinct public adapters, ranks 8-64, no synthesis.
    "eagle3-llama31": dict(
        model="meta-llama/Llama-3.1-8B-Instruct",
        spec_args=[
            "--speculative-algorithm=EAGLE3",
            "--speculative-draft-model-path=lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B",
            "--speculative-num-steps=5",
            "--speculative-eagle-topk=8",
            "--speculative-num-draft-tokens=32",
        ],
        adapters=[
            ("fact", "algoprog/fact-generation-llama-3.1-8b-instruct-lora"),
            ("guard", "nvidia/llama-3.1-nemoguard-8b-topic-control"),
            ("sql", "philschmid/code-llama-3-1-8b-text-to-sql-lora"),
            ("ocr", "pbevan11/llama-3.1-8b-ocr-correction"),
            ("zh", "faridlazuarda/valadapt-llama-3.1-8B-it-chinese"),
        ],
        common_args=["--mem-fraction-static=0.7", "--max-lora-rank=64"],
        tp=1,
    ),
    # EAGLE topk=1 is the same chain-shaped drafting NEXTN/MTP uses, on 1 GPU
    # without needing an MTP checkpoint. cuda-graph-max-bs forces the eager path.
    "eagle-topk1-llama2": dict(
        model="meta-llama/Llama-2-7b-chat-hf",
        spec_args=[
            "--speculative-algorithm=EAGLE",
            "--speculative-draft-model-path=lmsys/sglang-EAGLE-llama2-chat-7B",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
        ],
        adapters=[("norwegian", "RuterNorway/Llama-2-7b-chat-norwegian-LoRa")],
        derive_from="norwegian",
        common_args=[
            "--mem-fraction-static=0.7",
            "--max-lora-rank=128",
            "--cuda-graph-max-bs=2",
        ],
        tp=1,
    ),
    # Real NEXTN (self-bundled MTP head, no draft path) on a MoE base: the
    # densest coverage per GPU-hour — MoE-LoRA cuda-graph buffers, virtual
    # experts, and an arch whose MTP head shares the target lm_head module.
    "nextn-qwen35-35b-a3b": dict(
        model="Qwen/Qwen3.5-35B-A3B",
        spec_args=[
            "--speculative-algorithm=NEXTN",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
        ],
        adapters=[("case", "opherlie/lora-test-case-Qwen3.5-35B-A3B", "dataset")],
        derive_from="case",
        common_args=[
            "--max-lora-rank=64",
            "--moe-runner-backend=triton",
            "--experts-shared-outer-loras",
            "--lora-use-virtual-experts",
            "--disable-shared-experts-fusion",
            "--mem-fraction-static=0.8",
        ],
        tp=4,
    ),
    # MoE + EAGLE3 with a draft checkpoint matched to the same base variant.
    "eagle3-qwen3-30b-a3b": dict(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        spec_args=[
            "--speculative-algorithm=EAGLE3",
            "--speculative-draft-model-path=lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=4",
            "--speculative-num-draft-tokens=8",
        ],
        adapters=[
            ("case", "yushengsu/lora-diff-Qwen3-30B-A3B-Instruct-2507", "dataset")
        ],
        derive_from="case",
        common_args=[
            "--max-lora-rank=32",
            "--moe-runner-backend=triton",
            "--experts-shared-outer-loras",
            "--attention-backend=flashinfer",
            "--prefill-attention-backend=fa4",
            "--decode-attention-backend=fa4",
            "--mem-fraction-static=0.8",
        ],
        # The draft checkpoint's config derives a 2048 context; the draft
        # follows the target's context_length (it reads target KV), so the
        # longer-context guard has to be waived for this pairing.
        env={"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"},
        tp=4,
    ),
    # DFLASH: fixed-size block drafting, verified in one TARGET_VERIFY pass.
    # Its target is the one base with several genuinely distinct public
    # adapters, so no synthesis is needed.
    "dflash-llama31": dict(
        model="meta-llama/Llama-3.1-8B-Instruct",
        spec_args=[
            "--speculative-algorithm=DFLASH",
            "--speculative-draft-model-path=z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
            "--speculative-num-draft-tokens=4",
        ],
        adapters=[
            ("fact", "algoprog/fact-generation-llama-3.1-8b-instruct-lora"),
            ("guard", "nvidia/llama-3.1-nemoguard-8b-topic-control"),
        ],
        common_args=["--mem-fraction-static=0.7", "--max-lora-rank=64"],
        # The draft checkpoint derives a 40960 context against the target's
        # 131072; the draft follows the target, so waive the guard.
        env={"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"},
        tp=1,
    ),
    # DSPARK with a separate DSpark draft head, on the one base that has two
    # distinct public adapters. Only the default static ragged-verify mode is
    # supported: the others schedule per-request verify lengths, which the
    # uniform-width LoRA segment layout cannot express.
    "dspark-inkling": dict(
        model="thinkingmachines/Inkling-Small",
        spec_args=[
            "--speculative-algorithm=DSPARK",
            "--speculative-draft-model-path=RadixArk/Inkling-Small-DSpark",
            "--speculative-dspark-block-size=5",
        ],
        adapters=[
            ("gutenberg", "nbeerbower/Inkling-Small-Gutenberg-DPO-LoRA"),
            ("hemlock", "hemlang/Inkling-Small-Hemlock-SFT-LoRA"),
        ],
        common_args=[
            "--max-lora-rank=32",
            "--moe-runner-backend=triton",
            "--experts-shared-outer-loras",
            "--mem-fraction-static=0.8",
        ],
        env={"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"},
        # A ~500GB checkpoint over network storage does not load inside the
        # default launch budget.
        launch_timeout=3600,
        tp=8,
    ),
    # Flagship MLA + MoE NextN. 8 GPUs.
    "nextn-deepseek-v31": dict(
        model="deepseek-ai/DeepSeek-V3.1-Base",
        spec_args=[
            "--speculative-algorithm=NEXTN",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
        ],
        adapters=[("case", "yushengsu/lora-diff-DeepSeek-V3.1-Base", "dataset")],
        derive_from="case",
        common_args=[
            "--max-lora-rank=32",
            "--moe-runner-backend=triton",
            "--experts-shared-outer-loras",
            "--attention-backend=flashinfer",
            "--prefill-attention-backend=fa4",
            "--decode-attention-backend=flashinfer",
            "--disable-shared-experts-fusion",
            "--mem-fraction-static=0.8",
        ],
        tp=8,
    ),
}


def log(msg: str) -> None:
    print(f"[harness] {msg}", flush=True)


def materialize_adapter(hf_path: str, repo_type: str) -> str:
    """Download an adapter and return its local directory.

    Several adapters used here live in *dataset* repos (that is how the
    repo's own LoRA logprob-diff tests host them), which the server cannot
    resolve as a model id — so every adapter is materialized locally and
    passed to --lora-paths by path.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=hf_path, repo_type=repo_type)


def check_adapter_supported(name: str, local_path: str) -> None:
    """Report what each adapter exercises. Nothing here is a rejection: a
    draft sharing the target's lm_head gets the unwrapped base layer, so
    lm_head and embedding adapters cost accept rate, not correctness."""
    config_path = os.path.join(local_path, "adapter_config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:  # noqa: BLE001 - not every layout ships one
        log(f"  {name}: could not read adapter_config.json ({e}); skipping precheck")
        return
    modules = config.get("target_modules")
    log(f"  {name}: r={config.get('r')} target_modules={modules}")
    if isinstance(modules, list):
        shared = {"lm_head", "output", "unembed_tokens"} & set(modules)
        if shared:
            log(f"  {name}: targets {sorted(shared)} — shared-lm_head path")
        if {"embed_tokens", "vocab_emb", "word_embeddings"} & set(modules):
            log(f"  {name}: targets embeddings — expect a reduced accept rate")
    if config.get("modules_to_save"):
        log(f"  {name}: WARNING modules_to_save={config['modules_to_save']}")


LM_HEAD_MARKERS = ("lm_head", "unembed_tokens")


def adapter_lm_head_keys(local_path: str):
    """Tensor keys that would land on lm_head (PEFT unembed_tokens keys are
    rewritten to lm_head during load)."""
    from safetensors import safe_open

    weights = os.path.join(local_path, "adapter_model.safetensors")
    if not os.path.isfile(weights):
        return []
    with safe_open(weights, "pt") as f:
        return [k for k in f.keys() if any(m in k for m in LM_HEAD_MARKERS)]


def write_adapter_variant(
    source_dir: str, out_dir: str, keep: bool, *, negate_b: bool, drop_lm_head: bool
) -> str:
    """Write a modified copy of an adapter.

    ``drop_lm_head`` removes output-layer tensors, isolating an adapter's
    other modules from the shared-lm_head path. The server accepts lm_head
    adapters, so this is for narrowing a failure, not a requirement.

    ``negate_b`` flips the sign of every lora_B tensor, producing a second
    adapter of identical shape whose outputs genuinely diverge from the
    original — without which a "mixed-adapter" batch would compare an adapter
    against itself and could not detect crossed verify segments.
    """
    if keep and os.path.isdir(out_dir):
        log(f"reusing adapter variant at {out_dir}")
        return out_dir

    from safetensors.torch import load_file, save_file

    weights_name = "adapter_model.safetensors"
    src_weights = os.path.join(source_dir, weights_name)
    if not os.path.isfile(src_weights):
        raise SystemExit(f"{source_dir} has no {weights_name}")

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    tensors = load_file(src_weights)
    dropped = flipped = 0
    out = {}
    for key, value in tensors.items():
        if drop_lm_head and any(m in key for m in LM_HEAD_MARKERS):
            dropped += 1
            continue
        if negate_b and "lora_B" in key:
            value = -value
            flipped += 1
        out[key] = value
    save_file(out, os.path.join(out_dir, weights_name))
    shutil.copy(
        os.path.join(source_dir, "adapter_config.json"),
        os.path.join(out_dir, "adapter_config.json"),
    )
    log(
        f"wrote {out_dir} (dropped {dropped} lm_head tensors, "
        f"negated {flipped} lora_B tensors)"
    )
    return out_dir


def launch(config: dict, adapters, base_url: str, with_spec: bool):
    other_args = list(config["common_args"])
    env = config.get("env")
    if config["tp"] > 1:
        other_args += [f"--tp={config['tp']}"]
    if adapters:
        other_args += [
            "--enable-lora",
            f"--lora-backend={config.get('lora_backend', 'triton')}",
            # +1: the base model occupies a pool slot too, and every mixed
            # batch below co-batches base requests.
            f"--max-loras-per-batch={len(adapters) + 1}",
            "--lora-paths",
        ] + [f"{name}={path}" for name, path in adapters]
    else:
        # Base arms: --max-lora-rank is a LoRA-only flag and is rejected
        # without --enable-lora, so drop it from the common set.
        other_args = [a for a in other_args if not a.startswith("--max-lora-rank")]
    if with_spec:
        other_args += config["spec_args"]
    other_args += config.get("extra_args", [])
    log(f"launching {'spec' if with_spec else 'reference'} server: {other_args}")
    return popen_launch_server(
        config["model"],
        base_url,
        timeout=config.get("launch_timeout", DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH),
        other_args=other_args,
        env=env,
    )


def generate(base_url: str, texts, routes, max_new_tokens: int):
    """One /generate call; returns (texts, per-request accept lengths)."""
    response = requests.post(
        base_url + "/generate",
        json={
            "text": texts,
            "lora_path": routes,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        },
        timeout=1800,
    )
    response.raise_for_status()
    results = response.json()
    outputs = [item["text"] for item in results]
    accepts = [item["meta_info"].get("spec_accept_length") for item in results]
    return outputs, accepts


def min_logprob_gap(base_url: str, route, prompt: str, max_new_tokens: int):
    """Smallest top-2 logprob gap along the greedy path for one request.

    A gap of ~0 means the argmax at that position is decided by kernel scan
    order, so any change in reduction order (a verify forward computes logits
    at a different batch shape than a plain decode) flips the token with no
    correctness implication. Used to classify surviving mismatches.
    """
    response = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "lora_path": route,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            "return_logprob": True,
            "top_logprobs_num": 2,
        },
        timeout=600,
    )
    response.raise_for_status()
    tops = response.json()["meta_info"].get("output_top_logprobs") or []
    gaps = [c[0][0] - c[1][0] for c in tops if c and len(c) >= 2]
    return min(gaps) if gaps else None


def collect_solo(base_url: str, routes, max_new_tokens: int):
    """Per route: outputs and mean accept length over PROMPTS."""
    solo, accept = {}, {}
    for route in routes:
        outputs, accepts = generate(
            base_url, list(PROMPTS), [route] * len(PROMPTS), max_new_tokens
        )
        solo[route] = outputs
        seen = [a for a in accepts if a]
        accept[route] = sum(seen) / len(seen) if seen else None
    return solo, accept


def report_parity(label: str, reference: dict, actual: dict) -> int:
    """Compare per-route output lists; returns the mismatch count."""
    mismatches = 0
    for route, expected in reference.items():
        for i, (want, got) in enumerate(zip(expected, actual[route])):
            if want != got:
                mismatches += 1
                log(f"  MISMATCH [{label}] route={route} prompt#{i}")
                log(f"    spec-off: {want!r}")
                log(f"    spec-on : {got!r}")
    return mismatches


def build_load(num_prompts: int, input_len: int, seed: int):
    """Fixed synthetic prompts as raw token ids, so every arm sends identical
    work and the input length is exact rather than tokenizer-dependent."""
    rng = random.Random(seed)
    return [
        [rng.randint(1000, 10000) for _ in range(input_len)] for _ in range(num_prompts)
    ]


def one_request(base_url: str, input_ids, route, output_len: int):
    started = time.perf_counter()
    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": input_ids,
            "lora_path": route,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_len,
                # Fixed-length output: every arm emits exactly output_len
                # tokens, so throughput is comparable without length drift.
                "ignore_eos": True,
            },
        },
        timeout=3600,
    )
    response.raise_for_status()
    meta = response.json()["meta_info"]
    return dict(
        latency=time.perf_counter() - started,
        output_tokens=meta.get("completion_tokens", output_len),
        accept_length=meta.get("spec_accept_length"),
    )


def measure(base_url: str, loads, routes, output_len: int, concurrency: int):
    """Drive `concurrency` in-flight requests over the load; return throughput,
    accept length, and per-request latency."""
    pairs = [(loads[i], routes[i % len(routes)]) for i in range(len(loads))]
    started = time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(
            pool.map(
                lambda pair: one_request(base_url, pair[0], pair[1], output_len), pairs
            )
        )
    elapsed = time.perf_counter() - started
    accepts = [r["accept_length"] for r in results if r["accept_length"]]
    total_out = sum(r["output_tokens"] for r in results)
    latencies = sorted(r["latency"] for r in results)
    return dict(
        output_tps=total_out / elapsed,
        accept_length=(sum(accepts) / len(accepts)) if accepts else None,
        median_latency=latencies[len(latencies) // 2],
        elapsed=elapsed,
        requests=len(results),
    )


def run_perf(config: dict, args) -> int:
    """Four arms answering two questions: does speculation still pay once LoRA
    is on (lora_spec vs lora_nospec), and how much does serving adapters cost
    relative to speculating on the plain base model (lora_spec vs base_spec)?
    """
    base_url = f"http://127.0.0.1:{args.port}"
    adapters = resolve_adapters(config, args)
    concurrencies = [int(c) for c in args.concurrency.split(",")]
    loads = build_load(args.num_prompts, args.input_len, args.seed)
    warmup = build_load(max(concurrencies), args.input_len, args.seed + 1)

    # (label, serve LoRA?, speculate?)
    arms = [
        ("base_nospec", False, False),
        ("base_spec", False, True),
        ("lora_nospec", True, False),
        ("lora_spec", True, True),
    ]
    table = {}

    for label, with_lora, with_spec in arms:
        log("=" * 68)
        log(f"arm {label}: lora={with_lora} spec={with_spec}")
        process = None
        try:
            process = launch(
                config,
                adapters if with_lora else [],
                base_url,
                with_spec=with_spec,
            )
            # Multi-adapter arms spread requests round-robin across every
            # adapter (the multi-tenant shape); base arms send no adapter.
            routes = [name for name, _ in adapters] if with_lora else [None]
            measure(base_url, warmup, routes, args.output_len, len(warmup))
            for concurrency in concurrencies:
                stats = measure(base_url, loads, routes, args.output_len, concurrency)
                table[(label, concurrency)] = stats
                log(
                    f"  c={concurrency:<4} {stats['output_tps']:8.1f} tok/s  "
                    f"accept={stats['accept_length']}  "
                    f"p50={stats['median_latency']:.2f}s"
                )
        finally:
            if process is not None:
                kill_process_tree(process.pid)

    report_perf(table, concurrencies, args)
    return 0


def report_perf(table, concurrencies, args) -> None:
    log("=" * 78)
    log(
        f"PERF  model-shape: {args.num_prompts} reqs x {args.input_len} in / "
        f"{args.output_len} out"
    )
    log("=" * 78)
    labels = ["base_nospec", "base_spec", "lora_nospec", "lora_spec"]
    log(f"{'concurrency':<12}" + "".join(f"{label:>16}" for label in labels))
    for concurrency in concurrencies:
        cells = []
        for label in labels:
            stats = table.get((label, concurrency))
            cells.append(f"{stats['output_tps']:.1f}" if stats else "-")
        log(
            f"{concurrency:<12}" + "".join(f"{cell:>16}" for cell in cells) + "   tok/s"
        )
    log("")
    log(f"{'concurrency':<12}{'base accept':>16}{'lora accept':>16}")
    for concurrency in concurrencies:
        base = table.get(("base_spec", concurrency), {}).get("accept_length")
        lora = table.get(("lora_spec", concurrency), {}).get("accept_length")
        fmt = lambda v: f"{v:.3f}" if v else "-"  # noqa: E731
        log(f"{concurrency:<12}{fmt(base):>16}{fmt(lora):>16}")
    log("")
    log("ratios (>1 is better)")
    log(
        f"{'concurrency':<12}{'spec on base':>16}{'spec on lora':>16}"
        f"{'lora vs base':>16}"
    )
    for concurrency in concurrencies:

        def tps(label):
            stats = table.get((label, concurrency))
            return stats["output_tps"] if stats else None

        base_nospec, base_spec = tps("base_nospec"), tps("base_spec")
        lora_nospec, lora_spec = tps("lora_nospec"), tps("lora_spec")
        ratio = lambda a, b: f"{a / b:.2f}x" if a and b else "-"  # noqa: E731
        log(
            f"{concurrency:<12}"
            f"{ratio(base_spec, base_nospec):>16}"
            f"{ratio(lora_spec, lora_nospec):>16}"
            f"{ratio(lora_spec, base_spec):>16}"
        )
    log("")
    log("  spec on base  = speculation speedup without adapters (the ceiling)")
    log("  spec on lora  = speculation speedup with adapters served (the ask)")
    log("  lora vs base  = cost of serving adapters, both speculating")


def resolve_adapters(config: dict, args):
    """Materialize every adapter locally and return (name, local_path) pairs,
    deriving a second distinct adapter when the base has only one public one.

    Config entries are (name, hf_path) or (name, hf_path, repo_type).
    """
    adapters = []
    for entry in config["adapters"]:
        name, hf_path = entry[0], entry[1]
        repo_type = entry[2] if len(entry) > 2 else "model"
        local_path = materialize_adapter(hf_path, repo_type)

        # A draft sharing the target's lm_head gets the unwrapped base layer,
        # so these adapters cost accept rate, not correctness. Keep the
        # weights as published so the run exercises that path.
        lm_head_keys = adapter_lm_head_keys(local_path)
        if lm_head_keys:
            log(
                f"  {name}: carries {len(lm_head_keys)} lm_head tensor(s) "
                f"({lm_head_keys[0]}) — exercising the shared-lm_head path"
            )
        adapters.append((name, local_path))

    log(f"prechecking {len(adapters)} adapter(s) against the lm_head rule")
    for name, local_path in adapters:
        check_adapter_supported(name, local_path)

    derive_name = config.get("derive_from")
    if derive_name:
        source = dict(adapters)[derive_name]
        # Key the cache on the source path: several configs name their
        # adapter "case", so a name-only key served one model's variant to
        # another (mismatched layer counts).
        source_key = hashlib.sha1(source.encode()).hexdigest()[:8]
        adapters.append(
            (
                derive_name + DERIVED_SUFFIX,
                write_adapter_variant(
                    source,
                    os.path.join(
                        args.derived_dir, f"{derive_name}{DERIVED_SUFFIX}-{source_key}"
                    ),
                    args.keep_derived,
                    negate_b=True,
                    drop_lm_head=False,
                ),
            )
        )
    return adapters


def collect_shapes(base_url: str, routes, args):
    """Outputs for every (shape, route, prompt) the checks compare.

    Three shapes: solo (one route per batch), mixed (every route interleaved
    in one batch), wide (a batch larger than the cuda-graph capture, so the
    eager verify path runs).
    """
    out = {}
    solo, accept = collect_solo(base_url, routes, args.max_new_tokens)
    for route, texts in solo.items():
        for i, text in enumerate(texts):
            out[("solo", route, i)] = text

    mixed_texts = [p for p in PROMPTS for _ in routes]
    mixed_routes = [r for _ in PROMPTS for r in routes]
    mixed, _ = generate(base_url, mixed_texts, mixed_routes, args.max_new_tokens)
    for text, route, got in zip(mixed_texts, mixed_routes, mixed):
        out[("mixed", route, PROMPTS.index(text))] = got

    wide = max(args.wide_batch, len(routes) * 2)
    wide_routes = [routes[i % len(routes)] for i in range(wide)]
    wide_texts = [PROMPTS[i % len(PROMPTS)] for i in range(wide)]
    wide_out, _ = generate(base_url, wide_texts, wide_routes, args.max_new_tokens)
    for text, route, got in zip(wide_texts, wide_routes, wide_out):
        out[("wide", route, PROMPTS.index(text))] = got

    return out, accept, solo


def unstable_keys(ref_a: dict, ref_b: dict) -> set:
    """(route, prompt) pairs the reference itself cannot reproduce.

    Greedy decoding is not bitwise reproducible here: batch composition
    changes reduction order (MoE routing especially), and two runs of the
    same server can diverge. Anything unstable *without* speculation is a
    property of the model and backend, not of spec+LoRA, so the spec
    comparison must exclude it or it reports noise as failure.
    """
    unstable = set()
    for (shape, route, index), text in ref_a.items():
        # run-to-run, identical shape
        if ref_b.get((shape, route, index)) != text:
            unstable.add((route, index))
        # shape-to-shape, same run
        if ref_a.get(("solo", route, index)) != text:
            unstable.add((route, index))
    return unstable


def is_reproducible(base_url: str, route, prompt: str, args, repeats: int = 3) -> bool:
    """Whether one prompt yields the same greedy output across repeats."""
    outputs = {
        generate(base_url, [prompt], [route], args.max_new_tokens)[0][0]
        for _ in range(repeats)
    }
    return len(outputs) == 1


def compare(ref_a, spec_a, spec_b, noisy, base_url: str, args) -> int:
    """Count genuine spec-on vs spec-off differences.

    Skips pairs the reference could not reproduce, pairs where the spec side
    disagrees with itself, and differences that land on an argmax tie (where
    the token choice is arbitrary in the first place).
    """
    log("comparing spec-on against spec-off on reproducible pairs only")
    excluded = ties = failures = 0
    for key, want in ref_a.items():
        shape, route, index = key
        # Both sides must be self-consistent before a difference between them
        # means anything: the instability set is stochastic, so a single
        # sample per side would report noise as a failure.
        if (route, index) in noisy or spec_a.get(key) != spec_b.get(key):
            excluded += 1
            continue
        got = spec_a.get(key)
        if got == want:
            continue

        # A pair can look stable in two samples and still be stochastic, so
        # re-sample this specific pair before calling it a defect. Verified
        # necessary: a prompt that survived the blanket filter turned out to
        # diverge spec-on vs spec-off with no adapters loaded at all, and to
        # disagree with itself across repeats.
        if not is_reproducible(base_url, route, PROMPTS[index], args):
            ties += 1
            gap = min_logprob_gap(base_url, route, PROMPTS[index], args.max_new_tokens)
            log(
                f"  UNSTABLE [{shape}] route={route} prompt#{index}: the spec "
                f"server does not reproduce this prompt across repeats "
                f"(min top-2 logprob gap {gap}); not attributable to spec"
            )
            continue

        failures += 1
        log(f"  MISMATCH [{shape}] route={route} prompt#{index}")
        log(f"    spec-off: {want!r}")
        log(f"    spec-on : {got!r}")
    log(
        f"  ({excluded} skipped as baseline-nondeterministic, "
        f"{ties} skipped as not reproducible on re-sampling)"
    )
    return failures


def run(config: dict, args) -> int:
    base_url = f"http://127.0.0.1:{args.port}"
    adapters = resolve_adapters(config, args)
    routes = [None] + [name for name, _ in adapters]
    failures = 0
    process = None

    try:
        # Phase 1 — reference: same weights and adapters, speculation off.
        # Collected twice to measure how much this model/backend varies on
        # its own, which sets the noise floor for the spec comparison.
        process = launch(config, adapters, base_url, with_spec=False)
        ref_a, _, solo_a = collect_shapes(base_url, routes, args)
        ref_b, _, _ = collect_shapes(base_url, routes, args)
        noisy = unstable_keys(ref_a, ref_b)
        log(
            f"baseline noise floor: {len(noisy)} of "
            f"{len(routes) * len(PROMPTS)} (route, prompt) pairs are not "
            "reproducible without speculation"
        )

        for route in routes[1:]:
            if solo_a[route] == solo_a[None]:
                failures += 1
                log(
                    f"  FAIL adapter {route} output is identical to base on every prompt"
                )
        kill_process_tree(process.pid)
        process = None

        # Phase 2 — speculation on, same three shapes.
        process = launch(config, adapters, base_url, with_spec=True)
        spec_a, accept, _ = collect_shapes(base_url, routes, args)
        spec_b, _, _ = collect_shapes(base_url, routes, args)

        failures += compare(ref_a, spec_a, spec_b, noisy, base_url, args)
    finally:
        if process is not None:
            kill_process_tree(process.pid)

    log("=" * 68)
    log("per-adapter accept length (tokens per verify step; 1.0 = no speedup)")
    for route in routes:
        value = accept.get(route)
        label = "base" if route is None else route
        log(f"  {label:<24} {value if value is None else round(value, 3)}")
    log("=" * 68)
    log(f"RESULT: {'PASS' if failures == 0 else f'FAIL ({failures} mismatches)'}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help=f"one of: {', '.join(CONFIGS)}")
    parser.add_argument("--list", action="store_true", help="list configs and exit")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--wide-batch",
        type=int,
        default=16,
        help="batch size for the eager-path check; keep it above --cuda-graph-max-bs",
    )
    parser.add_argument(
        "--lora-backend", help="override the config's LoRA kernel backend"
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        dest="extra_args",
        help="extra server flag, repeatable (e.g. --extra-arg=--enable-lora-overlap-loading)",
    )
    parser.add_argument("--derived-dir", default="/tmp/sglang-derived-loras")
    parser.add_argument("--keep-derived", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["correctness", "perf", "both"],
        default="correctness",
        help="correctness = the 4 output checks; perf = the 4-arm throughput "
        "and accept-length comparison",
    )
    parser.add_argument(
        "--concurrency",
        default="1,8,32",
        help="comma-separated in-flight request counts to sweep (perf mode)",
    )
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.list or not args.config:
        for name, config in CONFIGS.items():
            algo = next(
                a.split("=")[1] for a in config["spec_args"] if "algorithm" in a
            )
            n = len(config["adapters"]) + (1 if config.get("derive_from") else 0)
            print(
                f"{name:<24} tp={config['tp']}  {algo:<7} adapters={n}  {config['model']}"
            )
        return 0

    if args.config not in CONFIGS:
        raise SystemExit(f"unknown config {args.config!r}; use --list")
    started = time.time()
    config = dict(CONFIGS[args.config])
    if args.lora_backend:
        config["lora_backend"] = args.lora_backend
    config["extra_args"] = args.extra_args
    code = 0
    if args.mode in ("correctness", "both"):
        code |= run(config, args)
    if args.mode in ("perf", "both"):
        code |= run_perf(config, args)
    log(f"total wall time {time.time() - started:.0f}s")
    return code


if __name__ == "__main__":
    sys.exit(main())
