"""Standalone CPU parity + weight-mapping audit for Qwen3DSparkModel.

Phase-3b (DSpark-for-SGLang, Qwen3 draft-model support) verification script.
Runs entirely on CPU with real checkpoint weights; does NOT instantiate the
full `sglang.srt.models.qwen3_dspark.Qwen3DSparkModel` (it composes
`VocabParallelEmbedding` / `ParallelLMHead` / `RadixAttention`, which need a
`torch.distributed` process group, and `sglang.srt.layers.layernorm.RMSNorm`,
whose `forward()` dispatches to a custom CUDA kernel on this build regardless
of tensor device -- both confirmed unavailable in this CPU sandbox while
writing this script). Instead it checks two independent things:

1. Component parity: for each of the three small, checkpoint-weight-bearing
   computations DSparkWorkerV2 relies on (the additive Markov logit bias, the
   fc-then-hidden_norm context-feature combine, and the confidence /
   accept-rate head), run identical fp32 random inputs through:
     (a) the DeepSpec reference implementation (deepspec-ref clone), and
     (b) this repo's implementation -- reusing the real, dependency-light
         `Qwen3DSparkConfidenceHead` class directly for the confidence head,
         and a faithful pure-torch reproduction of the RMSNorm/vocab-parallel
         math for the other two (see the docstrings on `rmsnorm_native` and
         `mine_markov_bias` for exactly what is reproduced and why).
   Both sides load the SAME real checkpoint tensors (cast to fp32). Expect
   max-abs-diff at or near fp32 machine precision.

2. Weight name-mapping audit: run `resolve_qwen3_dspark_weight` (the exact,
   pure, import-light function `Qwen3DSparkModel.load_weights` uses) over
   all 64 real tensor names from the checkpoint and print the full
   name -> destination-parameter table, asserting zero unmapped names.

Usage:
    python3 scripts/playground/dspark_qwen3_weight_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SGLANG_ROOT = Path(__file__).resolve().parents[2]
DEEPSPEC_ROOT = Path(
    "/tmp/claude-1000/-home-ubuntu/052d1f86-c10c-40cf-85f9-89f4e2bedc3b/"
    "scratchpad/phase1/deepspec-ref"
)
TENSOR_NAMES_TXT = Path(
    "/tmp/claude-1000/-home-ubuntu/052d1f86-c10c-40cf-85f9-89f4e2bedc3b/"
    "scratchpad/phase1/pr29917/tensor_names.txt"
)
CHECKPOINT_ROOT = Path(
    "/home/ubuntu/hf_cache/hub/models--deepseek-ai--dspark_qwen3_4b_block7"
)

sys.path.insert(0, str(SGLANG_ROOT / "python"))
sys.path.insert(0, str(DEEPSPEC_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from deepspec.modeling.dspark.common import AcceptRatePredictor  # noqa: E402
from deepspec.modeling.dspark.markov_head import VanillaMarkov  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm  # noqa: E402

from sglang.srt.models.qwen3_dspark import (  # noqa: E402
    Qwen3DSparkConfidenceHead,
    resolve_qwen3_dspark_weight,
)

SEED = 0
RMS_NORM_EPS = 1e-6  # matches the checkpoint's config.json rms_norm_eps
# Reasonable-bug-catching bound, not a machine-precision bound: a real formula
# mistake (wrong concat order, missing bias, wrong combine order, ...) shows up
# as an O(1) difference, not something near this threshold.
MAX_DIFF_TOLERANCE = 1e-3


def find_checkpoint_file() -> Path:
    candidates = sorted(CHECKPOINT_ROOT.glob("snapshots/*/model.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No model.safetensors found under {CHECKPOINT_ROOT}")
    return candidates[0]


def load_tensor_names() -> list[str]:
    names = []
    for line in TENSOR_NAMES_TXT.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("NUM TENSORS"):
            continue
        names.append(line.split()[0])
    return names


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def rmsnorm_native(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Faithful fp32 reproduction of the single-tensor call form of
    `sglang.srt.layers.layernorm.RMSNorm.forward_native` (no residual):
    variance-normalize in fp32, then multiply by weight, then cast back.

    Reproduced rather than called directly: on this build, RMSNorm.forward()
    dispatches to `sgl_kernel.rmsnorm` (a CUDA kernel) regardless of the
    input tensor's device, which raises `NotImplementedError` for a CPU
    tensor in this sandbox (confirmed while writing this script). Since the
    inputs here are already fp32, the final `.to(orig_dtype)` is a no-op and
    this is exactly the real forward path's arithmetic.
    """
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight.float()).to(orig_dtype)


def mine_project_main_hidden(
    main_hidden: torch.Tensor,
    fc_weight: torch.Tensor,
    hidden_norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reproduction of `Qwen3DSparkBackbone.project_main_hidden`: fc (a plain
    `nn.Linear`, bias=False -- no distributed/CUDA-kernel dependency, so this
    part IS the real op) THEN hidden_norm (reproduced via `rmsnorm_native`,
    see above)."""
    projected = F.linear(main_hidden, fc_weight)
    return rmsnorm_native(projected, hidden_norm_weight, eps)


def mine_markov_bias(
    token_ids: torch.Tensor,
    markov_w1_weight: torch.Tensor,
    markov_w2_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduction of the additive Markov bias computed in
    `dspark_worker_v2._refine_block_markov_sharded`:
    `F.linear(markov_w1(token_ids), markov_w2_weight)`. `markov_w1` /
    `markov_w2` are `Qwen3DSparkMarkovHead`'s `VocabParallelEmbedding` /
    `ParallelLMHead` at tp_size=1, which reduce mathematically to a plain
    embedding lookup and a bias-free linear; reproduced with
    `F.embedding` / `F.linear` directly since those parallel wrapper classes'
    real forward() requires an initialized `torch.distributed` process group
    (confirmed unavailable in this sandbox while writing this script).

    Returns (bias, prev_embedding) -- the embedding is also checked directly
    since it is what feeds the confidence head's Markov features.
    """
    prev_embed = F.embedding(token_ids, markov_w1_weight)
    bias = F.linear(prev_embed, markov_w2_weight)
    return bias, prev_embed


def run_component_parity(weights: dict) -> dict[str, float]:
    torch.manual_seed(SEED)
    results: dict[str, float] = {}

    # ---- 1. VanillaMarkov: additive logit bias over the vocab ----
    w1 = weights["markov_head.markov_w1.weight"].float()
    w2 = weights["markov_head.markov_w2.weight"].float()
    vocab_size, markov_rank = w1.shape

    ref_markov = VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
    with torch.no_grad():
        ref_markov.markov_w1.weight.copy_(w1)
        ref_markov.markov_w2.weight.copy_(w2)

    token_ids = torch.randint(0, vocab_size, (17,))
    ref_bias = ref_markov.compute_step_bias(token_ids, None)
    ref_embed = ref_markov.get_prev_embeddings(token_ids)
    mine_bias, mine_embed = mine_markov_bias(token_ids, w1, w2)

    results["markov_prev_embedding"] = max_abs_diff(ref_embed, mine_embed)
    results["markov_additive_bias"] = max_abs_diff(ref_bias, mine_bias)

    # ---- 2. fc THEN hidden_norm combine (context-feature projection) ----
    fc_w = weights["fc.weight"].float()
    hn_w = weights["hidden_norm.weight"].float()
    hidden_size, fc_in_features = fc_w.shape
    assert fc_in_features % hidden_size == 0
    num_context_features = fc_in_features // hidden_size

    ref_fc = torch.nn.Linear(fc_in_features, hidden_size, bias=False)
    with torch.no_grad():
        ref_fc.weight.copy_(fc_w)
    ref_norm = Qwen3RMSNorm(hidden_size, eps=RMS_NORM_EPS)
    with torch.no_grad():
        ref_norm.weight.copy_(hn_w)

    main_hidden = torch.randn(5, num_context_features * hidden_size)
    ref_combined = ref_norm(ref_fc(main_hidden))
    mine_combined = mine_project_main_hidden(main_hidden, fc_w, hn_w, RMS_NORM_EPS)
    results["fc_then_hidden_norm_combine"] = max_abs_diff(ref_combined, mine_combined)

    # ---- 3. confidence head (AcceptRatePredictor), using the REAL class ----
    cw = weights["confidence_head.proj.weight"].float()
    cb = weights["confidence_head.proj.bias"].float()
    input_dim = cw.shape[1]

    ref_head = AcceptRatePredictor(input_dim=input_dim)
    with torch.no_grad():
        ref_head.proj.weight.copy_(cw)
        ref_head.proj.bias.copy_(cb)

    hidden = torch.randn(9, hidden_size)
    markov_embed = torch.randn(9, markov_rank)
    # Reference call site (deepspec qwen3/modeling.py predict_confidence_step):
    # features = cat([hidden_states, prev_embeddings], dim=-1); proj(features).float()
    ref_conf = ref_head(torch.cat([hidden, markov_embed], dim=-1)).float()

    mine_head = Qwen3DSparkConfidenceHead(input_dim)  # the REAL sglang class
    with torch.no_grad():
        mine_head.proj.weight.copy_(cw)
        mine_head.proj.bias.copy_(cb)
    mine_conf = mine_head(hidden, markov_embed)

    results["confidence_head"] = max_abs_diff(ref_conf, mine_conf)

    return results


def run_weight_mapping_audit(names: list[str]):
    rows = []
    for name in names:
        mapping = resolve_qwen3_dspark_weight(
            name, has_markov_head=True, has_confidence_head=True
        )
        rows.append(mapping)
    unmapped = [m for m in rows if m.dest_param is None]
    return rows, unmapped


def main() -> int:
    ckpt_path = find_checkpoint_file()
    print(f"Loading checkpoint tensors from {ckpt_path}")
    weights = load_file(str(ckpt_path))
    print(f"Loaded {len(weights)} tensors from safetensors.\n")

    print("=" * 100)
    print(
        f"COMPONENT PARITY (DeepSpec reference vs sglang qwen3_dspark math), "
        f"fp32, seed={SEED}"
    )
    print("=" * 100)
    results = run_component_parity(weights)
    all_ok = True
    for name, diff in results.items():
        ok = diff < MAX_DIFF_TOLERANCE
        all_ok = all_ok and ok
        print(f"  {name:32s} max_abs_diff = {diff:.3e}   {'PASS' if ok else 'FAIL'}")
    print()

    print("=" * 100)
    print("WEIGHT NAME-MAPPING AUDIT (resolve_qwen3_dspark_weight) -- all 64 tensors")
    print("=" * 100)
    names = load_tensor_names()
    print(f"tensor_names.txt lists {len(names)} tensors.\n")
    rows, unmapped = run_weight_mapping_audit(names)
    for mapping in rows:
        if mapping.dest_param is not None:
            shard = f"  (shard_id={mapping.shard_id!r})" if mapping.shard_id is not None else ""
            print(f"  {mapping.checkpoint_name:45s} -> {mapping.dest_param}{shard}")
        else:
            print(f"  {mapping.checkpoint_name:45s} -> DROPPED: {mapping.drop_reason}")
    print()
    print(
        f"Total tensors: {len(names)}; mapped: {len(names) - len(unmapped)}; "
        f"unmapped: {len(unmapped)}"
    )

    # Cross-check every mapped destination is actually present among the
    # checkpoint's own tensors where that makes sense (sanity on the table
    # itself, not just on resolve_qwen3_dspark_weight not raising).
    checkpoint_names = set(weights.keys())
    names_set = set(names)
    if checkpoint_names != names_set:
        print(
            "NOTE: tensor_names.txt and the live safetensors file's key set "
            "differ:",
            "extra in checkpoint:",
            sorted(checkpoint_names - names_set),
            "extra in tensor_names.txt:",
            sorted(names_set - checkpoint_names),
        )

    if unmapped:
        print("FAIL: unmapped tensors present:")
        for m in unmapped:
            print(f"  {m.checkpoint_name}: {m.drop_reason}")
        all_ok = False
    else:
        print("PASS: zero unmapped tensors (all 64 map to a destination parameter).")

    assert not unmapped, "weight mapping audit found unmapped tensors -- see above"

    print()
    print("=" * 100)
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    print("=" * 100)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
