# Microbenchmark for the DeepSeek-V3-style biased grouped top-k router.
#
# Compares the three CUDA paths in `biased_grouped_topk_gpu` against the
# pure-PyTorch reference, across realistic configs and a token-count sweep
# spanning decode (1) → prefill (32K).
#
# Why this exists:
#   `biased_grouped_topk_gpu` (python/sglang/srt/layers/moe/topk.py) walks a
#   backend ladder: FlashInfer `fused_topk_deepseek` → sgl_kernel
#   `moe_fused_gate` → AMD aiter → PyTorch reference. Each backend has
#   different constraints (power-of-two #experts, experts_per_group ≤ 32,
#   etc.). Today there is no microbenchmark that isolates each path; this
#   file fills that gap and provides the data needed to motivate (or
#   refute) further router-kernel work.
#
# Usage:
#   python3 benchmark/kernels/grouped_topk/benchmark_grouped_topk.py
#   python3 benchmark/kernels/grouped_topk/benchmark_grouped_topk.py --json out.json
#   python3 benchmark/kernels/grouped_topk/benchmark_grouped_topk.py --config deepseek_v3
#   python3 benchmark/kernels/grouped_topk/benchmark_grouped_topk.py --backend flashinfer pytorch

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Callable, Optional

import torch

# ---- Backend imports (each guarded — missing backends are skipped) ----------

_flashinfer_err: Optional[str] = None
try:
    from flashinfer.fused_moe import fused_topk_deepseek  # type: ignore
except Exception as _e:  # noqa: BLE001
    fused_topk_deepseek = None
    _flashinfer_err = f"{type(_e).__name__}: {_e}"

_sgl_kernel_err: Optional[str] = None
try:
    from sgl_kernel import moe_fused_gate  # type: ignore
except Exception as _e:  # noqa: BLE001
    moe_fused_gate = None
    _sgl_kernel_err = f"{type(_e).__name__}: {_e}"


# Inlined PyTorch reference (mirrors sglang.srt.layers.moe.topk.biased_grouped_topk_impl).
# Inlined rather than imported so the benchmark stays runnable without
# pulling in sglang's full ~30-dep import chain (torchvision, transformers, etc.).
def biased_grouped_topk_impl(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool = True,
):
    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    scores_for_choice = scores + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


# ---- Configs ----------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    name: str
    num_experts: int
    num_expert_group: int
    topk_group: int
    topk: int

    @property
    def experts_per_group(self) -> int:
        return self.num_experts // self.num_expert_group


CONFIGS = {
    "deepseek_v3": Config("deepseek_v3", 256, 8, 4, 8),
    "bailing_hybrid": Config("bailing_hybrid", 256, 8, 4, 8),
    "kimi_k2": Config("kimi_k2", 384, 1, 1, 8),
    # Boundary stress: experts_per_group=64 > 32 → forces PyTorch fallback today.
    "boundary_epg64": Config("boundary_epg64", 512, 8, 4, 8),
    # Non-power-of-two: also forces PyTorch fallback.
    "nonpow2_200": Config("nonpow2_200", 200, 8, 4, 8),
}

TOKEN_COUNTS = [1, 8, 64, 512, 4096, 32768]


# ---- Backend wrappers (uniform signature for timing) ------------------------


def _flashinfer_call(gating_output, correction_bias, cfg: Config):
    assert fused_topk_deepseek is not None
    n = gating_output.shape[0]
    topk_weights = torch.empty(
        (n, cfg.topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty(
        (n, cfg.topk), dtype=torch.int32, device=gating_output.device
    )
    fused_topk_deepseek(
        gating_output.to(torch.float32),
        correction_bias,
        cfg.num_expert_group,
        cfg.topk_group,
        cfg.topk,
        1.0,  # scaling_factor
        topk_weights,
        topk_ids,
        True,  # renormalize
    )
    return topk_weights, topk_ids


def _moe_fused_gate_call(gating_output, correction_bias, cfg: Config):
    assert moe_fused_gate is not None
    return moe_fused_gate(
        gating_output.to(torch.float32),
        correction_bias,
        cfg.num_expert_group,
        cfg.topk_group,
        cfg.topk,
        0,  # num_fused_shared_experts
        1.0,  # routed_scaling_factor
        False,
    )


def _pytorch_call(gating_output, correction_bias, cfg: Config):
    return biased_grouped_topk_impl(
        gating_output,
        correction_bias,
        topk=cfg.topk,
        num_expert_group=cfg.num_expert_group,
        topk_group=cfg.topk_group,
        renormalize=True,
    )


# ---- Constraint matrix (matches biased_grouped_topk_gpu dispatcher) ---------


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _flashinfer_supports(cfg: Config) -> tuple[bool, str]:
    if fused_topk_deepseek is None:
        return False, "backend_unavailable"
    if not _is_pow2(cfg.num_experts):
        return False, "num_experts_not_power_of_two"
    if cfg.topk > 8:
        return False, "topk>8"
    if cfg.topk_group > cfg.num_expert_group:
        return False, "topk_group>num_expert_group"
    if cfg.topk_group * cfg.num_expert_group < cfg.topk:
        return False, "topk_group*num_expert_group<topk"
    if cfg.num_expert_group > 1:
        if cfg.experts_per_group > 32:
            return False, "experts_per_group>32"
        if cfg.experts_per_group * cfg.topk_group > 128:
            return False, "experts_per_group*topk_group>128"
        return True, ""
    if cfg.num_experts > 384:
        return False, "num_experts>384 (single group)"
    return True, ""


def _moe_fused_gate_supports(cfg: Config) -> tuple[bool, str]:
    if moe_fused_gate is None:
        return False, "backend_unavailable"
    if cfg.experts_per_group > 32:
        return False, "experts_per_group>32"
    if not _is_pow2(cfg.num_experts):
        return False, "num_experts_not_power_of_two"
    return True, ""


BACKENDS: dict[str, tuple[Callable, Callable]] = {
    "flashinfer": (_flashinfer_call, _flashinfer_supports),
    "moe_fused_gate": (_moe_fused_gate_call, _moe_fused_gate_supports),
    "pytorch": (_pytorch_call, lambda cfg: (True, "")),
}


# ---- Timing -----------------------------------------------------------------


def time_kernel(
    fn: Callable,
    *args,
    warmup: int = 5,
    iters: int = 50,
) -> float:
    """Return median latency in microseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record()
        fn(*args)
        e.record()
    torch.cuda.synchronize()

    times_us = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return times_us[len(times_us) // 2]


# ---- Runner -----------------------------------------------------------------


def make_inputs(num_tokens: int, cfg: Config, device: str = "cuda"):
    torch.manual_seed(0)
    gating_output = torch.randn(
        num_tokens, cfg.num_experts, dtype=torch.float32, device=device
    )
    correction_bias = torch.randn(cfg.num_experts, dtype=torch.float32, device=device)
    return gating_output, correction_bias


def run(
    selected_configs: list[Config],
    selected_backends: list[str],
    token_counts: list[int],
):
    rows: list[dict] = []
    for cfg in selected_configs:
        print(
            f"\n=== {cfg.name} "
            f"(num_experts={cfg.num_experts}, n_group={cfg.num_expert_group}, "
            f"topk_group={cfg.topk_group}, topk={cfg.topk}, "
            f"experts_per_group={cfg.experts_per_group}, "
            f"power_of_two={_is_pow2(cfg.num_experts)}) ==="
        )
        header = f"{'tokens':>8} | " + " | ".join(f"{b:>16}" for b in selected_backends)
        print(header)
        print("-" * len(header))

        for n in token_counts:
            gating, bias = make_inputs(n, cfg)
            cells = []
            for b in selected_backends:
                fn, supports = BACKENDS[b]
                ok, reason = supports(cfg)
                if not ok:
                    label = (
                        "unavailable" if reason == "backend_unavailable" else "skipped"
                    )
                    cells.append(f"{label:>16}")
                    rows.append(
                        {
                            "config": cfg.name,
                            "tokens": n,
                            "backend": b,
                            "us": None,
                            "supported": False,
                            "skip_reason": reason,
                        }
                    )
                    continue
                try:
                    us = time_kernel(fn, gating, bias, cfg)
                    cells.append(f"{us:>13.2f} µs")
                    rows.append(
                        {
                            "config": cfg.name,
                            "tokens": n,
                            "backend": b,
                            "us": us,
                            "supported": True,
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    cells.append(f"  ERR: {type(e).__name__}")
                    rows.append(
                        {
                            "config": cfg.name,
                            "tokens": n,
                            "backend": b,
                            "us": None,
                            "supported": True,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
            print(f"{n:>8} | " + " | ".join(f"{c:>16}" for c in cells))
    return rows


def print_speedup_summary(rows: list[dict]):
    """Speedup of fast backends vs PyTorch reference, per (config, tokens)."""
    print("\n=== Speedup vs PyTorch reference ===")
    by_key: dict[tuple, dict] = {}
    for r in rows:
        if r.get("us") is None:
            continue
        by_key.setdefault((r["config"], r["tokens"]), {})[r["backend"]] = r["us"]

    fast_backends = [b for b in BACKENDS if b != "pytorch"]
    header = f"{'config':<18} {'tokens':>8} | " + " | ".join(
        f"{b:>16}" for b in fast_backends
    )
    print(header)
    print("-" * len(header))
    for (cfg_name, tokens), bydict in sorted(by_key.items()):
        ref = bydict.get("pytorch")
        if ref is None:
            continue
        cells = []
        for b in fast_backends:
            v = bydict.get(b)
            cells.append(f"{ref/v:>14.2f}x" if v else f"{'—':>15}")
        print(f"{cfg_name:<18} {tokens:>8} | " + " | ".join(f"{c:>16}" for c in cells))


# ---- Main -------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument(
        "--config",
        nargs="+",
        choices=list(CONFIGS) + ["all"],
        default=["all"],
        help="Which config(s) to run.",
    )
    p.add_argument(
        "--backend",
        nargs="+",
        choices=list(BACKENDS),
        default=list(BACKENDS),
        help="Which backend(s) to time.",
    )
    p.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=TOKEN_COUNTS,
        help="Token counts to sweep.",
    )
    p.add_argument("--json", type=str, default=None, help="Optional JSON output path.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    cfgs = (
        list(CONFIGS.values())
        if "all" in args.config
        else [CONFIGS[c] for c in args.config]
    )

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(
        f"Backends available:  flashinfer={fused_topk_deepseek is not None}, "
        f"moe_fused_gate={moe_fused_gate is not None}"
    )
    if _flashinfer_err:
        print(f"  flashinfer import error: {_flashinfer_err}")
    if _sgl_kernel_err:
        print(f"  sgl_kernel import error: {_sgl_kernel_err}")

    rows = run(cfgs, args.backend, args.tokens)
    print_speedup_summary(rows)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
