"""T40 parity: the fused q_a/kv_a RMSNorm (SGLANG_OPT_FUSED_MLA_QKV_A_NORM) against the unfused
path it replaces, at every decode graph-bucket row count of the T38 decode (EAGLE 3/1/4, graph
bs <= 24 per rank: verify / draft-extend rows 4 * bs, draft-decode rows bs).

Unfused = the two sgl_kernel ``rmsnorm`` launches RMSNorm.forward_cuda runs, on the projection
output (plain BF16, or the TGV split-K route's summed output); fused = ``mla_qkv_a_rmsnorm`` on
that output or on the route's FP32 partials. Also writes both ``k_nope`` into the NVFP4 decode
KV store (T38 352-B rows) and compares the stored rows.
"""

import json
import os

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="SM 10.x only",
)

from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (  # noqa: E402
    tgv_splitk_bf16_gemm_out,
    tgv_splitk_bf16_gemm_partials,
)
from sglang.kernels.ops.kvcache import dsa_nvfp4_mla_kv as kv  # noqa: E402
from sglang.kernels.ops.layernorm.mla_qkv_a_rmsnorm import (  # noqa: E402
    mla_qkv_a_rmsnorm,
)
from sglang.srt.layers.quantization.unquant import (  # noqa: E402
    bf16_splitk_heuristic_route,
)

DEV = "cuda"
HIDDEN, Q_LORA, KV_LORA, ROPE = 6144, 2048, 512, 64
WIDTH = Q_LORA + KV_LORA + ROPE  # 2624, fused_qkv_a output
EPS = 1e-5  # GLM-5.2 rms_norm_eps
GRAPH_BS = list(range(1, 9)) + list(range(10, 25, 2))
ROWS = sorted(set(range(1, 25)) | {4 * b for b in GRAPH_BS})
# Forced SM counts: Rubin (VR200) 212 as measured on Hecate (T10 REPORT, job 905821); add 224 (full die)
# through T40_FORCE_SMS="212,224". A forced count equal to this GPU's own is not repeated.
RUBIN_SMS = tuple(int(v) for v in os.environ.get("T40_FORCE_SMS", "212").split(",") if v)
BF16_ULP = 2.0**-7  # relative spacing of bf16 at the top of a binade


def _record(kind: str, **fields) -> None:
    path = os.environ.get("T40_PARITY_JSONL")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"test": "fused_qkv_a_norm", "kind": kind, **fields}) + "\n")


def _inputs(m: int, seed: int):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(m, HIDDEN, device=DEV, generator=gen).to(torch.bfloat16)
    w = (torch.randn(WIDTH, HIDDEN, device=DEV, generator=gen) * 0.02).to(torch.bfloat16)
    qw = (1.0 + 0.1 * torch.randn(Q_LORA, device=DEV, generator=gen)).to(torch.bfloat16)
    kw = (1.0 + 0.1 * torch.randn(KV_LORA, device=DEV, generator=gen)).to(torch.bfloat16)
    return x, w, qw, kw


def _unfused(latent: torch.Tensor, qw, kw):
    from sgl_kernel import rmsnorm  # RMSNorm.forward_cuda's kernel; imported here so CPU collection works

    q, latent_cache = latent.split([Q_LORA, KV_LORA + ROPE], dim=-1)
    k_nope = latent_cache[..., :KV_LORA]
    return rmsnorm(q.contiguous(), qw, EPS), rmsnorm(k_nope.contiguous(), kw, EPS)


def _ulp_stats(a: torch.Tensor, b: torch.Tensor):
    a32, b32 = a.float(), b.float()
    diff = (a32 - b32).abs()
    scale = torch.maximum(a32.abs(), b32.abs()).clamp_min(1e-30)
    return {
        "max_abs": diff.max().item(),
        "max_rel": (diff / scale).max().item(),
        "mismatch_frac": (a != b).float().mean().item(),
    }


def _assert_norm_close(name: str, fused: torch.Tensor, ref: torch.Tensor):
    assert torch.isfinite(fused).all(), f"{name}: non-finite fused output"
    st = _ulp_stats(fused, ref)
    # Same math, different reduction order of the row's sum of squares: at most one
    # bf16 rounding step apart, on a small fraction of elements.
    assert st["max_rel"] <= 2 * BF16_ULP, (name, st)
    assert st["mismatch_frac"] <= 0.05, (name, st)
    return st


def _nvfp4_rows(k_nope: torch.Tensor, k_rope: torch.Tensor) -> torch.Tensor:
    rows = torch.zeros(k_nope.shape[0], 1, kv.ROW_BYTES, dtype=torch.uint8, device=DEV)
    loc = torch.arange(k_nope.shape[0], device=DEV, dtype=torch.int64)
    g = torch.ones(1, dtype=torch.float32, device=DEV)  # T38: global scales 78/78 == 1
    kv.dsa_nvfp4_quantize_into(k_nope, k_rope, rows, loc, g)
    return rows


@pytest.mark.parametrize("m", ROWS)
def test_fused_norm_plain_projection(m):
    """Projection without split-K (M <= 32 warp route, or any unrouted M)."""
    x, w, qw, kw = _inputs(m, seed=1000 + m)
    latent = torch.nn.functional.linear(x, w)
    q_ref, k_ref = _unfused(latent, qw, kw)
    out_latent, q, k = mla_qkv_a_rmsnorm(latent, qw, EPS, kw, EPS)
    assert out_latent.data_ptr() == latent.data_ptr()
    sq = _assert_norm_close("q", q, q_ref)
    sk = _assert_norm_close("k_nope", k, k_ref)
    # NVFP4 decode KV: the stored 352-B rows of the fused and unfused k_nope.
    k_rope = latent[:, Q_LORA + KV_LORA :]
    rows_f, rows_u = _nvfp4_rows(k, k_rope), _nvfp4_rows(k_ref, k_rope)
    byte_mismatch = (rows_f != rows_u).float().mean().item()
    deq_f = kv.dequant_reference(rows_f, 1.0, torch.float32)[:, :KV_LORA]
    deq_u = kv.dequant_reference(rows_u, 1.0, torch.float32)[:, :KV_LORA]
    kn = latent[:, Q_LORA : Q_LORA + KV_LORA].float()
    exact = kn * torch.rsqrt(kn.pow(2).mean(-1, keepdim=True) + EPS) * kw.float()
    err_f = (deq_f - exact).pow(2).mean().sqrt().item()
    err_u = (deq_u - exact).pow(2).mean().sqrt().item()
    assert torch.isfinite(deq_f).all()
    # A bf16 ulp can move a value across an E2M1 decision boundary: rare, and the stored row
    # must stay as close to the exact norm as the unfused one.
    assert byte_mismatch <= 0.01, byte_mismatch
    assert err_f <= err_u * 1.01 + 1e-6, (err_f, err_u)
    _record("plain", m=m, q=sq, k_nope=sk, nvfp4_byte_mismatch=byte_mismatch,
            nvfp4_rmse_fused=err_f, nvfp4_rmse_unfused=err_u)


def _splitk_cases():
    """Only (M, SM count) whose fused_qkv_a route hands partials to the fused kernel."""
    own = torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 152
    for m in ROWS:
        for sms in (None,) + tuple(v for v in RUBIN_SMS if v != own):
            r = bf16_splitk_heuristic_route(m=m, n=WIDTH, k=HIDDEN, num_sms=sms or own)
            if r is not None and r.kernel == "tgv" and r.split_k > 1:
                yield m, sms


@pytest.mark.parametrize("m,sms", list(_splitk_cases()))
def test_fused_norm_sums_splitk_partials(m, sms):
    """fused_qkv_a on the TGV split-K route: the fused kernel sums the FP32 partials.

    ``sms=None`` is this GPU's own route; RUBIN_SMS force the split Rubin would pick.
    """
    num_sms = sms or torch.cuda.get_device_properties(0).multi_processor_count
    route = bf16_splitk_heuristic_route(m=m, n=WIDTH, k=HIDDEN, num_sms=num_sms)
    assert route is not None and route.kernel == "tgv" and route.split_k > 1, route
    x, w, qw, kw = _inputs(m, seed=2000 + m)
    # Unfused production path: the route's own sum kernel, then the two norms.
    latent_ref = torch.empty(m, WIDTH, dtype=torch.bfloat16, device=DEV)
    tgv_splitk_bf16_gemm_out(x, w, latent_ref, tactic=route.tactic, split_k=route.split_k)
    q_ref, k_ref = _unfused(latent_ref, qw, kw)
    partials = tgv_splitk_bf16_gemm_partials(x, w, tactic=route.tactic, split_k=route.split_k)
    assert partials.shape == (route.split_k, m, WIDTH)
    latent, q, k = mla_qkv_a_rmsnorm(partials, qw, EPS, kw, EPS)
    # The in-kernel sum adds the partials in index order and rounds once, as the sum kernel.
    assert torch.equal(latent, latent_ref), "fused split-K sum != route sum kernel"
    sq = _assert_norm_close("q", q, q_ref)
    sk = _assert_norm_close("k_nope", k, k_ref)
    _record("splitk", m=m, num_sms=num_sms, split_k=route.split_k, tactic=route.tactic,
            q=sq, k_nope=sk)
