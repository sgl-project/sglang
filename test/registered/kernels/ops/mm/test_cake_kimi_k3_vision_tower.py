"""Cake Kimi-K3 vision tower (MoonViT-3D + PatchMergerV2) through sglang.kernels.

Checks: the registry resolves the explicit FlashInfer backend for the one-shot
entry, the prepared runner and the weight preparation; the facade results are
bitwise identical to calling FlashInfer directly (and the prepared runner is
bitwise identical to the one-shot form across a CUDA-graph replay); and the
tower output passes FlashInfer's oracle-fairness gate for the complete call
on a one-layer stack and the smallest grid.

Oracle and bound (FlashInfer ``tests/experimental/test_cake_kimi_k3_vision_tower.py``
and ``flashinfer/experimental/kimi_k3_vision_tower/README.md`` at
``46340689a5ab``): FlashInfer checks every *stage* against the FP32 oracle of
its operator at ``atol = rtol = 1e-2``, but validates the *complete call*
with the oracle-fairness gate of the Cake evaluation contract -- the error
against the pure-torch FP32 HF chain (patch embed + positional rows,
per-layer RMSNorm / QKV / 2-D RoPE / segment attention / out-proj / GELU-tanh
MLP, final RMSNorm, 2x2 temporal-pool merge, GELU merger, projector RMSNorm)
must be no worse than the HF BF16 chain's: ``violations(1e-2 band) <= 1.1 x
chain + 16``, ``mean error <= 1.05 x chain``, ``max error <= 1.5 x chain``,
all outputs finite. A ~10-round-point BF16 chain cannot meet a flat 1e-2
band against FP32 on every element, so the flat band is not the gate here.
Skips when FlashInfer lacks the module, the GPU is not sm_100a / sm_103a, or
the generated program is not registered for the arch.
"""

import math
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import mm as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.mm.cake import (
    cake_kimi_k3_vision_tower,
    cake_prepare_kimi_k3_vision_tower,
    cake_prepare_kimi_k3_vision_weights,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=900, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

PATCH, PATCH_DIM, HIDDEN, QKV_HIDDEN, HEADS = 14, 588, 1024, 1536, 12
HEAD_DIM, FFN, MERGED_DIM, TEXT_HIDDEN = 128, 4096, 4096, 7168
NORM_EPS, PROJECTOR_EPS = 2.0**-7, 1e-5
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIM)
# Per-operator band of the Cake evaluation contract (FlashInfer ATOL/RTOL).
ATOL = RTOL = 1e-2
SMALL_GRID = [(1, 2, 2)]
MIXED_GRIDS = [(1, 2, 6), (1, 10, 4), (2, 4, 4), (1, 6, 30)]


@pytest.mark.parametrize(
    "op",
    [
        "mm.kimi_k3_vision_tower",
        "mm.prepare_kimi_k3_vision_tower",
        "mm.prepare_kimi_k3_vision_weights",
    ],
)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.mm:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake.FI_MODULE, cake.FI_BACKEND_MODULE, cake.FI_JIT_MODULE
    ):
        pytest.skip("installed FlashInfer lacks flashinfer.kimi_k3_vision")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(f"Cake Kimi-K3 vision tower needs sm_100a/103a, device is {cc}")
    from flashinfer.experimental.kimi_k3_vision_tower import cake_backend

    if not cake_backend.generated_program_available(torch.device("cuda")):
        pytest.skip(f"FlashInfer does not register the vision-tower program for {cc}")


def _make_weights(device, layers, seed):
    from flashinfer.experimental.kimi_k3_vision_tower.cake_backend import (
        sincos_time_table,
    )

    g = torch.Generator(device=device).manual_seed(seed)

    def normal(shape, std):
        return (
            torch.empty(shape, dtype=torch.float32, device=device)
            .normal_(0.0, std, generator=g)
            .to(torch.bfloat16)
        )

    def uniform(shape, lo, hi):
        return (
            torch.empty(shape, dtype=torch.float32, device=device)
            .uniform_(lo, hi, generator=g)
            .to(torch.bfloat16)
        )

    weights = {
        "patch_proj": normal((HIDDEN, PATCH_DIM), 0.02),
        "pos_emb": normal((64, 64, HIDDEN), 0.02),
        "time_weight": sincos_time_table().to(device=device, dtype=torch.bfloat16),
        "final_norm": uniform((HIDDEN,), 0.9, 1.1),
        "merger_proj0": normal((MERGED_DIM, MERGED_DIM), math.sqrt(2.0 / MERGED_DIM)),
        "merger_proj1": normal((TEXT_HIDDEN, MERGED_DIM), math.sqrt(2.0 / MERGED_DIM)),
        "post_norm": uniform((TEXT_HIDDEN,), 0.9, 1.1),
        "layers": [
            {
                "norm0": uniform((HIDDEN,), 0.9, 1.1),
                "wqkv": normal((3 * QKV_HIDDEN, HIDDEN), 0.02),
                "wo": normal((HIDDEN, QKV_HIDDEN), 0.02),
                "norm1": uniform((HIDDEN,), 0.9, 1.1),
                "fc0": normal((FFN, HIDDEN), math.sqrt(2.0 / HIDDEN)),
                "fc1": normal((HIDDEN, FFN), math.sqrt(2.0 / FFN)),
            }
            for _ in range(layers)
        ],
    }
    return weights


def _inputs(grids, layers, seed):
    from flashinfer.experimental.kimi_k3_vision_tower.cake_backend import (
        cu_seqlens_of,
        merged_tokens,
        pos_emb_rows,
        rope_cos_sin,
    )

    device = torch.device("cuda", 0)
    weights = _make_weights(device, layers, seed)
    total = cu_seqlens_of(grids)[-1]
    g = torch.Generator(device=device).manual_seed(seed + 1)
    pixels = torch.empty(
        (total, 3, PATCH, PATCH), dtype=torch.bfloat16, device=device
    ).uniform_(-1.0, 1.0, generator=g)
    cos, sin = rope_cos_sin(grids, device)
    pos_rows = pos_emb_rows(weights["pos_emb"], weights["time_weight"], grids)
    out = torch.full(
        (merged_tokens(grids), TEXT_HIDDEN),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    return device, weights, pixels, cos, sin, pos_rows, out, tuple(cu_seqlens_of(grids))


# --- FP32 oracle and the HF BF16 reference chain (FlashInfer's _tower) ---------


def _rms_norm(x, weight, eps):
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rstd * weight.float()).to(x.dtype)


def _apply_rope(q, k, cos, sin):
    def rot(x):
        xf = x.float().reshape(*x.shape[:-1], HEAD_DIM // 2, 2)
        a, b = xf[..., 0], xf[..., 1]
        c = cos.reshape(cos.shape[0], 1, HEAD_DIM // 2)
        s = sin.reshape(sin.shape[0], 1, HEAD_DIM // 2)
        return (
            torch.stack([a * c - b * s, a * s + b * c], dim=-1)
            .reshape(x.shape)
            .to(x.dtype)
        )

    return rot(q), rot(k)


def _attention_fp32(q, k, v, cu, out_dtype):
    """Exact FP32 noncausal segment attention."""
    out = torch.empty(q.shape, dtype=out_dtype, device=q.device)
    for a, b in zip(cu, cu[1:]):
        if b <= a:
            continue
        logits = (
            torch.einsum("qhd,khd->hqk", q[a:b].float(), k[a:b].float()) * SOFTMAX_SCALE
        )
        probs = torch.softmax(logits, dim=-1)
        out[a:b] = torch.einsum("hqk,khd->qhd", probs, v[a:b].float()).to(out_dtype)
    return out


def _attention_bf16(q, k, v, cu, out_dtype=None):
    """BF16 tensor-core attention per segment (P rounded to BF16 before PV)."""
    out = torch.empty_like(q)
    for a, b in zip(cu, cu[1:]):
        if b <= a:
            continue
        qs, ks, vs = (t[a:b].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        out[a:b] = (
            F.scaled_dot_product_attention(qs, ks, vs, scale=SOFTMAX_SCALE)
            .squeeze(0)
            .transpose(0, 1)
        )
    return out


def _tpool_merge(x, grids):
    outputs, start = [], 0
    for t, h, w in grids:
        n = t * h * w
        seq = x[start : start + n]
        start += n
        nh, nw = h // 2, w // 2
        r = (
            seq.view(t, nh, 2, nw, 2, x.shape[-1])
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .mean(dim=0)
        )
        outputs.append(r.reshape(nh * nw, 4 * x.shape[-1]))
    return torch.cat(outputs, dim=0)


def _tower(pixels, grids, weights, cos, sin, pos_rows, cu, *, fp32, attention):
    """The HF chain; ``fp32=True`` keeps every parameter and activation in FP32."""
    cast = (lambda t: t.float()) if fp32 else (lambda t: t)  # noqa: E731
    x = F.linear(
        cast(pixels).reshape(pixels.shape[0], PATCH_DIM), cast(weights["patch_proj"])
    ) + cast(pos_rows)
    for lw in weights["layers"]:
        n = _rms_norm(x, cast(lw["norm0"]), NORM_EPS)
        qkv = F.linear(n, cast(lw["wqkv"])).view(x.shape[0], 3, HEADS, HEAD_DIM)
        q, k, v = qkv.unbind(dim=1)
        q, k = _apply_rope(q, k, cos, sin)
        a = attention(q.contiguous(), k.contiguous(), v.contiguous(), cu, x.dtype)
        x = x + F.linear(a.reshape(x.shape[0], QKV_HIDDEN), cast(lw["wo"]))
        n = _rms_norm(x, cast(lw["norm1"]), NORM_EPS)
        x = x + F.linear(
            F.gelu(F.linear(n, cast(lw["fc0"])), approximate="tanh"), cast(lw["fc1"])
        )
    x = _rms_norm(x, cast(weights["final_norm"]), NORM_EPS)
    m = _tpool_merge(x, grids)
    y = F.linear(
        F.gelu(F.linear(m, cast(weights["merger_proj0"]))),
        cast(weights["merger_proj1"]),
    )
    return _rms_norm(y, cast(weights["post_norm"]), PROJECTOR_EPS)


def _fairness(actual, chain, oracle):
    """Oracle-fairness gate of the Cake contract: no worse than the BF16 chain."""
    tol = ATOL + RTOL * oracle.abs()
    err_a = (actual.float() - oracle).abs()
    err_c = (chain.float() - oracle).abs()
    viol_a, viol_c = int((err_a > tol).sum()), int((err_c > tol).sum())
    return dict(
        finite=bool(torch.isfinite(actual.float()).all()),
        violations=(viol_a, viol_c),
        mean=(float(err_a.mean()), float(err_c.mean())),
        max=(float(err_a.max()), float(err_c.max())),
        passed=bool(torch.isfinite(actual.float()).all())
        and viol_a <= 1.1 * viol_c + 16
        and float(err_a.mean()) <= 1.05 * max(float(err_c.mean()), 1e-12)
        and float(err_a.max()) <= 1.5 * max(float(err_c.max()), 1e-6),
    )


def test_matches_flashinfer_and_fp32_oracle():
    _skip_unless_supported()
    grids = SMALL_GRID
    device, weights, pixels, cos, sin, pos_rows, out, cu = _inputs(
        grids, layers=1, seed=3
    )
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        assert cake.supports_kimi_k3_vision_tower(pixels, grids, weights, out)
        result = cake_kimi_k3_vision_tower(pixels, grids, weights, out)
        assert result is out
        from flashinfer.kimi_k3_vision import kimi_k3_vision_tower as fi_direct

        direct = fi_direct(pixels, grids, weights)
        torch.cuda.synchronize()
        assert direct.shape == out.shape and torch.equal(out, direct)
        assert torch.isfinite(out.float()).all()
        oracle = _tower(
            pixels,
            grids,
            weights,
            cos,
            sin,
            pos_rows,
            cu,
            fp32=True,
            attention=_attention_fp32,
        )
        chain = _tower(
            pixels,
            grids,
            weights,
            cos,
            sin,
            pos_rows,
            cu,
            fp32=False,
            attention=_attention_bf16,
        )
        report = _fairness(out, chain, oracle)
        assert report["passed"], report
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def test_prepared_runner_matches_one_shot_and_replays():
    _skip_unless_supported()
    grids = MIXED_GRIDS
    device, weights, pixels, cos, sin, pos_rows, out, cu = _inputs(
        grids, layers=2, seed=5
    )
    assert cake.supports_prepare_kimi_k3_vision_weights(weights)
    prepared = cake_prepare_kimi_k3_vision_weights(weights)
    from flashinfer.experimental.kimi_k3_vision_tower.cake_backend import (
        prepare_kimi_k3_vision_weights as fi_prepare_weights,
    )

    prepared_fi = fi_prepare_weights(weights)
    assert torch.equal(prepared.patch_proj, prepared_fi.patch_proj)
    assert all(
        torch.equal(a[k], b[k])
        for a, b in zip(prepared.layers, prepared_fi.layers)
        for k in a
    )

    runner = cake_prepare_kimi_k3_vision_tower(
        pixels, grids, prepared, out, pos_rows=pos_rows
    )
    runner.launch()
    torch.cuda.synchronize()
    eager = out.clone()
    one_shot = cake_kimi_k3_vision_tower(pixels, grids, prepared, pos_rows=pos_rows)
    torch.cuda.synchronize()
    assert torch.equal(eager, one_shot)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        runner.launch()
    torch.cuda.current_stream().wait_stream(stream)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    pixels.mul_(-1.0)
    runner.launch()
    torch.cuda.synchronize()
    expected = out.clone()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)


def test_admission_rejects_out_of_contract():
    _skip_unless_supported()
    device, weights, pixels, cos, sin, pos_rows, out, cu = _inputs(SMALL_GRID, 1, 1)
    assert cake.supports_kimi_k3_vision_tower(pixels, SMALL_GRID, weights, out)
    assert not cake.supports_kimi_k3_vision_tower(pixels.float(), SMALL_GRID, weights)
    assert not cake.supports_kimi_k3_vision_tower(
        pixels, [(1, 2, 4)], weights
    )  # T mismatch
    assert not cake.supports_kimi_k3_vision_tower(pixels, [(1, 1, 4)], weights)  # odd h
    assert not cake.supports_kimi_k3_vision_tower(
        pixels, SMALL_GRID, weights, out[:, :HIDDEN].contiguous()
    )
    assert not cake.supports_kimi_k3_vision_tower(
        pixels, SMALL_GRID, {k: v for k, v in weights.items() if k != "layers"}
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
