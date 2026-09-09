"""Native MXFP4 Triton decode kernel vs a pure-Torch golden.

``mxfp4_decode_attention_fwd`` reads the packed E2M1 + E8M0 pool buffers and
dequantizes inline; the golden (``torch_mxfp4_radix_decode_reference``) reads the
SAME packed bytes, dequantizes with the production eager codec (independently
validated bit-exact against an OCP reference in the codec test), and runs explicit
fp32 GQA decode attention. Both sides therefore consume identical dequantized
values, so the residual is kernel accumulation order only.

Also checks the MXFP4 Triton kernels capture into a CUDA graph and replay
byte-deterministically (the kernel-level half of CUDA-graph support).
"""

import unittest
from dataclasses import dataclass
from itertools import accumulate

import torch

from sglang.kernels.ops.attention.decode_attention_mxfp4_sm120 import (
    mxfp4_decode_attention_fwd,
)
from sglang.srt.layers.quantization.kvfp4_tensor import MXFP4KVQuantizeUtil
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

_HAS_REQUIREMENTS = torch.cuda.is_available()
_DEVICE = "cuda"


def decode_output_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Comparable metrics for decode outputs: max abs, mean abs, relative L2,
    cosine, norm ratio. Both inputs flattened in fp32 (device-agnostic)."""
    e = expected.detach().float().reshape(-1)
    a = actual.detach().float().reshape(-1).to(e.device)
    assert a.shape == e.shape, f"shape mismatch: {a.shape} vs {e.shape}"
    diff = a - e
    norm_a = a.norm().item()
    norm_e = e.norm().item()
    cos = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rel_l2": (diff.norm() / (e.norm() + 1e-12)).item(),
        "cosine": cos,
        "norm_ratio": norm_a / (norm_e + 1e-12),
    }


def format_metrics(metrics: dict) -> str:
    return (
        f"max_abs={metrics['max_abs']:.3e} mean_abs={metrics['mean_abs']:.3e} "
        f"rel_l2={metrics['rel_l2']:.3e} cos={metrics['cosine']:.6f} "
        f"norm_ratio={metrics['norm_ratio']:.5f}"
    )


# ---------------------------------------------------------------------------
# Pure-Torch decode golden (dsv4_attention.py style): dequantize the same packed
# bytes with the production eager codec, then run explicit fp32 GQA attention.
# Never calls the Triton kernel, so a mismatch localizes to the kernel.
# ---------------------------------------------------------------------------


@dataclass
class DecodeMathDiagnostics:
    """Codec-independent fp32 decode-attention intermediates."""

    k_dequant: torch.Tensor  # (q_heads, seq, dim) fp32
    v_dequant: torch.Tensor  # (q_heads, seq, dim) fp32
    scores: torch.Tensor  # (q_heads, seq) fp32
    probs: torch.Tensor  # (q_heads, seq) fp32
    output: torch.Tensor  # (q_heads, head_dim) fp32


def torch_radix_decode_from_effective_kv(
    q: torch.Tensor,
    k_effective: torch.Tensor,
    v_effective: torch.Tensor,
    *,
    scaling: float,
    return_diagnostics: bool = False,
):
    """Run the shared decode math on already dequantized logical K/V.

    ``k_effective`` and ``v_effective`` are logical sequences shaped
    ``(seq_len, num_kv_heads, head_dim)``. This owns all codec-independent
    behavior: GQA head expansion, fp32 QK, softmax, and fp32 PV.
    """
    q32 = q.to(torch.float32)
    k32 = k_effective.to(torch.float32).transpose(0, 1)
    v32 = v_effective.to(torch.float32).transpose(0, 1)

    num_q_heads = q32.shape[0]
    num_kv_heads = k32.shape[0]
    assert num_q_heads % num_kv_heads == 0, "GQA requires q % kv == 0"
    group = num_q_heads // num_kv_heads
    if group > 1:
        k32 = k32.repeat_interleave(group, dim=0)
        v32 = v32.repeat_interleave(group, dim=0)

    scores = torch.einsum("hd,hsd->hs", q32, k32) * scaling
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hs,hsd->hd", probs, v32)

    if not return_diagnostics:
        return out
    return out, DecodeMathDiagnostics(
        k_dequant=k32,
        v_dequant=v32,
        scores=scores,
        probs=probs,
        output=out,
    )


@dataclass
class Mxfp4DecodeDiagnostics:
    k_packed: torch.Tensor
    v_packed: torch.Tensor
    k_scales: torch.Tensor
    v_scales: torch.Tensor
    k_dequant: torch.Tensor
    v_dequant: torch.Tensor
    scores: torch.Tensor
    probs: torch.Tensor
    output: torch.Tensor


def torch_mxfp4_radix_decode_reference(
    q: torch.Tensor,
    k_packed_cache: torch.Tensor,
    v_packed_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices,
    seq_lens,
    *,
    scaling: float,
    logical_dim: int | None = None,
    return_diagnostics: bool = False,
):
    """Decode one request from a paged MXFP4 cache with shared Torch math."""
    logical_dim = q.shape[-1] if logical_dim is None else logical_dim
    seq_len = int(seq_lens)
    locs = req_to_token[int(req_pool_indices), :seq_len].long()
    k_packed = k_packed_cache[locs]
    v_packed = v_packed_cache[locs]
    k_scales = k_scale_cache[locs]
    v_scales = v_scale_cache[locs]
    k_effective = MXFP4KVQuantizeUtil.batched_dequantize(
        k_packed, k_scales, logical_dim=logical_dim, dtype=torch.float32
    )
    v_effective = MXFP4KVQuantizeUtil.batched_dequantize(
        v_packed, v_scales, logical_dim=logical_dim, dtype=torch.float32
    )
    out, math_diag = torch_radix_decode_from_effective_kv(
        q,
        k_effective,
        v_effective,
        scaling=scaling,
        return_diagnostics=True,
    )
    if not return_diagnostics:
        return out
    assert isinstance(math_diag, DecodeMathDiagnostics)
    return out, Mxfp4DecodeDiagnostics(
        k_packed=k_packed,
        v_packed=v_packed,
        k_scales=k_scales,
        v_scales=v_scales,
        k_dequant=math_diag.k_dequant,
        v_dequant=math_diag.v_dequant,
        scores=math_diag.scores,
        probs=math_diag.probs,
        output=math_diag.output,
    )


# Hard caps bounding IMPLEMENTATION equivalence (Triton native vs the Torch
# golden on the SAME packed cache); never relaxed to mask a scale/page/pack bug.
REL_L2_CAP = 2e-2
COSINE_CAP = 0.999
NORM_RATIO_RANGE = (0.98, 1.02)

# Frozen per-path thresholds (worst * 1.25 from an offline 20-seed sweep on
# SM120 / torch 2.13.0+cu130). MHA (per-head kernel, fp32 elementwise): the
# native kernel dequantizes inline in fp32, so both sides consume numerically
# identical values and the residual is fp32 accumulation order only -- four
# orders of magnitude tighter than a BF16-materializing (PLAIN) path.
MXFP4_TRITON_FROZEN_REL_L2 = 3.5e-7
MXFP4_TRITON_FROZEN_COSINE = 0.9999997
MXFP4_TRITON_FROZEN_NORM_RATIO = (0.9999998, 1.0000002)

# GQA/MQA (grouped kernel): serves the whole query group with tl.dot over bf16
# tiles (stock-kernel semantics; the K/V unpack is shared across the group
# instead of re-read per query head). The bf16 cast is lossless for dequantized
# E2M1 values; the residual comes from p->bf16 rounding before the PV dot plus
# the tensor-core accumulation order.
MXFP4_TRITON_GROUPED_FROZEN_REL_L2 = 1.85e-3
MXFP4_TRITON_GROUPED_FROZEN_COSINE = 0.9999986
MXFP4_TRITON_GROUPED_FROZEN_NORM_RATIO = (0.9994, 1.0006)


def _assert_caps(testcase, metrics, context):
    testcase.assertLessEqual(
        metrics["rel_l2"],
        REL_L2_CAP,
        f"[{context}] rel_l2 {metrics['rel_l2']:.3e} > cap {REL_L2_CAP} "
        f"({format_metrics(metrics)})",
    )
    testcase.assertGreaterEqual(
        metrics["cosine"],
        COSINE_CAP,
        f"[{context}] cosine {metrics['cosine']:.6f} < cap {COSINE_CAP} "
        f"({format_metrics(metrics)})",
    )
    lo, hi = NORM_RATIO_RANGE
    testcase.assertTrue(
        lo <= metrics["norm_ratio"] <= hi,
        f"[{context}] norm_ratio {metrics['norm_ratio']:.5f} outside "
        f"[{lo}, {hi}] ({format_metrics(metrics)})",
    )


def _assert_frozen(testcase, metrics, context, grouped=False):
    rel_cap = (
        MXFP4_TRITON_GROUPED_FROZEN_REL_L2 if grouped else MXFP4_TRITON_FROZEN_REL_L2
    )
    cos_cap = (
        MXFP4_TRITON_GROUPED_FROZEN_COSINE if grouped else MXFP4_TRITON_FROZEN_COSINE
    )
    lo, hi = (
        MXFP4_TRITON_GROUPED_FROZEN_NORM_RATIO
        if grouped
        else MXFP4_TRITON_FROZEN_NORM_RATIO
    )
    testcase.assertLessEqual(
        metrics["rel_l2"],
        rel_cap,
        f"[{context}] rel_l2 {metrics['rel_l2']:.3e} > frozen {rel_cap} "
        f"({format_metrics(metrics)})",
    )
    testcase.assertGreaterEqual(
        metrics["cosine"],
        cos_cap,
        f"[{context}] cosine {metrics['cosine']:.6f} < frozen {cos_cap} "
        f"({format_metrics(metrics)})",
    )
    testcase.assertTrue(
        lo <= metrics["norm_ratio"] <= hi,
        f"[{context}] norm_ratio {metrics['norm_ratio']:.5f} outside frozen "
        f"[{lo}, {hi}] ({format_metrics(metrics)})",
    )


# ---------------------------------------------------------------------------
# Decode differential harness
# ---------------------------------------------------------------------------


def _build_req_locs(layout, seq_lens, num_slots, seed):
    """Physical slot lists per request. Slot 0 stays reserved (production
    contract). Page-size > 1 does not constrain the layouts: the kernel's
    page/tok address math is an identity for these buffers, which is exactly
    what the page_size cases below exercise."""
    total = sum(seq_lens)
    if num_slots < total + 2:
        raise ValueError("slot pool too small")
    g = torch.Generator().manual_seed(seed)

    def _split(pool):
        reqs, cur = [], 0
        for s in seq_lens:
            reqs.append(pool[cur : cur + s])
            cur += s
        return reqs

    if layout == "contiguous":
        return _split(list(range(1, 1 + total)))
    if layout == "shuffled_pages":
        perm = torch.randperm(num_slots - 1, generator=g).tolist()
        return _split([i + 1 for i in perm[:total]])
    if layout == "interleaved_pages":
        perm = torch.randperm(num_slots - 1, generator=g).tolist()
        pool = [i + 1 for i in perm]
        reqs = [[] for _ in seq_lens]
        idx = 0
        for slot in pool:
            while idx < len(reqs) and len(reqs[idx]) >= seq_lens[idx]:
                idx += 1
            if idx >= len(reqs):
                break
            reqs[idx].append(slot)
        if any(len(r) != s for r, s in zip(reqs, seq_lens)):
            raise ValueError("interleaved layout ran out of slots")
        return reqs
    if layout == "non_monotonic":
        reqs = _split(list(range(1, 1 + total)))
        return [list(reversed(r)) for r in reqs]
    raise ValueError(f"unknown loc layout {layout!r}")


def _run_decode_case(
    testcase,
    *,
    num_q_heads,
    num_kv_heads,
    head_dim,
    page_size,
    prefix_lens,
    layout="shuffled_pages",
    seed=20260903,
    zero_first_request=False,
    max_kv_splits=8,
    assert_level="frozen",
):
    torch.manual_seed(seed)
    seq_lens = [p + 1 for p in prefix_lens]  # decode: current token included
    batch = len(seq_lens)
    total = sum(seq_lens)
    num_slots = total + page_size + 16

    packed_dim = head_dim // 2
    num_blocks = head_dim // 32
    scaling = head_dim**-0.5

    k_pool = torch.zeros(
        (num_slots, num_kv_heads, packed_dim), dtype=torch.uint8, device=_DEVICE
    )
    v_pool = torch.zeros(
        (num_slots, num_kv_heads, packed_dim), dtype=torch.uint8, device=_DEVICE
    )
    ks_pool = torch.zeros(
        (num_slots, num_kv_heads, num_blocks), dtype=torch.uint8, device=_DEVICE
    )
    vs_pool = torch.zeros(
        (num_slots, num_kv_heads, num_blocks), dtype=torch.uint8, device=_DEVICE
    )

    req_locs = _build_req_locs(layout, seq_lens, num_slots, seed)
    req_to_token = torch.zeros(
        (batch, max(seq_lens)), dtype=torch.int64, device=_DEVICE
    )
    for i, locs in enumerate(req_locs):
        locs_t = torch.tensor(locs, dtype=torch.int64, device=_DEVICE)
        req_to_token[i, : len(locs)] = locs_t
        k_logical = torch.randn((len(locs), num_kv_heads, head_dim), device=_DEVICE).to(
            torch.bfloat16
        )
        v_logical = torch.randn((len(locs), num_kv_heads, head_dim), device=_DEVICE).to(
            torch.bfloat16
        )
        if zero_first_request and i == 0:
            # amax == 0 blocks: scale byte 0 (2^-127), all-zero E2M1 data.
            k_logical = torch.zeros_like(k_logical)
            v_logical = torch.zeros_like(v_logical)
        k_packed, k_scales = MXFP4KVQuantizeUtil.batched_quantize(k_logical)
        v_packed, v_scales = MXFP4KVQuantizeUtil.batched_quantize(v_logical)
        k_pool[locs_t] = k_packed
        ks_pool[locs_t] = k_scales
        v_pool[locs_t] = v_packed
        vs_pool[locs_t] = v_scales

    q = torch.randn((batch, num_q_heads, head_dim), device=_DEVICE).to(torch.bfloat16)
    o = torch.zeros((batch, num_q_heads, head_dim), dtype=torch.float32, device=_DEVICE)

    kv_indptr = torch.tensor(
        [0] + list(accumulate(seq_lens)), dtype=torch.int32, device=_DEVICE
    )
    kv_indices = torch.tensor(
        [loc for locs in req_locs for loc in locs], dtype=torch.int32, device=_DEVICE
    )
    mxfp4_decode_attention_fwd(
        q,
        k_pool,
        v_pool,
        ks_pool,
        vs_pool,
        o,
        kv_indptr,
        kv_indices,
        scaling,
        page_size=page_size,
        max_kv_splits=max_kv_splits,
    )

    refs = [
        torch_mxfp4_radix_decode_reference(
            q[i],
            k_pool,
            v_pool,
            ks_pool,
            vs_pool,
            req_to_token,
            i,
            seq_lens[i],
            scaling=scaling,
            logical_dim=head_dim,
        )
        for i in range(batch)
    ]
    ref = torch.stack(refs, dim=0)

    metrics = decode_output_metrics(o, ref)
    grouped = num_q_heads != num_kv_heads
    if assert_level == "frozen":
        _assert_frozen(
            testcase, metrics, f"{layout}_p{page_size}_hd{head_dim}", grouped
        )
    else:
        _assert_caps(testcase, metrics, f"{layout}_p{page_size}_hd{head_dim}")
    return metrics


# (name, q_heads, kv_heads, head_dim, page_size, prefix_lens, layout, kwargs).
# Representative coverage: MHA per-head path, GQA/MQA grouped path, the Qwen
# 24q/4kv/256 shape, page-size and physical-layout variants, a long multi-split
# sequence, and an all-zero (amax==0) request.
_DECODE_CASES = (
    ("mha_hd64_p1_contiguous", 4, 4, 64, 1, (31,), "contiguous", {}),
    ("gqa_hd64_p16_boundary", 4, 2, 64, 16, (14, 15, 16), "shuffled_pages", {}),
    ("mqa_hd64_p16_bsz1", 4, 1, 64, 16, (7,), "shuffled_pages", {}),
    ("qwen_24_4_256_p16", 24, 4, 256, 16, (31,), "shuffled_pages", {}),
    ("qwen_24_4_256_multisplit_long", 24, 4, 256, 16, (255,), "shuffled_pages", {}),
    (
        "qwen_24_4_256_zero_req",
        24,
        4,
        256,
        16,
        (15, 16),
        "shuffled_pages",
        {"zero_first_request": True},
    ),
)


@unittest.skipUnless(_HAS_REQUIREMENTS, "CUDA is required")
class TestTritonMxfp4NativeDecode(CustomTestCase):
    def test_triton_mxfp4_native_decode_cases(self):
        for (
            name,
            q_heads,
            kv_heads,
            head_dim,
            page_size,
            prefix_lens,
            layout,
            kwargs,
        ) in _DECODE_CASES:
            with self.subTest(case=name):
                metrics = _run_decode_case(
                    self,
                    num_q_heads=q_heads,
                    num_kv_heads=kv_heads,
                    head_dim=head_dim,
                    page_size=page_size,
                    prefix_lens=prefix_lens,
                    layout=layout,
                    **kwargs,
                )
                print(f"[mxfp4-triton-decode] {name}: {format_metrics(metrics)}")


@unittest.skipUnless(_HAS_REQUIREMENTS, "CUDA is required")
class TestMxfp4TritonCudaGraphSafety(CustomTestCase):
    """Both MXFP4 Triton kernels must capture into a CUDA graph and replay
    deterministically: outputs byte-identical across replays AND identical to the
    eager run (same deterministic kernel, same static buffers). This is the
    kernel-level half of CUDA-graph support; the service-level half is the
    end-to-end server test."""

    def _capture(self, fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()  # warmup: triton JIT must not compile during capture
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g

    def test_decode_kernel_capture_replay(self):
        torch.manual_seed(20260905)
        batch, q_heads, kv_heads, head_dim, page_size = 2, 8, 4, 64, 16
        seq_lens = (37, 96)
        slots = sum(seq_lens) + page_size + 8
        scaling = head_dim**-0.5

        packed_dim, num_blocks = head_dim // 2, head_dim // 32
        k_pool = torch.randint(
            0, 256, (slots, kv_heads, packed_dim), dtype=torch.uint8, device=_DEVICE
        )
        v_pool = torch.randint(
            0, 256, (slots, kv_heads, packed_dim), dtype=torch.uint8, device=_DEVICE
        )
        ks_pool = torch.randint(
            120, 135, (slots, kv_heads, num_blocks), dtype=torch.uint8, device=_DEVICE
        )
        vs_pool = torch.randint(
            120, 135, (slots, kv_heads, num_blocks), dtype=torch.uint8, device=_DEVICE
        )
        q = torch.randn((batch, q_heads, head_dim), device=_DEVICE).to(torch.bfloat16)

        max_kv_splits = 8
        splits = [max(1, min(max_kv_splits, (s + 31) // 32)) for s in seq_lens]
        num_kv_splits = torch.tensor(splits, dtype=torch.int32, device=_DEVICE)
        attn_logits = torch.empty(
            (batch, q_heads, max_kv_splits, head_dim),
            dtype=torch.float32,
            device=_DEVICE,
        )
        attn_lse = torch.empty(
            (batch, q_heads, max_kv_splits), dtype=torch.float32, device=_DEVICE
        )

        kv_indptr = torch.tensor(
            [0] + list(accumulate(seq_lens)), dtype=torch.int32, device=_DEVICE
        )
        total = sum(seq_lens)
        kv_indices = torch.tensor(
            list(range(1, 1 + total)), dtype=torch.int32, device=_DEVICE
        )

        def run(out):
            mxfp4_decode_attention_fwd(
                q,
                k_pool,
                v_pool,
                ks_pool,
                vs_pool,
                out,
                kv_indptr,
                kv_indices,
                scaling,
                page_size=page_size,
                max_kv_splits=max_kv_splits,
                attn_logits=attn_logits,
                attn_lse=attn_lse,
                num_kv_splits=num_kv_splits,
            )

        o_eager = torch.zeros(
            (batch, q_heads, head_dim), dtype=torch.float32, device=_DEVICE
        )
        run(o_eager)

        o_graph = torch.zeros_like(o_eager)
        graph = self._capture(lambda: run(o_graph))
        graph.replay()
        out1 = o_graph.clone()
        graph.replay()
        out2 = o_graph.clone()

        self.assertTrue(torch.equal(out1, out2), "replay is not deterministic")
        self.assertTrue(
            torch.equal(out1, o_eager),
            f"graph output differs from eager: max diff "
            f"{(out1 - o_eager).abs().max().item()}",
        )

    def test_fused_write_kernel_capture_replay(self):
        from sglang.kernels.ops.quantization.mxfp4_quant import quant_store_kv_mxfp4

        torch.manual_seed(20260906)
        tokens, kv_heads, head_dim = 5, 4, 128
        slots = tokens + 8
        k = (torch.randn((tokens, kv_heads, head_dim), device=_DEVICE) * 1.5).to(
            torch.bfloat16
        )
        v = (torch.randn((tokens, kv_heads, head_dim), device=_DEVICE) * 1.5).to(
            torch.bfloat16
        )
        loc = torch.arange(1, tokens + 1, dtype=torch.int64, device=_DEVICE)

        def fresh_pools():
            return (
                torch.zeros(
                    (slots, kv_heads, head_dim // 2), dtype=torch.uint8, device=_DEVICE
                ),
                torch.zeros(
                    (slots, kv_heads, head_dim // 2), dtype=torch.uint8, device=_DEVICE
                ),
                torch.zeros(
                    (slots, kv_heads, head_dim // 32), dtype=torch.uint8, device=_DEVICE
                ),
                torch.zeros(
                    (slots, kv_heads, head_dim // 32), dtype=torch.uint8, device=_DEVICE
                ),
            )

        kd_e, vd_e, ks_e, vs_e = fresh_pools()
        quant_store_kv_mxfp4(k, v, loc, kd_e, vd_e, ks_e, vs_e)

        kd_g, vd_g, ks_g, vs_g = fresh_pools()
        graph = self._capture(
            lambda: quant_store_kv_mxfp4(k, v, loc, kd_g, vd_g, ks_g, vs_g)
        )
        graph.replay()
        snap1 = (kd_g.clone(), vd_g.clone(), ks_g.clone(), vs_g.clone())
        graph.replay()
        snap2 = (kd_g.clone(), vd_g.clone(), ks_g.clone(), vs_g.clone())

        for a, b in zip(snap1, snap2):
            self.assertTrue(
                torch.equal(a, b), "fused write replay is not deterministic"
            )
        for got, want in zip(snap1, (kd_e, vd_e, ks_e, vs_e)):
            self.assertTrue(
                torch.equal(got, want),
                "fused write graph output differs from eager bytes",
            )


if __name__ == "__main__":
    unittest.main()
