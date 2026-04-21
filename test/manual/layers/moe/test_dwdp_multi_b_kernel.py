"""Standalone correctness test for the flashinfer CuteDSL multi-B MoE kernel,
mirroring SGLang's DWDP call pattern.

Purpose
-------
Isolate whether DWDP output divergence is caused by the flashinfer multi-B
kernel itself, or by the SGLang integration (prefetch stream / IPC / weight
view reconstruction).

What this test does (strictly matching SGLang forward_dwdp)
-----------------------------------------------------------
1. Build a full single-tensor MoE problem (num_experts, hidden, inter).
2. Run baseline: ``cute_dsl_fused_moe_nvfp4`` with single tensors.
3. Simulate DWDP prefetch exactly as
   ``sglang/srt/layers/moe/dwdp/prefetch_buffer.py`` does:
       - Split expert range into ``dwdp_size`` equal chunks
         (num_experts_per_worker = num_experts // dwdp_size).
       - For each chunk, allocate a flat uint8 buffer of size
         ``num_prefetch_experts * per_expert_bytes``.
       - Copy the contiguous physical slice from the original tensor into
         the flat buffer (emulating cudaMemcpyAsync+cudaStreamSynchronize).
       - Reconstruct a strided view via
         ``torch.as_strided(typed, view_shape, view_strides, storage_offset=0)``
         where ``view_strides`` are the ORIGINAL full tensor strides and
         ``view_shape`` is the full shape with the expert dim replaced by
         ``num_prefetch_experts``.
4. Run multi-B: same ``cute_dsl_fused_moe_nvfp4`` call, but with each of
   w13_weight / w13_weight_sf / w1_alpha / w2_weight / w2_weight_sf / w2_alpha
   as a list of ``dwdp_size`` reconstructed tensors.
5. Assert bitwise / numerically-close equivalence.

Reference: TRT-LLM PR #12136 (
https://github.com/NVIDIA/TensorRT-LLM/pull/12136) — multi-B dispatch and
strided-view reconstruction patterns.
"""

import math
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Availability gates
# ---------------------------------------------------------------------------


def _is_sm100_family() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10


def _is_cute_dsl_available() -> bool:
    try:
        from flashinfer.cute_dsl import is_cute_dsl_available
    except Exception:
        return False
    return is_cute_dsl_available()


cute_dsl_available = pytest.mark.skipif(
    not _is_cute_dsl_available(), reason="CuteDSL not available"
)
sm100_required = pytest.mark.skipif(
    not _is_sm100_family(), reason="Requires SM100 family (Blackwell)"
)


# ---------------------------------------------------------------------------
# Tensor factory (matches flashinfer's own test style)
# ---------------------------------------------------------------------------


def _interleave_linear_and_gate(
    x: torch.Tensor, group_size: int = 64, dim: int = -1
) -> torch.Tensor:
    sizes = x.size()
    dim = dim % x.dim()
    assert sizes[dim] % (group_size * 2) == 0
    prev = sizes[:dim]
    post = sizes[dim + 1 :]
    x = x.view(*prev, 2, sizes[dim] // (group_size * 2), group_size, *post)
    x = x.transpose(dim, dim + 1).contiguous().view(*sizes)
    return x


def _create_moe_tensors(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    torch.manual_seed(seed)
    sf_vec_size = 16

    # Activations
    x_bf16 = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) / 10
    )
    a1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    x_fp4, x_sf = fp4_quantize(
        x_bf16, global_scale=a1_gs, sf_vec_size=sf_vec_size, is_sf_swizzled_layout=False
    )
    x_sf = x_sf.unsqueeze(-1)

    # Routing
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    routing = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing, selected = torch.topk(routing, top_k, dim=-1)
    routing = routing / routing.sum(dim=-1, keepdim=True)
    routing = routing.float()
    selected = selected.to(torch.int32)

    # GEMM1 weights
    w1_bf16 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    w1_bf16_il = _interleave_linear_and_gate(w1_bf16, group_size=64, dim=1)
    w1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w1_flat = w1_bf16_il.view(num_experts * 2 * intermediate_size, hidden_size)
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_flat, global_scale=w1_gs, sf_vec_size=sf_vec_size, is_sf_swizzled_layout=True
    )
    w1_q = w1_q_flat.view(num_experts, 2 * intermediate_size, hidden_size // 2)
    w1_sf_mma = convert_sf_to_mma_layout(
        w1_sf_flat,
        m=2 * intermediate_size,
        k=hidden_size,
        num_groups=num_experts,
        sf_vec_size=sf_vec_size,
    )
    w1_alpha = torch.ones(num_experts, device=device, dtype=torch.float32)

    # GEMM2 weights
    w2_bf16 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    w2_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w2_flat = w2_bf16.view(num_experts * hidden_size, intermediate_size)
    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_flat, global_scale=w2_gs, sf_vec_size=sf_vec_size, is_sf_swizzled_layout=True
    )
    w2_q = w2_q_flat.view(num_experts, hidden_size, intermediate_size // 2)
    w2_sf_mma = convert_sf_to_mma_layout(
        w2_sf_flat,
        m=hidden_size,
        k=intermediate_size,
        num_groups=num_experts,
        sf_vec_size=sf_vec_size,
    )
    w2_alpha = torch.ones(num_experts, device=device, dtype=torch.float32)

    fc2_input_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    return {
        "x": x_fp4,
        "x_sf": x_sf,
        "token_selected_experts": selected,
        "token_final_scales": routing,
        "w1_weight": w1_q,
        "w1_weight_sf": w1_sf_mma,
        "w1_alpha": w1_alpha,
        "fc2_input_scale": fc2_input_scale,
        "w2_weight": w2_q,
        "w2_weight_sf": w2_sf_mma,
        "w2_alpha": w2_alpha,
    }


# ---------------------------------------------------------------------------
# DWDP-style split: flat uint8 buffer + torch.as_strided view reconstruction
# ---------------------------------------------------------------------------


def _compute_expert_dim_and_per_expert_bytes(
    t: torch.Tensor, num_experts_per_worker: int
) -> Tuple[int, int]:
    """Mirror DwdpPrefetchBuffer.__init__ expert-dim detection."""
    shape = tuple(t.shape)
    strides = t.stride()
    candidates = [i for i, s in enumerate(shape) if s == num_experts_per_worker]
    assert candidates, (
        f"No dim with size num_experts_per_worker={num_experts_per_worker} "
        f"for tensor with shape {shape}"
    )
    expert_dim = max(candidates, key=lambda i: strides[i])
    assert strides[expert_dim] == max(strides), (
        f"Expert dim {expert_dim} is not the outermost physical dim "
        f"(shape={shape}, strides={strides})"
    )
    per_expert_bytes = strides[expert_dim] * t.element_size()
    return expert_dim, per_expert_bytes


def _split_as_dwdp_prefetch(
    t: torch.Tensor, dwdp_size: int
) -> List[torch.Tensor]:
    """Emulate SGLang DWDP prefetch+reconstruct:

    - Take the first num_experts_per_worker experts' physical bytes and copy
      them into a fresh flat uint8 buffer (1 buffer per "rank").
    - Reconstruct the strided view via torch.as_strided using the ORIGINAL
      tensor's strides and shape (with expert dim replaced by
      num_experts_per_worker), exactly as prefetch_buffer.get_buffer_views.

    Returns a list of ``dwdp_size`` reconstructed tensors ordered by rank
    0..dwdp_size-1.
    """
    shape = tuple(t.shape)
    strides = t.stride()
    dtype = t.dtype

    num_experts_total = None
    for d, s in enumerate(shape):
        if strides[d] == max(strides):
            num_experts_total = s
            expert_dim = d
            break
    assert num_experts_total is not None
    assert num_experts_total % dwdp_size == 0, (
        f"num_experts={num_experts_total} not divisible by dwdp_size={dwdp_size}"
    )

    num_per_worker = num_experts_total // dwdp_size
    per_expert_bytes = strides[expert_dim] * t.element_size()

    # Source: view underlying storage as flat uint8 starting at this tensor's
    # data_ptr. tensor.storage() contains the original allocation; we need a
    # byte-level slice starting at storage_offset.
    # For the convert_sf_to_mma_layout strided view we want to copy the
    # *physical* contiguous region that backs the first N experts, which is
    # num_per_worker * per_expert_bytes bytes.
    src_base_ptr = t.data_ptr()

    view_shape = list(shape)
    view_shape[expert_dim] = num_per_worker
    view_shape = torch.Size(view_shape)
    view_strides = strides  # identical for MMA SF (expert dim outermost)

    out: List[torch.Tensor] = []
    for rank in range(dwdp_size):
        src_offset_bytes = rank * num_per_worker * per_expert_bytes

        # Fresh flat uint8 buffer (simulates the prefetch buffer slot for this
        # rank). Copy the physical byte range into it.
        buf = torch.empty(
            num_per_worker * per_expert_bytes,
            dtype=torch.uint8,
            device=t.device,
        )
        from cuda import cudart

        err = cudart.cudaMemcpy(
            buf.data_ptr(),
            src_base_ptr + src_offset_bytes,
            num_per_worker * per_expert_bytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        )
        if isinstance(err, tuple):
            err = err[0]
        assert err == cudart.cudaError_t.cudaSuccess, (
            f"cudaMemcpy failed (rank={rank}): {err}"
        )

        typed = buf.view(dtype)
        view = torch.as_strided(typed, view_shape, view_strides, storage_offset=0)
        out.append(view)

    torch.cuda.synchronize()
    return out


# ---------------------------------------------------------------------------
# Reference / helpers
# ---------------------------------------------------------------------------


def _check_close(
    actual: torch.Tensor, expected: torch.Tensor, name: str, percent: float = 0.99
):
    actual = actual.float()
    expected = expected.float()
    output_scale = max(expected.std().item(), 0.01)
    atol = max(0.05, 1.5 * output_scale)
    rtol = 0.5
    abs_diff = (actual - expected).abs()
    rel_diff = abs_diff / (expected.abs() + 1e-8)
    ok = (abs_diff < atol) | (rel_diff < rtol)
    frac = ok.float().mean().item()
    assert frac >= percent, (
        f"{name}: only {frac * 100:.2f}% within tolerance "
        f"(atol={atol:.4f}, max|Δ|={abs_diff.max().item():.4e})"
    )


def _rebuild_single_from_list(
    parts: List[torch.Tensor], expert_dim: int
) -> torch.Tensor:
    """Concat the reconstructed views back into a single tensor (logical).

    Used to verify the reconstruction itself preserves values.
    """
    return torch.cat(parts, dim=expert_dim)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@cute_dsl_available
@sm100_required
class TestDwdpMultiBKernel:
    """Strict SGLang-DWDP-style multi-B correctness tests.

    These do NOT simulate IPC or double-buffering — the prefetch itself is
    effectively synchronous (cudaStreamSynchronize in prefetch_buffer.py) so
    what matters for the kernel is the reconstructed tensor layout.
    """

    @pytest.mark.parametrize(
        "num_tokens,hidden_size,intermediate_size,num_experts,top_k,dwdp_size",
        [
            # Small config: 4-way split matches DWDP deepseek-v3 layout.
            (128, 512, 512, 8, 2, 4),
            (256, 512, 512, 8, 4, 4),
            # Larger expert count for dwdp_size=4.
            (128, 512, 512, 16, 2, 4),
            # 2-way split (reduced dwdp size).
            (128, 512, 512, 8, 2, 2),
        ],
    )
    def test_dwdp_style_multi_b_matches_single_b(
        self,
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        dwdp_size,
    ):
        """Replicate SGLang forward_dwdp: flat uint8 buf + as_strided view."""
        from flashinfer import cute_dsl_fused_moe_nvfp4

        t = _create_moe_tensors(
            num_tokens, hidden_size, intermediate_size, num_experts, top_k
        )

        # Baseline: single-tensor call.
        out_single = cute_dsl_fused_moe_nvfp4(
            x=t["x"],
            x_sf=t["x_sf"],
            token_selected_experts=t["token_selected_experts"],
            token_final_scales=t["token_final_scales"],
            w1_weight=t["w1_weight"],
            w1_weight_sf=t["w1_weight_sf"],
            w1_alpha=t["w1_alpha"],
            fc2_input_scale=t["fc2_input_scale"],
            w2_weight=t["w2_weight"],
            w2_weight_sf=t["w2_weight_sf"],
            w2_alpha=t["w2_alpha"],
            num_experts=num_experts,
            top_k=top_k,
        )
        assert not torch.isnan(out_single).any()

        # DWDP-style multi-B reconstruction.
        w1_list = _split_as_dwdp_prefetch(t["w1_weight"], dwdp_size)
        w1_sf_list = _split_as_dwdp_prefetch(t["w1_weight_sf"], dwdp_size)
        w1_alpha_list = _split_as_dwdp_prefetch(t["w1_alpha"], dwdp_size)
        w2_list = _split_as_dwdp_prefetch(t["w2_weight"], dwdp_size)
        w2_sf_list = _split_as_dwdp_prefetch(t["w2_weight_sf"], dwdp_size)
        w2_alpha_list = _split_as_dwdp_prefetch(t["w2_alpha"], dwdp_size)

        # Sanity: reconstructing single tensor from parts should be bit-exact.
        _check_close(
            _rebuild_single_from_list(w1_list, expert_dim=0),
            t["w1_weight"],
            "w1 reconstruction",
            percent=1.0,
        )
        _check_close(
            _rebuild_single_from_list(w1_sf_list, expert_dim=5),
            t["w1_weight_sf"],
            "w1_sf reconstruction",
            percent=1.0,
        )

        out_multi = cute_dsl_fused_moe_nvfp4(
            x=t["x"],
            x_sf=t["x_sf"],
            token_selected_experts=t["token_selected_experts"],
            token_final_scales=t["token_final_scales"],
            w1_weight=w1_list,
            w1_weight_sf=w1_sf_list,
            w1_alpha=w1_alpha_list,
            fc2_input_scale=t["fc2_input_scale"],
            w2_weight=w2_list,
            w2_weight_sf=w2_sf_list,
            w2_alpha=w2_alpha_list,
            num_experts=num_experts,
            top_k=top_k,
        )
        assert not torch.isnan(out_multi).any()

        _check_close(out_multi, out_single, "DWDP multi-B vs single-B")

    @pytest.mark.parametrize("num_iters", [5])
    def test_dwdp_style_multi_b_determinism(self, num_iters):
        """Same inputs across repeated calls must yield the same outputs.

        This exercises the exact path where DWDP non-determinism was observed.
        If the kernel is deterministic for fixed inputs, the bug is elsewhere
        (prefetch/stream/IPC in SGLang).
        """
        from flashinfer import cute_dsl_fused_moe_nvfp4

        num_tokens, hidden_size, intermediate_size = 128, 512, 512
        num_experts, top_k, dwdp_size = 8, 2, 4

        t = _create_moe_tensors(
            num_tokens, hidden_size, intermediate_size, num_experts, top_k
        )

        w1_list = _split_as_dwdp_prefetch(t["w1_weight"], dwdp_size)
        w1_sf_list = _split_as_dwdp_prefetch(t["w1_weight_sf"], dwdp_size)
        w1_alpha_list = _split_as_dwdp_prefetch(t["w1_alpha"], dwdp_size)
        w2_list = _split_as_dwdp_prefetch(t["w2_weight"], dwdp_size)
        w2_sf_list = _split_as_dwdp_prefetch(t["w2_weight_sf"], dwdp_size)
        w2_alpha_list = _split_as_dwdp_prefetch(t["w2_alpha"], dwdp_size)

        outs = []
        for _ in range(num_iters):
            out = cute_dsl_fused_moe_nvfp4(
                x=t["x"],
                x_sf=t["x_sf"],
                token_selected_experts=t["token_selected_experts"],
                token_final_scales=t["token_final_scales"],
                w1_weight=w1_list,
                w1_weight_sf=w1_sf_list,
                w1_alpha=w1_alpha_list,
                fc2_input_scale=t["fc2_input_scale"],
                w2_weight=w2_list,
                w2_weight_sf=w2_sf_list,
                w2_alpha=w2_alpha_list,
                num_experts=num_experts,
                top_k=top_k,
            )
            torch.cuda.synchronize()
            outs.append(out.clone())

        # FP4 atomic scatter-add in the finalize kernel can have small
        # non-determinism, but repeat calls with identical inputs should not
        # diverge significantly.
        for i in range(1, num_iters):
            max_diff = (outs[0] - outs[i]).abs().max().item()
            assert max_diff < 1e-3, (
                f"Repeated multi-B call diverged: iter0 vs iter{i}, "
                f"max|Δ|={max_diff:.4e}"
            )

    def test_dwdp_fresh_buffers_every_call(self):
        """Prefetch allocates NEW flat buffers each forward — test that
        reconstructing views from freshly-allocated buffers (same content)
        gives the same output as the baseline.

        This checks whether the kernel caches anything by pointer address
        (the multi-B path uses b_tensor_l_sizes as a cache key — sizes only,
        not pointers, so this should be fine; verifying here).
        """
        from flashinfer import cute_dsl_fused_moe_nvfp4

        num_tokens, hidden_size, intermediate_size = 128, 512, 512
        num_experts, top_k, dwdp_size = 8, 2, 4

        t = _create_moe_tensors(
            num_tokens, hidden_size, intermediate_size, num_experts, top_k
        )

        # First call: build views once.
        w1_list_a = _split_as_dwdp_prefetch(t["w1_weight"], dwdp_size)
        w1_sf_list_a = _split_as_dwdp_prefetch(t["w1_weight_sf"], dwdp_size)
        w1_alpha_list_a = _split_as_dwdp_prefetch(t["w1_alpha"], dwdp_size)
        w2_list_a = _split_as_dwdp_prefetch(t["w2_weight"], dwdp_size)
        w2_sf_list_a = _split_as_dwdp_prefetch(t["w2_weight_sf"], dwdp_size)
        w2_alpha_list_a = _split_as_dwdp_prefetch(t["w2_alpha"], dwdp_size)

        out_a = cute_dsl_fused_moe_nvfp4(
            x=t["x"], x_sf=t["x_sf"],
            token_selected_experts=t["token_selected_experts"],
            token_final_scales=t["token_final_scales"],
            w1_weight=w1_list_a, w1_weight_sf=w1_sf_list_a,
            w1_alpha=w1_alpha_list_a,
            fc2_input_scale=t["fc2_input_scale"],
            w2_weight=w2_list_a, w2_weight_sf=w2_sf_list_a,
            w2_alpha=w2_alpha_list_a,
            num_experts=num_experts, top_k=top_k,
        )
        torch.cuda.synchronize()

        # Second call: reallocate and refill buffers (NEW data_ptrs) with
        # identical contents.
        w1_list_b = _split_as_dwdp_prefetch(t["w1_weight"], dwdp_size)
        w1_sf_list_b = _split_as_dwdp_prefetch(t["w1_weight_sf"], dwdp_size)
        w1_alpha_list_b = _split_as_dwdp_prefetch(t["w1_alpha"], dwdp_size)
        w2_list_b = _split_as_dwdp_prefetch(t["w2_weight"], dwdp_size)
        w2_sf_list_b = _split_as_dwdp_prefetch(t["w2_weight_sf"], dwdp_size)
        w2_alpha_list_b = _split_as_dwdp_prefetch(t["w2_alpha"], dwdp_size)

        # Confirm the pointers actually changed.
        assert w1_list_a[0].data_ptr() != w1_list_b[0].data_ptr()
        assert w1_sf_list_a[0].data_ptr() != w1_sf_list_b[0].data_ptr()

        out_b = cute_dsl_fused_moe_nvfp4(
            x=t["x"], x_sf=t["x_sf"],
            token_selected_experts=t["token_selected_experts"],
            token_final_scales=t["token_final_scales"],
            w1_weight=w1_list_b, w1_weight_sf=w1_sf_list_b,
            w1_alpha=w1_alpha_list_b,
            fc2_input_scale=t["fc2_input_scale"],
            w2_weight=w2_list_b, w2_weight_sf=w2_sf_list_b,
            w2_alpha=w2_alpha_list_b,
            num_experts=num_experts, top_k=top_k,
        )
        torch.cuda.synchronize()

        max_diff = (out_a - out_b).abs().max().item()
        assert max_diff < 1e-3, (
            f"Fresh-pointer multi-B outputs diverged: max|Δ|={max_diff:.4e}. "
            f"Kernel may be caching by pointer instead of by b_tensor_l_sizes."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
