"""Cake MiniMax-H3 QKV quantize-and-pack through sglang.kernels.

Checks: the registry resolves the explicit FlashInfer backend for the prepared
and one-shot entries; the facade results are bitwise identical to calling
FlashInfer directly; and the fused one-pass pack is **bitwise** identical to
the segmented torch + FlashInfer quantizer chain (destination-major copy, then
``flashinfer.fp4_quantize`` with the static global scale or
``flashinfer.mxfp8_quantize`` per destination), which is the contract of the
kernel. Skips when FlashInfer lacks the module or the GPU is not sm_100a /
sm_103a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import diffusion_minimax_h3_pre_attention as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.diffusion.cake import (
    cake_minimax_h3_qkv_quantize_pack,
    cake_prepare_minimax_h3_qkv_quantize_pack,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=400, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

NUM_HEADS, HEAD_DIM, KINDS = 56, 128, 3


@pytest.mark.parametrize(
    "op",
    [
        "diffusion.prepare_minimax_h3_qkv_quantize_pack",
        "diffusion.minimax_h3_qkv_quantize_pack",
    ],
)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith(
        "sglang.kernels.cake_kernels.diffusion_minimax_h3_pre_attention:"
    )


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake.FI_QKV_PACK_MODULE, cake.FI_QKV_PACK_JIT_MODULE
    ):
        pytest.skip("installed FlashInfer lacks cake_minimax_h3_qkv_pack")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(f"Cake MiniMax-H3 QKV pack needs sm_100a/103a, device is {cc}")


def _round_up(v, a):
    return -(-v // a) * a


def _scale_cols(fmt):
    return HEAD_DIM // 16 if fmt == "nvfp4" else HEAD_DIM // 32


def _sources(m, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    qkv = torch.empty(
        (m, NUM_HEADS, KINDS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    qkv.normal_(0.0, 1.0, generator=g)
    qkv[::7, :, 0, :] = 0  # exact-zero blocks (zero scale byte, zero codes)
    return [qkv[:, :, kind, :] for kind in range(KINDS)]  # strided kind slices


def _global_scale(sources):
    amax = torch.stack([s.float().abs().amax() for s in sources]).amax()
    return ((448.0 * 6.0) / amax).reshape(1).float()


def _destination_major(sources, p):
    m = sources[0].shape[0]
    stacked = torch.stack(tuple(sources), dim=2)
    return (
        stacked.view(m, p, NUM_HEADS // p, KINDS, HEAD_DIM)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )


def _swizzled_padding_mask(rows, cols, device):
    padded_rows, padded_cols = _round_up(rows, 128), _round_up(cols, 4)
    index = torch.arange(padded_rows * padded_cols, dtype=torch.int64, device=device)
    tile, within = index // (128 * padded_cols), index % (128 * padded_cols)
    col_group, rem = within // 512, within % 512
    row_in_32, rem = rem // 16, rem % 16
    row_group, col = rem // 4, col_group * 4 + rem % 4
    row = tile * 128 + row_group * 32 + row_in_32
    return (row >= rows) | (col >= cols)


def _segmented_reference(sources, p, fmt, global_scale):
    import flashinfer

    destination = _destination_major(sources, p)
    rows = destination.shape[1] * destination.shape[2] * destination.shape[3]
    padding = _swizzled_padding_mask(rows, _scale_cols(fmt), destination.device)
    values, scales = [], []
    for shard in destination:
        flat = shard.reshape(rows, HEAD_DIM)
        if fmt == "nvfp4":
            x_q, sf = flashinfer.fp4_quantize(
                flat,
                global_scale,
                sf_vec_size=16,
                sf_use_ue8m0=False,
                is_sf_swizzled_layout=True,
            )
            values.append(
                x_q.view(torch.uint8).reshape(*shard.shape[:-1], HEAD_DIM // 2)
            )
        else:
            x_q, sf = flashinfer.mxfp8_quantize(flat, is_sf_swizzled_layout=True)
            values.append(
                x_q.view(torch.float8_e4m3fn).reshape(*shard.shape[:-1], HEAD_DIM)
            )
        sf = sf.reshape(-1).view(torch.uint8).clone()
        sf[padding] = 0
        scales.append(sf)
    return torch.stack(values), torch.stack(scales)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("m,p", [(129, 8), (64, 4)])
def test_matches_flashinfer_and_segmented_chain(m, p, fmt):
    _skip_unless_supported()
    device = torch.device("cuda")
    q, k, v = _sources(m, device, seed=6090 + m + p)
    shapes = cake.minimax_h3_qkv_pack_output_shapes(m, p, fmt)
    out_q = torch.empty(*shapes["out_q"][0], dtype=shapes["out_q"][1], device=device)
    out_sf = torch.empty(*shapes["out_sf"][0], dtype=shapes["out_sf"][1], device=device)
    gs = _global_scale((q, k, v)) if fmt == "nvfp4" else None
    assert cake.supports_prepare_minimax_h3_qkv_quantize_pack(
        q=q, k=k, v=v, out_q=out_q, out_sf=out_sf, P=p, format=fmt, out_global_scale=gs
    )
    runner = cake_prepare_minimax_h3_qkv_quantize_pack(
        q=q, k=k, v=v, out_q=out_q, out_sf=out_sf, P=p, format=fmt, out_global_scale=gs
    )
    got_q, got_sf = runner()
    torch.cuda.synchronize()
    assert got_q is out_q and got_sf is out_sf

    expected_q, expected_sf = _segmented_reference((q, k, v), p, fmt, gs)
    assert torch.equal(out_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(out_sf, expected_sf)

    # Direct FlashInfer prepared runner: bitwise identical to the facade.
    from flashinfer.diffusion_ops.cake_minimax_h3_qkv_pack import (
        prepare_minimax_h3_qkv_quantize_pack as fi_prepare,
    )

    fi_q, fi_sf = torch.empty_like(out_q), torch.empty_like(out_sf)
    fi_prepare(
        q=q, k=k, v=v, out_q=fi_q, out_sf=fi_sf, P=p, format=fmt, out_global_scale=gs
    )()
    torch.cuda.synchronize()
    assert torch.equal(out_q.view(torch.uint8), fi_q.view(torch.uint8))
    assert torch.equal(out_sf, fi_sf)

    # One-shot facade entry (allocating form) agrees bitwise as well.
    assert cake.supports_minimax_h3_qkv_quantize_pack(q, k, v, p, fmt, gs)
    one_q, one_sf = cake_minimax_h3_qkv_quantize_pack(q, k, v, p, fmt, gs)
    torch.cuda.synchronize()
    assert one_q.shape == out_q.shape and one_q.dtype == out_q.dtype
    assert torch.equal(one_q.view(torch.uint8), out_q.view(torch.uint8))
    assert torch.equal(one_sf, out_sf)


def test_prepared_runner_is_graph_capturable():
    _skip_unless_supported()
    device = torch.device("cuda")
    m, p, fmt = 129, 8, "mxfp8"
    q, k, v = _sources(m, device, seed=11)
    shapes = cake.minimax_h3_qkv_pack_output_shapes(m, p, fmt)
    out_q = torch.empty(*shapes["out_q"][0], dtype=shapes["out_q"][1], device=device)
    out_sf = torch.empty(*shapes["out_sf"][0], dtype=shapes["out_sf"][1], device=device)
    runner = cake_prepare_minimax_h3_qkv_quantize_pack(
        q=q, k=k, v=v, out_q=out_q, out_sf=out_sf, P=p, format=fmt
    )
    runner()
    torch.cuda.synchronize()
    eager_q, eager_sf = out_q.clone(), out_sf.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        runner()
    torch.cuda.current_stream().wait_stream(stream)
    out_q.view(torch.uint8).fill_(255)
    out_sf.fill_(255)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out_q.view(torch.uint8), eager_q.view(torch.uint8))
    assert torch.equal(out_sf, eager_sf)


def test_admission_rejects_out_of_contract():
    _skip_unless_supported()
    device = torch.device("cuda")
    q, k, v = _sources(16, device, seed=3)
    shapes = cake.minimax_h3_qkv_pack_output_shapes(16, 8, "mxfp8")
    out_q = torch.empty(*shapes["out_q"][0], dtype=shapes["out_q"][1], device=device)
    out_sf = torch.empty(*shapes["out_sf"][0], dtype=shapes["out_sf"][1], device=device)
    ok = dict(q=q, k=k, v=v, out_q=out_q, out_sf=out_sf, P=8, format="mxfp8")
    assert cake.supports_prepare_minimax_h3_qkv_quantize_pack(**ok)
    assert not cake.supports_prepare_minimax_h3_qkv_quantize_pack(**dict(ok, P=3))
    assert not cake.supports_prepare_minimax_h3_qkv_quantize_pack(
        **dict(ok, format="fp8")
    )
    # nvfp4 needs a global scale; mxfp8 must not get one.
    assert not cake.supports_prepare_minimax_h3_qkv_quantize_pack(
        **dict(ok, out_global_scale=torch.ones(1, device=device))
    )
    assert not cake.supports_minimax_h3_qkv_quantize_pack(q, k, v, 8, "nvfp4")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
