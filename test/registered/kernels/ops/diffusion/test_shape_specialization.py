"""Per-request sizes must not select a new Triton specialization.

Triton compiles one variant per class of each specialized integer argument
(divisible by 16, equal to 1, other). Sequence lengths and row counts change
with the resolution, the prompt and the input image, so a class the server's
warmup did not hit compiles on the first request that does: 0.3-0.5 s per
kernel on a cold cache. These kernels use such sizes only for row indices and
row masks, so they take them unspecialized, at no cost to vectorization.
The Wan VAE layout kernels keep their channel count and element total
specialized, which stay in one class at every resolution with channels a
multiple of 16. The strides of a channels-last tensor are multiples of the
channels too; those of a channels-first one change with the resolution, so
they go unspecialized, with the vector width passed as a layout constexpr.
"""

import sys
from contextlib import contextmanager

import pytest
import torch
import triton

from sglang.kernels.kda_kernels.layernorm_modulate_triton import (
    fused_layernorm_modulate_raw,
    fused_qk_head_layernorm,
)
from sglang.kernels.ops.diffusion import (
    fuse_layernorm_scale_shift_gate_select01_kernel,
    fuse_residual_layernorm_scale_shift_gate_select01_kernel,
    fuse_scale_shift_kernel,
)
from sglang.kernels.ops.diffusion.layout.ulysses_qkv_triton import (
    pack_qkv_destination_major,
)
from sglang.kernels.ops.diffusion.layout.wan_causal_cache_triton import (
    cat_pad_channels_last_3d,
    dup_up3d_add,
)
from sglang.kernels.ops.diffusion.modulate.wan_temb_table_slices_triton import (
    fused_temb_table_slices,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
HIDDEN = 512


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if getattr(triton, "knobs", None) is None:
        pytest.skip("needs triton.knobs compile hooks")
    torch.cuda.manual_seed(0)


@contextmanager
def _compiled_kernels():
    compiled = []
    previous = triton.knobs.runtime.jit_post_compile_hook

    def record(*, repr, **kwargs):
        compiled.append(repr)
        if previous is not None:
            return previous(repr=repr, **kwargs)
        return None

    triton.knobs.runtime.jit_post_compile_hook = record
    try:
        yield compiled
    finally:
        triton.knobs.runtime.jit_post_compile_hook = previous


def _bf16(*shape):
    return torch.randn(*shape, device=DEVICE, dtype=torch.bfloat16)


def _select01(seq_len, *, residual):
    x = _bf16(1, seq_len, HIDDEN)
    mods = {
        name: _bf16(1, HIDDEN)
        for name in ("scale0", "shift0", "gate0", "scale1", "shift1", "gate1")
    }
    index = torch.randint(0, 2, (1, seq_len), device=DEVICE, dtype=torch.int32)
    common = dict(
        weight=_bf16(HIDDEN), bias=_bf16(HIDDEN), index=index, eps=1e-6, **mods
    )
    if residual:
        fuse_residual_layernorm_scale_shift_gate_select01_kernel(
            x, residual=torch.randn_like(x), residual_gate=torch.randn_like(x), **common
        )
    else:
        fuse_layernorm_scale_shift_gate_select01_kernel(x, **common)


def _scale_shift(seq_len, *, batch=1):
    x = _bf16(batch, seq_len, HIDDEN)
    fuse_scale_shift_kernel(x, _bf16(batch, seq_len, HIDDEN), _bf16(batch, 1, HIDDEN))


def _scale_shift_4d(seq_len):
    x = _bf16(1, seq_len, HIDDEN)
    fuse_scale_shift_kernel(x, _bf16(1, 1, 1, HIDDEN), torch.randn_like(x))


def _layernorm_modulate(seq_len):
    fused_layernorm_modulate_raw(
        _bf16(1, seq_len, HIDDEN), _bf16(1, HIDDEN), _bf16(1, HIDDEN), 1e-6
    )


def _qk_head_layernorm(seq_len):
    q = _bf16(1, seq_len, 3, 128)
    fused_qk_head_layernorm(q, torch.randn_like(q), 1e-6)


def _temb_table_slices(seq_len):
    temb = _bf16(1, seq_len, 6, HIDDEN)
    table = torch.randn(1, 6, HIDDEN, device=DEVICE, dtype=torch.float32)
    fused_temb_table_slices(table, temb)


def _pack_qkv(seq_len):
    q = _bf16(seq_len, 4, 128)
    pack_qkv_destination_major(q, torch.randn_like(q), torch.randn_like(q), 2)


def _cl3d(*shape):
    return _bf16(*shape).contiguous(memory_format=torch.channels_last_3d)


def _cf3d(b, c, t, h, w):
    # the frames-major view WanResample hands the next block
    return _bf16(b, t, c, h, w).permute(0, 2, 1, 3, 4)


# One-frame chunks are real (the Wan VAE decodes frame by frame), one-pixel
# frames are not: their size-1 dims get unit strides, and the strides stay
# specialized on purpose.
def _wan_cat_pad(n):
    # a Wan causal conv input: n frames, no cache, k3 padding
    hw = max(n, 2)
    cat_pad_channels_last_3d(_cl3d(1, 96, n, hw, hw), None, (1, 1, 1, 1, 2, 0))


def _wan_dup_up3d(n):
    # WanResample 2x upsample: frames and height vary, width fixed
    hw = max(n, 2)
    src = _cl3d(1, 128, n, hw, 16)
    main = _cl3d(1, 64, 2 * n, 2 * hw, 32)
    dup_up3d_add(main, src, 2, 2, 4, False)


def _wan_cat_pad_channels_first(n):
    # a first-chunk conv input behind a channels-first upsample
    hw = max(n, 2)
    cat_pad_channels_last_3d(_cf3d(1, 96, 1, hw, hw), None, (1, 1, 1, 1, 2, 0))


def _wan_dup_up3d_layouts(n, main_layout, src_layout):
    # even source widths keep the upsampled rows a multiple of 4 wide, as
    # they are at every Wan2.2 resolution
    hw = n + n % 2
    make = {"cl": _cl3d, "cf": _cf3d}
    src = make[src_layout](1, 128, 1, hw, hw)
    main = make[main_layout](1, 128, 2, 2 * hw, 2 * hw)
    dup_up3d_add(main, src, 2, 2, 8, False)


LAUNCHES = {
    "select01": lambda n: _select01(n, residual=False),
    "residual_select01": lambda n: _select01(n, residual=True),
    "scale_shift": _scale_shift,
    "scale_shift_4d": _scale_shift_4d,
    "layernorm_modulate": _layernorm_modulate,
    "qk_head_layernorm": _qk_head_layernorm,
    "temb_table_slices": _temb_table_slices,
    "pack_qkv": _pack_qkv,
    "wan_cat_pad": _wan_cat_pad,
    "wan_cat_pad_channels_first": _wan_cat_pad_channels_first,
    "wan_dup_up3d": _wan_dup_up3d,
    "wan_dup_up3d_cf_cf": lambda n: _wan_dup_up3d_layouts(n, "cf", "cf"),
    "wan_dup_up3d_cf_cl": lambda n: _wan_dup_up3d_layouts(n, "cf", "cl"),
    "wan_dup_up3d_cl_cf": lambda n: _wan_dup_up3d_layouts(n, "cl", "cf"),
}


@pytest.mark.parametrize("name", LAUNCHES)
def test_a_new_length_class_compiles_nothing(name):
    launch = LAUNCHES[name]
    launch(64)
    torch.cuda.synchronize()
    with _compiled_kernels() as compiled:
        # not divisible by 16, then equal to 1: both are classes of their own
        launch(57)
        launch(1)
        torch.cuda.synchronize()
    assert compiled == []


def test_scale_shift_batch_size_compiles_nothing():
    _scale_shift(64, batch=1)
    torch.cuda.synchronize()
    with _compiled_kernels() as compiled:
        _scale_shift(64, batch=2)
        torch.cuda.synchronize()
    assert compiled == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
