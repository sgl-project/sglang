"""CPU checks for CP decode attention TP: which linear attributes are sliced, the
128x4-swizzled MXFP8 scale, and the Hopper BF16 weight cache."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _ctx(rank: int, size: int) -> CpDecodeAttnTpContext:
    ctx = CpDecodeAttnTpContext.__new__(CpDecodeAttnTpContext)
    ctx.decode_tp_rank = rank
    ctx.decode_tp_size = size
    ctx.use_decode_attn_tp = False
    ctx._slice_cache = {}
    return ctx


def _linear(cls, n: int, k: int):
    """A linear with the attributes decode attention TP reads, no process group."""
    linear = cls.__new__(cls)
    torch.nn.Module.__init__(linear)
    linear.weight = torch.nn.Parameter(
        torch.randn(n, k).to(torch.float8_e4m3fn), requires_grad=False
    )
    linear.weight_scale_inv = torch.nn.Parameter(
        torch.rand(n // 32, k // 32), requires_grad=False
    )
    linear.input_size_per_partition = k
    linear.output_size_per_partition = n
    linear.use_decode_attn_tp = False
    return linear


def _swizzle_ref(scale: torch.Tensor) -> torch.Tensor:
    """FlashInfer block_scale_interleave layout: 128x4 tiles, 32x4x4 inside each."""
    rows, cols = scale.shape
    pr, pc = -(-rows // 128) * 128, -(-cols // 4) * 4
    padded = scale.new_zeros(pr, pc)
    padded[:rows, :cols] = scale
    tiles = padded.reshape(pr // 128, 4, 32, pc // 4, 4).permute(0, 3, 2, 1, 4)
    return tiles.reshape(-1).contiguous()


def _narrow(t: torch.Tensor, dim: int, rank: int, size: int) -> torch.Tensor:
    chunk = t.shape[dim] // size
    return t.narrow(dim, rank * chunk, chunk)


@contextmanager
def _enter(ctx: CpDecodeAttnTpContext, modules):
    def force_decode(forward_batch):
        ctx.use_decode_attn_tp = True

    with patch.object(ctx, "set_decode_attn_tp", side_effect=force_decode):
        with ctx.maybe_use_decode_attn_tp(SimpleNamespace(), modules):
            yield


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.data_ptr() == b.data_ptr() and a.shape == b.shape


@pytest.mark.parametrize("cls", [ColumnParallelLinear, RowParallelLinear])
def test_plain_linear_slices_weight_and_scale_only(cls):
    # The FP8 linear every CP model uses: no derived caches, no swizzled scale.
    # Behaves the same before and after the derived-copy slicing was added.
    ctx = _ctx(rank=1, size=4)
    linear = _linear(cls, 256, 512)
    dim = 1 if cls is RowParallelLinear else 0
    attrs = ctx._get_linear_attrs(linear)
    assert [(a[1], a[2]) for a in attrs] == [
        ("weight", dim),
        ("weight_scale_inv", dim),
    ]

    weight, scale = linear.weight, linear.weight_scale_inv
    weight_data, scale_data = weight.data, scale.data
    with _enter(ctx, [linear]):
        assert linear.weight is weight
        assert torch.equal(
            linear.weight.data.float(), _narrow(weight_data, dim, 1, 4).float()
        )
        assert torch.equal(linear.weight_scale_inv.data, _narrow(scale_data, dim, 1, 4))
    assert _same(linear.weight.data, weight_data)
    assert _same(linear.weight_scale_inv.data, scale_data)
    assert linear.input_size_per_partition == 512
    assert linear.output_size_per_partition == 256
    assert linear.use_decode_attn_tp is False


@pytest.mark.parametrize("cls", [ColumnParallelLinear, RowParallelLinear])
def test_hopper_bf16_weight_cache_is_sliced_and_restored(cls):
    ctx = _ctx(rank=2, size=4)
    linear = _linear(cls, 256, 512)
    cache = torch.randn(256, 512, dtype=torch.bfloat16)
    linear.register_buffer("_block_fp8_bf16_weight", cache, persistent=False)
    dim = 1 if cls is RowParallelLinear else 0

    with _enter(ctx, [linear]):
        first = linear._block_fp8_bf16_weight
        assert torch.equal(first, _narrow(cache, dim, 2, 4))
        size_attr = "input" if cls is RowParallelLinear else "output"
        assert first.shape[dim] == getattr(linear, f"{size_attr}_size_per_partition")
    assert linear._block_fp8_bf16_weight is cache
    # The shard is cached so captured CUDA graphs keep reading the same storage.
    with _enter(ctx, [linear]):
        assert linear._block_fp8_bf16_weight is first
    assert linear._block_fp8_bf16_weight is cache


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("dim,n,k", [(0, 1024, 512), (0, 1024, 96), (1, 256, 1024)])
def test_swizzled_scale_slice_matches_swizzled_shard(size, dim, n, k):
    # Slicing the swizzled scale must equal swizzling the logical scale's shard.
    scale = torch.randint(0, 256, (n, k // 32), dtype=torch.uint8)
    swizzled = _swizzle_ref(scale)
    for rank in range(size):
        ctx = _ctx(rank=rank, size=size)
        got = ctx._slice_sf_swizzled_scale(swizzled, dim, torch.Size((n, k)))
        assert torch.equal(got, _swizzle_ref(_narrow(scale, dim, rank, size)))


@pytest.mark.parametrize("cls", [ColumnParallelLinear, RowParallelLinear])
def test_swizzled_scale_is_activated_with_the_weight_shape(cls):
    ctx = _ctx(rank=3, size=4)
    n, k = 1024, 2048
    linear = _linear(cls, n, k)
    scale = torch.randint(0, 256, (n, k // 32), dtype=torch.uint8)
    swizzled = torch.nn.Parameter(_swizzle_ref(scale), requires_grad=False)
    linear.weight_scale_inv_swizzled = swizzled
    swizzled_data = swizzled.data
    dim = 1 if cls is RowParallelLinear else 0

    attrs = ctx._get_linear_attrs(linear)
    assert attrs[-1][1:] == ("weight_scale_inv_swizzled", dim, torch.Size((n, k)))
    with _enter(ctx, [linear]):
        assert linear.weight_scale_inv_swizzled is swizzled
        assert torch.equal(
            linear.weight_scale_inv_swizzled.data,
            _swizzle_ref(_narrow(scale, dim, 3, 4)),
        )
    assert _same(linear.weight_scale_inv_swizzled.data, swizzled_data)


def test_swizzled_scale_rejects_partial_tiles():
    ctx = _ctx(rank=0, size=4)
    # 256 rows split 4 ways leaves 64 rows per rank, half an SF tile.
    scale = torch.zeros(256, 16, dtype=torch.uint8)
    with pytest.raises(AssertionError, match="cannot be split"):
        ctx._slice_sf_swizzled_scale(_swizzle_ref(scale), 0, torch.Size((256, 512)))
    with pytest.raises(AssertionError, match="does not match"):
        ctx._slice_sf_swizzled_scale(
            torch.zeros(512, dtype=torch.uint8), 0, torch.Size((512, 512))
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("n,k", [(1024, 512), (512, 96)])
def test_swizzle_ref_matches_flashinfer(n, k):
    flashinfer = pytest.importorskip("flashinfer")
    scale = torch.randint(0, 256, (n, k // 32), dtype=torch.uint8, device="cuda")
    got = flashinfer.block_scale_interleave(scale.contiguous())
    assert torch.equal(got.reshape(-1).cpu(), _swizzle_ref(scale.cpu()))
