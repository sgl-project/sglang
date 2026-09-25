"""CP decode attention TP shards of FP8 Marlin linears (CPU, repack mocked)."""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from sglang.srt.layers.cp.cp_decode_attn_tp import (
    CpDecodeAttnTpContext,
    CpDecodeAttnTpShard,
)
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization import marlin_utils_fp8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _prepare(layer, size_k_first):
    """Run prepare_fp8_layer_for_marlin with an identity stand-in for the CUDA
    repack; returns the (size_k, size_n) of each repack call, in call order."""
    calls = []

    def fake_repack(b_q_weight, perm, size_k, size_n, num_bits):
        calls.append((size_k, size_n))
        return b_q_weight.clone()

    with (
        patch.object(
            marlin_utils_fp8,
            "marlin_make_workspace",
            return_value=torch.empty(0, dtype=torch.int32),
        ),
        patch.object(marlin_utils_fp8, "gptq_marlin_repack", fake_repack, create=True),
    ):
        marlin_utils_fp8.prepare_fp8_layer_for_marlin(layer, size_k_first)
    return calls


def _make_layer(n, k, size_k_first, block, scale_numel=None, shard=None):
    layer = torch.nn.Module()
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.orig_dtype = torch.bfloat16
    shape = (k, n) if size_k_first else (n, k)
    layer.weight = (torch.randn(shape) * 0.5).to(torch.float8_e4m3fn)
    if block is not None:
        layer.weight_block_size = block
        scale_shape = (n // block[0], k // block[1])
        if size_k_first:
            scale_shape = scale_shape[::-1]
        layer.weight_scale_inv = torch.rand(scale_shape) + 0.5
    else:
        layer.weight_scale = torch.rand(scale_numel) + 0.5
    if shard is not None:
        layer.cp_decode_attn_tp_shard = shard
    return layer


def _logical_slice_layer(full, split, rank, size, size_k_first, block):
    """Build a layer holding the shard sliced from the logical weight."""
    n = full.output_size_per_partition
    k = full.input_size_per_partition
    n_dim, k_dim = (1, 0) if size_k_first else (0, 1)
    dim = n_dim if split == "output" else k_dim
    if split == "output":
        n //= size
    else:
        k //= size
    layer = torch.nn.Module()
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.orig_dtype = full.orig_dtype
    layer.weight = full.weight.chunk(size, dim)[rank]
    if block is not None:
        layer.weight_block_size = block
        layer.weight_scale_inv = full.weight_scale_inv.chunk(size, dim)[rank]
    elif split == "output" and full.weight_scale.numel() > 1:
        layer.weight_scale = full.weight_scale.chunk(size)[rank]
    else:
        layer.weight_scale = full.weight_scale
    return layer


class TestFp8MarlinDecodeShard(CustomTestCase):
    SIZE = 4

    def _check(self, split, size_k_first, block, scale_numel=None):
        n, k = 512, 512
        for rank in range(self.SIZE):
            full = _make_layer(
                n,
                k,
                size_k_first,
                block,
                scale_numel,
                shard=CpDecodeAttnTpShard(split=split, rank=rank, size=self.SIZE),
            )
            ref = _logical_slice_layer(
                full, split, rank, self.SIZE, size_k_first, block
            )
            full_calls = _prepare(full, size_k_first)
            (ref_call,) = _prepare(ref, size_k_first)

            # The shard is packed first, then the full (prefill) weight.
            self.assertEqual(full_calls, [ref_call, (k, n)])
            views = full.cp_decode_attn_tp_packed_views
            self.assertEqual(set(views), {"weight", "weight_scale"})
            self.assertTrue(torch.equal(views["weight"], ref.weight))
            torch.testing.assert_close(views["weight_scale"], ref.weight_scale)

    def test_block_quant_output_split(self):
        self._check("output", size_k_first=False, block=[128, 128])

    def test_block_quant_input_split(self):
        self._check("input", size_k_first=False, block=[128, 128])

    def test_channelwise_output_split(self):
        self._check("output", size_k_first=True, block=None, scale_numel=512)

    def test_channelwise_input_split(self):
        self._check("input", size_k_first=True, block=None, scale_numel=512)

    def test_per_tensor_output_split(self):
        self._check("output", size_k_first=True, block=None, scale_numel=1)

    def test_no_shard_keeps_behavior(self):
        layer = _make_layer(256, 256, False, [128, 128])
        _prepare(layer, False)
        self.assertFalse(hasattr(layer, "cp_decode_attn_tp_packed_views"))

    def test_rejects_split_quant_block(self):
        # 256 output rows over 4 ranks = 64-row shards, splitting 128-row blocks.
        layer = _make_layer(
            256,
            256,
            False,
            [128, 128],
            shard=CpDecodeAttnTpShard(split="output", rank=0, size=4),
        )
        with self.assertRaisesRegex(AssertionError, "quantization block"):
            _prepare(layer, False)


def _bare_linear(cls, n, k):
    linear = cls.__new__(cls)
    torch.nn.Module.__init__(linear)
    linear.output_size_per_partition = n
    linear.input_size_per_partition = k
    linear.use_decode_attn_tp = False
    linear.weight = torch.nn.Parameter(torch.zeros(3, 5), requires_grad=False)
    linear.weight_scale = torch.nn.Parameter(torch.ones(2, 5), requires_grad=False)
    return linear


def _bare_ctx(rank, size):
    ctx = CpDecodeAttnTpContext.__new__(CpDecodeAttnTpContext)
    ctx.decode_tp_rank = rank
    ctx.decode_tp_size = size
    ctx.use_decode_attn_tp = False
    ctx._slice_cache = {}
    return ctx


@contextmanager
def _decode_attn_tp(ctx, modules):
    def _set(_):
        ctx.use_decode_attn_tp = True

    with patch.object(ctx, "set_decode_attn_tp", _set):
        with ctx.maybe_use_decode_attn_tp(None, modules):
            yield


class TestCpDecodeAttnTpPackedViews(CustomTestCase):
    def test_register_linears(self):
        ctx = _bare_ctx(rank=2, size=4)
        col = _bare_linear(ColumnParallelLinear, 8, 8)
        row = _bare_linear(RowParallelLinear, 8, 8)
        ctx.register_linears([col, row, torch.nn.Linear(2, 2)])
        self.assertEqual(
            col.cp_decode_attn_tp_shard, CpDecodeAttnTpShard("output", 2, 4)
        )
        self.assertEqual(
            row.cp_decode_attn_tp_shard, CpDecodeAttnTpShard("input", 2, 4)
        )

        disabled = _bare_ctx(rank=None, size=None)
        other = _bare_linear(ColumnParallelLinear, 8, 8)
        disabled.register_linears([other])
        self.assertFalse(hasattr(other, "cp_decode_attn_tp_shard"))

    def test_swaps_packed_views_and_restores(self):
        ctx = _bare_ctx(rank=1, size=4)
        col = _bare_linear(ColumnParallelLinear, 64, 32)
        row = _bare_linear(RowParallelLinear, 32, 64)
        col_views = {
            "weight": torch.full((7,), 1.0),
            "weight_scale": torch.full((2,), 2.0),
        }
        row_views = {
            "weight": torch.full((9,), 3.0),
            "weight_scale": torch.full((4,), 4.0),
        }
        col.cp_decode_attn_tp_packed_views = col_views
        row.cp_decode_attn_tp_packed_views = row_views
        col_w, row_w = col.weight.data, row.weight.data

        for _ in range(2):  # the second pass goes through the cache
            with _decode_attn_tp(ctx, [col, row]):
                self.assertEqual(col.weight.data_ptr(), col_views["weight"].data_ptr())
                self.assertEqual(
                    col.weight_scale.data_ptr(), col_views["weight_scale"].data_ptr()
                )
                self.assertEqual(row.weight.data_ptr(), row_views["weight"].data_ptr())
                self.assertEqual(
                    row.weight_scale.data_ptr(), row_views["weight_scale"].data_ptr()
                )
                self.assertEqual(col.output_size_per_partition, 16)
                self.assertEqual(row.input_size_per_partition, 16)
                self.assertTrue(row.use_decode_attn_tp)
            self.assertEqual(col.weight.data_ptr(), col_w.data_ptr())
            self.assertEqual(row.weight.data_ptr(), row_w.data_ptr())
            self.assertEqual(col.output_size_per_partition, 64)
            self.assertEqual(row.input_size_per_partition, 64)
            self.assertFalse(row.use_decode_attn_tp)

    def test_unregistered_marlin_linear_fails_loudly(self):
        ctx = _bare_ctx(rank=0, size=4)
        col = _bare_linear(ColumnParallelLinear, 64, 32)
        col.quant_method = type("M", (), {"use_marlin": True})()
        with self.assertRaisesRegex(AssertionError, "register_linears"):
            with _decode_attn_tp(ctx, [col]):
                pass


if __name__ == "__main__":
    unittest.main(verbosity=3)
