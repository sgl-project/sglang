"""--dcp-replicate-q-proj with a 128x128 block-FP8 q_b_proj.

ModelRunner._gather_block_fp8_linear rebuilds the full-head q_b_proj from each
DCP rank's row shard: it recovers the logical fp32 block scales (undoing the
DeepGEMM UE8M0 packing), gathers weight and scales along the output dim, and
re-runs the layer's own post-load step so the replica runs the same GEMM. The
check: the replica matches the unsharded layer bit for bit on every FP8 dense
backend of the device, and matches the concatenation of the per-shard outputs.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
from sglang.srt.model_executor.model_runner import (
    _gather_block_fp8_linear,
    _is_block_fp8,
)
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import (
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-small")

FP8_MAX = 448.0
DCP_SIZE = 4
# Full-head (N, K): per-rank shard N / DCP_SIZE must stay a multiple of the
# 128-row block, and K a multiple of 128 for DeepGEMM.
N, K = 1024, 512
BATCH_SIZES = [1, 8, 48]


def _fp8_block_backends():
    sm = get_device_sm()
    if 100 <= sm < 110:
        return ["triton", "deep_gemm", "flashinfer_trtllm", "flashinfer_cutlass"]
    if sm >= 120:
        return ["triton", "cutlass"]
    if sm == 90:
        return ["triton", "deep_gemm", "flashinfer_deepgemm"]
    return []


def _quantize_fp8_blockwise(w: torch.Tensor, block: int = 128):
    """Checkpoint-format (w_fp8 [N, K], scale_inv fp32 [N/block, K/block])."""
    n, k = w.shape
    tiles = w.float().reshape(n // block, block, k // block, block)
    amax = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = amax / FP8_MAX
    w_fp8 = (tiles / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return w_fp8.reshape(n, k), scale


def _block_fp8_layer(n: int, k: int, w_fp8: torch.Tensor, scale_inv: torch.Tensor):
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    layer = make_tp1_column_parallel_linear(
        quant_config, n, k, skip_block_quant_check=True
    )
    load_linear_weights(layer, weight=w_fp8, weight_scale_inv=scale_inv)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


class _FakeDcpGroup:
    """all_gather over the row shards' post-load state, as the DCP group would.

    The helper gathers the fp8 weight and the recovered fp32 block scales; each
    shard contributes the same two tensors, recovered the same way.
    """

    def __init__(self, shards):
        self.shards = shards
        self.world_size = len(shards)

    def all_gather(self, t: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0
        if t.dtype == torch.float8_e4m3fn:
            parts = [s.weight.data for s in self.shards]
        else:
            parts = [self._logical_scale(s) for s in self.shards]
        return torch.cat(parts, dim=0)

    @staticmethod
    def _logical_scale(layer) -> torch.Tensor:
        scale = layer.weight_scale_inv.data
        if scale.dtype == torch.int32:
            scale = fp8_utils.inverse_transform_scale_ue8m0(
                scale, mn=layer.weight.shape[0]
            )
        return scale


@unittest.skipIf(get_device_sm() < 90, "FP8 GEMM backends require SM90+")
class TestDcpReplicateQProjBlockFp8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def _check_backend(self, backend: str):
        if backend not in _fp8_block_backends():
            self.skipTest(f"{backend} not in SM{get_device_sm()} backend set")
        torch.manual_seed(7)
        with mock.patch.object(
            fp8_utils, "FP8_GEMM_RUNNER_BACKEND", Fp8GemmRunnerBackend(backend)
        ):
            w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) / 10
            w_fp8, scale_inv = _quantize_fp8_blockwise(w)
            full = _block_fp8_layer(N, K, w_fp8, scale_inv)

            rows, srows = N // DCP_SIZE, (N // 128) // DCP_SIZE
            shards = [
                _block_fp8_layer(
                    rows,
                    K,
                    w_fp8[r * rows : (r + 1) * rows],
                    scale_inv[r * srows : (r + 1) * srows],
                )
                for r in range(DCP_SIZE)
            ]
            self.assertTrue(_is_block_fp8(shards[0]))

            replica = _gather_block_fp8_linear(shards[0], _FakeDcpGroup(shards))

            # Same post-load format as an unsharded layer: scale dtype (int32
            # UE8M0-packed on DeepGEMM, fp32 otherwise) and the MXFP8 view flag.
            self.assertEqual(tuple(replica.weight.shape), (N, K))
            self.assertEqual(
                replica.weight_scale_inv.dtype, full.weight_scale_inv.dtype
            )
            self.assertEqual(
                tuple(replica.weight_scale_inv.shape),
                tuple(full.weight_scale_inv.shape),
            )
            self.assertEqual(
                getattr(replica, "block_fp8_mxfp8_ready", None),
                getattr(full, "block_fp8_mxfp8_ready", None),
            )
            torch.testing.assert_close(
                replica.weight.view(torch.int8), full.weight.view(torch.int8)
            )

            for m in BATCH_SIZES:
                with self.subTest(backend=backend, m=m):
                    x = torch.randn((m, K), device="cuda", dtype=torch.bfloat16)
                    with torch.inference_mode():
                        out = replica.quant_method.apply(replica, x)
                        ref_full, _ = full(x)
                        ref_shards = torch.cat([s(x)[0] for s in shards], dim=-1)
                    self.assertEqual(out.dtype, torch.bfloat16)
                    torch.testing.assert_close(out, ref_full, rtol=0, atol=0)
                    torch.testing.assert_close(out, ref_shards, rtol=0, atol=0)

    def test_triton(self):
        self._check_backend("triton")

    def test_deep_gemm(self):
        self._check_backend("deep_gemm")

    def test_flashinfer_trtllm(self):
        self._check_backend("flashinfer_trtllm")

    def test_flashinfer_cutlass(self):
        self._check_backend("flashinfer_cutlass")

    def test_flashinfer_deepgemm(self):
        self._check_backend("flashinfer_deepgemm")

    def test_cutlass(self):
        self._check_backend("cutlass")

    def test_bf16_layer_is_not_block_fp8(self):
        layer = make_tp1_column_parallel_linear(None, 256, 128)
        self.assertFalse(_is_block_fp8(layer))

    def test_replicated_forward_views_full_heads(self):
        """q_b_proj_replicated_forward reshapes to num_local_heads * dcp_size."""
        from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

        attn = DeepseekV2AttentionMLA.__new__(DeepseekV2AttentionMLA)
        attn.num_local_heads, attn.dcp_size, attn.qk_head_dim = 2, DCP_SIZE, 64
        attn.q_b_proj_qrep = None
        attn.q_b_proj_qrep_weight = torch.randn(
            (2 * DCP_SIZE * 64, 32), device="cuda", dtype=torch.bfloat16
        )
        x = torch.randn((3, 32), device="cuda", dtype=torch.bfloat16)
        q = attn.q_b_proj_replicated_forward(x)
        self.assertEqual(tuple(q.shape), (3, 2 * DCP_SIZE, 64))


if __name__ == "__main__":
    unittest.main()
