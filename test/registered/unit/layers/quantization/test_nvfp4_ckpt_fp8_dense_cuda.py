"""--nvfp4-ckpt-fp8-dense-modules on device: BF16 weights through the real loader, quantized at load, vs the BF16 matmul."""

import unittest
from unittest import mock

import torch

import sglang.srt.layers.quantization.modelopt_quant as mq
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptOnlineBlockFp8LinearMethod,
)
from sglang.srt.model_executor.model_runner import _gather_block_fp8_linear
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _unsupported_derived_weight_cache_error,
)
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import (
    assert_output_close,
    init_single_process_dist,
    make_tp1_column_parallel_linear,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
register_cuda_ci(est_time=10, stage="base-b", runner_config="4-gpu-b200")

# (M, N, K); N=192 leaves a ragged 64-row last weight block.
SHAPES = [(5, 512, 1024), (64, 192, 512), (128, 2048, 2048)]
MODULES = ["o_proj", "gate_proj", "up_proj"]


def _nvfp4_config():
    qc = ModelOptFp4Config.from_config(
        {
            "quantization": {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "exclude_modules": ["model.layers.*.self_attn*", "model.layers.*.mlp*"],
            }
        }
    )
    qc.update_packed_modules_mapping({})
    return qc


@unittest.skipIf(get_device_sm() < 90, "block-FP8 linear requires SM90+")
class TestFp8OnlineBlockLinear(CustomTestCase):
    modules = MODULES

    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def setUp(self):
        patcher = mock.patch.object(
            mq, "_online_fp8_dense_modules", return_value=self.modules
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        torch.manual_seed(0)

    def _load(self, layer, w, shard_id=None):
        # Exercise the loader chosen by construction, not weight_loader_v2 directly.
        self.assertEqual(layer.weight.weight_loader, layer.weight_loader_v2)
        if shard_id is None:
            layer.weight.weight_loader(layer.weight, w)
        else:
            layer.weight.weight_loader(layer.weight, w, shard_id)

    def _check(self, layer, w, m):
        self.assertIsInstance(layer.quant_method, ModelOptOnlineBlockFp8LinearMethod)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        layer.quant_method.process_weights_after_loading(layer)
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        x = torch.randn((m, w.shape[1]), device="cuda", dtype=torch.bfloat16) / 10
        out, _ = layer(x)
        ref = x.float() @ w.float().T
        assert_output_close(self, out, ref, cos_threshold=0.995)
        # Cosine alone misses a uniform scale error such as returning 2 * ref.
        relative_error = (out.float() - ref).norm() / ref.norm()
        self.assertLess(relative_error.item(), 0.08)
        self.assertTrue(torch.isfinite(out).all().item())

    def test_column_linear(self):
        for m, n, k in SHAPES:
            with self.subTest(shape=(m, n, k)):
                layer = make_tp1_column_parallel_linear(
                    _nvfp4_config(), n, k, prefix="model.layers.0.self_attn.o_proj"
                )
                w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
                self._load(layer, w)
                self._check(layer, w, m)

    def test_row_linear(self):
        layer = RowParallelLinear(
            input_size=1024,
            output_size=512,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=_nvfp4_config(),
            prefix="model.layers.0.self_attn.o_proj",
            parallel_group="replicated",
        ).cuda()
        w = torch.randn((512, 1024), device="cuda", dtype=torch.bfloat16) / 10
        self._load(layer, w)
        self._check(layer, w, 32)

    def test_merged_shards_loaded_separately(self):
        n, k = 1024, 1024
        layer = MergedColumnParallelLinear(
            input_size=k,
            output_sizes=[n, n],
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=_nvfp4_config(),
            prefix="model.layers.0.mlp.gate_up_proj",
            parallel_group="replicated",
        ).cuda()
        gate, up = (
            torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
            for _ in range(2)
        )
        self._load(layer, gate, shard_id=0)
        self._load(layer, up, shard_id=1)
        self._check(layer, torch.cat([gate, up]), 64)

    def test_dcp_replica_matches_layer(self):
        # The DCP gather re-runs post-load on the FP8 replica; it must not requantize.
        layer = make_tp1_column_parallel_linear(
            _nvfp4_config(), 1024, 512, prefix="model.layers.0.self_attn.o_proj"
        )
        w = torch.randn((1024, 512), device="cuda", dtype=torch.bfloat16) / 10
        self._load(layer, w)
        layer.quant_method.process_weights_after_loading(layer)
        replica = _gather_block_fp8_linear(layer, get_tp_group())
        x = torch.randn((16, 512), device="cuda", dtype=torch.bfloat16) / 10
        torch.testing.assert_close(
            layer.quant_method.apply(replica, x),
            layer.quant_method.apply(layer, x),
            rtol=0,
            atol=0,
        )

    def test_online_weight_updates_rejected(self):
        # Reloading BF16 values into the FP8 weight would serve them under stale scales.
        model = torch.nn.Module()
        model.o_proj = make_tp1_column_parallel_linear(
            _nvfp4_config(), 512, 512, prefix="model.layers.0.self_attn.o_proj"
        )
        w = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
        self._load(model.o_proj, w)
        model.o_proj.quant_method.process_weights_after_loading(model.o_proj)
        error = _unsupported_derived_weight_cache_error(model)
        self.assertIsNotNone(error)
        self.assertIn("--nvfp4-ckpt-fp8-dense-modules", error)


class TestFp8OnlineBlockLinearGlobs(TestFp8OnlineBlockLinear):
    modules = ["model.layers.*.self_attn.o_proj", "*.mlp.gate_proj", "*.mlp.up_proj"]


if __name__ == "__main__":
    unittest.main()
