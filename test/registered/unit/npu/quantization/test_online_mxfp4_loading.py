"""CPU lifecycle tests using native Linear/MoE shard loaders and mocked conversion."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUMXFP4W4A8LinearMethod,
)
from sglang.srt.hardware_backend.npu.quantization.online_moe_methods import (
    NPUW4A8MXFP4OnlineMoEMethod,
)
from sglang.srt.layers import linear
from sglang.srt.layers.moe.fused_moe_triton import layer as moe_layer
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.npu_mxfp4 import Mxfp4W4A8Config
from sglang.srt.model_loader.post_load import stage_module_for_post_load
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOnlineMXFP4Loading(unittest.TestCase):
    def setUp(self):
        self.snapshots = {}
        self.source_handles = []
        for target, kwargs in (
            (
                "sglang.srt.layers.quantization.unquant.get_moe_runner_backend",
                {"return_value": MoeRunnerBackend.AUTO},
            ),
            ("sglang.srt.layers.linear._is_cpu", {"new": False}),
            ("sglang.srt.layers.parameter._is_cpu", {"new": False}),
            (
                "sglang.srt.layers.linear.get_group_rank_size",
                {"side_effect": lambda group: (group.rank, group.size)},
            ),
        ):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def prepare(self, layer, name):
        source = getattr(layer, name)
        self.snapshots.setdefault(name, []).append(source.detach().clone())
        self.source_handles.append(source)  # Deliberately retain stale handles.
        packed = torch.zeros(
            *source.shape[:-1], source.shape[-1] // 2, dtype=torch.uint8
        )
        setattr(layer, name, torch.nn.Parameter(packed, requires_grad=False))

    def dense(self, cls=linear.MergedColumnParallelLinear, dtype=torch.bfloat16):
        layer = cls.__new__(cls)
        torch.nn.Module.__init__(layer)
        layer.tp_group = SimpleNamespace(rank=1, size=2)
        layer.kv_tp_group = SimpleNamespace(rank=0, size=1)
        layer.use_presharded_weights = False
        layer.output_sizes = [4, 4]
        layer.num_heads, layer.total_num_heads = 2, 4
        layer.num_kv_heads = layer.total_num_kv_heads = layer.num_kv_head_replicas = 1
        layer.head_size = layer.v_head_size = 2
        method = NPUMXFP4W4A8LinearMethod()
        method._quantize_weights = Mock(
            side_effect=lambda _: self.prepare(layer, "weight")
        )
        widths = [4, 2, 2] if cls is linear.QKVParallelLinear else [2, 2]
        method.create_weights(
            layer,
            64,
            widths,
            128 if cls is linear.RowParallelLinear else 64,
            sum(widths) * 2,
            dtype,
            weight_loader=layer.weight_loader,
        )
        return layer, method

    def moe(self, fused=False, with_bias=False):
        layer = moe_layer.FusedMoE.__new__(moe_layer.FusedMoE)
        torch.nn.Module.__init__(layer)
        layer.quant_config = Mxfp4W4A8Config()
        layer.scheme = None
        layer.uses_expert_location = False
        layer.layer_id = 0
        layer._expert_storage_rank = 1
        layer._num_local_routed = layer.num_local_experts = 2
        layer._num_global_routed = 4
        layer._has_fused_shared = False
        layer.num_fused_shared_experts = 0
        layer.moe_ep_size = layer.moe_tp_size = 2
        layer.moe_tp_rank = 1
        layer.use_triton_kernels = layer.use_flashinfer_trtllm_moe = False
        layer.use_padded_loading = layer.use_presharded_weights = False
        layer.moe_runner_config = SimpleNamespace(is_gated=True)
        method = NPUW4A8MXFP4OnlineMoEMethod(layer.quant_config)
        layer.quant_method = method
        method.create_weights(
            layer,
            2,
            64,
            64,
            torch.bfloat16,
            with_bias=with_bias,
            weight_loader=layer.weight_loader_fused if fused else layer.weight_loader,
        )
        for prefix in ("w13", "w2"):
            setattr(
                layer,
                f"{prefix}_kernel",
                SimpleNamespace(
                    process_weights_after_loading=lambda layer, prefix: self.prepare(
                        layer, f"{prefix}_weight"
                    )
                ),
            )
        return layer, method

    def test_dense_waits_for_all_shards_releases_source_and_reloads(self):
        layer, method = self.dense()
        stale_meta = layer.weight
        self.assertTrue(stale_meta.is_meta)
        for cycle in range(2):
            source = (
                torch.arange(4 * 64, dtype=torch.float32)
                .reshape(4, 64)
                .to(torch.bfloat16)
                + cycle
            )
            load = layer.weight.weight_loader
            load(stale_meta, source, 1)
            self.assertEqual(method._quantize_weights.call_count, cycle)
            with self.assertRaisesRegex(RuntimeError, "not completely loaded"):
                method.process_weights_after_loading(layer)
            load(stale_meta, source + 10, 0)
            expected = torch.cat([(source + 10)[2:], source[2:]])
            torch.testing.assert_close(self.snapshots["weight"][-1], expected)
            self.assertEqual(self.source_handles[-1].numel(), 0)
            self.assertEqual(layer.weight.dtype, torch.uint8)
            self.assertTrue(callable(layer.weight.weight_loader))
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                load(stale_meta, source, 0)
            method.process_weights_after_loading(layer)
            method.process_weights_after_loading(layer)
            self.assertEqual(method._quantize_weights.call_count, cycle + 1)

    def test_dense_column_and_row_tp_and_presharded(self):
        for cls in (linear.ColumnParallelLinear, linear.RowParallelLinear):
            for presharded in (False, True):
                with self.subTest(cls=cls, presharded=presharded):
                    layer, method = self.dense(cls, torch.float16)
                    layer.use_presharded_weights = presharded
                    source = (
                        torch.arange(8 * 128, dtype=torch.float32)
                        .reshape(8, 128)
                        .half()
                    )
                    source = (
                        source[:4]
                        if cls is linear.RowParallelLinear
                        else source[:, :64]
                    )
                    expected = (
                        source[:, 64:]
                        if cls is linear.RowParallelLinear
                        else source[4:]
                    )
                    layer.weight.weight_loader(
                        layer.weight, expected if presharded else source
                    )
                    torch.testing.assert_close(self.snapshots["weight"][-1], expected)
                    method.process_weights_after_loading(layer)

    def test_dense_qkv_replicated_kv_and_fused_checkpoint(self):
        for fused in (False, True):
            layer, method = self.dense(linear.QKVParallelLinear)
            q = torch.arange(8 * 64).reshape(8, 64).to(torch.bfloat16)
            k, v = q[:2] + 10, q[:2] + 20
            if fused:
                layer.weight.weight_loader(layer.weight, torch.cat([q, k, v]))
            else:
                for shard, value in (("k", k), ("q", q), ("v", v)):
                    layer.weight.weight_loader(layer.weight, value, shard)
            torch.testing.assert_close(
                self.snapshots["weight"][-1], torch.cat([q[4:], k, v])
            )
            method.process_weights_after_loading(layer)

    def test_native_loader_slices_before_device_copy(self):
        layer, method = self.dense(linear.ColumnParallelLinear)
        source = torch.ones(8, 64, dtype=torch.bfloat16)
        with patch.object(
            torch.Tensor, "to", side_effect=AssertionError("global tensor was staged")
        ):
            layer.weight.weight_loader(layer.weight, source)
        method.process_weights_after_loading(layer)

    def test_dense_merged_tuple_shard_uses_native_tp_slicing(self):
        layer, method = self.dense()
        source = torch.arange(8 * 64).reshape(8, 64).to(torch.bfloat16)
        layer.weight.weight_loader(layer.weight, source, (0, 1))
        torch.testing.assert_close(
            self.snapshots["weight"][-1], torch.cat([source[2:4], source[6:8]])
        )
        method.process_weights_after_loading(layer)

    def test_conversion_failure_is_not_retried_or_accepted(self):
        layer, method = self.dense(linear.ColumnParallelLinear)
        method._quantize_weights.side_effect = RuntimeError("conversion failed")
        source = torch.ones(8, 64, dtype=torch.bfloat16)
        load = layer.weight.weight_loader
        with self.assertRaisesRegex(RuntimeError, "conversion failed"):
            load(layer.weight, source)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            load(layer.weight, source)
        with self.assertRaisesRegex(RuntimeError, "not completely loaded"):
            method.process_weights_after_loading(layer)
        method._quantize_weights.assert_called_once()

    def test_dense_duplicate_partial_shard_and_bypassed_reload_rejected(self):
        layer, method = self.dense()
        source = torch.ones(4, 64, dtype=torch.bfloat16)
        load = layer.weight.weight_loader
        load(layer.weight, source, 0)
        with self.assertRaisesRegex(RuntimeError, "Duplicate shard"):
            load(layer.weight, source, 0)
        method._quantize_weights.assert_not_called()
        load(layer.weight, source, 1)
        method.process_weights_after_loading(layer)
        layer.weight = torch.nn.Parameter(
            torch.ones(4, 64, dtype=torch.bfloat16), requires_grad=False
        )
        with self.assertRaisesRegex(RuntimeError, "registered weight loader"):
            method.process_weights_after_loading(layer)

    def test_dense_concurrent_shards_convert_once(self):
        layer, method = self.dense()
        param = layer.weight
        load = param.weight_loader
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    load,
                    param,
                    torch.full((4, 64), float(i + 1), dtype=torch.bfloat16),
                    i,
                )
                for i in range(2)
            ]
            for future in futures:
                future.result()
        method._quantize_weights.assert_called_once()
        torch.testing.assert_close(
            self.snapshots["weight"][0][:2], torch.ones(2, 64, dtype=torch.bfloat16)
        )
        torch.testing.assert_close(
            self.snapshots["weight"][0][2:],
            torch.full((2, 64), 2.0, dtype=torch.bfloat16),
        )

    def test_invalid_dtype_and_missing_weights_fail(self):
        with self.assertRaisesRegex(ValueError, "FP16/BF16"):
            self.dense(dtype=torch.float32)
        layer, method = self.dense()
        with self.assertRaisesRegex(RuntimeError, "not completely loaded"):
            method.process_weights_after_loading(layer)
        layer, method = self.moe()
        with self.assertRaisesRegex(RuntimeError, "not completely loaded"):
            method.process_weights_after_loading(layer)

    def test_moe_independent_completion_nonlocal_tp_source_release_and_reload(self):
        layer, method = self.moe()
        stale = {p: getattr(layer, f"{p}_weight") for p in ("w13", "w2")}
        self.assertTrue(all(w.is_meta for w in stale.values()))
        load = stale["w13"].weight_loader
        for cycle in range(2):
            # Nonlocal experts are no-ops even after conversion and at reload.
            before = layer.w13_weight
            load(stale["w13"], torch.ones(128, 64), "w13_weight", "w1", 0)
            self.assertIs(layer.w13_weight, before)
            previous_w2 = layer.w2_weight
            expected = torch.empty(2, 128, 64, dtype=torch.bfloat16)
            for shard in ("w3", "w1"):
                for local, expert in enumerate((2, 3)):
                    value = (
                        torch.arange(128 * 64).reshape(128, 64).to(torch.bfloat16)
                        + cycle
                        + local
                    )
                    if shard == "w1":
                        expected[local, :64] = value[64:]
                    else:
                        expected[local, 64:] = value[64:]
                    load(stale["w13"], value, "w13_weight", shard, expert)
            torch.testing.assert_close(self.snapshots["w13_weight"][-1], expected)
            self.assertEqual(layer.w13_weight.dtype, torch.uint8)
            self.assertIs(layer.w2_weight, previous_w2)
            self.assertEqual(self.source_handles[-1].numel(), 0)
            prepared = layer.w13_weight
            load(stale["w13"], torch.ones(128, 64), "w13_weight", "w1", 0)
            self.assertIs(layer.w13_weight, prepared)
            for expert in (2, 3):
                down = torch.arange(64 * 128).reshape(64, 128).to(torch.bfloat16)
                load(stale["w2"], down, "w2_weight", "w2", expert)
            torch.testing.assert_close(
                self.snapshots["w2_weight"][-1], down[:, 64:].expand(2, -1, -1)
            )
            method.process_weights_after_loading(layer)
            method.process_weights_after_loading(layer)
            self.assertEqual(len(self.snapshots["w13_weight"]), cycle + 1)
            self.assertEqual(len(self.snapshots["w2_weight"]), cycle + 1)

    def test_moe_duplicate_expert_cannot_complete_partial_weight(self):
        layer, method = self.moe()
        load = layer.w13_weight.weight_loader
        source = torch.ones(128, 64, dtype=torch.bfloat16)
        load(layer.w13_weight, source, "w13_weight", "w1", 2)
        with self.assertRaisesRegex(RuntimeError, "Duplicate shard"):
            load(layer.w13_weight, source, "w13_weight", "w1", 2)
        self.assertNotIn("w13_weight", self.snapshots)
        with self.assertRaisesRegex(RuntimeError, "not completely loaded"):
            method.process_weights_after_loading(layer)

    def test_prepared_weights_survive_post_load_device_staging_without_copy(self):
        for is_moe in (False, True):
            with self.subTest(is_moe=is_moe):
                if is_moe:
                    layer, method = self.moe(fused=True)
                    for prefix, shape in (
                        ("w13", (2, 256, 64)),
                        ("w2", (2, 64, 128)),
                    ):
                        param = getattr(layer, f"{prefix}_weight")
                        param.weight_loader(
                            param,
                            torch.ones(shape, dtype=torch.bfloat16),
                            f"{prefix}_weight",
                            prefix,
                        )
                else:
                    layer, method = self.dense()
                    layer.weight.weight_loader(
                        layer.weight, torch.ones(8, 64, dtype=torch.bfloat16)
                    )
                prepared = {
                    name: param.data_ptr() for name, param in layer.named_parameters()
                }
                with patch.object(
                    torch.Tensor,
                    "copy_",
                    side_effect=AssertionError("unexpected copy_"),
                ):
                    with stage_module_for_post_load(layer, torch.device("cpu:0")):
                        method.process_weights_after_loading(layer)
                for name, param in layer.named_parameters():
                    self.assertEqual(param.dtype, torch.uint8)
                    self.assertEqual(param.data_ptr(), prepared[name])

    def test_moe_fused_loader_and_bias_allocation(self):
        layer, method = self.moe(fused=True, with_bias=True)
        self.assertFalse(layer.w13_weight_bias.is_meta)
        self.assertFalse(layer.w2_weight_bias.is_meta)
        self.assertEqual(layer.w13_weight_bias.weight_loader, layer.weight_loader_fused)
        for prefix, shape in (("w13", (2, 256, 64)), ("w2", (2, 64, 128))):
            source = torch.ones(shape, dtype=torch.bfloat16)
            param = getattr(layer, f"{prefix}_weight")
            param.weight_loader(param, source, f"{prefix}_weight", prefix)
        method.process_weights_after_loading(layer)

    def test_expert_destinations_preserve_eplb_replication_shared_and_global_slots(
        self,
    ):
        layer, _ = self.moe()
        param = layer.w13_weight
        metadata = SimpleNamespace(
            num_logical_experts=4, logical_to_all_physical=Mock(return_value=[0, 2, 3])
        )
        with patch.object(
            moe_layer, "get_global_expert_location_metadata", return_value=metadata
        ):
            # Explicitly disabled location metadata must not be consulted.
            self.assertEqual(list(layer._weight_loader_expert_ids(param, 2)), [0])
            metadata.logical_to_all_physical.assert_not_called()
            layer.uses_expert_location = True
            self.assertEqual(list(layer._weight_loader_expert_ids(param, 1)), [0, 1])
            param._sglang_require_global_experts = True
            self.assertEqual(list(layer._weight_loader_expert_ids(param, 1)), [0, 2, 3])
            layer._has_fused_shared = True
            layer.num_fused_shared_experts = 1
            layer.num_local_experts = 3
            with patch.object(
                moe_layer, "uses_per_rank_fused_shared_slots", return_value=True
            ):
                self.assertEqual(
                    list(layer._weight_loader_expert_ids(param, 4)), [2, 5]
                )
            param._sglang_require_global_experts = False
            self.assertEqual(list(layer._weight_loader_expert_ids(param, 4)), [2])


if __name__ == "__main__":
    unittest.main()
