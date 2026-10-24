# SPDX-License-Identifier: Apache-2.0
"""Layout declarations must agree with native loaders, not just with themselves."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.weight_cache.model_adapters import (
    WeightAdapterContext,
    get_weight_layout_adapter,
    validate_adapter_config,
)
from sglang.srt.weight_cache.model_adapters.layers import LayerLayoutBuilder
from sglang.srt.weight_cache.weight_layout import WeightLayoutBuilder
from sglang.srt.weight_cache.weight_runtime_manifest import WeightManifestError
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

HAS_MOONCAKE_RESHARD = all(
    importlib.util.find_spec(name) is not None
    for name in ("mooncake", "mooncake.reshard", "mooncake.reshard.weight")
)


def write_declared_regions(plan, checkpoint):
    plan.initialize_local_regions()
    with torch.no_grad():
        for region in plan.views:
            slices = tuple(
                slice(o, o + n)
                for o, n in zip(region.global_offset, region.local_shape)
            )
            values = checkpoint[region.tensor_id][slices].contiguous().flatten()
            region.parameter.view(-1).narrow(
                0, region.byte_offset // region.parameter.element_size(), values.numel()
            ).copy_(values)


class TestWeightLayout(unittest.TestCase):
    def test_linear_helpers_match_native_loaders(self):
        for kind in ("replicated", "column", "row", "merged"):
            for size in (1,) if kind == "replicated" else (1, 2, 4):
                for rank in range(size):
                    with self.subTest(kind=kind, size=size, rank=rank):
                        group = SimpleNamespace(rank_in_group=rank, world_size=size)
                        with patch(
                            "sglang.srt.layers.linear._resolve_linear_group",
                            return_value=group,
                        ):
                            cls = {
                                "replicated": ReplicatedLinear,
                                "column": ColumnParallelLinear,
                                "row": RowParallelLinear,
                                "merged": MergedColumnParallelLinear,
                            }[kind]
                            module = cls(
                                8,
                                [8, 16] if kind == "merged" else 12,
                                bias=True,
                                params_dtype=torch.float32,
                            )
                        checkpoint = {}
                        for name, rows, shard in (
                            (("gate", 8, 0), ("up", 16, 1))
                            if kind == "merged"
                            else (("linear", 12, None),)
                        ):
                            for attr, shape in (
                                ("weight", (rows, 8)),
                                ("bias", (rows,)),
                            ):
                                values = torch.arange(
                                    torch.tensor(shape).prod().item()
                                ).float().reshape(shape) + (1000 * (shard or 0))
                                checkpoint[f"{name}.{attr}"] = values
                                param = getattr(module, attr)
                                if kind == "merged":
                                    module.weight_loader(param, values, shard)
                                else:
                                    module.weight_loader(param, values)
                        expected = {
                            n: p.detach().clone() for n, p in module.named_parameters()
                        }
                        builder = LayerLayoutBuilder(module)
                        if kind == "merged":
                            builder.merged_column(module, ("gate", "up"))
                        else:
                            getattr(
                                builder,
                                kind
                                + ("_linear" if kind == "replicated" else "_parallel"),
                            )(module, "linear")
                        plan = builder.build()
                        for name, param in module.named_parameters():
                            torch.testing.assert_close(
                                param, expected[name], rtol=0, atol=0
                            )
                            param.data.fill_(-99)
                        write_declared_regions(plan, checkpoint)
                        for name, param in module.named_parameters():
                            torch.testing.assert_close(
                                param, expected[name], rtol=0, atol=0
                            )

    def test_coverage_alias_and_storage_rebinding(self):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.arange(8).reshape(2, 4).float())
        model.alias = model.weight
        model.register_buffer("cache", torch.ones(2), persistent=False)
        b = LayerLayoutBuilder(model)
        b.replicated(model.weight, "weight")
        b.replicated(model.alias, "weight")
        with self.assertRaisesRegex(WeightManifestError, "complete tensor: cache"):
            b.build()
        b.local_buffer("cache")
        plan = b.build()
        self.assertEqual(len(plan.views), 1)
        model.weight.data = model.weight.data.clone()
        with self.assertRaisesRegex(WeightManifestError, "storage changed"):
            plan.validate_bindings(model)

    def test_moe_helper_matches_native_expert_loader(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        # Allocate small CPU weights using the actual native classes and loader.
        # Construction-time distributed/kernel setup is irrelevant to this contract.
        for ep_size, tp_size in ((1, 2), (2, 2), (4, 1)):
            for ep in range(ep_size):
                for tp in range(tp_size):
                    with self.subTest(ep_size=ep_size, tp_size=tp_size, ep=ep, tp=tp):
                        module = FusedMoE.__new__(FusedMoE)
                        torch.nn.Module.__init__(module)
                        method = UnquantizedFusedMoEMethod.__new__(
                            UnquantizedFusedMoEMethod
                        )
                        torch.nn.Module.__init__(method)
                        for name in (
                            "use_triton_kernels",
                            "use_flashinfer_trtllm_moe",
                            "use_deep_gemm",
                            "with_bias",
                            "w13_swiglu_interleaved",
                        ):
                            setattr(method, name, False)
                        method.runner = SimpleNamespace(
                            runner_backend=SimpleNamespace(is_triton=lambda: True)
                        )
                        module.quant_method = method
                        module.quant_config = None
                        module.num_fused_shared_experts = 0
                        module._has_fused_shared = False
                        module._num_global_routed = 4
                        module._num_local_routed = module.num_local_experts = (
                            4 // ep_size
                        )
                        module._expert_storage_rank = module.moe_ep_rank = ep
                        module.moe_tp_rank, module.moe_tp_size = tp, tp_size
                        module.hidden_size = 4
                        module.intermediate_size_per_partition = 8 // tp_size
                        module.use_triton_kernels = module.use_flashinfer_trtllm_moe = (
                            module.use_presharded_weights
                        ) = False
                        module.moe_runner_config = SimpleNamespace(is_gated=True)
                        module.w13_weight = torch.nn.Parameter(
                            torch.empty(4 // ep_size, 16 // tp_size, 4)
                        )
                        module.w2_weight = torch.nn.Parameter(
                            torch.empty(4 // ep_size, 4, 8 // tp_size)
                        )
                        checkpoint = {}
                        with (
                            torch.no_grad(),
                            patch(
                                "sglang.srt.layers.moe.fused_moe_triton.layer.get_global_expert_location_metadata",
                                return_value=None,
                            ),
                            patch(
                                "sglang.srt.layers.quantization.unquant.get_moe_runner_backend",
                                return_value=SimpleNamespace(
                                    is_flashinfer_trtllm_routed=lambda: False
                                ),
                            ),
                        ):
                            for expert in range(4):
                                for part, shard, shape in (
                                    ("gate_proj", "w1", (8, 4)),
                                    ("up_proj", "w3", (8, 4)),
                                    ("down_proj", "w2", (4, 8)),
                                ):
                                    value = (
                                        torch.arange(32).reshape(shape).float()
                                        + expert * 100
                                        + (1000 if shard == "w3" else 0)
                                    )
                                    checkpoint[f"experts.{expert}.{part}.weight"] = (
                                        value
                                    )
                                    parameter = (
                                        module.w2_weight
                                        if shard == "w2"
                                        else module.w13_weight
                                    )
                                    module.weight_loader(
                                        parameter, value, "weight", shard, expert
                                    )
                        expected = {
                            n: p.detach().clone() for n, p in module.named_parameters()
                        }
                        with patch(
                            "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
                            return_value=None,
                        ):
                            builder = LayerLayoutBuilder(module)
                            builder.fused_moe(module, "experts")
                            plan = builder.build()
                            self.assertEqual(
                                len(plan.views), 3 * module.num_local_experts
                            )
                            for parameter in module.parameters():
                                parameter.data.fill_(-99)
                            write_declared_regions(plan, checkpoint)
                            for name, parameter in module.named_parameters():
                                torch.testing.assert_close(
                                    parameter, expected[name], rtol=0, atol=0
                                )
                            method.w13_swiglu_interleaved = True
                            with self.assertRaisesRegex(
                                WeightManifestError, "unsupported MoE layout"
                            ):
                                LayerLayoutBuilder(module).fused_moe(module, "experts")

    def test_gap_overlap_and_bounds_fail(self):
        model = torch.nn.Linear(4, 2, bias=False)
        for offset in (4, -4, 100):
            with self.subTest(offset=offset), self.assertRaises(WeightManifestError):
                b = WeightLayoutBuilder(model)
                b.region(model.weight, "w", global_shape=(2, 4), byte_offset=offset)
                b.build()
        b = WeightLayoutBuilder(model)
        b.region(model.weight, "w", global_shape=(2, 4))
        b.region(model.weight, "other", global_shape=(2, 4))
        with self.assertRaisesRegex(WeightManifestError, "overlap"):
            b.build()
        b = WeightLayoutBuilder(model)
        b.region(model.weight, "w", global_shape=(2, 4), local_shape=(1, 4))
        with self.assertRaisesRegex(WeightManifestError, "complete tensor"):
            b.build()

    def test_derived_state_must_be_rebuilt_without_replacing_parameters(self):
        model = torch.nn.Linear(4, 2, bias=False)
        model.derived = None
        b = LayerLayoutBuilder(model)
        b.replicated(model.weight, "w")
        b.derived("derived")
        plan = b.build()
        with self.assertRaisesRegex(WeightManifestError, "not rebuilt"):
            plan.validate_derived_state(model)
        model.derived = model.weight.detach().T.contiguous()
        plan.validate_derived_state(model)
        plan.validate_bindings(model)
        model.extra = torch.nn.Parameter(torch.empty(1))
        with self.assertRaisesRegex(WeightManifestError, "names changed"):
            plan.validate_bindings(model)

    def test_distinct_overlapping_storage_views_are_rejected(self):
        model = torch.nn.Module()
        storage = torch.empty(8)
        model.first = torch.nn.Parameter(storage[:6])
        model.second = torch.nn.Parameter(storage[4:])
        builder = LayerLayoutBuilder(model)
        builder.replicated(model.first, "first")
        builder.replicated(model.second, "second")
        with self.assertRaisesRegex(WeightManifestError, "overlapping registered"):
            builder.build()

    @unittest.skipUnless(HAS_MOONCAKE_RESHARD, "requires mooncake.reshard.weight")
    def test_incompatible_layout_does_not_write_target_or_padding(self):
        from unittest.mock import MagicMock

        from sglang.srt.weight_cache.weight_heterogeneous_transfer import (
            WeightParallelLayout,
            WeightTransferSession,
            pull_weights_from_source,
        )
        from sglang.srt.weight_cache.weight_runtime_manifest import (
            ImmutableWeightRuntimeManifestBuilder,
            WeightParallelTopology,
        )

        models, plans, manifests = [], [], []
        for source in (True, False):
            model = torch.nn.Module()
            model.weight = torch.nn.Parameter(torch.full((6 if source else 8,), -99.0))
            b = WeightLayoutBuilder(model)
            b.region(
                model.weight,
                "weight",
                global_shape=(6,),
                local_shape=(6,),
                layout_fingerprint="sglang:row-major"
                if source
                else "unsupported-packed",
            )
            if not source:
                b.initialize(model.weight, byte_offset=24, nbytes=8)
            plan = b.build()
            manifest = ImmutableWeightRuntimeManifestBuilder(
                model=model,
                layout_plan=plan,
                topology=WeightParallelTopology(),
                allowed_devices=("cpu",),
            ).build(
                model_id="model",
                revision="test",
                instance_id=str(source),
                worker_id=str(source),
                endpoint="127.0.0.1:1",
            )
            models.append(model)
            plans.append(plan)
            manifests.append(manifest)
        engine = MagicMock()
        session = WeightTransferSession(
            engine, manifests[1], WeightParallelLayout(1), (), {}, layout_plan=plans[1]
        )
        with self.assertRaisesRegex(ValueError, "layout mismatch"):
            pull_weights_from_source(
                target_session=session,
                source_runtime_manifests=(manifests[0],),
                source_parallel_layout=WeightParallelLayout(1),
                target_runtime_manifests=(manifests[1],),
            )
        engine.batch_transfer_sync_read.assert_not_called()
        torch.testing.assert_close(
            models[1].weight, torch.full((8,), -99.0), rtol=0, atol=0
        )

    def test_registry_rejects_unknown_types_and_deepseek_variants(self):
        for architecture in ("Unknown", "DeepseekV3ForCausalLM", "Qwen3MoeForCausalLM"):
            with self.assertRaisesRegex(
                WeightManifestError, "no weight-layout adapter"
            ):
                validate_adapter_config(SimpleNamespace(architectures=[architecture]))
        with self.assertRaisesRegex(WeightManifestError, "V2-Lite"):
            validate_adapter_config(
                SimpleNamespace(architectures=["DeepseekV2ForCausalLM"], q_lora_rank=16)
            )
        with self.assertRaisesRegex(WeightManifestError, "exact type"):
            get_weight_layout_adapter(
                torch.nn.Linear(1, 1),
                WeightAdapterContext(
                    SimpleNamespace(architectures=["Qwen3ForCausalLM"])
                ),
            )


if __name__ == "__main__":
    unittest.main()
