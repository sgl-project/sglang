# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU contracts for the in-tree Kimi K3 NVIDIA megakernel."""

from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.kernels.plugin_kernels.nvidia import spec as registry
from sglang.kernels.plugin_kernels.nvidia.spec import (
    MEGAKERNEL_API_VERSION,
    BuildContext,
    Feature,
    NumericsKind,
)
from sglang.srt.kernel_plugins.nvidia.binding import (
    HybridPoolView,
    HybridStepView,
    KdaPoolView,
    KdaStateView,
    KdaStepView,
    MlaPoolView,
    MlaStepView,
    PreparationView,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64)


class _Mode:
    def __init__(self, name: str):
        self.name = name

    def is_decode(self) -> bool:
        return self.name == "decode"


def _batch(mode: str, batch_size: int):
    return SimpleNamespace(forward_mode=_Mode(mode), batch_size=batch_size)


class TestKimiK3Contract(CustomTestCase):
    maxDiff = None

    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)

    def test_spec_is_exact_api2_tp8_sm103_surface(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.spec import (
            REQUIRED_MEGAKERNEL_API_VERSION,
            SPEC,
        )

        self.assertEqual(REQUIRED_MEGAKERNEL_API_VERSION, 2)
        self.assertEqual(SPEC.api_version, 2)
        self.assertEqual(MEGAKERNEL_API_VERSION, 2)
        self.assertEqual(SPEC.requires.isas, frozenset({"sm_103"}))
        self.assertEqual(SPEC.requires.tp_sizes, frozenset({8}))
        self.assertEqual(SPEC.requires.resident_blocks, 152)
        self.assertEqual(SPEC.requires.dtypes, frozenset({"bfloat16"}))
        self.assertEqual(SPEC.requires.decode_graph_backends, frozenset({"full"}))
        self.assertEqual(SPEC.features, frozenset({Feature.MULTI_NODE_TP}))
        self.assertEqual(SPEC.max_rows, 64)
        self.assertIsNotNone(SPEC.prepare)
        self.assertEqual(
            SPEC.requires.server_args,
            {
                "kv_cache_dtype": frozenset({"fp8_e4m3"}),
                "moe_runner_backend": frozenset({"flashinfer_mxfp4"}),
                "flashinfer_mxfp4_moe_precision": frozenset({"default"}),
                "prefill_attention_backend": frozenset({"trtllm_mla"}),
                "decode_attention_backend": frozenset({"trtllm_mla"}),
                "page_size": frozenset({64}),
                "dp_size": frozenset({1}),
                "enable_unified_memory": frozenset({False}),
                "enable_page_major_kv_layout": frozenset({False}),
            },
        )
        self.assertEqual(
            SPEC.requires.hf_config,
            {"language_only": True, "encoder_only": False},
        )
        self.assertEqual(SPEC.numerics.kind, NumericsKind.TOLERANCE)
        self.assertEqual(
            SPEC.numerics.tolerances,
            {"forward.selected_logprobs.max_abs": 0.5},
        )

    def test_registration_is_lazy_and_api_mismatch_fails_closed(self):
        # Restore modules and the parent attribute after isolating the lazy path.
        parent = importlib.import_module("sglang.srt.kernel_plugins.nvidia")
        root = "sglang.srt.kernel_plugins.nvidia.kimi_k3"
        architecture = "KimiK3ForConditionalGeneration"
        with (
            patch.dict(sys.modules),
            patch.object(
                parent, "kimi_k3", getattr(parent, "kimi_k3", None), create=True
            ),
        ):
            for name in tuple(sys.modules):
                if name == root or name.startswith(root + "."):
                    del sys.modules[name]
            self.assertTrue(registry.builtin_spec_registered([architecture]))
            self.assertTrue(registry.spec_registered([architecture]))
            self.assertIn(architecture, registry.registered_architectures())
            self.assertFalse(
                any(name == root or name.startswith(root + ".") for name in sys.modules)
            )
            spec, reason = registry.find_spec(
                architectures=[architecture], isa="sm_103"
            )
            self.assertIsNotNone(spec)
            self.assertEqual(reason, "")
            self.assertEqual(registry._BUILTIN_SPECS[architecture], root + ".spec:SPEC")
            self.assertFalse(
                any(name.startswith(root + ".cutedsl") for name in sys.modules)
            )
            self.assertNotIn("cutlass", sys.modules)
            self.assertNotIn("sgl_nvidia_mega", sys.modules)
            with patch.object(registry, "MEGAKERNEL_API_VERSION", 3):
                incompatible, reason = registry.find_spec(
                    architectures=[architecture], isa="sm_103"
                )
            self.assertIsNone(incompatible)
            self.assertIn("uses megakernel API version 2", reason)
            self.assertIn("requires version 3", reason)

    def test_build_filters_the_dense_capture_range_to_frozen_widths(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3 import compat, providers
        from sglang.srt.kernel_plugins.nvidia.kimi_k3 import spec as kimi_spec

        model = SimpleNamespace(language_model=SimpleNamespace())
        context = BuildContext(
            widths=tuple(range(1, 65)),
            tokens_per_request=1,
            deadline_s=30.0,
            binding=object(),
        )
        resources = object()
        with (
            patch.object(
                compat,
                "validate_runtime_compatibility",
                return_value="runtime-contract",
            ),
            patch.object(
                kimi_spec,
                "get_model",
                return_value=SimpleNamespace(context_length=8192),
            ),
            patch.object(providers, "_cuda_memory", return_value={}),
            patch.object(
                providers,
                "build_resources",
                return_value=resources,
            ) as build_resources,
        ):
            built = kimi_spec._build(model, SimpleNamespace(), context)
        self.assertEqual(built.widths, WIDTHS)
        self.assertIsNone(built.draft_sampler)
        self.assertIn("TP8", built.summary())
        build_resources.assert_called_once_with(
            model.language_model,
            context_length=8192,
            memory_before={},
            deadline_ns=30_000_000_000,
        )

    def test_exact_widths_plan_and_every_other_call_falls_back(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.spec import WIDTHS, _Built

        built = object.__new__(_Built)
        built.widths = WIDTHS
        for width in WIDTHS:
            step = built.plan(SimpleNamespace(shape=(width,)), _batch("decode", width))
            self.assertEqual((step.mode, step.rows), ("decode", width))
        for mode, batch_size, rows, args, kwargs in (
            ("decode", 9, 9, (), {}),
            ("decode", 8, 9, (), {}),
            ("extend", 1, 1, (), {}),
            ("prefill", 1, 1, (), {}),
            ("target_verify", 1, 1, (), {}),
            ("decode", 1, 1, (True,), {}),
            ("decode", 1, 1, (False, object()), {}),
            ("decode", 1, 1, (), {"get_embedding": True}),
            ("decode", 1, 1, (), {"pp_proxy_tensors": object()}),
        ):
            with self.subTest(mode=mode, batch_size=batch_size, rows=rows):
                self.assertIsNone(
                    built.plan(
                        SimpleNamespace(shape=(rows,)),
                        _batch(mode, batch_size),
                        *args,
                        **kwargs,
                    )
                )

    def test_configuration_and_loaded_model_refusals_are_named(self):
        from sglang.srt.arg_groups.model_override_base import resolving_view
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.spec import (
            _config_disable_reasons,
            _model_disable_reasons,
        )

        cfg = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="full")),
            context_length=8192,
            speculative_algorithm=None,
            enable_return_aux_hidden_states=False,
            aux_hidden_state_capture=None,
            dcp_size=1,
            nnodes=1,
            attn_cp_size=1,
            enable_prefill_cp=False,
        )
        for extended in (False, True):
            with (
                self.subTest(extended=extended),
                patch(
                    "sglang.srt.kernel_plugins.nvidia.kimi_k3.spec.EXTENDED_PROFILE",
                    extended,
                ),
            ):
                self.assertEqual(_config_disable_reasons(cfg, SimpleNamespace()), [])
                for method in ("dflash", "dspark"):
                    with self.subTest(capture=method):
                        inputs = SimpleNamespace(**vars(cfg))
                        inputs.enable_return_aux_hidden_states = True
                        inputs.speculative_algorithm = method.upper()
                        # Match handle_aux_hidden_state_capture: this is capture
                        # without a draft, so a NoSpec check alone cannot reject it.
                        inputs._resolved_overrides = [
                            (
                                "_handle_aux_hidden_state_capture",
                                {
                                    "speculative_algorithm": None,
                                    "aux_hidden_state_capture": method,
                                },
                            )
                        ]
                        resolved = resolving_view(inputs)
                        self.assertIsNone(resolved.speculative_algorithm)
                        self.assertEqual(resolved.aux_hidden_state_capture, method)
                        reasons = _config_disable_reasons(resolved, SimpleNamespace())
                        self.assertTrue(
                            any(
                                "--enable-return-aux-hidden-states" in reason
                                for reason in reasons
                            ),
                            reasons,
                        )
        cfg.context_length = 4096
        cfg.cuda_graph_config.decode.backend = "disabled"
        reasons = _config_disable_reasons(cfg, SimpleNamespace())
        self.assertTrue(any("FULL" in reason for reason in reasons))
        self.assertTrue(any("8192" in reason for reason in reasons))

        class PackedMXFP4Method:
            pass

        full_ids, linear_ids = tuple(range(24)), tuple(range(24, 93))
        layers = tuple(
            SimpleNamespace(
                _is_moe_layer=True,
                mlp=SimpleNamespace(
                    experts=SimpleNamespace(quant_method=PackedMXFP4Method())
                ),
            )
            for _ in range(93)
        )
        language_model = SimpleNamespace(
            model=SimpleNamespace(layers=layers),
            config=SimpleNamespace(
                full_attention_layer_ids=full_ids,
                linear_layer_ids=linear_ids,
                hidden_size=7168,
                num_hidden_layers=93,
                num_attention_heads=96,
                q_lora_rank=1536,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                mla_use_nope=True,
                mla_use_output_gate=True,
                hidden_act="situ",
                n_routed_experts=896,
                num_experts=896,
                num_experts_per_token=16,
                num_expert_group=1,
                topk_group=1,
                moe_router_activation_func="sigmoid",
                routed_expert_hidden_size=3584,
                moe_intermediate_size=3072,
                num_shared_experts=2,
                latent_moe_use_norm=True,
                attn_res_block_size=12,
                linear_attn_config={
                    "num_heads": 96,
                    "head_dim": 128,
                    "short_conv_kernel_size": 4,
                    "use_full_rank_gate": True,
                },
            ),
        )
        model = SimpleNamespace(language_model=language_model)
        self.assertEqual(_model_disable_reasons(model, SimpleNamespace()), [])
        language_model.model.layers = layers[:-1]
        self.assertTrue(
            any("93-layer" in reason for reason in _model_disable_reasons(model, None))
        )

    def test_candidate_payload_inventory(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.source_manifest import (
            payload_manifest,
            source_paths,
        )
        from sglang.test.kernels.kimi_k3_source_paths import RUNTIME_ROOT

        mapping = json.loads((RUNTIME_ROOT / "SOURCE_MAPPING.json").read_text())
        self.assertEqual(
            mapping["source_commit"], "459640ec1db2b0ffd5ebabc65a51dfd9524f4e44"
        )
        self.assertEqual(mapping["source_file_count"], 54)
        self.assertEqual(
            sum("/cutedsl/" in item["source"] for item in mapping["sources"]), 43
        )
        self.assertEqual(len(source_paths()), 72)
        manifest = payload_manifest()
        self.assertTrue(all(record["nbytes"] >= 0 for record in manifest.values()))


class _PreparedCore:
    def __init__(self):
        self.values = []

    def prepare_cache(self, cache):
        self.values.append(cache)

    def prepare_caches(self, caches):
        self.values.append(caches)


class TestKimiK3TypedBinding(CustomTestCase):
    def _fixture(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.providers import (
            PreparedAttention,
            ResourceSet,
        )

        live = [128]
        mla_cache = torch.zeros(3, 64, 4)
        conv = torch.zeros(4, 3)
        temporal = torch.zeros(4, 5)
        hybrid = HybridPoolView(
            mla=MlaPoolView(
                reserved_capacity=128,
                serving_capacity=lambda: live[0],
                post_capture_active=True,
                page_size=64,
                logical_dtype=torch.float8_e4m3fn,
                storage_dtype=torch.uint8,
                layer_mapping={1: 0},
                caches={1: mla_cache},
            ),
            kda=KdaPoolView(
                capacity=4,
                layer_mapping={0: 0},
                states={0: KdaStateView(conv=(conv,), temporal=temporal)},
            ),
        )
        model = SimpleNamespace(
            config=SimpleNamespace(
                full_attention_layer_ids=(1,), linear_layer_ids=(0,)
            ),
            modules=lambda: (),
            parameters=lambda: (),
            buffers=lambda: (),
        )
        mla_core, kda_core = _PreparedCore(), _PreparedCore()
        attention = {
            1: PreparedAttention(1, "mla", torch.empty(0), None, mla_core, None),
            0: PreparedAttention(0, "kda", torch.empty(0), None, kda_core, None),
        }
        memory = {
            key: 0
            for key in ("allocated", "reserved", "peak_allocated", "peak_reserved")
        }
        resources = ResourceSet(
            model=model,
            layers=(),
            attention=attention,
            feed_forward={},
            moe_front_weights={},
            before={},
            final=None,
            fused_before={},
            boxes={},
            scratch={},
            workspaces={},
            runtime_handles={},
            payload_modules={},
            compilation_inventory=[],
            memory_before=memory,
            memory_after=memory,
            deadline_ns=30_000_000_000,
        )
        return resources, hybrid, live, mla_cache, conv, temporal, mla_core, kda_core

    def test_typed_pool_binding_is_direct_idempotent_and_ignores_live_shrink(self):
        resources, hybrid, live, mla, conv, temporal, mla_core, kda_core = (
            self._fixture()
        )

        from sglang.srt.kernel_plugins.nvidia.kimi_k3.spec import SPEC, _Built

        built = object.__new__(_Built)
        built.resources = resources
        memory = {
            key: 0
            for key in ("allocated", "reserved", "peak_allocated", "peak_reserved")
        }
        with (
            patch(
                "sglang.srt.kernel_plugins.nvidia.kimi_k3.providers._cuda_memory",
                return_value=memory,
            ),
            patch.object(torch.cuda, "synchronize"),
            patch(
                "sglang.srt.kernel_plugins.nvidia.kimi_k3.providers._validate_pool_tensor"
            ),
        ):
            SPEC.prepare(built, PreparationView(hybrid=hybrid))
            fingerprint = resources.binding_fingerprint
            live[0] = 64
            SPEC.prepare(built, PreparationView(hybrid=hybrid))
        self.assertEqual(resources.binding_fingerprint, fingerprint)
        self.assertIs(resources.attention[1].cache, mla)
        self.assertIs(resources.attention[0].caches[0], conv)
        self.assertIs(resources.attention[0].caches[1], temporal)
        self.assertEqual(mla_core.values, [mla])
        self.assertEqual(kda_core.values, [(conv, temporal)])
        self.assertNotIn("serving_capacity", resources.binding)

    def test_typed_pool_replacement_is_refused(self):
        resources, hybrid, _, _, _, _, _, _ = self._fixture()

        from sglang.srt.kernel_plugins.nvidia.kimi_k3.spec import SPEC, _Built

        built = object.__new__(_Built)
        built.resources = resources

        memory = {
            key: 0
            for key in ("allocated", "reserved", "peak_allocated", "peak_reserved")
        }
        with (
            patch(
                "sglang.srt.kernel_plugins.nvidia.kimi_k3.providers._cuda_memory",
                return_value=memory,
            ),
            patch.object(torch.cuda, "synchronize"),
            patch(
                "sglang.srt.kernel_plugins.nvidia.kimi_k3.providers._validate_pool_tensor"
            ),
        ):
            SPEC.prepare(built, PreparationView(hybrid=hybrid))
            replaced = HybridPoolView(
                mla=MlaPoolView(
                    reserved_capacity=hybrid.mla.reserved_capacity,
                    serving_capacity=lambda: hybrid.mla.serving_capacity,
                    post_capture_active=hybrid.mla.post_capture_active,
                    page_size=hybrid.mla.page_size,
                    logical_dtype=hybrid.mla.logical_dtype,
                    storage_dtype=hybrid.mla.storage_dtype,
                    layer_mapping=hybrid.mla.layer_mapping,
                    caches={1: hybrid.mla.cache(1).clone()},
                ),
                kda=hybrid.kda,
            )
            with self.assertRaisesRegex(RuntimeError, "pool binding changed"):
                SPEC.prepare(built, PreparationView(hybrid=replaced))

    def test_spec_preparation_requires_a_hybrid_pool(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.spec import SPEC, _Built

        built = object.__new__(_Built)
        built.resources = SimpleNamespace(bind=Mock())
        with self.assertRaisesRegex(ValueError, "requires a hybrid MLA/KDA pool"):
            SPEC.prepare(built, PreparationView())
        built.resources.bind.assert_not_called()


class _NoOp:
    def run(self, *args, **kwargs):
        return args[0] if args else None


class _RecordingCore(_NoOp):
    def __init__(self):
        self.args = None
        self.kwargs = None

    def run(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        return super().run(*args, **kwargs)


class TestKimiK3TypedStep(CustomTestCase):
    def test_executor_uses_only_the_typed_step_and_tracks_every_kda_layer(self):
        for native in (False, True):
            with self.subTest(native=native):
                self._exercise_executor(native)

    def test_graph48_dispatches_every_layer_with_prepared_native_or_full_resources(
        self,
    ):
        for native in (False, True):
            with self.subTest(native=native):
                self._exercise_executor(native, batch=48)

    def _exercise_executor(self, native, batch=1):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.executor import (
            KimiK3MegakernelExecutor,
        )

        width = 7168
        scratch = {
            "prefix": torch.zeros(batch, 1),
            "blocks": torch.zeros(batch, 1),
            "hidden.0": torch.zeros(batch, width),
            "hidden.1": torch.zeros(batch, width),
            "hidden.out": torch.zeros(batch, width),
            "mla.projected": torch.zeros(batch, 1),
            "kda.projected": torch.zeros(batch, 1),
            "attention.local": torch.zeros(batch, width),
        }
        layers = tuple(SimpleNamespace(layer_idx=index) for index in range(93))
        mla_cache, mla_core = torch.zeros(1), _RecordingCore()
        mla_core.native_mla = native
        attention = {}
        for index in range(93):
            is_mla = index % 4 == 0
            attention[index] = SimpleNamespace(
                kind="mla" if is_mla else "kda",
                projection=_NoOp(),
                core=mla_core if is_mla else _NoOp(),
                output=_NoOp(),
                cache=mla_cache if is_mla else None,
                caches=None if is_mla else (torch.zeros(1), torch.zeros(1)),
            )
        resources = SimpleNamespace(
            binding_fingerprint="bound",
            batches=(batch,),
            native_mla=native,
            full_decode=batch == 48 and not native,
            model=SimpleNamespace(model=SimpleNamespace(embed_tokens=Mock())),
            layers=layers,
            attention=attention,
            scratch=scratch,
            before={index: _NoOp() for index in range(93)},
            fused_before={},
            feed_forward={index: _NoOp() for index in range(93)},
            final=_NoOp(),
        )
        tracked = []
        prepared = []
        view = HybridStepView(
            mla=MlaStepView(
                block_table=torch.zeros(batch, 1, dtype=torch.int32),
                sequence_lengths=torch.full(
                    (batch,), 0 if native else 1, dtype=torch.int32
                ),
                global_sequence_lengths=torch.ones(batch, dtype=torch.int32)
                if native
                else None,
                native_decode=Mock() if native else None,
                write_slots=torch.zeros(batch, dtype=torch.int64),
            ),
            kda=KdaStepView(
                state_indices=torch.zeros(batch, dtype=torch.int32),
                track_decode_state=tracked.append,
            ),
            prepare_layer=prepared.append if native or batch == 48 else None,
        )
        with patch(
            "sglang.srt.kernel_plugins.nvidia.kimi_k3.executor._validate_step_inputs"
        ):
            result = KimiK3MegakernelExecutor(resources).run(
                torch.zeros(batch, dtype=torch.int64), torch.zeros(batch, width), view
            )
        self.assertEqual(result.data_ptr(), scratch["hidden.out"].data_ptr())
        self.assertEqual(tuple(result.shape), (batch, width))
        self.assertEqual(tracked, [index for index in range(93) if index % 4])
        self.assertIs(mla_core.kwargs["cache"], mla_cache)
        self.assertIs(mla_core.kwargs["slots"], view.mla.write_slots)
        self.assertIs(mla_core.kwargs["block_table"], view.mla.block_table)
        self.assertIs(
            mla_core.kwargs["lengths"],
            view.mla.global_sequence_lengths if native else view.mla.sequence_lengths,
        )
        if native or batch == 48:
            self.assertEqual(prepared, list(range(93)))
        if native:
            mla_core.kwargs["native_decode"]("query", "output")
            view.mla.native_decode.assert_called_once_with(92, "query", "output")
        resources.model.model.embed_tokens.assert_not_called()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
