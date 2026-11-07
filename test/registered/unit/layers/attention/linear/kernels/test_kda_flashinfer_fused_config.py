"""Real resolution/env/state allocation must reach the optional Kimi consumer.

The source methods run on CPU; only CUDA metadata, allocator integration and
kernel launch boundaries are isolated. This does not qualify GPU arithmetic.
"""

import ast
import dataclasses
import logging
import os
import sys
import unittest
from abc import ABC
from contextlib import nullcontext
from enum import Enum, auto
from functools import cached_property
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import test_kda_flashinfer_fused as consumer
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
ROOT = consumer.ROOT


def _source_nodes(relative_path, names, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    namespace.setdefault("__name__", "_kda_config_contract")
    module = ModuleType(namespace["__name__"])
    module.__dict__.update(namespace)
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(
            compile(ast.fix_missing_locations(code), str(path), "exec"), module.__dict__
        )
    return module.__dict__


class _TritonLaunch:
    def decode(self, *args, **kwargs):
        return "triton"


class _FlashInferLaunch:
    def decode(self, *args, **kwargs):
        return "flashinfer-recurrent"


class TestFlashInferFusedKDAConfig(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.envs = consumer._load_file(
            "_kda_config_environ", ROOT / "python/sglang/srt/environ.py"
        ).envs
        common = dict(torch=torch, np=np, logger=logging.getLogger(__name__))
        cls.views = _source_nodes(
            "python/sglang/srt/arg_groups/model_override_base.py",
            (
                "ResolvedView",
                "ResolvingConfig",
                "resolving_view",
                "resolved_view",
                "_declaration_overlay",
            ),
            {},
        )
        declare = consumer._source_function(
            ROOT / "python/sglang/srt/arg_groups/overrides.py",
            "declare_resolution",
            dict(
                is_record=lambda _: False,
                get_context=lambda: SimpleNamespace(server_args=None),
            ),
        )
        cls.predicates = _source_nodes(
            "python/sglang/srt/configs/model_config.py",
            ("_hf_arch", "_hf_attr", "is_kimi_k3", "uses_kda_attention"),
            {},
        )
        cls.platform = SimpleNamespace(is_cuda=True, is_sm100=True, is_sm120=False)
        hook_globals = dict(
            **common,
            resolving_view=cls.views["resolving_view"],
            resolved_view=cls.views["resolved_view"],
            declare_resolution=declare,
            get_platform=lambda: cls.platform,
            model_config_of=lambda args: args._model_config,
            is_kimi_k3=cls.predicates["is_kimi_k3"],
            uses_kda_attention=cls.predicates["uses_kda_attention"],
            pp_spec_stable_rows_enabled=consumer._source_function(
                ROOT / "python/sglang/srt/layers/attention/linear/utils.py",
                "pp_spec_stable_rows_enabled",
                dict(envs=cls.envs),
            ),
        )
        cls.export_env = staticmethod(
            consumer._source_function(
                ROOT / "python/sglang/srt/arg_groups/serving_hook.py",
                "handle_environment_variables",
                dict(
                    **hook_globals,
                    envs=cls.envs,
                    handle_multimodal_feature_transport=lambda _: None,
                ),
            )
        )
        divide = _source_nodes(
            "python/sglang/srt/distributed/utils.py",
            ("ensure_divisibility", "divide"),
            {},
        )["divide"]
        cls.params = _source_nodes(
            "python/sglang/srt/configs/mamba_utils.py",
            (
                "Mamba2StateDType",
                "mamba2_state_dtype",
                "BaseLinearStateParams",
                "KimiLinearStateShape",
                "KimiLinearCacheParams",
            ),
            dict(
                **common,
                envs=cls.envs,
                dataclass=dataclasses.dataclass,
                field=dataclasses.field,
                ABC=ABC,
                divide=divide,
            ),
        )
        cls.pool = _source_nodes(
            "python/sglang/srt/mem_cache/memory_pool.py",
            ("get_tensor_size_bytes", "MambaPool"),
            dict(
                **common,
                envs=cls.envs,
                dataclasses=dataclasses,
                dataclass=dataclasses.dataclass,
                fields=dataclasses.fields,
                cached_property=cached_property,
                nullcontext=nullcontext,
                _is_npu=False,
                _is_cpu=True,
                _cpu_has_amx_support=False,
                GB=1024**3,
                GPU_MEMORY_TYPE_KV_CACHE="kv",
                TorchMemorySaverAdapter=SimpleNamespace(
                    create=lambda **_: SimpleNamespace(region=lambda _: nullcontext())
                ),
                maybe_init_custom_mem_pool=lambda **_: (False, None, None),
            ),
        )["MambaPool"]
        cls.algorithm = _source_nodes(
            "python/sglang/srt/speculative/spec_info.py",
            ("SpeculativeAlgorithm",),
            dict(Enum=Enum, auto=auto, CustomSpecAlgo=object),
        )["SpeculativeAlgorithm"]
        cls.backends = _source_nodes(
            "python/sglang/srt/layers/attention/linear/utils.py",
            ("LinearAttnKernelBackend",),
            dict(Enum=Enum),
        )["LinearAttnKernelBackend"]
        cls.dispatcher = _source_nodes(
            "python/sglang/srt/layers/attention/linear/kda_backend.py",
            ("KDAKernelDispatcher",),
            dict(
                torch=torch,
                TritonKDAKernel=_TritonLaunch,
                is_cuda=lambda: True,
                rank0_log=Mock(),
            ),
        )["KDAKernelDispatcher"]
        cls.resolve = staticmethod(
            consumer._source_function(
                ROOT / "python/sglang/srt/arg_groups/attention_hook.py",
                "handle_linear_attn_backend",
                dict(
                    **hook_globals,
                    mamba2_state_dtype=cls.params["mamba2_state_dtype"],
                    SpeculativeAlgorithm=cls.algorithm,
                ),
            )
        )

    def _args(
        self,
        *,
        dtype="float32",
        architecture="KimiK3ForConditionalGeneration",
        hf_dtype="bfloat16",
        **changes,
    ):
        hf_config = SimpleNamespace(
            architectures=[architecture],
            linear_attn_config={"kda_layers": [0]},
        )
        if hf_dtype is not None:
            hf_config.mamba_ssm_dtype = hf_dtype
        args = SimpleNamespace(
            linear_attn_decode_backend="flashinfer",
            linear_attn_backend="triton",
            linear_attn_prefill_backend=None,
            linear_attn_verify_backend=None,
            mamba_ssm_dtype=dtype,
            speculative_algorithm=None,
            enable_linear_replayssm=False,
            enable_linear_replayssm_spec=False,
            enable_torch_compile=False,
            disable_outlines_disk_cache=False,
            enable_deterministic_inference=False,
            debug_cuda_graph=False,
            enable_deepseek_v4_fp4_indexer=False,
            enable_fp4_indexer=False,
            _model_config=SimpleNamespace(hf_config=hf_config),
        )
        vars(args).update(changes)
        return args

    def _state_pool(self, args):
        self.export_env(args)
        # Match production KimiLinearConfig.mamba2_cache_params: it does not
        # pass a dtype, so the field default factory resolves env/default only.
        # Supplying an hf_config-derived dtype here would hide a real mismatch.
        params = self.params["KimiLinearCacheParams"](
            shape=self.params["KimiLinearStateShape"].create(
                tp_world_size=8, num_heads=96, head_dim=128
            ),
            layers=[0],
        )
        return self.pool(
            size=4,
            spec_state_size=0,
            cache_params=params,
            mamba_layer_ids=[0],
            device="cpu",
        )

    def _make_dispatcher(self, args, pool, *, recurrent_available):
        view = self.views["resolving_view"](args)
        recurrent = Mock()
        probe = Mock(return_value=(recurrent_available, recurrent))
        init = consumer._source_function(
            ROOT
            / "python/sglang/srt/layers/attention/linear/kernels/kda_flashinfer.py",
            "__init__",
            dict(_get_flashinfer_kda_kernel=probe, logger=logging.getLogger(__name__)),
        )
        # Keep the real FI constructor and its capability failure gate. Only its
        # external library probe is controlled; the FP32 route must not call it.
        fi_class = type("ActualFlashInferConstructor", (), {"__init__": init})
        create = consumer._source_function(
            ROOT / "python/sglang/srt/layers/attention/linear/kda_backend.py",
            "_create_kernel_dispatcher",
            dict(
                KDAKernelDispatcher=self.dispatcher,
                torch=torch,
                LinearAttnKernelBackend=self.backends,
                get_exec=lambda: SimpleNamespace(mamba=view),
                get_spec=lambda: SimpleNamespace(
                    speculative_algorithm=view.speculative_algorithm
                ),
                get_platform=lambda: self.platform,
                is_kimi_k3=self.predicates["is_kimi_k3"],
                uses_kda_attention=self.predicates["uses_kda_attention"],
                SpeculativeAlgorithm=self.algorithm,
            ),
        )
        owner = SimpleNamespace(
            state_pool=pool,
            _model_runner=SimpleNamespace(model_config=args._model_config),
        )
        selected = SimpleNamespace(
            decode=self.backends(
                view.linear_attn_decode_backend or view.linear_attn_backend
            ),
            prefill=self.backends(
                view.linear_attn_prefill_backend or view.linear_attn_backend
            ),
            verify=self.backends(view.linear_attn_verify_backend or "triton"),
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.layers.attention.linear.kernels.kda_flashinfer": SimpleNamespace(
                    FlashInferKDAKernel=fi_class
                )
            },
        ):
            return create(owner, selected), probe

    def _runtime_scope(self):
        return patch.dict(os.environ, {"SGLANG_OPT_FP8_WO_A_GEMM": "false"})

    def test_resolved_fp32_kimi_pool_reaches_flashinfer_fused_dispatch(self):
        for architecture in (
            "KimiK3ForConditionalGeneration",
            "KimiK31ForConditionalGeneration",
        ):
            with (
                self.subTest(architecture=architecture),
                self._runtime_scope(),
                patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
            ):
                args = self._args(architecture=architecture)
                self.resolve(args)
                view = self.views["resolving_view"](args)
                self.assertEqual(view.linear_attn_verify_backend, "triton")
                pool = self._state_pool(args)
                dispatcher, probe = self._make_dispatcher(
                    args, pool, recurrent_available=False
                )
                self.assertIsInstance(dispatcher.decode_kernel, _TritonLaunch)
                probe.assert_not_called()
                cache = pool.mamba2_layer_cache(0)
                self.assertEqual(cache.temporal.dtype, torch.float32)
                self.assertEqual(cache.conv[0].dtype, torch.bfloat16)
                model, backend, inputs = consumer._make_inputs()
                x, a, beta, _, _, indices, gate = inputs
                inputs = (x, a, beta, cache.conv[0], cache.temporal, indices, gate)
                backend.state_layer_cache = lambda _: cache
                backend.kernel_dispatcher = dispatcher
                kernel = Mock(return_value=torch.full((1, 3, 12, 128), 7.0))
                prepare = consumer._source_function(
                    ROOT / "python/sglang/srt/models/kimi_k3.py",
                    "_prepare_fused_decode",
                    dict(
                        torch=torch,
                        _is_hip=False,
                        _is_npu=False,
                        rank0_log=Mock(),
                        get_exec=lambda: SimpleNamespace(mamba=view),
                        get_spec=lambda: SimpleNamespace(
                            speculative_algorithm=view.speculative_algorithm
                        ),
                        SpeculativeAlgorithm=self.algorithm,
                        get_platform=lambda: SimpleNamespace(is_cuda=True),
                        mamba2_state_dtype=self.params["mamba2_state_dtype"],
                    ),
                )
                with (
                    patch.dict(
                        sys.modules,
                        {
                            consumer.ADAPTER_NAME: consumer.adapter,
                            "sglang.kernels.ops.attention": SimpleNamespace(
                                kda_fused_decode=SimpleNamespace(
                                    SUPPORTED_HEADS={3, 6, 12}
                                )
                            ),
                        },
                    ),
                    patch.object(
                        consumer.adapter, "_get_fused_kda_decode", return_value=kernel
                    ),
                    patch.object(
                        torch.cuda, "get_device_capability", return_value=(10, 3)
                    ),
                ):
                    prepare(model)
                decode, native, _, call = consumer._decode(model, backend, inputs)
                self.assertIs(decode(*call), kernel.return_value)
                native.assert_not_called()
                self.assertIs(kernel.call_args.kwargs["state"], cache.temporal)

    def test_existing_dtype_verify_and_spec_guards_remain(self):
        cases = (
            dict(dtype="float16"),
            dict(architecture="Qwen3NextForCausalLM"),
            dict(architecture="KimiLinearForCausalLM"),
            # Production KimiLinearCacheParams ignores the HF dtype; with no
            # CLI/env value the actual pool is FP32, so this must be rejected
            # instead of being admitted as BF16.
            dict(architecture="KimiLinearForCausalLM", dtype=None),
            dict(speculative_algorithm="DSPARK"),
            dict(linear_attn_verify_backend="flashinfer"),
        )
        with (
            self.envs.SGLANG_ENABLE_PP_SPEC.override(False),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
        ):
            for changes in cases:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    self.resolve(self._args(**changes))
            with (
                self.envs.SGLANG_ENABLE_PP_SPEC.override(True),
                self.assertRaisesRegex(ValueError, "decode-backend flashinfer"),
            ):
                self.resolve(self._args(linear_attn_verify_backend="flashinfer"))
            bf16 = self._args(dtype="bfloat16")
            self.resolve(bf16)
            self.assertIsNone(
                self.views["resolving_view"](bf16).linear_attn_verify_backend
            )
            defaults = self._args(linear_attn_decode_backend=None)
            self.resolve(defaults)
            self.assertIsNone(
                self.views["resolving_view"](defaults).linear_attn_decode_backend
            )
            with self.envs.SGLANG_MAMBA_SSM_DTYPE.override(None):
                # An HF-config BF16 value is not propagated into the Kimi cache
                # params; the actual pool is FP32, so K3 must take the FP32
                # fused route rather than the BF16 recurrent FI route.
                hf_bf16 = self._args(dtype=None)
                self.resolve(hf_bf16)
                self.assertEqual(
                    self.views["resolving_view"](hf_bf16).linear_attn_verify_backend,
                    "triton",
                )
                # The actual pool must also be FP32: this is the mismatch the
                # hf_config value previously hid.
                with self._runtime_scope():
                    hf_bf16_pool = self._state_pool(self._args(dtype=None))
                self.assertEqual(
                    hf_bf16_pool.mamba2_layer_cache(0).temporal.dtype,
                    torch.float32,
                )
                # The default FP32 pool (no CLI flag, no HF value) must reach
                # the explicit FP32 Kimi fused route.
                default_fp32 = self._args(dtype=None, hf_dtype=None)
                self.resolve(default_fp32)
                self.assertEqual(
                    self.views["resolving_view"](
                        default_fp32
                    ).linear_attn_verify_backend,
                    "triton",
                )
                # --speculative-algorithm NONE is a real no-spec value.
                spec_none = self._args(speculative_algorithm="NONE")
                self.resolve(spec_none)
                self.assertEqual(
                    self.views["resolving_view"](spec_none).linear_attn_verify_backend,
                    "triton",
                )
            with self.envs.SGLANG_MAMBA_SSM_DTYPE.override("float32"):
                # The dtype may also arrive through the environment without
                # the CLI flag; that is still the explicit FP32 route.
                env_fp32 = self._args(dtype=None, hf_dtype=None)
                self.resolve(env_fp32)
                self.assertEqual(
                    self.views["resolving_view"](env_fp32).linear_attn_verify_backend,
                    "triton",
                )
            with (
                patch.object(torch.version, "cuda", "12.9"),
                self.assertRaisesRegex(ValueError, "CUDA 13"),
            ):
                self.resolve(self._args(linear_attn_prefill_backend="flashinfer"))

    def test_fp32_fallback_does_not_construct_unused_fi_recurrence(self):
        with (
            self._runtime_scope(),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
        ):
            args = self._args()
            pool = self._state_pool(args)
            dispatcher, probe = self._make_dispatcher(
                args, pool, recurrent_available=False
            )
            self.assertIsInstance(dispatcher.decode_kernel, _TritonLaunch)
            probe.assert_not_called()
            result = dispatcher.decode(
                None,
                None,
                None,
                None,
                None,
                A_log=None,
                dt_bias=None,
                ssm_states=pool.mamba2_layer_cache(0).temporal,
                cache_indices=None,
                query_start_loc=None,
                lower_bound=-5.0,
            )
            self.assertEqual(result, "triton")
            bf16 = self._args(dtype="bfloat16")
            bf16_pool = self._state_pool(bf16)
            with self.assertRaisesRegex(RuntimeError, "not available"):
                self._make_dispatcher(bf16, bf16_pool, recurrent_available=False)
            dispatcher, probe = self._make_dispatcher(
                bf16, bf16_pool, recurrent_available=True
            )
            probe.assert_called_once()
            self.assertIs(
                dispatcher.decode_kernel._recurrent_kda, probe.return_value[1]
            )
            with (
                patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
                self.assertRaisesRegex(RuntimeError, "not available"),
            ):
                self._make_dispatcher(args, pool, recurrent_available=False)


if __name__ == "__main__":
    unittest.main()
