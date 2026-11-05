# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU contracts for the Kimi K3 A1 fail-early runtime guards."""

from __future__ import annotations

import dataclasses
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch
from msgspec.structs import replace

from sglang.kernels.plugin_kernels.nvidia.spec import BuildContext
from sglang.srt.kernel_plugins.nvidia.binding import (
    HybridPoolView,
    HybridStepView,
    KdaPoolView,
    KdaStateView,
    KdaStepView,
    MlaPoolView,
    MlaStepView,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ContractTensor(torch.Tensor):
    """A metadata-only tensor used to test CUDA contracts without a GPU."""

    @staticmethod
    def __new__(
        cls,
        *,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        device: str = "cuda:0",
        contiguous: bool = True,
        pointer: int = 16,
    ):
        value = torch.Tensor._make_subclass(cls, torch.empty(0), False)
        value._contract_dtype = dtype
        value._contract_shape = shape
        value._contract_stride = stride
        value._contract_device = torch.device(device)
        value._contract_contiguous = contiguous
        value._contract_pointer = pointer
        return value

    @property
    def dtype(self) -> torch.dtype:
        return self._contract_dtype

    @property
    def shape(self) -> torch.Size:
        return torch.Size(self._contract_shape)

    @property
    def device(self) -> torch.device:
        return self._contract_device

    def dim(self) -> int:
        return len(self._contract_shape)

    def stride(self, dim: int | None = None):
        if dim is None:
            return self._contract_stride
        return self._contract_stride[dim]

    def is_contiguous(self) -> bool:
        return self._contract_contiguous

    def data_ptr(self) -> int:
        return self._contract_pointer


def _module(name: str, *, package: bool = False) -> ModuleType:
    value = ModuleType(name)
    if package:
        value.__path__ = []
    return value


class TestKimiK3RuntimeCompatibility(CustomTestCase):
    def setUp(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.compat import (
            validate_runtime_compatibility,
        )

        validate_runtime_compatibility.cache_clear()
        self.addCleanup(validate_runtime_compatibility.cache_clear)

    def test_version_mismatches_fail_before_private_imports(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3 import compat

        with patch.object(
            compat.importlib.metadata,
            "version",
            return_value="4.9.0",
        ):
            with self.assertRaisesRegex(
                RuntimeError, "requires nvidia-cutlass-dsl 4.8.0"
            ):
                compat.validate_runtime_compatibility()

        compat.validate_runtime_compatibility.cache_clear()
        versions = {
            "nvidia-cutlass-dsl": "4.8.0",
            "flashinfer-python": "0.7.1",
            "flashinfer-cubin": "0.7.2",
            "flashinfer-jit-cache": "0.7.1+cu130",
        }
        with patch.object(
            compat.importlib.metadata,
            "version",
            side_effect=versions.__getitem__,
        ):
            with self.assertRaisesRegex(RuntimeError, "coherent FlashInfer wheel trio"):
                compat.validate_runtime_compatibility()

    def test_required_runtime_artifact_lists_cannot_be_empty(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.compat import _require_files

        with self.assertRaisesRegex(RuntimeError, "wrapper sources are absent"):
            _require_files((), "wrapper sources")

    def test_complete_private_abi_is_accepted_and_cached(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3 import compat

        cutlass = _module("cutlass", package=True)
        cutlass.__file__ = "/opt/cutlass/__init__.py"
        cute = _module("cutlass.cute", package=True)
        cute.compile = Mock()
        cute.jit = Mock()
        cute.kernel = Mock()
        cute.sym_int = Mock()
        experimental = _module("cutlass.cute.experimental")
        experimental.allocate = Mock()
        experimental.jit = Mock()
        experimental.kernel = Mock()
        experimental.make_tmem_layout_acc = Mock()
        arch = _module("cutlass.cute.arch")
        arch.store_async_dsmem = lambda *args: None
        nvgpu = _module("cutlass.cute.nvgpu", package=True)
        tcgen05 = _module("cutlass.cute.nvgpu.tcgen05")
        tcgen05.CtaGroup = object()
        cutlass_dsl = _module("cutlass.cutlass_dsl")
        cutlass_dsl.dsl_user_op = lambda value: value
        mlir = _module("cutlass._mlir", package=True)
        dialects = _module("cutlass._mlir.dialects", package=True)
        llvm = _module("cutlass._mlir.dialects.llvm")

        cutlass.cute = cute
        cute.experimental = experimental
        cute.arch = arch
        cute.nvgpu = nvgpu
        nvgpu.tcgen05 = tcgen05
        dialects.llvm = llvm

        flashinfer = _module("flashinfer", package=True)
        decode = _module("flashinfer.decode")
        gemm = _module("flashinfer.gemm", package=True)
        kernels = _module("flashinfer.gemm.kernels", package=True)
        splitk = _module("flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk")
        jit = _module("flashinfer.jit")
        utils = _module("flashinfer.utils")

        @dataclasses.dataclass(frozen=True)
        class SplitKTactic:
            mma_m: int
            mma_n: int
            split_k: int
            ab_stages: int

        def remote_store(
            value0,
            value1,
            value2,
            value3,
            smem_ptr,
            mbar_ptr,
            peer_cta_rank_in_cluster,
            loc,
            ip,
        ):
            return None

        def trtllm_mla(
            query,
            kv_cache,
            workspace_buffer,
            qk_nope_head_dim,
            kv_lora_rank,
            qk_rope_head_dim,
            block_tables,
            seq_lens,
            max_seq_len,
            out,
            bmm1_scale,
            bmm2_scale,
            enable_pdl,
            backend,
            multi_ctas_kv_counter_buffer,
        ):
            return None

        splitk.OWNER_RANK = 0
        splitk.SplitKTactic = SplitKTactic
        splitk._store_shared_remote_v4 = remote_store
        decode.trtllm_batch_decode_with_kv_cache_mla = trtllm_mla
        utils.get_trtllm_gen_multi_ctas_kv_counter_bytes = lambda *args: 256
        jit.gen_trtllm_gen_fmha_module = lambda: SimpleNamespace(
            name="trtllm_gen_fmha",
            sources=("/opt/flashinfer/wrapper.cu",),
            aot_artifacts=(SimpleNamespace(path="/opt/flashinfer/wrapper.so"),),
            aot_path="/opt/flashinfer/wrapper.so",
            extra_include_dirs=("/opt/flashinfer/include",),
            extra_cuda_cflags=("-DTLLM_RUBIN_FEATURES",),
        )

        modules = {
            module.__name__: module
            for module in (
                cutlass,
                cute,
                experimental,
                arch,
                nvgpu,
                tcgen05,
                cutlass_dsl,
                mlir,
                dialects,
                llvm,
                flashinfer,
                decode,
                gemm,
                kernels,
                splitk,
                jit,
                utils,
            )
        }
        versions = {
            "nvidia-cutlass-dsl": "4.8.0",
            "flashinfer-python": "0.7.1",
            "flashinfer-cubin": "0.7.1+cu130",
            "flashinfer-jit-cache": "0.7.1+cu130",
        }
        version = Mock(side_effect=versions.__getitem__)
        with (
            patch.dict(sys.modules, modules),
            patch.object(compat.importlib.metadata, "version", version),
            patch.object(compat, "_require_files"),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=148),
            ),
        ):
            first = compat.validate_runtime_compatibility()
            second = compat.validate_runtime_compatibility()

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertEqual(version.call_count, 4)

    def test_build_checks_compatibility_before_resource_construction(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3 import spec

        context = BuildContext(
            widths=(1,),
            tokens_per_request=1,
            deadline_s=2.5,
            binding=object(),
        )
        model = SimpleNamespace(language_model=object())
        events = []

        def compatible():
            events.append("compat")
            return "contract"

        def memory():
            events.append("memory")
            return {}

        def build(*args, **kwargs):
            events.append("build")
            return object()

        with (
            patch.object(
                spec,
                "get_model",
                return_value=SimpleNamespace(context_length=8192),
            ),
            patch(
                "sglang.srt.kernel_plugins.nvidia.kimi_k3.compat.validate_runtime_compatibility",
                side_effect=compatible,
            ),
            patch(
                "sglang.srt.kernel_plugins.nvidia.kimi_k3.providers._cuda_memory",
                side_effect=memory,
            ),
            patch(
                "sglang.srt.kernel_plugins.nvidia.kimi_k3.providers.build_resources",
                side_effect=build,
            ),
        ):
            spec._build(model, None, context)
        self.assertEqual(events, ["compat", "memory", "build"])

        with (
            patch(
                "sglang.srt.kernel_plugins.nvidia.kimi_k3.compat.validate_runtime_compatibility",
                side_effect=RuntimeError("incompatible runtime"),
            ),
            patch(
                "sglang.srt.kernel_plugins.nvidia.kimi_k3.providers._cuda_memory"
            ) as memory_mock,
            patch(
                "sglang.srt.kernel_plugins.nvidia.kimi_k3.providers.build_resources"
            ) as build_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "incompatible runtime"):
                spec._build(model, None, context)
        memory_mock.assert_not_called()
        build_mock.assert_not_called()

    def test_invalid_deadlines_stop_before_compatibility_validation(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3 import spec

        model = SimpleNamespace(language_model=object())
        for value in (None, False, 0, -1, float("nan"), float("inf"), 1e-20, 1e20):
            with self.subTest(value=value):
                context = BuildContext(
                    widths=(1,),
                    tokens_per_request=1,
                    deadline_s=value,
                    binding=object(),
                )
                with patch(
                    "sglang.srt.kernel_plugins.nvidia.kimi_k3.compat.validate_runtime_compatibility"
                ) as compatible:
                    with self.assertRaisesRegex(ValueError, "deadline"):
                        spec._build(model, None, context)
                compatible.assert_not_called()


class TestKimiK3PoolGuards(CustomTestCase):
    def _resources_and_pool(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.providers import (
            PreparedAttention,
            ResourceSet,
        )

        mla = torch.zeros(3, 64, 4)
        conv = torch.zeros(5, 3)
        temporal = torch.zeros(5, 12)
        hybrid = HybridPoolView(
            mla=MlaPoolView(
                reserved_capacity=128,
                serving_capacity=lambda: 128,
                post_capture_active=True,
                page_size=64,
                logical_dtype=torch.float8_e4m3fn,
                storage_dtype=torch.uint8,
                layer_mapping={1: 0},
                caches={1: mla},
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
        mla_core, kda_core = Mock(), Mock()
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
        return resources, hybrid, mla_core, kda_core

    def test_pool_tensor_contract_accepts_exact_metadata_and_names_each_rejection(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.providers import (
            _validate_pool_tensor,
        )

        expected = dict(
            dtype=torch.uint8,
            shape=(3, 64, 576),
            stride=(36864, 576, 1),
            device=torch.device("cuda:0"),
        )
        valid = _ContractTensor(**expected)
        _validate_pool_tensor("MLA cache", valid, **expected)

        cases = (
            ("not a tensor", object(), "not a tensor"),
            (
                "device",
                _ContractTensor(**{**expected, "device": "cuda:1"}),
                "must be on CUDA device",
            ),
            (
                "dtype",
                _ContractTensor(**{**expected, "dtype": torch.int8}),
                "has dtype",
            ),
            (
                "shape",
                _ContractTensor(**{**expected, "shape": (2, 64, 576)}),
                "has shape",
            ),
            (
                "stride",
                _ContractTensor(**{**expected, "stride": (1, 3, 5)}),
                "has stride",
            ),
            (
                "contiguous",
                _ContractTensor(**expected, contiguous=False),
                "not contiguous",
            ),
            (
                "alignment",
                _ContractTensor(**expected, pointer=4),
                "16-byte aligned",
            ),
        )
        for label, value, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_pool_tensor("MLA cache", value, **expected)

        with self.assertRaisesRegex(ValueError, "must be on CUDA device"):
            _validate_pool_tensor(
                "CPU cache",
                torch.zeros(3, 64, 576, dtype=torch.uint8),
                **expected,
            )

    def test_binding_requests_exact_pool_layouts(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3 import providers

        resources, hybrid, _, _ = self._resources_and_pool()
        with patch.object(providers, "_validate_pool_tensor") as validate:
            candidate = resources._binding_candidate(hybrid)

        self.assertEqual(candidate["page_size"], 64)
        self.assertEqual(candidate["logical_dtype"], "torch.float8_e4m3fn")
        self.assertEqual(candidate["storage_dtype"], "torch.uint8")
        self.assertEqual(
            validate.call_args_list,
            [
                call(
                    "layer 1 MLA cache",
                    hybrid.mla.cache(1),
                    dtype=torch.uint8,
                    shape=(3, 64, 576),
                    stride=(36864, 576, 1),
                    device=torch.device("cpu"),
                ),
                call(
                    "layer 0 KDA convolution state",
                    hybrid.kda.state(0).conv[0],
                    dtype=torch.bfloat16,
                    shape=(5, 3, 4608),
                    stride=(13824, 4608, 1),
                    device=torch.device("cpu"),
                ),
                call(
                    "layer 0 KDA temporal state",
                    hybrid.kda.state(0).temporal,
                    dtype=torch.float32,
                    shape=(5, 12, 128, 128),
                    stride=(196608, 16384, 128, 1),
                    device=torch.device("cpu"),
                ),
            ],
        )

    def test_binding_layout_rejection_precedes_provider_preparation(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3 import providers

        resources, hybrid, mla_core, kda_core = self._resources_and_pool()
        with (
            patch.object(
                providers,
                "_validate_pool_tensor",
                side_effect=ValueError("bad pool layout"),
            ),
            patch.object(providers, "_cuda_memory") as memory,
        ):
            with self.assertRaisesRegex(ValueError, "bad pool layout"):
                resources.bind(hybrid)
        mla_core.prepare_cache.assert_not_called()
        kda_core.prepare_caches.assert_not_called()
        memory.assert_not_called()


class TestKimiK3StepGuards(CustomTestCase):
    def _view(self, *, batch: int = 2, pages: int = 8):
        return HybridStepView(
            mla=MlaStepView(
                block_table=_ContractTensor(
                    dtype=torch.int32,
                    shape=(batch, pages),
                    stride=(pages, 1),
                ),
                sequence_lengths=_ContractTensor(
                    dtype=torch.int32, shape=(batch,), stride=(1,)
                ),
                write_slots=_ContractTensor(
                    dtype=torch.int64, shape=(batch,), stride=(1,)
                ),
            ),
            kda=KdaStepView(
                state_indices=_ContractTensor(
                    dtype=torch.int32, shape=(batch,), stride=(1,)
                ),
                track_decode_state=lambda layer_id: None,
            ),
        )

    def test_step_contract_accepts_exact_cuda_metadata(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.executor import (
            _validate_step_inputs,
        )

        batch = 2
        _validate_step_inputs(
            _ContractTensor(dtype=torch.int64, shape=(batch,), stride=(1,)),
            _ContractTensor(
                dtype=torch.bfloat16,
                shape=(batch, 7168),
                stride=(7168, 1),
            ),
            self._view(batch=batch),
            batch=batch,
            device=torch.device("cuda:0"),
        )

    def test_step_contract_rejects_cpu_execution_and_bad_metadata(self):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.executor import (
            _validate_step_inputs,
        )

        batch = 2
        with self.assertRaisesRegex(RuntimeError, "requires CUDA resources"):
            _validate_step_inputs(
                torch.zeros(batch, dtype=torch.int64),
                None,
                self._view(batch=batch),
                batch=batch,
                device=torch.device("cpu"),
            )

        input_ids = _ContractTensor(dtype=torch.int64, shape=(batch,), stride=(1,))
        cases = []
        view = self._view(batch=batch)
        wrong_write = replace(
            view,
            mla=replace(
                view.mla,
                write_slots=_ContractTensor(
                    dtype=torch.int32, shape=(batch,), stride=(1,)
                ),
            ),
        )
        cases.append(
            ("write dtype", input_ids, wrong_write, "MLA write slots has dtype")
        )

        empty_table = replace(
            view,
            mla=replace(
                view.mla,
                block_table=_ContractTensor(
                    dtype=torch.int32, shape=(batch, 0), stride=(1, 1)
                ),
            ),
        )
        cases.append(("empty table", input_ids, empty_table, "nonempty"))

        strided_table = replace(
            view,
            mla=replace(
                view.mla,
                block_table=_ContractTensor(
                    dtype=torch.int32, shape=(batch, 8), stride=(7, 1)
                ),
            ),
        )
        cases.append(("table stride", input_ids, strided_table, "unsupported stride"))

        cases.append(
            (
                "input device",
                _ContractTensor(
                    dtype=torch.int64,
                    shape=(batch,),
                    stride=(1,),
                    device="cuda:1",
                ),
                self._view(batch=batch),
                "input_ids is on",
            )
        )
        for label, ids, view, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, message):
                    _validate_step_inputs(
                        ids,
                        None,
                        view,
                        batch=batch,
                        device=torch.device("cuda:0"),
                    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
