"""CPU contract tests for the FlashInfer CAKE KDA adapter."""

import sys
import unittest
from dataclasses import FrozenInstanceError
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.linear.kernels import kda_flashinfer
from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (
    CakeKDAKernel,
    CakePackedDecodeAdmission,
    CakePackedDecodeReason,
    CakePrefillAdmission,
    CakePrefillReason,
)
from sglang.srt.mem_cache.allocator.mamba import (
    MambaSlotAllocator,
    MambaStateIndexReplayProvenance,
    _issue_state_index_contract,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashInferPackedKDALoader(CustomTestCase):
    def setUp(self):
        self._original_loader_state = (
            kda_flashinfer._flashinfer_packed_kda_available,
            kda_flashinfer._flashinfer_packed_kda_decode,
        )
        self.addCleanup(self._restore_loader_state)
        self._reset_loader_state()

    @staticmethod
    def _reset_loader_state():
        kda_flashinfer._flashinfer_packed_kda_available = None
        kda_flashinfer._flashinfer_packed_kda_decode = None

    def _restore_loader_state(self):
        (
            kda_flashinfer._flashinfer_packed_kda_available,
            kda_flashinfer._flashinfer_packed_kda_decode,
        ) = self._original_loader_state

    @staticmethod
    def _fake_flashinfer_modules(
        packed_decode: Mock,
        version_check: Mock,
        *,
        expose_api: bool = True,
    ) -> dict[str, ModuleType]:
        flashinfer = ModuleType("flashinfer")
        flashinfer.__path__ = []
        if expose_api:
            flashinfer.packed_kda_decode = packed_decode

        jit = ModuleType("flashinfer.jit")
        jit.__path__ = []
        cpp_ext = ModuleType("flashinfer.jit.cpp_ext")
        cpp_ext.is_cuda_version_at_least = version_check
        flashinfer.jit = jit
        jit.cpp_ext = cpp_ext
        return {
            "flashinfer": flashinfer,
            "flashinfer.jit": jit,
            "flashinfer.jit.cpp_ext": cpp_ext,
        }

    def test_loader_enforces_per_arch_cuda_floor_and_caches_result(self):
        for capability, required_version in (((10, 0), "12.8"), ((10, 3), "12.9")):
            with self.subTest(capability=capability):
                self._reset_loader_state()
                packed_decode = Mock(name="packed_kda_decode")
                version_check = Mock(return_value=True)
                modules = self._fake_flashinfer_modules(packed_decode, version_check)
                with (
                    patch.dict(sys.modules, modules),
                    patch.object(kda_flashinfer, "is_cuda", return_value=True),
                    patch.object(
                        torch.cuda,
                        "get_device_capability",
                        return_value=capability,
                    ),
                ):
                    first = kda_flashinfer._get_flashinfer_packed_kda_kernel()
                    second = kda_flashinfer._get_flashinfer_packed_kda_kernel()

                self.assertEqual(first, (True, packed_decode))
                self.assertEqual(second, first)
                version_check.assert_called_once_with(required_version)

    def test_loader_rejects_cuda_below_target_floor(self):
        packed_decode = Mock(name="packed_kda_decode")
        version_check = Mock(return_value=False)
        modules = self._fake_flashinfer_modules(packed_decode, version_check)
        with (
            patch.dict(sys.modules, modules),
            patch.object(kda_flashinfer, "is_cuda", return_value=True),
            patch.object(
                torch.cuda,
                "get_device_capability",
                return_value=(10, 3),
            ),
        ):
            available, loaded = kda_flashinfer._get_flashinfer_packed_kda_kernel()

        self.assertFalse(available)
        self.assertIsNone(loaded)
        version_check.assert_called_once_with("12.9")

    def test_loader_caches_missing_public_api(self):
        packed_decode = Mock(name="packed_kda_decode")
        version_check = Mock(return_value=True)
        modules = self._fake_flashinfer_modules(
            packed_decode,
            version_check,
            expose_api=False,
        )
        with (
            patch.dict(sys.modules, modules),
            patch.object(kda_flashinfer, "is_cuda", return_value=True),
            patch.object(
                torch.cuda,
                "get_device_capability",
                return_value=(10, 0),
            ),
        ):
            first = kda_flashinfer._get_flashinfer_packed_kda_kernel()
            modules["flashinfer"].packed_kda_decode = packed_decode
            second = kda_flashinfer._get_flashinfer_packed_kda_kernel()

        self.assertEqual(first, (False, None))
        self.assertEqual(second, first)
        version_check.assert_not_called()


class TestCakeKDAPrefillCheckpointAdapter(CustomTestCase):
    @staticmethod
    def _selector_tensor(shape, *, dtype=torch.bfloat16, is_cuda=True):
        tensor = Mock()
        tensor.shape = shape
        tensor.ndim = len(shape)
        tensor.dtype = dtype
        tensor.is_cuda = is_cuda
        return tensor

    def test_empty_interior_checkpoint_keeps_prefill_selector_parity(self):
        q = self._selector_tensor((1, 5, 12, 128))
        k = self._selector_tensor((1, 5, 12, 128))
        v = self._selector_tensor((1, 5, 12, 128))
        g = self._selector_tensor((1, 5, 12, 128))
        beta = self._selector_tensor((1, 5, 12))
        query_start_loc = Mock()
        query_start_loc.numel.return_value = 3
        A_log = Mock()
        dt_bias = Mock()

        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
        ):
            without_checkpoint = CakeKDAKernel._cake_prefill_is_supported(
                q,
                k,
                v,
                g,
                beta,
                A_log=A_log,
                dt_bias=dt_bias,
                query_start_loc=query_start_loc,
                lower_bound=-5.0,
                is_spec_decode=False,
                return_intermediate_states=False,
                track_ssm_h_src=None,
            )
            aligned_checkpoint = CakeKDAKernel._cake_prefill_is_supported(
                q,
                k,
                v,
                g,
                beta,
                A_log=A_log,
                dt_bias=dt_bias,
                query_start_loc=query_start_loc,
                lower_bound=-5.0,
                is_spec_decode=False,
                return_intermediate_states=True,
                track_ssm_h_src=torch.empty(0, dtype=torch.int64),
            )
            missing_checkpoint_track = CakeKDAKernel._cake_prefill_is_supported(
                q,
                k,
                v,
                g,
                beta,
                A_log=A_log,
                dt_bias=dt_bias,
                query_start_loc=query_start_loc,
                lower_bound=-5.0,
                is_spec_decode=False,
                return_intermediate_states=True,
                track_ssm_h_src=None,
            )
            interior_checkpoint = CakeKDAKernel._cake_prefill_is_supported(
                q,
                k,
                v,
                g,
                beta,
                A_log=A_log,
                dt_bias=dt_bias,
                query_start_loc=query_start_loc,
                lower_bound=-5.0,
                is_spec_decode=False,
                return_intermediate_states=True,
                track_ssm_h_src=torch.ones(1, dtype=torch.int64),
            )

        self.assertTrue(without_checkpoint)
        self.assertEqual(aligned_checkpoint, without_checkpoint)
        self.assertFalse(missing_checkpoint_track)
        self.assertFalse(interior_checkpoint)

    @staticmethod
    def _prefill_inputs():
        torch.manual_seed(7)
        tokens = 5
        return {
            "q": torch.randn(1, tokens, 12, 128, dtype=torch.bfloat16),
            "k": torch.randn(1, tokens, 12, 128, dtype=torch.bfloat16),
            "v": torch.randn(1, tokens, 12, 128, dtype=torch.bfloat16),
            "g": torch.randn(1, tokens, 12, 128, dtype=torch.bfloat16),
            "beta": torch.rand(1, tokens, 12, dtype=torch.float32),
            "cache_indices": torch.tensor([2, 0], dtype=torch.int32),
            "query_start_loc": torch.tensor([0, 2, 5], dtype=torch.int32),
            "A_log": torch.randn(1, 1, 12, 1, dtype=torch.float32),
            "dt_bias": torch.randn(12 * 128, dtype=torch.float32),
        }

    def test_aligned_checkpoint_uses_same_cake_final_state_bitwise(self):
        kernel = object.__new__(CakeKDAKernel)
        kernel._backend = "cake"
        kernel._gate_cache = {}
        cake_calls = []

        def fake_recurrent_kda(**kwargs):
            cake_calls.append(kwargs)
            return kwargs["v"].clone(), kwargs["initial_state"] + 1

        inputs = self._prefill_inputs()
        state_false = torch.randn(4, 12, 128, 128, dtype=torch.bfloat16)
        state_true = state_false.clone()
        with (
            patch.object(
                kernel,
                "_cake_prefill_admission",
                return_value=CakePrefillAdmission(True, CakePrefillReason.ELIGIBLE),
            ) as selector,
            patch.object(
                kernel,
                "_extend_triton",
                side_effect=AssertionError("aligned checkpoint fell back to Triton"),
            ),
            patch.object(
                kda_flashinfer,
                "_get_flashinfer_kda_prefill_kernel",
                return_value=(True, fake_recurrent_kda),
            ),
            patch.object(kda_flashinfer, "record_kda_terminal_route") as telemetry,
        ):
            output_false = kernel.extend(
                **inputs,
                ssm_states=state_false,
                lower_bound=-5.0,
                return_intermediate_states=False,
                layer_id=7,
            )
            output_true, h_empty = kernel.extend(
                **inputs,
                ssm_states=state_true,
                lower_bound=-5.0,
                return_intermediate_states=True,
                track_ssm_h_src=torch.empty(0, dtype=torch.int64),
                layer_id=7,
            )

        self.assertEqual(selector.call_count, 2)
        self.assertEqual(len(cake_calls), 2)
        self.assertEqual(telemetry.call_count, 2)
        for call in telemetry.call_args_list:
            self.assertEqual(call.kwargs["mode"], "prefill")
            self.assertEqual(call.kwargs["layer_id"], 7)
            self.assertTrue(call.kwargs["eligible"])
            self.assertTrue(call.kwargs["attempted_cake"])
            self.assertTrue(call.kwargs["cake_success"])
            self.assertFalse(call.kwargs["triton_fallback"])
            self.assertFalse(call.kwargs["fatal"])
            self.assertEqual(call.kwargs["reason"], CakePrefillReason.ELIGIBLE)
        self.assertTrue(torch.equal(output_true, output_false))
        self.assertTrue(torch.equal(state_true, state_false))
        self.assertEqual(h_empty.shape, (1, 0, 12, 128, 128))
        self.assertEqual(h_empty.dtype, torch.float32)


class TestCakeKDAIndexedStateAdapter(CustomTestCase):
    @staticmethod
    def _inputs(batch_size=2, num_heads=2, head_dim=4):
        return {
            "q": torch.randn(1, batch_size, num_heads, head_dim).bfloat16(),
            "k": torch.randn(1, batch_size, num_heads, head_dim).bfloat16(),
            "v": torch.randn(1, batch_size, num_heads, head_dim).bfloat16(),
            "a": torch.randn(batch_size, num_heads * head_dim).bfloat16(),
            "b": torch.randn(batch_size, num_heads).bfloat16(),
            "A_log": torch.randn(num_heads).float(),
            "dt_bias": torch.randn(num_heads * head_dim).float(),
            "ssm_states": torch.randn(
                5, num_heads, head_dim, head_dim, dtype=torch.bfloat16
            ),
        }

    @staticmethod
    def _kernel(calls):
        kernel = object.__new__(CakeKDAKernel)

        def fake_recurrent_kda(**kwargs):
            calls.append(kwargs)
            output = torch.zeros(
                kwargs["q"].shape[0],
                kwargs["q"].shape[1],
                kwargs["v"].shape[2],
                kwargs["v"].shape[3],
                dtype=kwargs["v"].dtype,
            )
            return output, None

        kernel._recurrent_kda = fake_recurrent_kda
        return kernel

    def test_direct_path_forwards_pool_and_unmodified_indices(self):
        calls = []
        kernel = self._kernel(calls)
        inputs = self._inputs()
        state_pool = inputs.pop("ssm_states")
        state_indices = torch.tensor([3, -1], dtype=torch.int32)

        with (
            patch.object(
                kernel,
                "_cake_direct_indexed_state_is_supported",
                return_value=True,
            ),
            patch.object(
                torch.Tensor,
                "index_select",
                side_effect=AssertionError("direct CAKE decode gathered state"),
            ),
            patch.object(
                torch.Tensor,
                "index_copy_",
                side_effect=AssertionError("direct CAKE decode scattered state"),
            ),
        ):
            kernel._decode_cake(
                **inputs,
                ssm_states=state_pool,
                cache_indices=state_indices,
                lower_bound=-5.0,
            )

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIs(call["initial_state"], state_pool)
        self.assertIs(call["ssm_state_indices"], state_indices)
        self.assertEqual(call["ssm_state_indices"].tolist(), [3, -1])
        self.assertEqual(call["ssm_state_indices"].dtype, torch.int32)
        self.assertEqual(call["backend"], "cake")
        self.assertFalse(call["output_final_state"])
        self.assertFalse(call["use_gate_in_kernel"])

    def test_unsupported_pool_retains_dense_gather_scatter_fallback(self):
        calls = []
        kernel = self._kernel(calls)
        inputs = self._inputs()
        state_pool = inputs.pop("ssm_states")
        state_before = state_pool.clone()
        state_indices = torch.tensor([3, 1], dtype=torch.int32)

        def update_dense_state(**kwargs):
            calls.append(kwargs)
            kwargs["initial_state"].add_(1)
            output = torch.zeros(
                kwargs["q"].shape[0],
                kwargs["q"].shape[1],
                kwargs["v"].shape[2],
                kwargs["v"].shape[3],
                dtype=kwargs["v"].dtype,
            )
            return output, None

        kernel._recurrent_kda = update_dense_state
        with patch.object(
            kernel,
            "_cake_direct_indexed_state_is_supported",
            return_value=False,
        ):
            kernel._decode_cake(
                **inputs,
                ssm_states=state_pool,
                cache_indices=state_indices,
                lower_bound=-5.0,
            )

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIsNone(call["ssm_state_indices"])
        self.assertTrue(call["initial_state"].is_contiguous())
        self.assertTrue(torch.equal(state_pool[3], state_before[3] + 1))
        self.assertTrue(torch.equal(state_pool[1], state_before[1] + 1))
        for slot in (0, 2, 4):
            self.assertTrue(torch.equal(state_pool[slot], state_before[slot]))


class TestCakeKDAPackedDecodeAdapter(CustomTestCase):
    @staticmethod
    def _inputs(batch_size=2):
        mixed_storage = torch.randn(batch_size, 4608 + 64, dtype=torch.bfloat16)
        return {
            "mixed_qkv": mixed_storage[:, :4608],
            "a": torch.randn(batch_size, 1, 12 * 128, dtype=torch.bfloat16),
            "b": torch.randn(1, batch_size, 12, dtype=torch.bfloat16),
            "A_log": torch.randn(1, 1, 12, 1, dtype=torch.float32),
            "dt_bias": torch.randn(12 * 128, dtype=torch.float32),
            "scale": 128**-0.5,
            "ssm_states": torch.randn(
                batch_size + 3,
                12,
                128,
                128,
                dtype=torch.bfloat16,
            ),
            "cache_indices": torch.tensor([3, -1], dtype=torch.int32),
            "num_v_heads": 12,
            "head_v_dim": 128,
            "lower_bound": -5.0,
        }

    @staticmethod
    def _strided_rows(batch_size, width, row_stride, *, storage_offset=7):
        storage = torch.empty(
            storage_offset + (batch_size - 1) * row_stride + width,
            dtype=torch.bfloat16,
        )
        return storage.as_strided(
            (batch_size, width),
            (row_stride, 1),
            storage_offset=storage_offset,
        )

    @classmethod
    def _row_strided_inputs(cls, batch_size=8):
        mixed_qkv = cls._strided_rows(batch_size, 4608, 4672)
        raw_gate = cls._strided_rows(batch_size, 12 * 128, 1600).unsqueeze(1)
        raw_beta = cls._strided_rows(batch_size, 12, 144).unsqueeze(0)
        return {
            "mixed_qkv": mixed_qkv,
            "a": raw_gate,
            "b": raw_beta,
            "A_log": torch.empty(1, 1, 12, 1, dtype=torch.float32),
            "dt_bias": torch.empty(12 * 128, dtype=torch.float32),
            "scale": 128**-0.5,
            "ssm_states": torch.empty(
                batch_size + 1,
                12,
                128,
                128,
                dtype=torch.bfloat16,
            ),
            "cache_indices": torch.arange(1, batch_size + 1, dtype=torch.int32),
            "num_v_heads": 12,
            "head_v_dim": 128,
            "lower_bound": -5.0,
        }

    @staticmethod
    def _kernel(packed_decode):
        kernel = object.__new__(CakeKDAKernel)
        kernel._packed_kda_decode = packed_decode
        kernel._last_packed_decode_admission = CakePackedDecodeAdmission(
            False, CakePackedDecodeReason.KERNEL_UNAVAILABLE
        )
        return kernel

    @staticmethod
    def _admission(inputs):
        with patch.object(
            CakeKDAKernel,
            "_cake_packed_decode_cuda_device_reason",
            return_value=None,
        ):
            return CakeKDAKernel._cake_packed_decode_admission(**inputs)

    def test_selector_accepts_all_h12_batches_and_beta_stride_144(self):
        for batch_size in (1, 8, 31, 32, 64, 128):
            with self.subTest(batch_size=batch_size):
                inputs = self._row_strided_inputs(batch_size)
                admission = self._admission(inputs)

                self.assertTrue(admission.eligible, admission)
                self.assertEqual(admission.reason, CakePackedDecodeReason.ELIGIBLE)
                self.assertEqual(
                    CakeKDAKernel._cake_packed_row_view(
                        inputs["b"], batch_size=batch_size, row_width=12
                    ).stride(),
                    (144, 1),
                )

    def test_selector_accepts_all_padding_capture_envelope(self):
        inputs = self._row_strided_inputs(batch_size=8)
        inputs["cache_indices"].fill_(-1)

        admission = self._admission(inputs)

        self.assertTrue(admission.eligible, admission)
        self.assertEqual(admission.reason, CakePackedDecodeReason.ELIGIBLE)

    def test_exact_contract_forwards_original_packed_inputs_and_owned_output(self):
        calls = []

        def fake_packed_decode(**kwargs):
            calls.append(kwargs)
            kwargs["output"].fill_(1)
            return kwargs["output"]

        kernel = self._kernel(fake_packed_decode)
        inputs = self._row_strided_inputs(batch_size=8)
        with (
            patch.object(
                CakeKDAKernel,
                "_cake_packed_decode_cuda_device_reason",
                return_value=None,
            ),
            patch.object(
                torch.Tensor,
                "contiguous",
                side_effect=AssertionError("packed CAKE decode copied an input"),
            ),
            patch.object(
                torch.Tensor,
                "index_select",
                side_effect=AssertionError("packed CAKE decode gathered state"),
            ),
            patch.object(
                torch.Tensor,
                "index_copy_",
                side_effect=AssertionError("packed CAKE decode scattered state"),
            ),
            patch.object(kda_flashinfer, "record_kda_terminal_route") as telemetry,
        ):
            output = kernel.packed_decode(**inputs, layer_id=7)

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIs(call["mixed_qkv"], inputs["mixed_qkv"])
        self.assertIs(call["state"], inputs["ssm_states"])
        self.assertIs(call["state_indices"], inputs["cache_indices"])
        self.assertEqual(call["raw_gate"].data_ptr(), inputs["a"].data_ptr())
        self.assertEqual(call["raw_beta"].data_ptr(), inputs["b"].data_ptr())
        self.assertEqual(
            call["raw_gate"].storage_offset(), inputs["a"].storage_offset()
        )
        self.assertEqual(
            call["raw_beta"].storage_offset(), inputs["b"].storage_offset()
        )
        self.assertEqual(call["raw_gate"].stride(), (1600, 1))
        self.assertEqual(call["raw_beta"].stride(), (144, 1))
        self.assertEqual(call["A_log"].data_ptr(), inputs["A_log"].data_ptr())
        self.assertEqual(call["dt_bias"].data_ptr(), inputs["dt_bias"].data_ptr())
        self.assertEqual(call["output"].shape, (8, 1, 12, 128))
        telemetry.assert_called_once_with(
            mode="decode",
            layer_id=7,
            eligible=True,
            attempted_cake=True,
            cake_success=True,
            triton_fallback=False,
            fatal=False,
            reason=CakePackedDecodeReason.ELIGIBLE,
            detail="",
            copy_count=0,
            copy_count_source="static_zero_copy_row_view",
        )
        self.assertEqual(call["output"].dtype, torch.bfloat16)
        self.assertTrue(call["output"].is_contiguous())
        self.assertEqual(output.shape, (1, 8, 12, 128))
        self.assertEqual(output.data_ptr(), call["output"].data_ptr())
        self.assertTrue(torch.equal(output, torch.ones_like(output)))
        self.assertEqual(
            kernel._last_packed_decode_admission.reason,
            CakePackedDecodeReason.ELIGIBLE,
        )

    def test_unsupported_contract_falls_back_to_triton_packed_decode(self):
        packed_decode = Mock(
            side_effect=AssertionError("unsupported contract reached CAKE")
        )
        kernel = self._kernel(packed_decode)
        inputs = self._inputs()
        sentinel = torch.empty(1, 2, 12, 128, dtype=torch.bfloat16)
        with (
            patch.object(
                kernel,
                "_cake_packed_decode_admission",
                return_value=CakePackedDecodeAdmission(
                    False, CakePackedDecodeReason.INNER_STRIDE, "raw_beta"
                ),
            ),
            patch.object(
                kernel,
                "_packed_decode_triton",
                return_value=sentinel,
            ) as fallback,
            patch.object(kda_flashinfer, "record_kda_terminal_route") as telemetry,
        ):
            output = kernel.packed_decode(**inputs, layer_id=7)

        self.assertIs(output, sentinel)
        fallback.assert_called_once()
        packed_decode.assert_not_called()
        self.assertEqual(
            kernel._last_packed_decode_admission.reason,
            CakePackedDecodeReason.INNER_STRIDE,
        )
        telemetry.assert_called_once_with(
            mode="decode",
            layer_id=7,
            eligible=False,
            attempted_cake=False,
            cake_success=False,
            triton_fallback=True,
            fatal=False,
            reason=CakePackedDecodeReason.INNER_STRIDE,
            detail="raw_beta",
        )

    def test_selector_reports_stable_stride_reasons(self):
        batch_size = 8
        inputs = self._row_strided_inputs(batch_size)

        inner_storage = torch.empty(
            (batch_size - 1) * 144 + (12 - 1) * 2 + 1,
            dtype=torch.bfloat16,
        )
        inputs["b"] = inner_storage.as_strided((batch_size, 12), (144, 2)).unsqueeze(0)
        self.assertEqual(
            self._admission(inputs).reason, CakePackedDecodeReason.INNER_STRIDE
        )

        inputs = self._row_strided_inputs(batch_size)
        inputs["b"] = (
            torch.empty(1, 12, dtype=torch.bfloat16).expand(batch_size, 12).unsqueeze(0)
        )
        self.assertEqual(
            self._admission(inputs).reason, CakePackedDecodeReason.ZERO_ROW_STRIDE
        )

        inputs = self._row_strided_inputs(batch_size)
        overlap_storage = torch.empty((batch_size - 1) * 11 + 12, dtype=torch.bfloat16)
        inputs["b"] = overlap_storage.as_strided((batch_size, 12), (11, 1)).unsqueeze(0)
        self.assertEqual(
            self._admission(inputs).reason,
            CakePackedDecodeReason.OVERLAPPING_ROW_STRIDE,
        )

        negative = CakeKDAKernel._cake_row_stride_admission("raw_beta", -1, 12)
        self.assertEqual(negative.reason, CakePackedDecodeReason.NEGATIVE_ROW_STRIDE)
        self.assertEqual(CakePackedDecodeReason.INNER_STRIDE, "inner_stride")
        self.assertEqual(CakePackedDecodeReason.ZERO_ROW_STRIDE, "zero_row_stride")
        self.assertEqual(
            CakePackedDecodeReason.OVERLAPPING_ROW_STRIDE,
            "overlapping_row_stride",
        )
        self.assertEqual(
            CakePackedDecodeReason.NEGATIVE_ROW_STRIDE, "negative_row_stride"
        )

    def test_selector_reports_storage_alias(self):
        inputs = self._row_strided_inputs(batch_size=2)
        inputs["mixed_qkv"] = (
            inputs["ssm_states"].view(-1).as_strided((2, 4608), (4608, 1))
        )

        admission = self._admission(inputs)

        self.assertEqual(admission.reason, CakePackedDecodeReason.STORAGE_ALIAS)
        self.assertEqual(admission.detail, "state:mixed_qkv")

    def test_selector_reports_oob_before_duplicate_indices(self):
        inputs = self._row_strided_inputs(batch_size=2)
        inputs["cache_indices"] = torch.tensor(
            [0, inputs["ssm_states"].shape[0]], dtype=torch.int32
        )
        self.assertEqual(
            self._admission(inputs).reason, CakePackedDecodeReason.CACHE_INDEX_OOB
        )

        inputs["cache_indices"] = torch.tensor([0, 1], dtype=torch.int32)
        self.assertEqual(
            self._admission(inputs).reason, CakePackedDecodeReason.CACHE_INDEX_OOB
        )

        inputs["cache_indices"] = torch.tensor([1, 1], dtype=torch.int32)
        self.assertEqual(
            self._admission(inputs).reason,
            CakePackedDecodeReason.CACHE_INDEX_DUPLICATE,
        )
        self.assertEqual(CakePackedDecodeReason.CACHE_INDEX_OOB, "cache_index_oob")
        self.assertEqual(
            CakePackedDecodeReason.CACHE_INDEX_DUPLICATE,
            "cache_index_duplicate",
        )

    def test_cuda_indices_require_exact_allocator_provenance(self):
        cuda_indices = SimpleNamespace(device=torch.device("cuda"))

        missing = CakeKDAKernel._cake_cache_index_source_admission(
            cuda_indices,
            cache_indices_cpu=None,
            cache_index_contract=None,
            batch_size=2,
            state_slots=3,
        )

        class FakeCudaIndices:
            device = torch.device("cuda")

            @staticmethod
            def data_ptr():
                return 1234

            @staticmethod
            def storage_offset():
                return 0

            @staticmethod
            def numel():
                return 2

        indices = FakeCudaIndices()
        allocator = MambaSlotAllocator(size=2, device="cpu")
        trusted_contract = _issue_state_index_contract(
            allocator,
            indices,
            active_prefix=2,
            state_slots=3,
            active_request_ids=("request-a", "request-b"),
        )
        with self.assertRaisesRegex(ValueError, "distinct active request"):
            _issue_state_index_contract(
                allocator,
                indices,
                active_prefix=2,
                state_slots=3,
                active_request_ids=("request-a", "request-a"),
            )
        with self.assertRaisesRegex(ValueError, "one host request identity"):
            _issue_state_index_contract(
                allocator,
                indices,
                active_prefix=2,
                state_slots=3,
                active_request_ids=("unexpanded-topk-request",),
            )
        with self.assertRaisesRegex(ValueError, "Unknown.*replay producer"):
            _issue_state_index_contract(
                allocator,
                indices,
                active_prefix=2,
                state_slots=3,
                active_request_ids=("request-a", "request-b"),
                replay_provenance=MambaStateIndexReplayProvenance(),
            )
        trusted = CakeKDAKernel._cake_cache_index_source_admission(
            indices,
            cache_indices_cpu=None,
            cache_index_contract=trusted_contract,
            batch_size=2,
            state_slots=3,
        )
        lookalike_tensor = CakeKDAKernel._cake_cache_index_source_admission(
            FakeCudaIndices(),
            cache_indices_cpu=None,
            cache_index_contract=trusted_contract,
            batch_size=2,
            state_slots=3,
        )
        wrong_state_envelope = CakeKDAKernel._cake_cache_index_source_admission(
            indices,
            cache_indices_cpu=None,
            cache_index_contract=trusted_contract,
            batch_size=2,
            state_slots=4,
        )

        self.assertEqual(missing.reason, CakePackedDecodeReason.CACHE_INDEX_UNVERIFIED)
        self.assertIsNone(trusted)
        with self.assertRaises(FrozenInstanceError):
            trusted_contract.active_prefix = 1
        self.assertEqual(
            lookalike_tensor.reason,
            CakePackedDecodeReason.CACHE_INDEX_UNVERIFIED,
        )
        self.assertEqual(
            wrong_state_envelope.reason,
            CakePackedDecodeReason.CACHE_INDEX_UNVERIFIED,
        )
        self.assertEqual(
            CakePackedDecodeReason.CACHE_INDEX_UNVERIFIED,
            "cache_index_unverified",
        )

    def test_invalid_indices_fall_back_with_zero_cake_activity(self):
        packed_decode = Mock(side_effect=AssertionError("invalid indices reached CAKE"))
        kernel = self._kernel(packed_decode)
        inputs = self._row_strided_inputs(batch_size=2)
        inputs["cache_indices"] = torch.tensor([1, 1], dtype=torch.int32)
        sentinel = torch.empty(1, 2, 12, 128, dtype=torch.bfloat16)
        with (
            patch.object(
                CakeKDAKernel,
                "_cake_packed_decode_cuda_device_reason",
                return_value=None,
            ),
            patch.object(
                kernel, "_packed_decode_triton", return_value=sentinel
            ) as fallback,
        ):
            output = kernel.packed_decode(**inputs, layer_id=7)

        self.assertIs(output, sentinel)
        self.assertEqual(fallback.call_count, 1)
        packed_decode.assert_not_called()
        self.assertEqual(
            kernel._last_packed_decode_admission.reason,
            CakePackedDecodeReason.CACHE_INDEX_DUPLICATE,
        )

    def test_selector_rejects_batch_outside_cuda_grid_y(self):
        class OversizedMixedQKV:
            ndim = 2
            shape = (65536, 4608)

        self.assertFalse(
            CakeKDAKernel._cake_packed_decode_is_supported(
                OversizedMixedQKV(),
                None,
                None,
                A_log=None,
                dt_bias=None,
                scale=128**-0.5,
                ssm_states=None,
                cache_indices=None,
                num_v_heads=12,
                head_v_dim=128,
                lower_bound=-5.0,
            )
        )

    def test_replayssm_triton_safe_gate_error_propagates(self):
        packed_decode = Mock(side_effect=AssertionError("ReplaySSM reached CAKE"))
        kernel = self._kernel(packed_decode)
        inputs = self._inputs()
        replay_args = {
            "replayssm_d": torch.empty(5, 12, 4, 128, dtype=torch.bfloat16),
            "replayssm_k": torch.empty(5, 12, 4, 128, dtype=torch.bfloat16),
            "replayssm_g": torch.empty(5, 12, 4, 128, dtype=torch.bfloat16),
            "replayssm_write_pos": torch.zeros(2, dtype=torch.int32),
            "replayssm_force_flush": torch.zeros(2, dtype=torch.bool),
        }
        safe_gate_error = (
            "KDA safe gate (lower_bound) is not implemented in the ReplaySSM "
            "decode kernel"
        )
        with (
            patch.object(
                kernel,
                "_packed_decode_triton",
                side_effect=NotImplementedError(safe_gate_error),
            ) as fallback,
            patch.object(kda_flashinfer, "record_kda_terminal_route") as telemetry,
            self.assertRaisesRegex(NotImplementedError, "KDA safe gate"),
        ):
            kernel.packed_decode(**replay_args, **inputs, layer_id=7)

        fallback.assert_called_once()
        for name, tensor in replay_args.items():
            self.assertIs(fallback.call_args.kwargs[name], tensor)
        self.assertEqual(fallback.call_args.kwargs["lower_bound"], -5.0)
        packed_decode.assert_not_called()
        self.assertEqual(
            kernel._last_packed_decode_admission.reason,
            CakePackedDecodeReason.REPLAYSSM_REQUESTED,
        )
        telemetry.assert_called_once_with(
            mode="decode",
            layer_id=7,
            eligible=False,
            attempted_cake=False,
            cake_success=False,
            triton_fallback=False,
            fatal=True,
            reason="triton_fallback_exception",
            detail="builtins.NotImplementedError",
        )


if __name__ == "__main__":
    unittest.main()
