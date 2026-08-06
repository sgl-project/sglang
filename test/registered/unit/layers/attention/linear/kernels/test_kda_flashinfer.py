"""CPU contract tests for the FlashInfer CAKE KDA adapter."""

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.linear.kernels import kda_flashinfer
from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import CakeKDAKernel
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
    def _kernel(packed_decode):
        kernel = object.__new__(CakeKDAKernel)
        kernel._packed_kda_decode = packed_decode
        return kernel

    def test_exact_contract_forwards_original_packed_inputs_and_owned_output(self):
        calls = []

        def fake_packed_decode(**kwargs):
            calls.append(kwargs)
            kwargs["output"].fill_(1)
            return kwargs["output"]

        kernel = self._kernel(fake_packed_decode)
        inputs = self._inputs()
        with (
            patch.object(
                kernel,
                "_cake_packed_decode_is_supported",
                return_value=True,
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
        ):
            output = kernel.packed_decode(**inputs)

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIs(call["mixed_qkv"], inputs["mixed_qkv"])
        self.assertIs(call["state"], inputs["ssm_states"])
        self.assertIs(call["state_indices"], inputs["cache_indices"])
        self.assertEqual(call["raw_gate"].data_ptr(), inputs["a"].data_ptr())
        self.assertEqual(call["raw_beta"].data_ptr(), inputs["b"].data_ptr())
        self.assertEqual(call["A_log"].data_ptr(), inputs["A_log"].data_ptr())
        self.assertEqual(call["dt_bias"].data_ptr(), inputs["dt_bias"].data_ptr())
        self.assertEqual(call["output"].shape, (2, 1, 12, 128))
        self.assertEqual(call["output"].dtype, torch.bfloat16)
        self.assertTrue(call["output"].is_contiguous())
        self.assertEqual(output.shape, (1, 2, 12, 128))
        self.assertEqual(output.data_ptr(), call["output"].data_ptr())
        self.assertTrue(torch.equal(output, torch.ones_like(output)))

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
                "_cake_packed_decode_is_supported",
                return_value=False,
            ),
            patch.object(
                kernel,
                "_packed_decode_triton",
                return_value=sentinel,
            ) as fallback,
        ):
            output = kernel.packed_decode(**inputs)

        self.assertIs(output, sentinel)
        fallback.assert_called_once()
        packed_decode.assert_not_called()

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
                "_cake_packed_decode_is_supported",
                return_value=True,
            ),
            patch.object(
                kernel,
                "_packed_decode_triton",
                side_effect=NotImplementedError(safe_gate_error),
            ) as fallback,
            self.assertRaisesRegex(NotImplementedError, "KDA safe gate"),
        ):
            kernel.packed_decode(**replay_args, **inputs)

        fallback.assert_called_once()
        for name, tensor in replay_args.items():
            self.assertIs(fallback.call_args.kwargs[name], tensor)
        self.assertEqual(fallback.call_args.kwargs["lower_bound"], -5.0)
        packed_decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
