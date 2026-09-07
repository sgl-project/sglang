import unittest
from unittest.mock import ANY, MagicMock, patch

import torch

from sglang.srt.arg_groups.attention_hook import handle_linear_attn_backend
from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher
from sglang.srt.layers.attention.linear.kernels.kda_helion import HelionKDAKernel
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestHelionKDADispatcher(unittest.TestCase):
    def _make_dispatcher(self, decode_backend, prefill_backend):
        helion_kernel = MagicMock(supports_packed_decode=True)
        with (
            patch(
                "sglang.srt.layers.attention.linear.kda_backend.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.linear.kernels.kda_helion.HelionKDAKernel",
                return_value=helion_kernel,
            ) as constructor,
        ):
            dispatcher = KDAKernelDispatcher(
                decode_backend=decode_backend,
                prefill_backend=prefill_backend,
                verify_backend=LinearAttnKernelBackend.TRITON,
            )
        return dispatcher, helion_kernel, constructor

    def test_combined_backend_reuses_adapter_and_keeps_triton_verify(self):
        dispatcher, helion_kernel, constructor = self._make_dispatcher(
            LinearAttnKernelBackend.HELION,
            LinearAttnKernelBackend.HELION,
        )

        constructor.assert_called_once_with(
            triton_fallback=ANY,
            enable_decode=True,
            enable_prefill=True,
        )
        self.assertIs(dispatcher.decode_kernel, helion_kernel)
        self.assertIs(dispatcher.extend_kernel, helion_kernel)
        self.assertIsInstance(dispatcher.verify_kernel, TritonKDAKernel)
        self.assertTrue(dispatcher.supports_packed_decode)

    def test_decode_only_keeps_triton_prefill_and_verify(self):
        dispatcher, helion_kernel, constructor = self._make_dispatcher(
            LinearAttnKernelBackend.HELION,
            LinearAttnKernelBackend.TRITON,
        )

        constructor.assert_called_once_with(
            triton_fallback=ANY,
            enable_decode=True,
            enable_prefill=False,
        )
        self.assertIs(dispatcher.decode_kernel, helion_kernel)
        self.assertIsInstance(dispatcher.extend_kernel, TritonKDAKernel)
        self.assertIsInstance(dispatcher.verify_kernel, TritonKDAKernel)

    def test_prefill_only_keeps_triton_decode_and_verify(self):
        dispatcher, helion_kernel, constructor = self._make_dispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.HELION,
        )

        constructor.assert_called_once_with(
            triton_fallback=ANY,
            enable_decode=False,
            enable_prefill=True,
        )
        self.assertIsInstance(dispatcher.decode_kernel, TritonKDAKernel)
        self.assertIs(dispatcher.extend_kernel, helion_kernel)
        self.assertIsInstance(dispatcher.verify_kernel, TritonKDAKernel)

    def test_enum_recognizes_helion(self):
        backend = LinearAttnKernelBackend("helion")
        self.assertIs(backend, LinearAttnKernelBackend.HELION)
        self.assertTrue(backend.is_helion())

    def test_replayssm_decode_uses_native_helion_kernel(self):
        kernel = HelionKDAKernel.__new__(HelionKDAKernel)
        kernel._packed_decode = MagicMock()
        kernel._replayssm_decode = MagicMock()
        kernel._triton = MagicMock()
        mixed_qkv = torch.empty(2, 20)
        a = torch.empty(2, 8)
        b = torch.empty(2, 1)
        state = torch.empty(2, 1, 4, 8)
        indices = torch.arange(2, dtype=torch.int32)
        force_flush = torch.zeros(2, dtype=torch.int32)
        replay_args = {
            "replayssm_d": torch.empty(2, 1, 4, 4),
            "replayssm_k": torch.empty(2, 1, 4, 8),
            "replayssm_g": torch.empty(2, 1, 4, 8),
            "replayssm_write_pos": torch.zeros(2, dtype=torch.int32),
            "replayssm_force_flush": force_flush,
        }

        result = kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=torch.empty(1),
            dt_bias=torch.empty(8),
            scale=0.5,
            ssm_states=state,
            cache_indices=indices,
            num_v_heads=1,
            head_v_dim=4,
            lower_bound=-5.0,
            **replay_args,
        )

        self.assertEqual(result.shape, (1, 2, 1, 4))
        kernel._packed_decode.assert_not_called()
        kernel._replayssm_decode.assert_called_once()
        self.assertIs(
            kernel._replayssm_decode.call_args.kwargs["force_flush"], force_flush
        )
        self.assertEqual(kernel._replayssm_decode.call_args.kwargs["lower_bound"], -5.0)
        kernel._triton.packed_decode.assert_not_called()

    def test_packed_decode_forwards_lower_bound(self):
        kernel = HelionKDAKernel.__new__(HelionKDAKernel)
        kernel._packed_decode = MagicMock()
        kernel._triton = MagicMock()
        mixed_qkv = torch.empty(2, 16)
        a = torch.empty(2, 8)
        b = torch.empty(2, 1)
        a_log = torch.empty(1)
        dt_bias = torch.empty(8)
        state = torch.empty(2, 1, 4, 8)
        indices = torch.arange(2, dtype=torch.int32)

        kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=0.5,
            ssm_states=state,
            cache_indices=indices,
            num_v_heads=1,
            head_v_dim=4,
            lower_bound=-5.0,
        )

        kernel._packed_decode.assert_called_once()
        self.assertEqual(kernel._packed_decode.call_args.kwargs["lower_bound"], -5.0)

    def test_replayssm_accepts_helion_and_rejects_other_backends(self):
        with (
            override_platform(is_sm100=False),
            override_platform(is_cuda=False),
        ):
            helion_args = ServerArgs(
                model_path="dummy",
                linear_attn_decode_backend="helion",
                enable_linear_replayssm=True,
            )
            handle_linear_attn_backend(helion_args)

            flashinfer_args = ServerArgs(
                model_path="dummy",
                linear_attn_decode_backend="flashinfer",
                enable_linear_replayssm=True,
            )
            with self.assertRaisesRegex(ValueError, "Triton, or Helion"):
                handle_linear_attn_backend(flashinfer_args)

    def test_explicit_base_backend_is_not_replaced_by_flashinfer(self):
        args = ServerArgs(
            model_path="dummy",
            linear_attn_backend="helion",
            mamba_ssm_dtype="bfloat16",
        )
        with (
            override_platform(is_sm100=True),
            override_platform(is_cuda=False),
        ):
            handle_linear_attn_backend(args)

        self.assertIsNone(args.linear_attn_decode_backend)
        self.assertEqual(args.linear_attn_backend, "helion")


if __name__ == "__main__":
    unittest.main()
