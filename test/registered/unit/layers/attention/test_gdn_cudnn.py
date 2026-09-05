import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from packaging.version import Version

from sglang.srt.arg_groups.attention_hook import handle_linear_attn_backend
from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import GDNKernelDispatcher
from sglang.srt.layers.attention.linear.kernels import gdn_cudnn
from sglang.srt.layers.attention.linear.kernels.gdn_cudnn import CudnnGDNKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import LINEAR_ATTN_KERNEL_BACKEND_CHOICES, ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCudnnGDNKernel(CustomTestCase):
    def test_backend_is_a_public_choice(self):
        self.assertIn("cudnn", LINEAR_ATTN_KERNEL_BACKEND_CHOICES)
        self.assertTrue(LinearAttnKernelBackend("cudnn").is_cudnn())

    def test_runtime_requires_frontend_128_and_cutlass_47(self):
        supported = (Version("1.28.0"), Version("4.7.0"))
        with (
            patch.object(gdn_cudnn, "_distribution_version", side_effect=supported),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
        ):
            gdn_cudnn._validate_cudnn_gdn_runtime()

        with patch.object(
            gdn_cudnn,
            "_distribution_version",
            return_value=Version("1.27.0"),
        ):
            with self.assertRaisesRegex(RuntimeError, "frontend>=1.28.0"):
                gdn_cudnn._validate_cudnn_gdn_runtime()

        with (
            patch.object(
                gdn_cudnn,
                "_distribution_version",
                side_effect=(Version("1.28.0"), Version("4.6.2")),
            ),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
        ):
            with self.assertRaisesRegex(RuntimeError, "cutlass-dsl>=4.7.0"):
                gdn_cudnn._validate_cudnn_gdn_runtime()

    def test_dispatcher_uses_cudnn_for_prefill_only(self):
        cudnn_kernel = MagicMock(uses_state_checkpoints=True)
        with (
            patch.object(gdn_backend, "is_cuda", return_value=True),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_cudnn.CudnnGDNKernel",
                return_value=cudnn_kernel,
            ),
        ):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.TRITON,
                LinearAttnKernelBackend.CUDNN,
            )
        self.assertIs(dispatcher.extend_kernel, cudnn_kernel)

        with self.assertRaisesRegex(ValueError, "prefill-only"):
            GDNKernelDispatcher(
                LinearAttnKernelBackend.CUDNN,
                LinearAttnKernelBackend.TRITON,
            )

    def test_base_backend_keeps_cudnn_prefill_and_resolves_triton_decode(self):
        args = ServerArgs(
            model_path="dummy",
            linear_attn_backend="cudnn",
            mamba_ssm_dtype="float32",
        )
        with (
            override_platform(is_sm100=False),
            override_platform(is_cuda=False),
        ):
            handle_linear_attn_backend(args)

        self.assertEqual(args.linear_attn_backend, "cudnn")
        self.assertEqual(resolved_view(args).linear_attn_decode_backend, "triton")

    def test_explicit_cudnn_decode_is_rejected(self):
        args = ServerArgs(
            model_path="dummy",
            linear_attn_decode_backend="cudnn",
        )
        with (
            override_platform(is_sm100=False),
            override_platform(is_cuda=False),
        ):
            with self.assertRaisesRegex(ValueError, "prefill-only"):
                handle_linear_attn_backend(args)

    def test_checkpoint_plan_keeps_native_indices(self):
        kernel = object.__new__(CudnnGDNKernel)
        source_indices = torch.tensor([1, 4], dtype=torch.int64)
        metadata = SimpleNamespace(
            track_ssm_h_src=source_indices,
            state_checkpoint_every_n_tokens=0,
        )
        with patch.object(gdn_cudnn, "mamba_cache_chunk_size", return_value=64):
            kernel.prepare_state_checkpoint_plan(None, metadata, "cpu")

        self.assertIs(metadata.track_ssm_h_src, source_indices)
        self.assertEqual(metadata.state_checkpoint_every_n_tokens, 64)

    def test_extend_adapts_pool_state_to_cudnn_thd_api(self):
        kernel = object.__new__(CudnnGDNKernel)
        captured = {}
        total_tokens = 3
        q_heads = 1
        value_heads = 2
        dim = 128

        q = torch.randn(1, total_tokens, q_heads, dim, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, total_tokens, value_heads, dim, dtype=torch.bfloat16)
        g = torch.randn(1, total_tokens, value_heads, dtype=torch.float32)
        beta = torch.rand_like(g)
        states = torch.stack(
            [torch.full((value_heads, dim, dim), float(slot)) for slot in range(4)]
        )
        cache_indices = torch.tensor([1, -1], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1, 3], dtype=torch.int64)
        checkpoints = torch.randn(2, value_heads, dim, dim, dtype=torch.bfloat16)

        def fake_gated_delta_net(**kwargs):
            captured.update(kwargs)
            output = torch.randn(total_tokens, value_heads, dim, dtype=torch.bfloat16)
            return output, kwargs["initial_state"] + 10, checkpoints

        kernel._gated_delta_net = fake_gated_delta_net
        output, last_state, h = kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            state_checkpoint_every_n_tokens=64,
            batch_invariant=True,
        )

        self.assertEqual(output.shape, (1, total_tokens, value_heads, dim))
        self.assertIsNone(last_state)
        torch.testing.assert_close(h, checkpoints.unsqueeze(0))
        self.assertEqual(captured["q"].shape, (total_tokens, q_heads, dim))
        self.assertEqual(captured["v"].shape, (total_tokens, value_heads, dim))
        self.assertEqual(captured["g"].dtype, torch.float32)
        self.assertEqual(captured["beta"].dtype, torch.float32)
        self.assertEqual(captured["cu_seqlens"].dtype, torch.int32)
        self.assertEqual(captured["plan_name"], "gdn_frost")
        self.assertTrue(captured["use_qk_l2norm_in_kernel"])
        self.assertEqual(captured["checkpoint_every_n_tokens"], 64)
        self.assertTrue(captured["batch_invariant"])
        torch.testing.assert_close(
            captured["initial_state"][:, 0, 0, 0], torch.tensor([1.0, 3.0])
        )
        torch.testing.assert_close(states[[1, 3], 0, 0, 0], torch.tensor([11.0, 13.0]))

    def test_extend_rejects_non_fp32_state(self):
        kernel = object.__new__(CudnnGDNKernel)
        tensor = torch.empty(1, 1, 1, 128, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "float32 recurrent states"):
            kernel.extend(
                tensor,
                tensor,
                tensor,
                torch.empty(1, 1, 1),
                torch.empty(1, 1, 1),
                ssm_states=torch.empty(1, 1, 128, 128, dtype=torch.bfloat16),
                cache_indices=torch.tensor([0]),
                query_start_loc=torch.tensor([0, 1]),
            )


if __name__ == "__main__":
    unittest.main()
