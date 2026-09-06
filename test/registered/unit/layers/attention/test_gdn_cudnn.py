import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.arg_groups.attention_hook import handle_linear_attn_backend
from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import GDNKernelDispatcher
from sglang.srt.layers.attention.linear.kernels import gdn_cudnn
from sglang.srt.layers.attention.linear.kernels.gdn_cudnn import CudnnGDNKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import LINEAR_ATTN_KERNEL_BACKEND_CHOICES, ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestCudnnGDNKernel(CustomTestCase):
    def test_backend_is_a_public_choice(self):
        self.assertIn("cudnn", LINEAR_ATTN_KERNEL_BACKEND_CHOICES)
        self.assertTrue(LinearAttnKernelBackend("cudnn").is_cudnn())

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

    def test_cudnn_prefill_rejects_context_parallelism(self):
        cases = (
            {"linear_attn_prefill_backend": "cudnn"},
            {"linear_attn_backend": "cudnn"},
        )
        for fields in cases:
            with self.subTest(fields=fields):
                args = ServerArgs(
                    model_path="dummy",
                    enable_prefill_cp=True,
                    cp_strategy="zigzag",
                    attn_cp_size=2,
                    mamba_ssm_dtype="float32",
                    **fields,
                )
                with (
                    override_platform(is_sm100=False),
                    override_platform(is_cuda=False),
                ):
                    with self.assertRaisesRegex(
                        ValueError, "does not support prefill context parallelism"
                    ):
                        handle_linear_attn_backend(args)

    def test_cudnn_prefill_allows_tensor_parallelism(self):
        args = ServerArgs(
            model_path="dummy",
            linear_attn_prefill_backend="cudnn",
            linear_attn_decode_backend="triton",
            tp_size=2,
            enable_prefill_cp=False,
            mamba_ssm_dtype="float32",
        )
        with (
            override_platform(is_sm100=False),
            override_platform(is_cuda=False),
        ):
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

    @unittest.skipUnless(torch.cuda.is_available(), "requires an NVIDIA CUDA GPU")
    def test_cudnn_prefill_matches_default_gdn_backend(self):
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor
        if sm not in gdn_cudnn._SUPPORTED_SMS:
            self.skipTest(f"cuDNN FROST GDN does not support SM{sm}")

        # Qwen3.5 TP2 local shape: the model's 16 GDN heads are split into
        # eight heads per rank. Unequal sequence lengths exercise the packed
        # THD boundary adapter; non-adjacent slots exercise state gather/scatter.
        seq_lens = (95, 129)
        total_tokens = sum(seq_lens)
        num_heads = 8
        head_dim = 128
        pool_size = 4
        device = "cuda"
        generator = torch.Generator(device=device).manual_seed(42)

        def randn(*shape, dtype=torch.bfloat16):
            return torch.randn(
                *shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )

        q = randn(1, total_tokens, num_heads, head_dim)
        k = randn(1, total_tokens, num_heads, head_dim)
        v = randn(1, total_tokens, num_heads, head_dim)
        g = -torch.nn.functional.softplus(
            randn(1, total_tokens, num_heads, dtype=torch.float32)
        )
        beta = torch.sigmoid(randn(1, total_tokens, num_heads, dtype=torch.float32))
        query_start_loc = torch.tensor(
            [0, seq_lens[0], total_tokens], device=device, dtype=torch.int32
        )
        cache_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
        initial_states = (
            randn(
                pool_size,
                num_heads,
                head_dim,
                head_dim,
                dtype=torch.float32,
            )
            * 0.05
        )
        triton_states = initial_states.clone()
        cudnn_states = initial_states.clone()

        default_dispatcher = GDNKernelDispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.TRITON,
        )
        cudnn_dispatcher = GDNKernelDispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.CUDNN,
        )
        default_output = default_dispatcher.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=triton_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )[0]
        cudnn_output = cudnn_dispatcher.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=cudnn_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )[0]
        torch.cuda.synchronize()

        tol = 5e-2
        for name, actual, expected in (
            ("output", cudnn_output, default_output),
            (
                "state",
                cudnn_states[cache_indices],
                triton_states[cache_indices],
            ),
        ):
            torch.testing.assert_close(
                actual,
                expected,
                atol=tol,
                msg=f"cuDNN/default GDN {name} error must be < {tol}",
            )

        untouched_slots = torch.tensor([0, 2], device=device)
        torch.testing.assert_close(
            cudnn_states[untouched_slots], initial_states[untouched_slots]
        )


if __name__ == "__main__":
    unittest.main()
