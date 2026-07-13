import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import torch

from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNKernelDispatcher,
    maybe_set_default_flashinfer_gdn_prefill,
)
from sglang.srt.layers.attention.linear.kernels import gdn_prefill
from sglang.srt.layers.attention.linear.kernels.gdn_prefill import GDNQKVShape
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_runner(
    *,
    state_dtype=torch.bfloat16,
    key_dim=128,
    value_dim=128,
    **arg_overrides,
):
    args = SimpleNamespace(
        linear_attn_backend="triton",
        linear_attn_prefill_backend=None,
        uses_mamba_radix_cache=False,
        enable_page_major_kv_layout=False,
        mamba_radix_cache_strategy="no_buffer",
        enable_dynamic_chunking=False,
        chunked_prefill_size=8192,
    )
    for name, value in arg_overrides.items():
        setattr(args, name, value)

    return SimpleNamespace(
        server_args=args,
        hybrid_gdn_config=SimpleNamespace(
            linear_key_head_dim=key_dim,
            linear_value_head_dim=value_dim,
        ),
        req_to_token_pool=SimpleNamespace(
            mamba_pool=SimpleNamespace(
                mamba_cache=SimpleNamespace(temporal=SimpleNamespace(dtype=state_dtype))
            )
        ),
    )


class TestFlashInferGDNPrefillBackendPolicy(CustomTestCase):
    def apply_policy(
        self,
        runner,
        *,
        cuda=True,
        capability=(10, 0),
        cuda_version="13.0",
        flashinfer_available=True,
    ):
        with (
            patch.object(gdn_backend, "is_cuda", return_value=cuda),
            patch.object(torch.cuda, "get_device_capability", return_value=capability),
            patch.object(torch.version, "cuda", cuda_version),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                "is_flashinfer_gdn_prefill_available",
                return_value=flashinfer_available,
            ),
        ):
            maybe_set_default_flashinfer_gdn_prefill(runner)
        return runner.server_args.linear_attn_prefill_backend

    def test_selects_flashinfer_for_supported_sm100_gdn(self):
        self.assertEqual(self.apply_policy(make_runner()), "flashinfer")

    def test_selects_flashinfer_for_no_buffer_radix_cache(self):
        runner = make_runner(
            uses_mamba_radix_cache=True,
            mamba_radix_cache_strategy="no_buffer",
        )
        self.assertEqual(self.apply_policy(runner), "flashinfer")

    def test_preserves_explicit_prefill_override(self):
        for backend in ("triton", "flashinfer", "cutedsl"):
            with self.subTest(backend=backend):
                runner = make_runner(linear_attn_prefill_backend=backend)
                self.assertEqual(self.apply_policy(runner), backend)

    def test_rejects_unsupported_capability(self):
        cases = (
            ("non_cuda", {}, {"cuda": False}),
            ("hopper", {}, {"capability": (9, 0)}),
            ("future_sm", {}, {"capability": (12, 0)}),
            ("cuda_12", {}, {"cuda_version": "12.9"}),
            ("fp32_state", {"state_dtype": torch.float32}, {}),
            ("key_dim", {"key_dim": 64}, {}),
            ("value_dim", {"value_dim": 64}, {}),
            ("missing_api", {}, {"flashinfer_available": False}),
        )
        for name, runner_args, hardware in cases:
            with self.subTest(name=name):
                self.assertIsNone(
                    self.apply_policy(make_runner(**runner_args), **hardware)
                )

    def test_rejects_gdn_config_without_qwen_head_dims(self):
        runner = make_runner()
        runner.hybrid_gdn_config = SimpleNamespace()
        self.assertIsNone(self.apply_policy(runner))

    def test_rejects_unvalidated_runtime_modes(self):
        cases = (
            ("non_triton_base", {"linear_attn_backend": "cutedsl"}),
            ("page_major_kv", {"enable_page_major_kv_layout": True}),
            (
                "extra_buffer",
                {
                    "uses_mamba_radix_cache": True,
                    "mamba_radix_cache_strategy": "extra_buffer",
                },
            ),
            (
                "extra_buffer_lazy",
                {
                    "uses_mamba_radix_cache": True,
                    "mamba_radix_cache_strategy": "extra_buffer_lazy",
                },
            ),
            ("dynamic_chunk", {"enable_dynamic_chunking": True}),
            ("unchunked", {"chunked_prefill_size": -1}),
            ("unknown_chunk", {"chunked_prefill_size": None}),
            ("large_chunk", {"chunked_prefill_size": 8193}),
        )
        for name, runner_args in cases:
            with self.subTest(name=name):
                self.assertIsNone(self.apply_policy(make_runner(**runner_args)))

    def test_tree_verify_uses_triton_kernel(self):
        flashinfer_kernel = MagicMock(supports_target_verify=True)
        with (
            patch.object(gdn_backend, "is_cuda", return_value=True),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                "FlashInferGDNKernel",
                return_value=flashinfer_kernel,
            ),
        ):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.TRITON,
                LinearAttnKernelBackend.FLASHINFER,
            )

        self.assertIsInstance(dispatcher.tree_verify_kernel, TritonGDNKernel)

        tensor = sentinel.tensor
        with patch.object(
            dispatcher.tree_verify_kernel, "target_verify"
        ) as tree_verify:
            dispatcher.target_verify(
                *([tensor] * 7),
                ssm_states=tensor,
                cache_indices=tensor,
                query_start_loc=tensor,
                retrieve_parent_token=sentinel.parent_token,
            )

        tree_verify.assert_called_once()
        flashinfer_kernel.target_verify.assert_not_called()


class TestGDNPrefillPackedDispatch(CustomTestCase):
    def test_delegates_complete_packed_route_to_resolved_kernel(self):
        result = (sentinel.output, sentinel.last_recurrent_state, sentinel.h)
        shape = GDNQKVShape(
            num_q_heads=4,
            num_k_heads=4,
            num_v_heads=8,
            head_q_dim=128,
            head_k_dim=128,
            head_v_dim=128,
        )
        extend_kernel = MagicMock(
            supports_target_verify=True,
            supports_packed_decode=False,
        )
        extend_kernel.extend_packed.return_value = result
        with (
            patch.object(gdn_backend, "is_cuda", return_value=True),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                "FlashInferGDNKernel",
                return_value=extend_kernel,
            ),
        ):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.TRITON,
                LinearAttnKernelBackend.FLASHINFER,
            )

        self.assertIs(dispatcher.extend_kernel, extend_kernel)
        packed_kwargs = dict(
            A_log=sentinel.A_log,
            dt_bias=sentinel.dt_bias,
            shape=shape,
            ssm_states=sentinel.ssm_states,
            cache_indices=sentinel.cache_indices,
            query_start_loc=sentinel.query_start_loc,
            out=sentinel.out,
            prep=sentinel.prep,
            no_prefix=True,
        )
        actual = dispatcher.extend_prefill_from_packed(
            sentinel.mixed_qkv,
            sentinel.a,
            sentinel.b,
            **packed_kwargs,
        )

        self.assertIs(actual, result)
        extend_kernel.extend_packed.assert_called_once_with(
            sentinel.mixed_qkv,
            sentinel.a,
            sentinel.b,
            **packed_kwargs,
        )
        extend_kernel.extend.assert_not_called()

    def test_standard_kernel_uses_split_log_gate_fallback(self):
        total_tokens = 3
        num_q_heads, num_k_heads, num_v_heads = 2, 2, 4
        head_q_dim, head_k_dim, head_v_dim = 8, 8, 16
        shape = GDNQKVShape(
            num_q_heads=num_q_heads,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_q_dim=head_q_dim,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
        )
        mixed_qkv = torch.arange(
            total_tokens * shape.total_dim, dtype=torch.float32
        ).view(total_tokens, shape.total_dim)
        a = torch.randn(total_tokens, num_v_heads)
        b = torch.randn_like(a)
        A_log = torch.randn(num_v_heads)
        dt_bias = torch.randn(num_v_heads)

        dispatcher = GDNKernelDispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.TRITON,
        )
        extend_kernel = dispatcher.extend_kernel
        result = (sentinel.output, sentinel.last_recurrent_state, sentinel.h)

        with (
            patch.object(gdn_prefill, "is_cuda", return_value=False),
            patch.object(gdn_prefill, "is_hip", return_value=False),
            patch.object(extend_kernel, "extend", return_value=result) as extend,
            patch.object(
                gdn_backend,
                "fused_gdn_gating",
                return_value=(sentinel.g, sentinel.beta),
            ) as gating,
        ):
            actual = dispatcher.extend_prefill_from_packed(
                mixed_qkv,
                a,
                b,
                A_log=A_log,
                dt_bias=dt_bias,
                shape=shape,
                ssm_states=sentinel.ssm_states,
                cache_indices=sentinel.cache_indices,
                query_start_loc=sentinel.query_start_loc,
                out=sentinel.out,
                prep=sentinel.prep,
                no_prefix=False,
            )

        self.assertIs(actual, result)
        gating.assert_called_once_with(A_log, a, b, dt_bias)
        extend.assert_called_once()
        call = extend.call_args
        torch.testing.assert_close(
            call.args[0],
            mixed_qkv[:, : shape.q_dim].view(
                1, total_tokens, shape.num_q_heads, shape.head_q_dim
            ),
        )
        torch.testing.assert_close(
            call.args[1],
            mixed_qkv[:, shape.q_dim : shape.q_dim + shape.k_dim].view(
                1, total_tokens, shape.num_k_heads, shape.head_k_dim
            ),
        )
        torch.testing.assert_close(
            call.args[2],
            mixed_qkv[:, shape.q_dim + shape.k_dim :].view(
                1, total_tokens, shape.num_v_heads, shape.head_v_dim
            ),
        )
        self.assertIs(call.args[3], sentinel.g)
        self.assertIs(call.args[4], sentinel.beta)
        self.assertIs(call.kwargs["ssm_states"], sentinel.ssm_states)
        self.assertIs(call.kwargs["cache_indices"], sentinel.cache_indices)
        self.assertIs(call.kwargs["query_start_loc"], sentinel.query_start_loc)
        self.assertIs(call.kwargs["out"], sentinel.out)
        self.assertIs(call.kwargs["prep"], sentinel.prep)
        self.assertIs(call.kwargs["no_prefix"], False)


if __name__ == "__main__":
    unittest.main()
