"""Intel XPU linear-attn (GDN) backend dispatch: intel_xpu is opt-in only
(default stays triton, like every other platform) and fails fast rather than
silently degrading when misconfigured. Pure dispatch-logic tests -- no XPU
device required -- kept under test/registered/xpu to separate Intel-XPU-only
behavior from the platform-agnostic linear-attn dispatch tests.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch

from sglang.srt.arg_groups.choices import LINEAR_ATTN_KERNEL_BACKEND_CHOICES
from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    GDNKernelDispatcher,
)
from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher
from sglang.srt.layers.attention.linear.kernels import gdn_xpu
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.kernels.gdn_xpu import XpuGDNKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=5, suite="stage-b-test-1-gpu-xpu")


class TestIntelXpuGDNDispatch(CustomTestCase):
    def test_intel_xpu_is_a_registered_backend_choice(self):
        self.assertIn("intel_xpu", LINEAR_ATTN_KERNEL_BACKEND_CHOICES)

    def test_intel_xpu_requires_xpu_hardware(self):
        with patch.object(gdn_backend, "is_xpu", return_value=False):
            with self.assertRaisesRegex(ValueError, "requires Intel XPU"):
                GDNKernelDispatcher(
                    LinearAttnKernelBackend.INTEL_XPU,
                    LinearAttnKernelBackend.TRITON,
                )
            with self.assertRaisesRegex(ValueError, "requires Intel XPU"):
                GDNKernelDispatcher(
                    LinearAttnKernelBackend.TRITON,
                    LinearAttnKernelBackend.INTEL_XPU,
                )

    def test_intel_xpu_selects_standalone_kernels_by_phase(self):
        with patch.object(gdn_backend, "is_xpu", return_value=True):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.INTEL_XPU,
                LinearAttnKernelBackend.INTEL_XPU,
                LinearAttnKernelBackend.INTEL_XPU,
            )

        self.assertIsInstance(dispatcher.decode_kernel, XpuGDNKernel)
        self.assertIsInstance(dispatcher.extend_kernel, XpuGDNKernel)
        self.assertIsInstance(dispatcher.verify_kernel, XpuGDNKernel)
        self.assertIsInstance(dispatcher.tree_verify_kernel, XpuGDNKernel)

    def test_intel_xpu_verify_defaults_to_triton_when_not_selected(self):
        with patch.object(gdn_backend, "is_xpu", return_value=True):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.INTEL_XPU,
                LinearAttnKernelBackend.INTEL_XPU,
            )

        self.assertIsInstance(dispatcher.verify_kernel, TritonGDNKernel)

    @unittest.skipUnless(gdn_backend.is_xpu(), "Requires XPU platform")
    def test_triton_gdn_backend_still_uses_xpu_conv(self):
        from sgl_kernel.mamba import causal_conv1d_fn_xpu

        GDNKernelDispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.TRITON,
        )
        self.assertIs(gdn_backend.causal_conv1d_fn, causal_conv1d_fn_xpu)
        self.assertIs(
            gdn_backend.causal_conv1d_update,
            gdn_xpu.causal_conv1d_update_xpu_dispatch,
        )

    def test_conv_verify_adapter_preserves_linear_and_tree_links(self):
        with patch.object(gdn_xpu, "causal_conv1d_update_verify_xpu") as kernel:
            for links in (
                (None, None, None),
                (sentinel.next_token, sentinel.next_sibling, sentinel.parent_token),
            ):
                with self.subTest(links=links):
                    kernel.reset_mock()
                    result = gdn_xpu.causal_conv1d_verify_xpu(
                        sentinel.x,
                        sentinel.conv_state,
                        sentinel.weight,
                        sentinel.bias,
                        "silu",
                        conv_state_indices=sentinel.cache_indices,
                        intermediate_conv_window=sentinel.window,
                        intermediate_state_indices=sentinel.intermediate_indices,
                        retrieve_next_token=links[0],
                        retrieve_next_sibling=links[1],
                        retrieve_parent_token=links[2],
                    )
                    self.assertIs(result, kernel.return_value)
                    kernel.assert_called_once_with(
                        sentinel.x,
                        sentinel.conv_state,
                        sentinel.weight,
                        sentinel.window,
                        sentinel.intermediate_indices,
                        *links,
                        bias=sentinel.bias,
                        activation="silu",
                        cache_indices=sentinel.cache_indices,
                    )

    def test_conv_update_dispatches_decode_and_verify(self):
        with (
            patch.object(gdn_xpu, "causal_conv1d_update_xpu") as decode,
            patch.object(gdn_xpu, "causal_conv1d_verify_xpu") as verify,
        ):
            args = (
                sentinel.x,
                sentinel.conv_state,
                sentinel.weight,
                sentinel.bias,
                "silu",
            )
            result = gdn_xpu.causal_conv1d_update_xpu_dispatch(
                *args, conv_state_indices=sentinel.cache_indices
            )
            self.assertIs(result, decode.return_value)
            decode.assert_called_once_with(
                *args, conv_state_indices=sentinel.cache_indices
            )
            verify.assert_not_called()

            for links in (
                (None, None, None),
                (sentinel.next_token, sentinel.next_sibling, sentinel.parent_token),
            ):
                with self.subTest(links=links):
                    verify.reset_mock()
                    result = gdn_xpu.causal_conv1d_update_xpu_dispatch(
                        *args,
                        conv_state_indices=sentinel.cache_indices,
                        intermediate_conv_window=sentinel.window,
                        intermediate_state_indices=sentinel.intermediate_indices,
                        retrieve_next_token=links[0],
                        retrieve_next_sibling=links[1],
                        retrieve_parent_token=links[2],
                    )
                    self.assertIs(result, verify.return_value)
                    verify.assert_called_once_with(
                        *args,
                        conv_state_indices=sentinel.cache_indices,
                        intermediate_conv_window=sentinel.window,
                        intermediate_state_indices=sentinel.intermediate_indices,
                        retrieve_next_token=links[0],
                        retrieve_next_sibling=links[1],
                        retrieve_parent_token=links[2],
                    )
            decode.assert_called_once()

    def test_unaligned_tracked_prefill_uses_triton_intermediate_state(self):
        kernel = XpuGDNKernel()
        self.assertFalse(kernel.supports_track_state_snapshot)
        q = k = v = torch.zeros(1, 100, 1, 2)
        indices = torch.tensor([0], dtype=torch.int32)
        starts = torch.tensor([0, 100], dtype=torch.int32)
        h = torch.zeros(1, 2, 1, 2, 2)
        with (
            patch.object(
                TritonGDNKernel, "extend", return_value=(v, None, h)
            ) as triton,
            patch.object(gdn_xpu, "gated_delta_rule") as sycl,
        ):
            result = kernel.extend(
                q,
                k,
                v,
                sentinel.g,
                sentinel.beta,
                ssm_states=sentinel.states,
                cache_indices=indices,
                query_start_loc=starts,
                a=sentinel.a,
                b=sentinel.b,
                needs_track_state=True,
            )
        self.assertIs(result[2], h)
        triton.assert_called_once()
        sycl.assert_not_called()

    @unittest.skipUnless(torch.xpu.is_available(), "requires XPU")
    def test_tracked_prefill_returns_real_intermediate_state(self):
        kernel = XpuGDNKernel()
        q = torch.randn(1, 80, 1, 128, device="xpu", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        g = torch.full((1, 80, 1), -0.5, device="xpu")
        beta = torch.full_like(g, 0.3)
        cache_indices = torch.tensor([0], device="xpu", dtype=torch.int32)
        ssm_states = torch.zeros(1, 1, 128, 128, device="xpu")

        output, final, h = kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=torch.tensor([0, 80], device="xpu", dtype=torch.int32),
            needs_track_state=True,
        )
        self.assertEqual(output.shape, v.shape)
        self.assertIsNone(final)
        self.assertEqual(h.shape, (1, 2, 1, 128, 128))

        prefix_states = torch.zeros_like(ssm_states)
        _, _, prefix_h = kernel.extend(
            q[:, :64],
            k[:, :64],
            v[:, :64],
            g[:, :64],
            beta[:, :64],
            ssm_states=prefix_states,
            cache_indices=cache_indices,
            query_start_loc=torch.tensor([0, 64], device="xpu", dtype=torch.int32),
            needs_track_state=True,
        )
        torch.testing.assert_close(h[:, 0], prefix_h[:, 0])
        # h[:, 1] is the state after 64 tokens, the checkpoint prefix caching needs.
        self.assertGreater(h[:, 1].abs().max().item(), 0)
        torch.testing.assert_close(h[:, 1].float(), prefix_states, atol=2e-2, rtol=2e-2)

    def test_plain_prefill_keeps_sycl_kernel(self):
        kernel = XpuGDNKernel()
        q = k = v = torch.zeros(1, 100, 1, 2)
        indices = torch.tensor([0], dtype=torch.int32)
        starts = torch.tensor([0, 100], dtype=torch.int32)
        with (
            patch.object(TritonGDNKernel, "extend") as triton,
            patch.object(gdn_xpu, "gated_delta_rule") as sycl,
        ):
            result = kernel.extend(
                q,
                k,
                v,
                sentinel.g,
                sentinel.beta,
                ssm_states=sentinel.states,
                cache_indices=indices,
                query_start_loc=starts,
                a=sentinel.a,
                b=sentinel.b,
                A_log=sentinel.A_log,
                dt_bias=sentinel.dt_bias,
                extend_prefix_lens=torch.tensor([0, 3]),
                needs_track_state=False,
            )
        self.assertEqual(result[0].shape, v.shape)
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])
        sycl.assert_called_once()
        torch.testing.assert_close(
            sycl.call_args.kwargs["has_initial_state"], torch.tensor([False, True])
        )
        triton.assert_not_called()

    def test_decode_clamps_padded_state_indices(self):
        kernel = XpuGDNKernel()
        x = torch.zeros(1, 3, 1, 2)
        indices = torch.tensor([2, -1, -1], dtype=torch.int32)
        states = torch.arange(5, dtype=torch.float32)

        def update(*args, **kwargs):
            for index in args[10]:
                states[index] += 10

        with patch.object(gdn_xpu, "gated_delta_rule", side_effect=update):
            kernel.decode(
                x,
                x,
                x,
                x,
                x,
                A_log=sentinel.A_log,
                dt_bias=sentinel.dt_bias,
                ssm_states=states,
                cache_indices=indices,
                query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            )
        self.assertEqual(indices.tolist(), [2, -1, -1])
        self.assertEqual(states.tolist(), [20, 1, 12, 3, 4])

    def test_target_verify_clamps_padded_seed_indices(self):
        kernel = XpuGDNKernel()
        x = torch.zeros(1, 3, 1, 2)
        starts = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        indices = torch.tensor([4, -1, -1, -1], dtype=torch.int32)
        inter = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        with patch.object(gdn_xpu, "gated_delta_rule_verify") as verify:
            kernel.target_verify(
                sentinel.A_log,
                sentinel.dt_bias,
                x,
                x,
                x,
                sentinel.a,
                sentinel.b,
                ssm_states=sentinel.states,
                cache_indices=indices,
                query_start_loc=starts,
                intermediate_states_buffer=sentinel.buf,
                intermediate_state_indices=inter,
                cache_steps=1,
                retrieve_parent_token=None,
            )
        seeds = verify.call_args.args[9]
        self.assertEqual(seeds.tolist(), [4, 0, 0])
        self.assertEqual(indices.tolist(), [4, -1, -1, -1])

    def test_sycl_prefill_defers_gates_and_fallback_computes_them(self):
        kernel = XpuGDNKernel()
        self.assertTrue(kernel.defers_prefill_gates)
        self.assertFalse(kernel.supports_strided_target_verify_qkv)
        q = k = v = torch.zeros(1, 100, 1, 2)
        indices = torch.tensor([0], dtype=torch.int32)
        starts = torch.tensor([0, 100], dtype=torch.int32)
        with (
            patch.object(
                TritonGDNKernel, "extend", return_value=(v, None, None)
            ) as triton,
            patch.object(
                gdn_xpu, "fused_gdn_gating", return_value=(sentinel.g, sentinel.beta)
            ) as gating,
        ):
            kernel.extend(
                q,
                k,
                v,
                None,
                None,
                ssm_states=sentinel.states,
                cache_indices=indices,
                query_start_loc=starts,
                a=sentinel.a,
                b=sentinel.b,
                A_log=sentinel.A_log,
                dt_bias=sentinel.dt_bias,
                needs_track_state=True,
            )
        gating.assert_called_once_with(
            sentinel.A_log, sentinel.a, sentinel.b, sentinel.dt_bias
        )
        self.assertIs(triton.call_args.args[3], sentinel.g)
        self.assertIs(triton.call_args.args[4], sentinel.beta)

    def test_prefill_conv_uses_device_kernel_and_optional_block_table(self):
        backend = object.__new__(GDNAttnBackend)
        x = torch.zeros(3, 4)
        state = torch.zeros(1, 4, 2)
        weight = torch.zeros(4, 3)
        layer = SimpleNamespace(conv_weights=weight, bias=None, activation="silu")
        batch = SimpleNamespace(
            extend_prefix_lens=torch.zeros(1, dtype=torch.int32),
            extend_seq_lens_cpu=[3],
        )
        starts = torch.tensor([0, 3], dtype=torch.int32)
        indices = torch.tensor([0], dtype=torch.int32)
        kernel = Mock(side_effect=lambda x, *args, **kwargs: x)
        with patch.object(gdn_backend, "causal_conv1d_fn", kernel):
            for block_table in (None, torch.tensor([[0, 0]], dtype=torch.int32)):
                with self.subTest(block_table=block_table):
                    backend.forward_metadata = SimpleNamespace(
                        query_start_loc=starts, conv_block_table=block_table
                    )
                    kernel.reset_mock()
                    torch.testing.assert_close(
                        backend._convolve_prefill(layer, batch, x, state, indices), x
                    )
                    kwargs = kernel.call_args.kwargs
                    self.assertEqual("block_table" in kwargs, block_table is not None)
                    if block_table is not None:
                        self.assertIs(kwargs["block_table"], block_table)
                    self.assertIs(kwargs["conv_states"], state)
                    self.assertIs(kwargs["cache_indices"], indices)


class TestIntelXpuKDADispatch(CustomTestCase):
    def test_intel_xpu_is_not_a_supported_kda_backend(self):
        # Unlike GDN, KDA has no Intel XPU SYCL kernel: intel_xpu must not be
        # silently treated as Triton, it should fail fast like any other
        # backend KDA does not implement.
        with self.assertRaisesRegex(ValueError, "Unsupported KDA decode backend"):
            KDAKernelDispatcher(
                decode_backend=LinearAttnKernelBackend.INTEL_XPU,
                prefill_backend=LinearAttnKernelBackend.TRITON,
                verify_backend=LinearAttnKernelBackend.TRITON,
            )
        with self.assertRaisesRegex(ValueError, "Unsupported KDA prefill backend"):
            KDAKernelDispatcher(
                decode_backend=LinearAttnKernelBackend.TRITON,
                prefill_backend=LinearAttnKernelBackend.INTEL_XPU,
                verify_backend=LinearAttnKernelBackend.TRITON,
            )


if __name__ == "__main__":
    unittest.main()
