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
from sglang.srt.server_args import LINEAR_ATTN_KERNEL_BACKEND_CHOICES
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

    def test_prefill_conv_uses_selected_kernel_and_optional_block_table(self):
        backend = object.__new__(GDNAttnBackend)
        backend._causal_conv1d_fn = Mock(side_effect=lambda x, *args, **kwargs: x)
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
        for block_table in (None, torch.tensor([[0, 0]], dtype=torch.int32)):
            with self.subTest(block_table=block_table):
                backend.forward_metadata = SimpleNamespace(
                    query_start_loc=starts, conv_block_table=block_table
                )
                backend._causal_conv1d_fn.reset_mock()
                torch.testing.assert_close(
                    backend._convolve_prefill(layer, batch, x, state, indices), x
                )
                kwargs = backend._causal_conv1d_fn.call_args.kwargs
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
