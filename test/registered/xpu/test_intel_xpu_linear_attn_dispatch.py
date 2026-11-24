"""Intel XPU linear-attn (GDN) backend dispatch: intel_xpu is opt-in only
(default stays triton, like every other platform) and fails fast rather than
silently degrading when misconfigured. Pure dispatch-logic tests -- no XPU
device required -- plus XPU-device tests for the intel_xpu prefill views
path, kept under test/registered/xpu to separate Intel-XPU-only behavior
from the platform-agnostic linear-attn dispatch tests.
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

register_xpu_ci(est_time=5, suite="extra-test-1-gpu-xpu")


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

    @unittest.skipUnless(torch.xpu.is_available(), "requires XPU")
    def test_prefill_views_match_materialized_conv_recurrence_and_norm(self):
        from sgl_kernel.mamba import causal_conv1d_fn_xpu

        from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
        from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
            fused_qkv_split_gdn_prefill,
            qwen3_5_gdn_prefill_projection_views,
        )

        torch.manual_seed(42)
        dtype, key_heads, value_heads, width = torch.bfloat16, 4, 12, 128
        key_dim, value_dim = key_heads * width, value_heads * width
        indices = torch.tensor([1], device="xpu", dtype=torch.int32)
        weights = (
            torch.randn(2 * key_dim + value_dim, 4, device="xpu", dtype=dtype) * 0.1
        )
        norm = RMSNorm(width, device="xpu", dtype=dtype, norm_before_gate=True)
        for length, prefix in ((8192, 0), (2053, 17)):
            with self.subTest(length=length, prefix=prefix):
                packed = torch.randn(
                    length, 2 * key_dim + 2 * value_dim, device="xpu", dtype=dtype
                )
                gates = (
                    torch.randn(length, 2 * value_heads, device="xpu", dtype=dtype)
                    * 0.1
                )
                mixed, gate, beta, alpha = qwen3_5_gdn_prefill_projection_views(
                    packed, gates, key_heads, value_heads, width, width
                )
                original_gate, original_gates = gate.clone(), gates.clone()
                self.assertEqual(
                    mixed.untyped_storage().data_ptr(),
                    packed.untyped_storage().data_ptr(),
                )
                self.assertEqual(
                    gate.untyped_storage().data_ptr(),
                    packed.untyped_storage().data_ptr(),
                )
                self.assertFalse(beta.is_contiguous())
                starts = torch.tensor([0, length], device="xpu", dtype=torch.int32)
                seeded = torch.tensor([prefix > 0], device="xpu")
                conv = (
                    torch.randn(2, mixed.shape[1], 3, device="xpu", dtype=dtype) * 0.1
                )
                ssm = torch.randn(2, value_heads, width, width, device="xpu") * 0.01
                reference_conv, reference_ssm = conv.clone(), ssm.clone()
                outputs = []
                for is_reference, source, conv_state, recurrent_state in (
                    (True, mixed.contiguous(), reference_conv, reference_ssm),
                    (False, mixed, conv, ssm),
                ):
                    convolved = causal_conv1d_fn_xpu(
                        source.transpose(0, 1),
                        weights,
                        conv_states=conv_state,
                        query_start_loc=starts,
                        cache_indices=indices,
                        has_initial_state=seeded,
                        activation="silu",
                    ).transpose(0, 1)
                    query, key, value = fused_qkv_split_gdn_prefill(
                        convolved,
                        key_heads,
                        key_heads,
                        value_heads,
                        width,
                        width,
                        width,
                    )
                    output = torch.empty_like(value)
                    gdn_xpu.gated_delta_rule(
                        output.reshape(length, value_heads, width),
                        query.reshape(length, key_heads, width),
                        key.reshape(length, key_heads, width),
                        value.reshape(length, value_heads, width),
                        beta.contiguous() if is_reference else beta,
                        alpha.contiguous() if is_reference else alpha,
                        torch.zeros(value_heads, device="xpu"),
                        torch.zeros(value_heads, device="xpu", dtype=dtype),
                        recurrent_state,
                        starts,
                        indices,
                        num_prefills=1,
                        num_decodes=0,
                        has_initial_state=seeded,
                        token_indx=torch.arange(length, device="xpu", dtype=torch.int32)
                        if is_reference
                        else None,
                    )
                    output = output.reshape(-1, width)
                    norm_gate = (
                        original_gate.reshape(-1, width) if is_reference else gate
                    )
                    outputs.append(norm(output, norm_gate))
                torch.testing.assert_close(outputs[1], outputs[0], atol=0.05, rtol=0.03)
                torch.testing.assert_close(conv, reference_conv, atol=0, rtol=0)
                torch.testing.assert_close(ssm, reference_ssm, atol=0.0005, rtol=0.03)
                torch.testing.assert_close(gate, original_gate, atol=0, rtol=0)
                torch.testing.assert_close(gates, original_gates, atol=0, rtol=0)

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


class TestQwen35PrefillViewGate(CustomTestCase):
    def test_the_views_follow_the_resolved_prefill_kernel(self):
        """Both --linear-attn-prefill-backend override directions.

        The model's unpack choice reads the per-runner resolved prefill
        backend, so the strided views move with whichever kernel runs extend:
        an intel_xpu prefill override turns them on from a triton base, and a
        triton prefill override turns them off from an intel_xpu base. A gate
        answered from the base --linear-attn-backend leaf would freeze both.
        """
        from sglang.srt.layers.attention.linear.utils import LinearAttnBackends
        from sglang.srt.models.qwen3_5 import _resolved_prefill_is_intel_xpu

        def linear_backend(base, prefill):
            return SimpleNamespace(
                linear_attn_backends=LinearAttnBackends(
                    decode=LinearAttnKernelBackend(base),
                    prefill=LinearAttnKernelBackend(prefill),
                    verify=LinearAttnKernelBackend(base),
                )
            )

        # No override: an intel_xpu runner serves prefill itself.
        self.assertTrue(
            _resolved_prefill_is_intel_xpu(linear_backend("intel_xpu", "intel_xpu"))
        )
        self.assertFalse(
            _resolved_prefill_is_intel_xpu(linear_backend("intel_xpu", "triton"))
        )
        self.assertTrue(
            _resolved_prefill_is_intel_xpu(linear_backend("triton", "intel_xpu"))
        )


class _GdnBoundarySpy:
    """Stands in for RadixLinearAttention: records the unpacked boundary
    tensors and returns a layout-independent core output."""

    def __init__(self, record):
        self.record = record

    def __call__(self, forward_batch, *, mixed_qkv, a, b):
        self.record.append((mixed_qkv, a, b))
        row = torch.tanh(mixed_qkv.to(torch.float32)).sum(dim=-1, keepdim=True)
        return row.to(mixed_qkv.dtype).view(1, -1, 1, 1).expand(1, -1, 12, 128)


@unittest.skipUnless(torch.xpu.is_available(), "requires XPU")
class TestQwen35PrefillViewIntegration(CustomTestCase):
    """The intel_xpu views path beyond a single direct prefill: the model's
    own branch selection, ragged multi-sequence prefill through the SYCL
    conv+recurrence chain, and a speculative verify/commit cycle checked
    against the Triton recurrent reference."""

    def _gated_delta_net_layer(self, record, qkvz, ba):
        from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
        from sglang.srt.models import qwen3_5

        # Bare layer with only what forward() reads: the GEMM pair is the
        # given packed outputs, the RadixLinearAttention boundary is the spy.
        layer = object.__new__(qwen3_5.Qwen3_5GatedDeltaNet)
        torch.nn.Module.__init__(layer)
        layer.num_k_heads, layer.num_v_heads = 4, 12
        layer.attn_tp_size = 1
        layer.head_k_dim = layer.head_v_dim = 128
        layer.key_dim = 4 * 128
        layer.value_dim = 12 * 128
        layer.norm = RMSNorm(
            128, device="xpu", dtype=torch.bfloat16, norm_before_gate=True
        )
        layer.out_proj = lambda x: (x, None)
        layer.attn = _GdnBoundarySpy(record)
        layer._forward_input_proj = lambda _hidden: (qkvz, ba)
        return layer

    def test_model_forward_selects_unpack_by_resolved_prefill_and_mode(self):
        """The unpack the model's forward() picks follows the runner's
        resolved prefill kernel and the forward mode: strided views for an
        intel_xpu prefill extend whatever the base flag says, materialized
        for a triton prefill, and never under target_verify."""
        from sglang.srt.layers.attention.linear.utils import LinearAttnBackends
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang.srt.models import qwen3_5

        torch.manual_seed(3)
        tokens = 64
        qkvz = torch.randn(
            tokens,
            4 * 128 * 2 + 12 * 128 + 12 * 128,
            device="xpu",
            dtype=torch.bfloat16,
        )
        ba = torch.randn(tokens, 24, device="xpu", dtype=torch.bfloat16)
        cases = (
            (
                "triton_base_prefill_override",
                "triton",
                "intel_xpu",
                ForwardMode.EXTEND,
                True,
            ),
            ("intel_xpu_default", "intel_xpu", "intel_xpu", ForwardMode.EXTEND, True),
            (
                "intel_xpu_prefill_to_triton",
                "intel_xpu",
                "triton",
                ForwardMode.EXTEND,
                False,
            ),
            (
                "target_verify",
                "intel_xpu",
                "intel_xpu",
                ForwardMode.TARGET_VERIFY,
                False,
            ),
        )
        # The spy must run the real views function (split_fn returns its
        # tuple), so pass through instead of substituting a bare Mock.
        original_views = qwen3_5.qwen3_5_gdn_prefill_projection_views
        view_calls = []

        def spy_views(*args, **kwargs):
            view_calls.append(True)
            return original_views(*args, **kwargs)

        outputs = {}
        with patch.object(
            qwen3_5, "qwen3_5_gdn_prefill_projection_views", side_effect=spy_views
        ):
            for name, base, prefill, mode, views_expected in cases:
                with self.subTest(case=name):
                    view_calls.clear()
                    record = []
                    layer = self._gated_delta_net_layer(record, qkvz, ba)
                    backend = SimpleNamespace(
                        linear_attn_backends=LinearAttnBackends(
                            decode=LinearAttnKernelBackend(base),
                            prefill=LinearAttnKernelBackend(prefill),
                            verify=LinearAttnKernelBackend(base),
                        )
                    )
                    batch = SimpleNamespace(
                        forward_mode=mode, logical_forward_mode=mode
                    )
                    with forward_context(ForwardContext(attn_backend=backend)):
                        outputs[name] = layer.forward(torch.zeros(1, 1), batch)
                    self.assertEqual(len(view_calls), 1 if views_expected else 0)
                    mixed_qkv, a, b = record[0]
                    # Zero-copy boundaries alias the packed GEMM output; the
                    # materialized boundaries leave it untouched.
                    alias = views_expected
                    self.assertEqual(
                        mixed_qkv.untyped_storage().data_ptr()
                        == qkvz.untyped_storage().data_ptr(),
                        alias,
                    )
                    self.assertEqual(b.is_contiguous(), not alias)
                    self.assertEqual(a.is_contiguous(), not alias)
                    self.assertEqual(mixed_qkv.is_contiguous(), not alias)
        # The branch choice must not move the numbers any extend gets.
        for name in ("triton_base_prefill_override", "intel_xpu_prefill_to_triton"):
            torch.testing.assert_close(outputs[name], outputs["intel_xpu_default"])

    def test_ragged_prefill_views_match_materialized_sycl_chain(self):
        """Three ragged sequences through the real XPU prefill chain (in-place
        SYCL conv, triton split, SYCL recurrence) with strided views against
        the materialized contiguous inputs. A sequence-boundary aliasing bug
        or an OOB token read breaks the outputs or corrupts the state pools
        the views share storage with."""
        from sgl_kernel.mamba import causal_conv1d_fn_xpu

        from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
            fused_qkv_split_gdn_prefill,
            qwen3_5_gdn_prefill_projection_views,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_xpu import XpuGDNKernel

        torch.manual_seed(11)
        k_heads, v_heads, width = 4, 12, 128
        key_dim, value_dim = k_heads * width, v_heads * width
        extend_lens = (97, 7, 313)
        prefix_lens = (0, 5, 33)
        total = sum(extend_lens)
        qkv_dim = 2 * key_dim + value_dim
        # CPU cumsum promotes int32 to int64; the SYCL op expects int32.
        starts = torch.tensor(
            [0] + [sum(extend_lens[: i + 1]) for i in range(len(extend_lens))],
            device="xpu",
            dtype=torch.int32,
        )
        packed = torch.randn(
            total, qkv_dim + value_dim, device="xpu", dtype=torch.bfloat16
        )
        gates = (
            torch.randn(total, 2 * v_heads, device="xpu", dtype=torch.bfloat16) * 0.1
        )
        a_log = (-(torch.rand(v_heads, device="xpu") + 0.1)).float()
        dt_bias = torch.randn(v_heads, device="xpu", dtype=torch.bfloat16) * 0.1
        conv_weight = torch.randn(qkv_dim, 4, device="xpu", dtype=torch.bfloat16)
        slots = torch.tensor([1, 3, 0], device="xpu", dtype=torch.int32)
        has_initial = torch.tensor([p > 0 for p in prefix_lens], device="xpu")
        extend_prefix = torch.tensor(list(prefix_lens), dtype=torch.int32, device="xpu")
        z_before = packed[:, qkv_dim:].clone()
        gates_before = gates.clone()
        kernel = XpuGDNKernel()
        # Both runs start from the same pools: side-specific draw order
        # would differ in the slots no run writes.
        seed_conv = torch.randn(4, qkv_dim, 3, device="xpu", dtype=torch.bfloat16) * 0.1
        seed_ssm = torch.randn(4, v_heads, width, width, device="xpu") * 0.01

        def materialized_inputs():
            mixed = packed[:, :qkv_dim].contiguous()
            z = packed[:, qkv_dim:].contiguous().view(total, v_heads, width)
            b = gates[:, :v_heads].contiguous()
            a = gates[:, v_heads:].contiguous()
            return mixed, z, b, a

        def run_chain(mixed, b, a):
            conv_pool, ssm_pool = seed_conv.clone(), seed_ssm.clone()
            convolved = causal_conv1d_fn_xpu(
                mixed.transpose(0, 1),
                conv_weight,
                conv_states=conv_pool,
                query_start_loc=starts,
                cache_indices=slots,
                has_initial_state=has_initial,
                activation="silu",
            ).transpose(0, 1)
            query, key, value = fused_qkv_split_gdn_prefill(
                convolved, k_heads, k_heads, v_heads, width, width, width
            )
            out, _, _ = kernel.extend(
                query,
                key,
                value,
                None,
                None,
                ssm_states=ssm_pool,
                cache_indices=slots,
                query_start_loc=starts,
                a=a,
                b=b,
                A_log=a_log,
                dt_bias=dt_bias,
                extend_prefix_lens=extend_prefix,
            )
            return out, conv_pool, ssm_pool

        # Materialized side reads the original packed rows into fresh copies.
        mixed, _, b, a = materialized_inputs()
        out_contig, conv_contig, ssm_contig = run_chain(mixed, b, a)

        # Views side owns clones: the in-place SYCL conv writes the views'
        # packed copy, so the original packed rows stay untouched above.
        packed_views, gates_views = packed.clone(), gates.clone()
        mixed, _, b, a = qwen3_5_gdn_prefill_projection_views(
            packed_views, gates_views, k_heads, v_heads, width, width
        )
        out_views, conv_views, ssm_views = run_chain(mixed, b, a)

        torch.testing.assert_close(out_views, out_contig, atol=0.05, rtol=0.03)
        torch.testing.assert_close(conv_views, conv_contig, atol=0, rtol=0)
        torch.testing.assert_close(ssm_views, ssm_contig, atol=0.0005, rtol=0.03)
        # The in-place conv touches only the qkv columns, never the z/b/a
        # slices that share the packed rows the views wrap.
        torch.testing.assert_close(packed_views[:, qkv_dim:], z_before, atol=0, rtol=0)
        torch.testing.assert_close(gates_views, gates_before, atol=0, rtol=0)

    def test_speculative_verify_commit_matches_recurrent_reference(self):
        """A speculative verify/commit cycle on the standalone SYCL verifier:
        the outputs and per-draft-step intermediate states must match the
        Triton recurrent reference over the same draft chain, the committed
        pool rows must land on the accepted-length snapshots, and the
        committed pool itself must be untouched by verify."""
        from sglang.srt.hardware_backend.xpu.kernels.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update as recurrent_update,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_xpu import XpuGDNKernel

        torch.manual_seed(13)
        k_heads, v_heads, width = 4, 12, 128
        draft = 3
        q = torch.randn(1, draft, k_heads, width, device="xpu", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, draft, v_heads, width, device="xpu", dtype=torch.bfloat16)
        b = torch.randn(draft, v_heads, device="xpu", dtype=torch.bfloat16) * 0.5
        a = torch.randn(draft, v_heads, device="xpu", dtype=torch.bfloat16) * 0.5
        a_log = (-(torch.rand(v_heads, device="xpu") + 0.1)).float()
        dt_bias = torch.randn(v_heads, device="xpu", dtype=torch.bfloat16) * 0.1
        pool = torch.randn(8, v_heads, width, width, device="xpu") * 0.01
        pool_seed = pool.clone()
        cache_indices = torch.tensor([1], device="xpu", dtype=torch.int32)
        scratch_rows = torch.tensor([0], device="xpu", dtype=torch.int32)
        query_start_loc = torch.tensor([0, draft], device="xpu", dtype=torch.int32)

        def verify_round():
            # verify never mutates the committed pool, so a fresh scratch
            # buffer per round re-reads the same seed pool row.
            buf = torch.zeros(1, draft, v_heads, width, width, device="xpu")
            out = XpuGDNKernel().target_verify(
                a_log,
                dt_bias,
                q,
                k,
                v,
                a,
                b,
                ssm_states=pool,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=buf,
                intermediate_state_indices=scratch_rows,
                cache_steps=draft,
                retrieve_parent_token=None,
            )
            return out, buf

        out_full, buf_full = verify_round()
        out_part, buf_part = verify_round()

        def recurrent(count):
            ref_pool = pool_seed.clone()
            ref_out = recurrent_update(
                A_log=a_log,
                a=a[:count],
                dt_bias=dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q[:, :count],
                k=k[:, :count],
                v=v[:, :count],
                b=b[:count],
                initial_state_source=ref_pool,
                initial_state_indices=cache_indices,
                use_qk_l2norm_in_kernel=True,
            )
            return ref_out, ref_pool

        # verify rounds fully specified, so reference iterations are the
        # accepted-length prefixes of the same random draft chain.
        ref_full_out, ref_full_pool = recurrent(draft)
        ref_part_out, ref_part_pool = recurrent(2)

        # Commit: scatter_mamba_states_after_mtp_verify copies
        # intermediate[scratch_row, commit_lens - 1] into the work pool row.
        pool[5] = buf_full[0, draft - 1]
        pool[6] = buf_part[0, 1]

        torch.testing.assert_close(out_full, ref_full_out, atol=0.05, rtol=0.03)
        torch.testing.assert_close(out_part[:, :2], ref_part_out, atol=0.05, rtol=0.03)
        torch.testing.assert_close(pool[5], ref_full_pool[1], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(pool[6], ref_part_pool[1], atol=2e-2, rtol=2e-2)
        # verify never writes the committed pool; the scatter owns commits.
        torch.testing.assert_close(pool[1], pool_seed[1], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
