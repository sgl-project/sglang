import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear import gdn_backend, kda_backend
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    GDNKernelDispatcher,
)
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Above the fp16 midpoint 1 + 2^-11 (single-rounds up to 1 + 2^-10) but below
# the bf16 midpoint 1 + 2^-8 (rounds to 1.0, which then stays 1.0 in fp16):
# any fp32 -> bf16 -> fp16 double rounding loses the increment. The 2^-23 tail
# is the last fp32 mantissa bit at 1.x, so the probe is fp32-exact (a 2^-24
# tail would round back to the midpoint itself).
DOUBLE_ROUND_PROBE = 1.0 + 2.0**-11 + 2.0**-23


class TestTrackMambaStateDtype(CustomTestCase):
    """The fp32 track snapshot is cast to the pool dtype exactly once.

    ``_track_mamba_state_extend`` reads the in-kernel fp32 snapshot
    (``h_track_buf``) and casts it to ``ssm_states.dtype`` in a single ``.to``.
    This must hold for every ``--mamba-ssm-dtype``: fp32 keeps full precision,
    bf16 matches the (already correct) legacy path, and fp16 must not inherit
    the old double rounding through the bf16 per-chunk states ``h``.
    """

    @staticmethod
    def _run_track_copy(pool_dtype, h_track_buf, dst_slots, batch_rows):
        metadata = ForwardMetadata(
            has_mamba_track_mask=True,
            # Only numel() gates the copy on this path; the h-row values
            # themselves are unused when h_track_buf is given.
            track_ssm_h_src=torch.zeros(len(dst_slots), dtype=torch.long),
            track_ssm_h_dst=torch.tensor(dst_slots),
            track_ssm_h_batch_src=torch.tensor(batch_rows),
            track_ssm_final_src=torch.empty(0, dtype=torch.long),
            track_ssm_final_dst=torch.empty(0, dtype=torch.long),
            # Required by the dataclass; unused on this path.
            query_start_loc=torch.zeros(1, dtype=torch.int32),
            mamba_cache_indices=torch.zeros(1, dtype=torch.long),
        )
        ssm_states = torch.zeros(8, *h_track_buf.shape[1:], dtype=pool_dtype)
        # The method touches no `self` state; call it unbound so this stays a
        # pure bookkeeping test.
        MambaAttnBackendBase._track_mamba_state_extend(
            None, None, None, ssm_states, metadata, h_track_buf=h_track_buf
        )
        return ssm_states

    def test_snapshot_cast_once_to_pool_dtype(self):
        torch.manual_seed(0)
        h_track_buf = torch.randn(3, 2, 4, 4, dtype=torch.float32)
        h_track_buf[0, 0, 0, 0] = DOUBLE_ROUND_PROBE
        for pool_dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(pool_dtype=pool_dtype):
                ssm_states = self._run_track_copy(
                    pool_dtype, h_track_buf, dst_slots=[5, 2], batch_rows=[0, 2]
                )
                # Single rounding of the fp32 snapshot, in batch-row order.
                self.assertTrue(
                    torch.equal(ssm_states[5], h_track_buf[0].to(pool_dtype))
                )
                self.assertTrue(
                    torch.equal(ssm_states[2], h_track_buf[2].to(pool_dtype))
                )
                untouched = torch.ones(8, dtype=torch.bool)
                untouched[[5, 2]] = False
                self.assertTrue(torch.all(ssm_states[untouched] == 0))

    def test_fp16_pool_is_not_double_rounded_through_bf16(self):
        h_track_buf = torch.full((1, 1, 1, 1), DOUBLE_ROUND_PROBE)
        ssm_states = self._run_track_copy(
            torch.float16, h_track_buf, dst_slots=[3], batch_rows=[0]
        )
        # fp32 -> fp16 rounds the probe UP to 1 + 2^-10; the legacy path
        # (fp32 -> bf16 h -> fp16) collapsed it to exactly 1.0.
        self.assertEqual(ssm_states[3, 0, 0, 0].item(), 1.0 + 2.0**-10)

    def test_no_unaligned_rows_leaves_pool_untouched(self):
        # Aligned-only tracking: the h branch is gated off entirely.
        h_track_buf = torch.randn(2, 1, 1, 1, dtype=torch.float32)
        ssm_states = self._run_track_copy(
            torch.float16, h_track_buf, dst_slots=[], batch_rows=[]
        )
        self.assertTrue(torch.all(ssm_states == 0))

    def test_default_prefill_convolution_hooks_preserve_arguments(self):
        layer = SimpleNamespace(conv_weights=object(), bias=None, activation="silu")
        batch = SimpleNamespace(
            extend_prefix_lens=torch.tensor([0, 64]), extend_seq_lens_cpu=[2, 2]
        )
        qkv = torch.arange(24).view(4, 6)
        states = torch.zeros(8, 6, 2)
        indices = torch.tensor([6, 2])
        for backend_type, module in (
            (GDNAttnBackend, gdn_backend),
            (KDAAttnBackend, kda_backend),
        ):
            with self.subTest(backend=backend_type.__name__):
                backend = object.__new__(backend_type)
                starts = torch.tensor([0, 2, 4])
                backend.forward_metadata = SimpleNamespace(query_start_loc=starts)
                with patch.object(
                    module, "causal_conv1d_fn", return_value=qkv.T
                ) as conv:
                    output = backend._convolve_prefill(
                        layer, batch, qkv, states, indices
                    )
                torch.testing.assert_close(output, qkv)
                args, kwargs = conv.call_args
                torch.testing.assert_close(args[0], qkv.T)
                self.assertIs(args[1], layer.conv_weights)
                self.assertIsNone(args[2])
                self.assertEqual(kwargs["activation"], "silu")
                self.assertIs(kwargs["conv_states"], states)
                self.assertIs(kwargs["cache_indices"], indices)
                self.assertIs(kwargs["query_start_loc"], starts)
                self.assertEqual(kwargs["seq_lens_cpu"], [2, 2])
                torch.testing.assert_close(
                    kwargs["has_initial_state"], torch.tensor([False, True])
                )

    def test_gdn_prefill_hooks_and_checkpoint_routing(self):
        layer = SimpleNamespace(
            layer_id=3,
            q_dim=2,
            k_dim=2,
            v_dim=2,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=1,
            head_q_dim=2,
            head_k_dim=2,
            head_v_dim=2,
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_prefix_lens=torch.tensor([0, 64]),
        )
        qkv = torch.arange(24, dtype=torch.float32).view(4, 6)
        indices = torch.tensor([6, 2])
        metadata = ForwardMetadata(
            query_start_loc=torch.tensor([0, 2, 4]),
            mamba_cache_indices=indices,
            has_mamba_track_mask=True,
            conv_states_mask_indices=torch.tensor([4, 5]),
            track_conv_indices=torch.tensor([[0, 1], [2, 3]]),
            track_ssm_h_src=torch.tensor([0]),
            track_ssm_h_dst=torch.tensor([4]),
            track_ssm_h_batch_src=torch.tensor([1]),
            track_chunk_idx=torch.tensor([-1, 0]),
            track_ssm_final_src=torch.tensor([6]),
            track_ssm_final_dst=torch.tensor([5]),
        )
        for snapshot in (False, True):
            for contiguous in (True, False):
                with self.subTest(snapshot=snapshot, contiguous=contiguous):
                    backend = object.__new__(GDNAttnBackend)
                    backend.forward_metadata = metadata
                    backend.mis_metadata = None
                    backend.requires_contiguous_prefill_state = contiguous
                    conv_states = torch.zeros(16, 6, 2)[::2]
                    ssm_states = torch.zeros(16, 1, 2, 2)[::2]
                    cache = SimpleNamespace(conv=[conv_states], temporal=ssm_states)
                    backend.req_to_token_pool = SimpleNamespace(
                        mamba2_layer_cache=lambda _: cache
                    )
                    expected_indices = torch.arange(2) if contiguous else indices

                    def convolve(_layer, _batch, x, states, slots):
                        torch.testing.assert_close(slots, expected_indices)
                        self.assertEqual(states.is_contiguous(), contiguous)
                        states[slots] = 7
                        return x + 10

                    def extend(q, k, v, g, beta, **kwargs):
                        self.assertEqual(kwargs["layer_id"], layer.layer_id)
                        self.assertIs(
                            kwargs["extend_prefix_lens"], batch.extend_prefix_lens
                        )
                        torch.testing.assert_close(
                            kwargs["cache_indices"], expected_indices
                        )
                        self.assertEqual(
                            kwargs["ssm_states"].is_contiguous(), contiguous
                        )
                        kwargs["ssm_states"][kwargs["cache_indices"]] = 5
                        if snapshot:
                            self.assertIs(
                                kwargs["track_chunk_idx"], metadata.track_chunk_idx
                            )
                            track = kwargs["track_state"]
                            self.assertEqual(track.dtype, torch.float32)
                            self.assertEqual(track.shape, (2, 1, 2, 2))
                            track[0] = 42
                            track[1] = DOUBLE_ROUND_PROBE
                            h = None
                        else:
                            self.assertIsNone(kwargs["track_state"])
                            self.assertIsNone(kwargs["track_chunk_idx"])
                            h = torch.full((1, 1, 1, 2, 2), 2.0)
                        return v, None, h

                    backend._convolve_prefill = Mock(side_effect=convolve)
                    backend._prefill_gates = Mock(return_value=(None, None))
                    backend.kernel_dispatcher = object.__new__(GDNKernelDispatcher)
                    backend.kernel_dispatcher.extend_kernel = SimpleNamespace(
                        supports_track_state_snapshot=snapshot, extend=extend
                    )
                    # Exercise the GPU gather policy with CPU tensors and fake kernels.
                    with (
                        patch.object(gdn_backend, "is_cpu", return_value=False),
                        patch.object(gdn_backend, "is_cuda", return_value=False),
                        patch.object(gdn_backend, "is_hip", return_value=False),
                        patch.object(gdn_backend, "is_xpu", return_value=False),
                        patch.object(gdn_backend, "is_npu", return_value=False),
                    ):
                        output = backend.forward_extend(layer, batch, qkv, None, None)
                    backend._prefill_gates.assert_called_once_with(layer, None, None)
                    torch.testing.assert_close(output.flatten(0, 2), qkv[:, 4:] + 10)
                    torch.testing.assert_close(
                        conv_states[[4, 5]], qkv.view(2, 2, 6).transpose(1, 2)
                    )
                    self.assertTrue(torch.all(conv_states[indices] == 7))
                    self.assertTrue(torch.all(ssm_states[[6, 2, 5]] == 5))
                    expected = DOUBLE_ROUND_PROBE if snapshot else 2.0
                    self.assertTrue(torch.all(ssm_states[4] == expected))


if __name__ == "__main__":
    unittest.main()
