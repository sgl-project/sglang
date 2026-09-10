"""FlashKDA prefill wrapper vs envelope-strided Mamba state pools (CPU).

Derived property under test: ``FlashKDAKernel`` (the wrapper around the
external, contiguous-only ``flash_kda`` CUTLASS kernel) touches the SSM state
pool ONLY through torch advanced indexing — a gather into a contiguous local
copy before the kernel and a scatter write-back after. Advanced indexing is
layout-agnostic, so the wrapper works unchanged on the envelope-strided
temporal views used by --enable-page-major-kv-layout / --enable-unified-memory
(slot pitch == the multi-layer entry envelope, NOT H*V*K). This insulation is
the justification for allowing prefill=flashkda under the page-major backend
gate without ever teaching the external kernel about strides.

What turns this red: any "optimization" that hands the pool view to
``flash_kda.fwd`` directly, replaces the gather with a ``.view()`` / pointer
reshape that assumes the contiguous slot pitch, or drops the scatter
write-back. On a contiguous pool such a change is invisible; on the strided
pool it mis-addresses state exactly like the chunk_delta_h hardcoded-pitch bug
(GSM8K 0.17).

Runs on CPU — the external kernel is replaced by a stub; only the pool access
pattern (the code under test) executes.

    python -m pytest test/registered/unit/mem_cache/test_flashkda_strided_state_access.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import sys
import types
import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.kda_flashkda import FlashKDAKernel
from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mamba_views,
    mamba_entry_bytes,
)

_DEV = "cpu"

# Tiny KDA-like geometry (multi-layer so the envelope slot pitch != H*V*K).
_LAYERS = 3
_LAYER_UNDER_TEST = 1
_H = 2
_K = 4
_V = 4
_SLOTS = 8
_CONV_SHAPES = (
    (3, 8),
)  # KDA conv layout [kernel-1, dim]; bf16 region pads the envelope
_CONV_DTYPE = torch.bfloat16
_TEMPORAL_DTYPE = torch.float32
# FlashKDA fused-path window: per-seq len must be in [chunk_size, max_seq_len].
_SEQ_LEN = 128


def _make_strided_temporal_views():
    """Envelope-strided conv/temporal views, as UnifiedMambaPool / the
    page-major MambaPool serve them ((num_layers, max_slots, *inner))."""
    entry = mamba_entry_bytes(
        layer_num=_LAYERS,
        conv_state_shapes=_CONV_SHAPES,
        conv_dtype=_CONV_DTYPE,
        temporal_state_shape=(_H, _V, _K),
        temporal_dtype=_TEMPORAL_DTYPE,
    )
    raw = torch.zeros(_SLOTS * entry, dtype=torch.uint8, device=_DEV)
    conv_views, temporal = build_page_major_mamba_views(
        raw,
        layer_num=_LAYERS,
        conv_state_shapes=_CONV_SHAPES,
        conv_dtype=_CONV_DTYPE,
        temporal_state_shape=(_H, _V, _K),
        temporal_dtype=_TEMPORAL_DTYPE,
        max_slots=_SLOTS,
    )
    return conv_views, temporal


class _FakeFlashKDA:
    """Stand-in for the external ``flash_kda`` module. Records what the wrapper
    hands it and applies a deterministic state update so the write-back is
    checkable: final = 2 * initial + 1."""

    def __init__(self):
        self.calls = 0
        self.initial_state_was_contiguous = None
        self.initial_state_copy = None

    def fwd(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale,
        out_buf,
        A_log,
        dt_bias,
        lower_bound,
        *,
        initial_state,
        final_state,
        cu_seqlens,
    ):
        self.calls += 1
        self.initial_state_was_contiguous = initial_state.is_contiguous()
        self.initial_state_copy = initial_state.clone()
        final_state.copy_(initial_state * 2.0 + 1.0)
        out_buf.fill_(0.25)


class TestFlashKDAStridedStateAccess(unittest.TestCase):
    def setUp(self):
        self._saved_module = sys.modules.get("flash_kda")
        self.fake = _FakeFlashKDA()
        mod = types.ModuleType("flash_kda")
        mod.fwd = self.fake.fwd
        sys.modules["flash_kda"] = mod

    def tearDown(self):
        if self._saved_module is None:
            sys.modules.pop("flash_kda", None)
        else:
            sys.modules["flash_kda"] = self._saved_module

    def _run_extend(self, ssm_states, cache_indices):
        num_seqs = cache_indices.numel()
        packed = num_seqs * _SEQ_LEN
        torch.manual_seed(0)
        q = torch.randn(1, packed, _H, _K, dtype=torch.bfloat16)
        k = torch.randn(1, packed, _H, _K, dtype=torch.bfloat16)
        v = torch.randn(1, packed, _H, _V, dtype=torch.bfloat16)
        g = torch.randn(1, packed, _H, _K, dtype=torch.bfloat16)
        beta = torch.rand(1, packed, _H, dtype=torch.bfloat16) * 0.8 + 0.1
        query_start_loc = torch.arange(0, packed + 1, _SEQ_LEN, dtype=torch.int32)
        return FlashKDAKernel().extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=torch.randn(1, 1, _H, 1),
            dt_bias=torch.randn(_H * _K),
            lower_bound=-10.0,  # safe gate => fused path (no triton fallback)
            extend_seq_lens_cpu=[_SEQ_LEN] * num_seqs,
        )

    def test_gather_kernel_scatter_on_envelope_strided_pool(self):
        conv_views, temporal = _make_strided_temporal_views()
        ssm_states = temporal[_LAYER_UNDER_TEST]  # what mamba2_layer_cache serves

        # Precondition of the property: the pool really is envelope-strided.
        self.assertNotEqual(
            ssm_states.stride(0),
            _H * _V * _K,
            "test setup no longer produces a strided pool; the property below "
            "would be vacuous",
        )

        # Distinct value per (layer, slot); sentinel in the conv regions that
        # interleave the temporal regions inside each slot envelope.
        seed = (
            torch.arange(_LAYERS, dtype=torch.float32)[:, None] * 100.0
            + torch.arange(_SLOTS, dtype=torch.float32)[None, :]
        )
        temporal[:] = seed.view(_LAYERS, _SLOTS, 1, 1, 1) + 1.0
        for cv in conv_views:
            cv.fill_(3.0)
        temporal_before = temporal.clone()
        conv_before = [cv.clone() for cv in conv_views]

        cache_indices = torch.tensor([5, 2], dtype=torch.int32)
        out, intermediate_states = self._run_extend(ssm_states, cache_indices)

        # Routing: the fused path ran exactly once (a silent re-route to the
        # triton fallback would make every assertion below vacuous).
        self.assertEqual(self.fake.calls, 1)
        self.assertIsNone(intermediate_states)
        self.assertEqual(tuple(out.shape), (1, 2 * _SEQ_LEN, _H, _V))

        # Gather: the external kernel must receive a CONTIGUOUS copy whose rows
        # are the addressed slots of the strided pool.
        self.assertTrue(self.fake.initial_state_was_contiguous)
        self.assertTrue(
            torch.equal(
                self.fake.initial_state_copy,
                temporal_before[_LAYER_UNDER_TEST][cache_indices.long()],
            ),
            "gather mis-addressed the envelope-strided slots",
        )

        # Scatter: the committed state lands in exactly the addressed slots.
        expected = temporal_before[_LAYER_UNDER_TEST][cache_indices.long()] * 2.0 + 1.0
        self.assertTrue(
            torch.equal(ssm_states[cache_indices.long()], expected),
            "write-back mis-addressed the envelope-strided slots",
        )

        # Isolation: untouched slots of this layer, ALL slots of the other
        # layers, and the interleaved conv regions are byte-identical. A
        # contiguous-pitch (H*V*K) access pattern would corrupt these.
        touched = torch.zeros(_SLOTS, dtype=torch.bool)
        touched[cache_indices.long()] = True
        self.assertTrue(
            torch.equal(
                ssm_states[~touched],
                temporal_before[_LAYER_UNDER_TEST][~touched],
            ),
            "write-back leaked into unaddressed slots",
        )
        for layer in range(_LAYERS):
            if layer == _LAYER_UNDER_TEST:
                continue
            self.assertTrue(
                torch.equal(temporal[layer], temporal_before[layer]),
                f"write-back leaked into layer {layer}'s envelope region",
            )
        for cv, before in zip(conv_views, conv_before):
            self.assertTrue(
                torch.equal(cv, before),
                "write-back leaked into the conv region of the slot envelope",
            )


if __name__ == "__main__":
    unittest.main()
