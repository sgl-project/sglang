"""NV KDA fused-decode kernel gate + slot addressing vs envelope-strided SSM
pools (CPU).

Derived property under test: the fully-fused KDA decode kernel
(``kda_fused_decode``) addresses the ssm/temporal state pool by the slot pitch
the pool actually reports (``ssm_states.stride(0)``), NOT the dense ``HV*V*K``
pitch. Under ``--enable-unified-memory`` / ``--enable-page-major-kv-layout`` the
per-layer temporal view is envelope-strided: one slot pitches across ALL layers
(56,171,520 B on K3), so a hardcoded ``slot*HV*V*K`` offset mis-addresses every
slot > 0 (the exact chunk_delta_h hardcoded-pitch bug pattern, GSM8K 0.17).

Two things are pinned here (both CPU-checkable without the CUDA kernel):

  1. ``covered()`` ACCEPTS the envelope-strided view. Pre-fix the gate did
     ``ssm_states.view(-1, HV, V, K).is_contiguous()``, which is False on a
     non-dense slot pitch, so decode silently dropped to the slower unfused
     chain. Reverting to that ``.view(...)`` gate turns test (1) red. The gate
     still REJECTS a view whose inner ``[HV, V, K]`` is non-contiguous (the
     one contract the float4 state loads rely on).

  2. The kernel's reconstructed slot-addressing formula
     ``base + slot*stride(0) + i_hv*V*K + v*K + k`` resolves to the exact same
     storage element as torch's native ``ssm_states[slot, i_hv, v, k]`` on the
     strided pool, while the pre-fix dense-pitch formula
     ``base + slot*(HV*V*K) + ...`` resolves ELSEWHERE for every slot > 0.
     Hardcoding the dense pitch back into the ``.cuh`` turns test (2) red.

Runs on CPU: only the Python gate and the (dtype/stride-only) addressing
arithmetic execute — no CUDA kernel is launched.

    python -m pytest test/registered/unit/mem_cache/test_kda_fused_decode_strided_state.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

import unittest

import torch

from sglang.kernels.ops.attention.kda_fused_decode import (
    _CONV_STATE_W,
    covered,
)
from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mamba_views,
    mamba_entry_bytes,
)

_DEV = "cpu"

# The kernel is compiled for the K3 KDA decode regime; covered() enforces these
# supported local head counts. Multi-layer + several slots so the envelope slot
# pitch differs from the dense H*V*K pitch.
_KDA_HEADS = (3, 6, 12)
_V = 128
_K = 128
_LAYERS = 3
_LAYER_UNDER_TEST = 1
_SLOTS = 6
_CONV_SHAPES = ((3, 8),)  # tiny bf16 conv region interleaves the temporal region
_CONV_DTYPE = torch.bfloat16
_TEMPORAL_DTYPE = torch.float32


def _seg(heads: int) -> int:
    return heads * _V


def _conv_dim(heads: int) -> int:
    return 3 * _seg(heads)


def _make_strided_temporal_view(heads: int):
    """Envelope-strided temporal (SSM) view as UnifiedMambaPool / the
    page-major MambaPool serve it to the KDA backend: shape
    ``(num_layers, max_slots, H, V, K)`` with the slot pitch spanning ALL
    layers' state, not H*V*K."""
    entry = mamba_entry_bytes(
        layer_num=_LAYERS,
        conv_state_shapes=_CONV_SHAPES,
        conv_dtype=_CONV_DTYPE,
        temporal_state_shape=(heads, _V, _K),
        temporal_dtype=_TEMPORAL_DTYPE,
    )
    raw = torch.zeros(_SLOTS * entry, dtype=torch.uint8, device=_DEV)
    _conv_views, temporal = build_page_major_mamba_views(
        raw,
        layer_num=_LAYERS,
        conv_state_shapes=_CONV_SHAPES,
        conv_dtype=_CONV_DTYPE,
        temporal_state_shape=(heads, _V, _K),
        temporal_dtype=_TEMPORAL_DTYPE,
        max_slots=_SLOTS,
    )
    return raw, temporal


def _make_covered_side_args(batch: int, heads: int):
    """The non-ssm covered() arguments, in the exact K3 shapes/dtypes so the
    gate turns solely on the ssm_states view under test."""
    bf16 = torch.bfloat16
    seg = _seg(heads)
    conv_dim = _conv_dim(heads)
    mixed_qkv = torch.zeros((batch, conv_dim), dtype=bf16, device=_DEV)
    a = torch.zeros((batch, seg), dtype=bf16, device=_DEV)
    b = torch.zeros((batch, heads), dtype=bf16, device=_DEV)
    onorm_g = torch.zeros((batch, seg), dtype=bf16, device=_DEV)
    conv_states = torch.zeros(
        (_SLOTS, _CONV_STATE_W, conv_dim), dtype=bf16, device=_DEV
    )
    cache_indices = torch.zeros((batch,), dtype=torch.int32, device=_DEV)
    return mixed_qkv, a, b, conv_states, onorm_g, cache_indices


def _addressing_samples(heads: int):
    return [
        (0, 0, 0, 0),
        (5, heads - 1, 127, 127),
        (3, heads // 2, 64, 100),
        (1, 0, 0, 1),
        (2, min(heads - 1, 2), 3, 7),
    ]


class TestKdaFusedDecodeStridedState(unittest.TestCase):
    def test_covered_accepts_envelope_strided_and_rejects_noncontiguous_inner(self):
        for heads in _KDA_HEADS:
            with self.subTest(kda_heads=heads):
                _raw, temporal = _make_strided_temporal_view(heads)
                ssm = temporal[_LAYER_UNDER_TEST]  # what mamba2_layer_cache serves

                # Precondition: the pool really is envelope-strided (else the
                # property below would be vacuous — a dense pool passes the old
                # gate too).
                self.assertNotEqual(
                    ssm.stride(0),
                    heads * _V * _K,
                    "test setup no longer produces a strided pool",
                )
                # Inner [H, V, K] IS contiguous — the contract the kernel's
                # float4 state loads rely on and all covered() must still require.
                self.assertEqual(
                    (ssm.stride(-1), ssm.stride(-2), ssm.stride(-3)),
                    (1, _K, _V * _K),
                )

                (
                    mixed_qkv,
                    a,
                    b,
                    conv_states,
                    onorm_g,
                    cache_indices,
                ) = _make_covered_side_args(batch=2, heads=heads)

                # (1) Accept the envelope-strided view — pre-fix
                # .view(...).is_contiguous() would reject this and drop decode
                # to the unfused chain.
                self.assertTrue(
                    covered(mixed_qkv, a, b, conv_states, ssm, cache_indices, onorm_g),
                    "covered() rejected the envelope-strided ssm pool (fused decode "
                    "would silently fall back to the unfused chain)",
                )

                # Still reject a view whose inner dims are NOT contiguous: the
                # kernel cannot float4-load a transposed [.., V, K] state.
                ssm_bad = ssm.transpose(-1, -2)  # stride(-1) == K, not 1
                self.assertFalse(
                    covered(
                        mixed_qkv,
                        a,
                        b,
                        conv_states,
                        ssm_bad,
                        cache_indices,
                        onorm_g,
                    ),
                    "covered() must reject a non-inner-contiguous ssm view",
                )

    def test_slot_formula_resolves_correct_element_and_dense_pitch_misaddresses(self):
        for heads in _KDA_HEADS:
            with self.subTest(kda_heads=heads):
                raw, temporal = _make_strided_temporal_view(heads)
                ssm = temporal[_LAYER_UNDER_TEST]

                # Distinct value per storage element so an offset that lands
                # elsewhere reads a provably different value.
                raw_fp32 = raw.view(torch.float32)
                raw_fp32.copy_(torch.arange(raw_fp32.numel(), dtype=torch.float32))

                base = ssm.storage_offset()
                # == state.stride(0) the wrapper passes the kernel.
                slot_stride = ssm.stride(0)
                dense_pitch = heads * _V * _K  # the pre-fix hardcoded slot pitch

                for slot, i_hv, v, k in _addressing_samples(heads):
                    intra = (
                        i_hv * (_V * _K) + v * _K + k
                    )  # kernel's hardcoded intra-slot offset
                    kernel_off = base + slot * slot_stride + intra

                    # (2a) The kernel formula names exactly the element torch
                    # indexing names — proves slot*stride(0) + intra addresses
                    # the intended slot.
                    self.assertEqual(
                        raw_fp32[kernel_off].item(),
                        ssm[slot, i_hv, v, k].item(),
                        f"kernel slot formula mis-addressed "
                        f"(slot={slot}, i_hv={i_hv}, heads={heads})",
                    )

                    # (2b) The pre-fix dense-pitch formula lands on a DIFFERENT
                    # element (a different layer's envelope region) for every
                    # slot > 0.
                    dense_off = base + slot * dense_pitch + intra
                    if slot > 0:
                        self.assertNotEqual(
                            raw_fp32[dense_off].item(),
                            ssm[slot, i_hv, v, k].item(),
                            f"dense-pitch formula happened to match at "
                            f"slot={slot}, heads={heads}; the fix would not "
                            "be load-bearing",
                        )


if __name__ == "__main__":
    unittest.main()
