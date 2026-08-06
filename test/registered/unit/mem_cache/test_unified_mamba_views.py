# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Round-trip correctness of ``UnifiedKVPool._build_mamba_views`` — the
envelope-strided conv/temporal (SSM) state views that back ``UnifiedMambaPool``.

This isolates the unified-memory-pool Mamba STATE layout from the full model. It guards
against a class of correctness defect where Falcon-H1 greedy decode is garbled
under the unified memory pool, isolated to the Mamba conv/temporal state path: a
stride/offset/alignment bug in the view construction (analogous to the fixed
`_extract_kv_strides` MHA bug).

Within one slot's envelope the bytes are
``[conv[0]·L0 | conv[0]·L1 | ... | conv[1]·L0 | ... | temporal·L0 | ...]`` and
across slots the layout is envelope (slot stride == entry_bytes). Each returned
view is ``(num_layers, max_slots, *inner_shape)``. The conv dtype (bf16, 2 B)
and temporal dtype (fp32, 4 B) DIFFER, so the temporal view's byte offset must
be a multiple of the temporal itemsize — an alignment hazard that
``_build_mamba_views`` now asserts.

These tests prove the views:
  - round-trip every (tensor, layer, slot) element with the Falcon-like
    bf16-conv / fp32-temporal dtype mix (catches stride/offset/alignment bugs);
  - do NOT alias each other (conv[i] vs conv[j] vs temporal) or across
    layers/slots (catches envelope-overlap);
  - match a contiguous ``(num_layers, max_slots, *inner)`` reference exactly
    (the shape `MambaPool.State.conv[i]` / `.temporal` expose);
  - reject a deliberately mis-aligned spec via the alignment assert.

The round-trip class is skipped on CPU — those views back GPU kernels and we
mirror the GPU path. ``TestKDAFlashInferEnvelopeStateContract`` is pure stride
arithmetic and runs everywhere.

    python -m pytest test/registered/unit/mem_cache/test_shared_mamba_views.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

_HAS_CUDA = torch.cuda.is_available()
_DEV = "cuda" if _HAS_CUDA else "cpu"

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd")


def _make_pool(
    *,
    mamba_layer_num,
    conv_state_shapes,
    conv_dtype,
    temporal_state_shape,
    temporal_dtype,
    want_slots=8,
    device=_DEV,
):
    """Build a minimal 2-sub-pool ``UnifiedKVPool`` (a small MHA grow-up peer
    + the Mamba grow-down pool under test) sized to hold >= ``want_slots`` Mamba
    slots, and return ``(pool, mamba_spec)``."""
    from sglang.srt.mem_cache.unified_memory_pool import (
        MambaSubPoolSpec,
        MHASubPoolSpec,
        UnifiedKVPool,
    )

    mamba_spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=mamba_layer_num,
        grow_direction="down",
        conv_state_shapes=tuple(tuple(s) for s in conv_state_shapes),
        conv_dtype=conv_dtype,
        temporal_state_shape=tuple(temporal_state_shape),
        temporal_dtype=temporal_dtype,
    )
    # Tiny full-attention peer (required: exactly one grow-up + one grow-down).
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=1,
        head_num=1,
        head_dim=8,
        store_dtype=torch.bfloat16,
        grow_direction="up",
    )
    entry_mamba = mamba_spec.entry_bytes()
    entry_full = full_spec.entry_bytes()
    entry_max = max(entry_mamba, entry_full)
    # Need max_slots("mamba") = total // entry_mamba >= want_slots, and total
    # large enough that BOTH pools clear their min_slot_index. Add generous
    # headroom, then round up to a multiple of 8 (covers bf16/fp32 .view()).
    total_bytes = want_slots * entry_mamba + 8 * entry_max
    total_bytes = ((total_bytes + 7) // 8) * 8
    pool = UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, mamba_spec],
        device=device,
        enable_memory_saver=False,
    )
    return pool, mamba_spec


@unittest.skipUnless(_HAS_CUDA, "shared Mamba views back GPU kernels")
class TestUnifiedMambaViews(unittest.TestCase):
    # Falcon-H1-like dims: even conv_dim, bf16 conv, fp32 temporal, several
    # layers. (Mamba2 conv state is (conv_dim, kernel-1); temporal/SSM state is
    # (nheads, head_dim, ssm_state_size).)
    FALCON_KW = dict(
        mamba_layer_num=5,  # odd, to stress the temporal-offset alignment
        conv_state_shapes=[(48, 3)],  # conv_dim=48, kernel-1=3
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(6, 8, 16),  # nheads, head_dim, ssm_state
        temporal_dtype=torch.float32,
    )

    def _fill_and_roundtrip(self, pool, mamba_spec):
        """Write a distinct random tensor to each conv view + the temporal view
        (in their own dtypes), then read all back and assert exact equality.
        Writing ALL views first and reading ALL after means any envelope overlap
        (conv[i]/conv[j]/temporal aliasing) corrupts an earlier write → mismatch.
        """
        conv_views, temporal_view = pool.mamba_views_for("mamba")
        torch.manual_seed(0)
        refs = []
        for v in conv_views:
            r = torch.randn(v.shape, device=v.device).to(v.dtype)
            v.copy_(r)
            refs.append(r)
        rt = torch.randn(temporal_view.shape, device=temporal_view.device).to(
            temporal_view.dtype
        )
        temporal_view.copy_(rt)
        refs.append(rt)
        # Read back AFTER all writes.
        for i, v in enumerate(conv_views):
            self.assertTrue(
                torch.equal(v, refs[i]),
                f"conv view[{i}] round-trip mismatch (stride/offset/overlap "
                f"bug); shape={tuple(v.shape)} stride={v.stride()}",
            )
        self.assertTrue(
            torch.equal(temporal_view, refs[-1]),
            f"temporal view round-trip mismatch; shape={tuple(temporal_view.shape)} "
            f"stride={temporal_view.stride()}",
        )

    def test_roundtrip_falcon_like(self):
        pool, spec = _make_pool(**self.FALCON_KW)
        self._fill_and_roundtrip(pool, spec)

    def test_roundtrip_single_layer_single_slot_edges(self):
        # 1 layer, multiple conv tensors, same-dtype conv/temporal.
        pool, spec = _make_pool(
            mamba_layer_num=1,
            conv_state_shapes=[(16, 3), (8, 3)],
            conv_dtype=torch.float32,
            temporal_state_shape=(4, 8, 16),
            temporal_dtype=torch.float32,
            want_slots=4,
        )
        self._fill_and_roundtrip(pool, spec)

    def test_roundtrip_multi_conv_tensors(self):
        # Two conv tensors + bf16/fp32 mix — exercises the per-conv-tensor offset
        # accumulation in _build_mamba_views.
        pool, spec = _make_pool(
            mamba_layer_num=3,
            conv_state_shapes=[(32, 3), (16, 3)],
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(8, 8, 16),
            temporal_dtype=torch.float32,
            want_slots=6,
        )
        self._fill_and_roundtrip(pool, spec)

    def test_no_cross_region_overlap(self):
        """Zero buffer; write a sentinel to ONE view; every OTHER view must read
        all-zero. Pinpoints conv[i]/conv[j]/temporal aliasing if present."""
        pool, spec = _make_pool(**self.FALCON_KW)
        conv_views, temporal_view = pool.mamba_views_for("mamba")
        views = list(conv_views) + [temporal_view]
        names = [f"conv[{i}]" for i in range(len(conv_views))] + ["temporal"]
        for target in range(len(views)):
            pool._raw.zero_()
            views[target].fill_(7.0)
            for other in range(len(views)):
                if other == target:
                    self.assertTrue(
                        bool((views[other] == 7.0).all().item()),
                        f"write to {names[target]} did not fully land",
                    )
                    continue
                self.assertTrue(
                    bool((views[other] == 0).all().item()),
                    f"writing {names[target]} CORRUPTED {names[other]} "
                    f"(envelope regions overlap)",
                )

    def test_per_layer_per_slot_addressing(self):
        """Distinct value per (layer, slot) on the temporal view; verify exact
        addressing (no layer/slot aliasing). Uses small integers exactly
        representable in the view dtype.

        NB: ``temporal_view`` is a non-contiguous strided view, so we must NOT
        ``.reshape()`` it (that would COPY, breaking the alias) — we
        broadcast-assign into the view in place and read back via basic
        indexing (which keeps the view)."""
        pool, spec = _make_pool(**self.FALCON_KW)
        _, temporal_view = pool.mamba_views_for("mamba")
        N, S = temporal_view.shape[0], temporal_view.shape[1]
        inner_ndim = temporal_view.dim() - 2
        # value = layer*S + slot  (< N*S, small → exact in fp32)
        base = (
            torch.arange(N, device=temporal_view.device)[:, None] * S
            + torch.arange(S, device=temporal_view.device)[None, :]
        ).to(temporal_view.dtype)
        # Broadcast (N, S) over the inner dims, in place into the strided view.
        temporal_view[:] = base.view(N, S, *([1] * inner_ndim))
        # Read back the first inner element of every (layer, slot) via basic
        # indexing (stays a view).
        readback = temporal_view[(slice(None), slice(None)) + (0,) * inner_ndim]
        self.assertTrue(
            torch.equal(readback, base),
            "temporal (layer, slot) addressing wrong — layer/slot stride bug",
        )

    def test_matches_contiguous_reference(self):
        """The shared view must be a faithful relabeling of a contiguous
        ``(num_layers, max_slots, *inner)`` tensor: identical data written by the
        same logical index reads back identically."""
        pool, spec = _make_pool(**self.FALCON_KW)
        conv_views, temporal_view = pool.mamba_views_for("mamba")
        for v in conv_views + [temporal_view]:
            ref = torch.randn(v.shape, device=v.device).to(v.dtype)
            contig = ref.clone().contiguous()
            v.copy_(ref)
            self.assertEqual(tuple(v.shape), tuple(contig.shape))
            self.assertTrue(
                torch.equal(v.contiguous(), contig),
                "shared view not equivalent to its contiguous counterpart",
            )

    def test_alignment_guard_fires_on_misaligned_spec(self):
        """A spec whose conv region (bf16) is an odd multiple of 2 B makes the
        per-slot entry (= conv_region + N*temporal_row = 2 B + 4 B = 6 B) NOT a
        multiple of the temporal itemsize (fp32, 4 B). The temporal/SSM-state
        view's storage_offset is computed by integer-dividing a byte offset by
        the temporal itemsize, so this would silently mis-offset the view.
        ``_build_mamba_views`` must reject it with a loud alignment assert.

        NOTE: the ``entry_bytes % itemsize`` guard is what fires here, and it
        subsumes the conv-region offset check (see the comment in
        ``_build_mamba_views``). We assert on the shared "misaligned" wording
        rather than on which specific guard trips."""
        with self.assertRaises(AssertionError) as cm:
            _make_pool(
                mamba_layer_num=1,  # entry = 2 B conv + 4 B temporal = 6 B, not %4
                conv_state_shapes=[(1, 1)],
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(1,),
                temporal_dtype=torch.float32,
                want_slots=4,
            )
        self.assertIn("misalign", str(cm.exception).lower())

    def test_alignment_ok_for_aligned_spec(self):
        """An aligned spec (conv region a multiple of the temporal itemsize)
        must build and round-trip cleanly."""
        # conv region = N * conv_dim*(k-1) * 2 ; with conv_dim=2 -> per-layer 2*3*2=12,
        # times N=2 = 24, divisible by 4. Aligned.
        pool, spec = _make_pool(
            mamba_layer_num=2,
            conv_state_shapes=[(2, 3)],
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(2, 4, 4),
            temporal_dtype=torch.float32,
            want_slots=4,
        )
        self._fill_and_roundtrip(pool, spec)


def _k3_kda_mamba_geometry(heads_per_rank: int) -> dict:
    """Kimi K3 KDA per-rank state geometry: 69 KDA layers, K = V = 128,
    conv width 4 (=> 3 cached tokens), conv row ``(kernel-1, q+k+v dim)``
    in the KimiLinear layout (``KimiLinearStateShape.create`` with
    num_k_heads == num_heads, head_k_dim == head_dim — see
    ``models/kimi_linear.py``), temporal/SSM state ``(HV, V, K)``.
    ``heads_per_rank`` = 96 total KDA heads / attn_tp (12 at the TP8
    deployment shape, cf. ``kernels/ops/attention/kda_fused_decode.py``)."""
    h = heads_per_rank
    return dict(
        layer_num=69,
        conv_state_shapes=((3, 3 * h * 128),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(h, 128, 128),
        # FlashInfer recurrent_kda requires a bf16 state pool (the server-args
        # gate enforces --mamba-ssm-dtype bfloat16 for flashinfer decode).
        temporal_dtype=torch.bfloat16,
    )


class TestKDAFlashInferEnvelopeStateContract(unittest.TestCase):
    """Derived property: the envelope-strided KDA temporal view (unified memory
    / page-major layout) must satisfy the state contract of FlashInfer
    ``recurrent_kda`` (pinned ``flashinfer_python==0.6.14``), because the KDA
    flashinfer decode wrapper (``linear/kernels/kda_flashinfer.py``) passes the
    committed per-layer pool view straight into the kernel (in-place state
    update on the cu_seqlens path — no gather/scatter copy around the call).

    The kernel compiles its state argument as a CuTe fake tensor of shape
    ``[N, HV, V, K]`` with stride ``(sym_int64(divisibility=16), V*K, K, 1)``
    and ``assumed_align=32`` (flashinfer ``kda_kernels/recurrent_kda.py``), so
    a per-layer pool view is only readable by the kernel when:

      * its inner strides are exactly compact ``(V*K, K, 1)``;
      * its slot stride — the per-slot envelope pitch, NOT ``HV*V*K`` — is a
        multiple of 16 elements (32 bytes at bf16);
      * its base byte offset is 32-byte aligned (for every layer).

    Any envelope-layout change that breaks one of these (per-slot padding that
    is not a 32 B multiple, a conv-shape change misaligning the temporal
    region, a transposed/padded temporal inner layout) would silently
    mis-address every KDA state read/write on SM100 flashinfer decode; this
    test turns such a diff red without a GPU.
    """

    # 32 B: recurrent_kda's assumed_align AND its slot-stride divisibility
    # (16 elements * 2 B bf16). External-source literal from flashinfer
    # kda_kernels/recurrent_kda.py (S_batch = cute.sym_int64(divisibility=16),
    # make_fake_tensor(..., assumed_align=32)).
    _KERNEL_ALIGN_BYTES = 32

    @staticmethod
    def _build_tp8_views():
        """Real TP8 K3 KDA envelope views on CPU (2 slots suffice — the
        per-slot geometry is slot-count independent)."""
        from sglang.srt.mem_cache.layout.page_major import (
            build_page_major_mamba_views,
            mamba_entry_bytes,
        )

        geom = _k3_kda_mamba_geometry(12)  # 96 heads / TP8
        entry_bytes = mamba_entry_bytes(**geom)
        max_slots = 2
        raw = torch.empty(max_slots * entry_bytes, dtype=torch.uint8, device="cpu")
        _, temporal_view = build_page_major_mamba_views(
            raw, max_slots=max_slots, **geom
        )
        return geom, entry_bytes, temporal_view

    def test_k3_tp8_envelope_view_matches_recurrent_kda_contract(self):
        """Check every per-layer temporal view against the kernel contract."""
        geom, entry_bytes, temporal_view = self._build_tp8_views()

        itemsize = temporal_view.element_size()
        _, v, k = geom["temporal_state_shape"]
        for layer in (0, geom["layer_num"] - 1):
            view = temporal_view[layer]  # [slots, HV, V, K], what decode() gets
            self.assertEqual(
                view.stride()[1:],
                (v * k, k, 1),
                "temporal inner strides must stay compact (V*K, K, 1): "
                "recurrent_kda compiles them as constants",
            )
            self.assertEqual(
                view.stride(0),
                entry_bytes // itemsize,
                "slot stride must be the envelope pitch (entry_bytes)",
            )
            self.assertEqual(
                view.stride(0) % (self._KERNEL_ALIGN_BYTES // itemsize),
                0,
                "slot stride must satisfy recurrent_kda's "
                "sym_int64(divisibility=16) — 16 elements = 32 B at bf16",
            )
            self.assertEqual(
                (view.storage_offset() * itemsize) % self._KERNEL_ALIGN_BYTES,
                0,
                f"layer {layer} temporal view base is not 32 B aligned "
                "(recurrent_kda assumed_align=32)",
            )

    def test_k3_entry_and_temporal_offset_32B_multiples_across_tp(self):
        """The two byte quantities that feed the contract above — the per-slot
        envelope pitch and the temporal region's offset inside the envelope
        (= all-layers conv region, temporal comes last) — must be 32 B
        multiples for every plausible attn-TP shard of K3's 96 KDA heads."""
        import math

        from sglang.srt.mem_cache.layout.page_major import mamba_entry_bytes

        for heads_per_rank in (96, 48, 24, 12):  # attn_tp 1 / 2 / 4 / 8
            geom = _k3_kda_mamba_geometry(heads_per_rank)
            entry_bytes = mamba_entry_bytes(**geom)
            conv_region_bytes = (
                geom["layer_num"]
                * math.prod(geom["conv_state_shapes"][0])
                * geom["conv_dtype"].itemsize
            )
            self.assertEqual(
                entry_bytes % self._KERNEL_ALIGN_BYTES,
                0,
                f"tp shard h={heads_per_rank}: envelope pitch {entry_bytes} B "
                "breaks recurrent_kda's slot-stride divisibility",
            )
            self.assertEqual(
                conv_region_bytes % self._KERNEL_ALIGN_BYTES,
                0,
                f"tp shard h={heads_per_rank}: temporal region offset "
                f"{conv_region_bytes} B breaks assumed_align=32",
            )

    def test_wrapper_state_contract_check_matches_layout(self):
        """The KDA flashinfer decode wrapper enforces this same contract at
        runtime (``FlashInferKDAKernel._check_state_stride_contract``, called
        once per pool view before handing the pool to ``recurrent_kda``). A
        regression in that check would only surface on SM100 hardware, so pin
        its accept/reject behavior here: it must ACCEPT exactly what the
        layouts produce — the envelope-strided per-layer view and a plain
        contiguous pool — and REJECT views the kernel would silently
        mis-address (wrong inner strides; a slot stride off the divisibility)."""
        import types

        from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (
            FlashInferKDAKernel,
        )

        check = FlashInferKDAKernel._check_state_stride_contract

        def run(view):
            # Fresh stub per call: the real kernel caches approvals by id().
            check(types.SimpleNamespace(_state_contract_ok=set()), view)

        _, _, temporal_view = self._build_tp8_views()
        envelope = temporal_view[0]  # what forward_decode hands to the kernel
        run(envelope)  # must not raise

        contiguous = torch.empty(2, 12, 128, 128, dtype=torch.bfloat16)
        run(contiguous)  # locally-allocated pools must keep working

        with self.assertRaises(ValueError):
            run(envelope.transpose(-1, -2))  # inner strides not compact

        # Slot stride 196616 elements: envelope-like but % 16 != 0.
        flat = torch.empty(2 * 196616, dtype=torch.bfloat16)
        misaligned = flat.as_strided((2, 12, 128, 128), (196616, 16384, 128, 1))
        with self.assertRaises(ValueError):
            run(misaligned)


if __name__ == "__main__":
    unittest.main()
