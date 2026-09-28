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
"""Round-trip correctness of ``UnifiedKVPool._build_mamba_views`` -- the
envelope-strided conv/temporal (SSM) state views that back ``UnifiedMambaPool``.
Guards against garbled greedy decode from a stride/offset/alignment bug in the
state views. CPU-only.

Within one slot's envelope the bytes are
``[conv[0] L0 | conv[0] L1 | ... | conv[1] L0 | ... | temporal L0 | ...]``,
the slot stride is ``entry_bytes``, and each returned view is
``(num_layers, max_slots, *inner_shape)``. The conv dtype (bf16, 2 B) and
temporal dtype (fp32, 4 B) may DIFFER, so the temporal view's byte offset must
be a multiple of the temporal itemsize -- an alignment hazard that
``_build_mamba_views`` asserts.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")
register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd")


def _make_pool(
    *,
    mamba_layer_num,
    conv_state_shapes,
    conv_dtype,
    temporal_state_shape,
    temporal_dtype,
    want_slots=8,
    device="cpu",
):
    """Minimal 2-sub-pool ``UnifiedKVPool`` sized to hold >= ``want_slots``
    Mamba slots; returns ``(pool, mamba_spec)``."""
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
    # Headroom so BOTH pools clear their min_slot_index, then round up to a
    # multiple of 8 so bf16/fp32 ``.view()`` stays legal.
    total_bytes = want_slots * entry_mamba + 8 * entry_max
    total_bytes = ((total_bytes + 7) // 8) * 8
    pool = UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, mamba_spec],
        device=device,
        enable_memory_saver=False,
    )
    return pool, mamba_spec


class TestUnifiedMambaViews(unittest.TestCase):
    # Falcon-H1-like dims: even conv_dim, bf16 conv, fp32 temporal.
    FALCON_KW = dict(
        mamba_layer_num=5,  # odd, to stress the temporal-offset alignment
        conv_state_shapes=[(48, 3)],  # conv_dim=48, kernel-1=3
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(6, 8, 16),  # nheads, head_dim, ssm_state
        temporal_dtype=torch.float32,
    )

    def _fill_and_roundtrip(self, pool, mamba_spec):
        """Write a distinct random tensor to every view, then read all of them
        back: writing all first makes any envelope overlap corrupt an earlier
        write."""
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

    OTHER_GEOMETRIES = {
        # 1 layer, multiple conv tensors, same-dtype conv/temporal.
        "single_layer_single_slot_edges": dict(
            mamba_layer_num=1,
            conv_state_shapes=[(16, 3), (8, 3)],
            conv_dtype=torch.float32,
            temporal_state_shape=(4, 8, 16),
            temporal_dtype=torch.float32,
            want_slots=4,
        ),
        # Two conv tensors + bf16/fp32 mix: exercises the per-conv-tensor offset
        # accumulation in _build_mamba_views.
        "multi_conv_tensors": dict(
            mamba_layer_num=3,
            conv_state_shapes=[(32, 3), (16, 3)],
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(8, 8, 16),
            temporal_dtype=torch.float32,
            want_slots=6,
        ),
        # ALIGNED spec: conv region = 2 layers * 2*3*2 B = 24 B, a multiple of
        # the fp32 temporal itemsize, so it must build and round-trip.
        "alignment_ok": dict(
            mamba_layer_num=2,
            conv_state_shapes=[(2, 3)],
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(2, 4, 4),
            temporal_dtype=torch.float32,
            want_slots=4,
        ),
    }

    def test_roundtrip_geometries(self):
        """Round-trip every (tensor, layer, slot) element across the geometry
        shapes the view builder has to handle."""
        for name, kwargs in (
            ("falcon_like", self.FALCON_KW),
            *self.OTHER_GEOMETRIES.items(),
        ):
            with self.subTest(geometry=name):
                pool, spec = _make_pool(**kwargs)
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

    def test_alignment_guard_fires_on_misaligned_spec(self):
        """A per-slot entry that is not a multiple of the temporal itemsize would
        mis-offset the view and must trip the alignment assert; either guard can
        fire first, so match on the shared "misaligned" wording."""
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


def _k3_kda_mamba_geometry(heads_per_rank: int) -> dict:
    """Kimi K3 KDA per-rank state geometry: 69 KDA layers, K = V = 128, conv
    width 4 (=> 3 cached tokens), conv row ``(kernel-1, q+k+v dim)`` per
    ``KimiLinearStateShape.create``, temporal/SSM state ``(HV, V, K)``, and
    ``heads_per_rank`` = 96 total KDA heads / attn_tp."""
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
    """Derived property: the envelope-strided KDA temporal view must satisfy the
    state contract of FlashInfer ``recurrent_kda``, because the KDA flashinfer
    decode wrapper (``linear/kernels/kda_flashinfer.py``) passes the per-layer
    pool view straight into the kernel, with no gather/scatter copy.

    The kernel compiles its state argument as a CuTe fake tensor of shape
    ``[N, HV, V, K]`` with stride ``(sym_int64(divisibility=16), V*K, K, 1)``
    and ``assumed_align=32``, so a per-layer pool view is readable only when:

      * its inner strides are exactly compact ``(V*K, K, 1)``;
      * its slot stride -- the per-slot envelope pitch, NOT ``HV*V*K`` -- is a
        multiple of 16 elements (32 bytes at bf16);
      * its base byte offset is 32-byte aligned (for every layer).
    """

    # recurrent_kda's assumed_align AND its slot-stride divisibility (16
    # elements * 2 B bf16), from flashinfer kda_kernels/recurrent_kda.py.
    _KERNEL_ALIGN_BYTES = 32

    @staticmethod
    def _build_kda_views(heads_per_rank: int = 12):
        """Real K3 KDA envelope views on CPU for one attn-TP shard; 2 slots
        suffice, the per-slot geometry is slot-count independent."""
        from sglang.srt.mem_cache.layout.page_major import (
            build_page_major_mamba_views,
            mamba_entry_bytes,
        )

        geom = _k3_kda_mamba_geometry(heads_per_rank)
        entry_bytes = mamba_entry_bytes(**geom)
        max_slots = 2
        raw = torch.empty(max_slots * entry_bytes, dtype=torch.uint8, device="cpu")
        _, temporal_view = build_page_major_mamba_views(
            raw, max_slots=max_slots, **geom
        )
        return geom, entry_bytes, temporal_view

    def test_k3_tp8_envelope_view_matches_recurrent_kda_contract(self):
        """Check every per-layer temporal view against the kernel contract, for
        every plausible attn-TP shard of K3's 96 KDA heads."""
        for heads_per_rank in (96, 48, 24, 12):  # attn_tp 1 / 2 / 4 / 8
            with self.subTest(heads_per_rank=heads_per_rank):
                geom, entry_bytes, temporal_view = self._build_kda_views(heads_per_rank)
                itemsize = temporal_view.element_size()
                _, v, k = geom["temporal_state_shape"]
                for layer in (0, geom["layer_num"] - 1):
                    # [slots, HV, V, K], what decode() gets
                    view = temporal_view[layer]
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

    def test_wrapper_state_contract_check_matches_layout(self):
        """``FlashInferKDAKernel._check_state_stride_contract`` enforces the same
        contract at runtime, and a regression there would only surface on SM100
        hardware, so pin its accept/reject behavior on CPU."""
        import types

        from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (
            FlashInferKDAKernel,
        )

        check = FlashInferKDAKernel._check_state_stride_contract

        def run(view):
            # Fresh stub per call: the real kernel caches approvals by id().
            check(types.SimpleNamespace(_state_contract_ok=set()), view)

        _, _, temporal_view = self._build_kda_views()
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
