"""Unit tests for NIXL KV transfer from a prefill with pp_size > 1 to a decode
peer that is not pipelined, on plain-MLA models (MLATokenToKVPool)."""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_LAYERS = 78
ITEM_LEN = 576


def _pp_layer_split(total: int, pp: int) -> list:
    """(start_layer, num_layers) per PP rank, remainder to the last ranks."""
    base, rem = divmod(total, pp)
    out, start = [], 0
    for rank in range(pp):
        num = base + (1 if rank >= pp - rem else 0)
        out.append((start, num))
        start += num
    return out


class _StubKVManager:
    """Carries the real span primitive, so the cases exercise the shared mapping
    rather than a reimplementation of it."""

    _mla_kv_entry_span_with_pp = CommonKVManager._mla_kv_entry_span_with_pp
    get_mla_kv_ptrs_with_pp = CommonKVManager.get_mla_kv_ptrs_with_pp

    def __init__(
        self,
        *,
        pp_size: int,
        prefill_start_layer: int,
        n_src: int,
        is_mla_backend: bool = True,
        is_hybrid_mla_backend: bool = False,
        kv_layer_ids: list = (),
        mla_compression_ratios=None,
        item_len: int = ITEM_LEN,
    ):
        self.pp_size = pp_size
        self.is_mla_backend = is_mla_backend
        self.is_hybrid_mla_backend = is_hybrid_mla_backend
        self.kv_args = SimpleNamespace(
            prefill_start_layer=prefill_start_layer,
            kv_layer_ids=list(kv_layer_ids),
            kv_item_lens=[item_len] * n_src,
            mla_compression_ratios=mla_compression_ratios,
        )


def _peer(*, n_dst: int, dst_kv_layer_ids: list = (), item_len: int = ITEM_LEN):
    return SimpleNamespace(
        dst_kv_layer_ids=list(dst_kv_layer_ids),
        dst_kv_item_lens=[item_len] * n_dst,
    )


def _resolve(manager, peer, n_src, n_dst):
    return NixlKVManager._pp_layer_offset_dst_indices(
        manager, peer_info=peer, n_src=n_src, n_dst=n_dst
    )


class TestPpPrefillToUnpipelinedDecode(CustomTestCase):
    """Bug regression: a pp>1 prefill paired with a pp=1 decode never transferred
    KV on NIXL, and failed after /health was green so the request hung rather
    than the launch failing. Each stage must write into its own layer range of
    the peer's pool."""

    def test_stages_tile_the_decode_pool_exactly_once(self):
        covered = []
        for start, num in _pp_layer_split(NUM_LAYERS, 4):
            with self.subTest(start_layer=start):
                indices = _resolve(
                    _StubKVManager(pp_size=4, prefill_start_layer=start, n_src=num),
                    _peer(n_dst=NUM_LAYERS),
                    num,
                    NUM_LAYERS,
                )
                self.assertEqual(indices, list(range(start, start + num)))
                covered += indices
        self.assertEqual(sorted(covered), list(range(NUM_LAYERS)))
        self.assertEqual(len(covered), len(set(covered)))

    def test_decode_side_draft_regions_are_never_targeted(self):
        """Decode-only speculative decoding appends draft buffers after the
        target's, so the span must stay inside the target layer range."""
        start, num = _pp_layer_split(NUM_LAYERS, 4)[3]
        indices = _resolve(
            _StubKVManager(pp_size=4, prefill_start_layer=start, n_src=num),
            _peer(n_dst=NUM_LAYERS + 1),
            num,
            NUM_LAYERS + 1,
        )
        self.assertEqual(indices, list(range(start, start + num)))
        self.assertNotIn(NUM_LAYERS, indices)

    def test_cell_geometry_disagreement_raises(self):
        """A per-layer item_len mismatch means the peers did not build the same
        KV geometry, which must fail loudly instead of corrupting the pool."""
        start, num = _pp_layer_split(NUM_LAYERS, 4)[3]
        with self.assertRaisesRegex(RuntimeError, "geometry differs"):
            _resolve(
                _StubKVManager(pp_size=4, prefill_start_layer=start, n_src=num),
                _peer(n_dst=NUM_LAYERS, item_len=ITEM_LEN + 80),
                num,
                NUM_LAYERS,
            )


class TestExistingPairingIsUnchanged(CustomTestCase):
    """Derived property: the span is reachable only from the geometry the layer-id
    pairing cannot express. Every other deployment must fall through to
    build_transfer_entry_pairs, which the resolver signals with None."""

    def test_unpipelined_prefill_defers(self):
        for n_dst in (NUM_LAYERS, NUM_LAYERS + 1):
            with self.subTest(n_dst=n_dst):
                self.assertIsNone(
                    _resolve(
                        _StubKVManager(
                            pp_size=1, prefill_start_layer=0, n_src=NUM_LAYERS
                        ),
                        _peer(n_dst=n_dst),
                        NUM_LAYERS,
                        n_dst,
                    )
                )

    def test_matched_pp_defers(self):
        for start, num in _pp_layer_split(NUM_LAYERS, 4):
            with self.subTest(start_layer=start):
                self.assertIsNone(
                    _resolve(
                        _StubKVManager(pp_size=4, prefill_start_layer=start, n_src=num),
                        _peer(n_dst=num),
                        num,
                        num,
                    )
                )

    def test_published_layer_ids_defer(self):
        start, num = _pp_layer_split(NUM_LAYERS, 4)[2]
        self.assertIsNone(
            _resolve(
                _StubKVManager(
                    pp_size=4,
                    prefill_start_layer=start,
                    n_src=num,
                    kv_layer_ids=range(start, start + num),
                ),
                _peer(n_dst=NUM_LAYERS),
                num,
                NUM_LAYERS,
            )
        )
        self.assertIsNone(
            _resolve(
                _StubKVManager(pp_size=4, prefill_start_layer=start, n_src=num),
                _peer(n_dst=NUM_LAYERS, dst_kv_layer_ids=range(NUM_LAYERS)),
                num,
                NUM_LAYERS,
            )
        )

    def test_pools_that_are_not_one_region_per_layer_defer(self):
        start, num = _pp_layer_split(NUM_LAYERS, 4)[2]
        for label, kwargs, n_dst in (
            ("mha", {"is_mla_backend": False}, 2 * NUM_LAYERS),
            ("hybrid_mla", {"is_hybrid_mla_backend": True}, NUM_LAYERS),
            (
                "compressed_mla",
                {"mla_compression_ratios": [4] * NUM_LAYERS},
                2 * NUM_LAYERS,
            ),
        ):
            with self.subTest(pool=label):
                self.assertIsNone(
                    _resolve(
                        _StubKVManager(
                            pp_size=4,
                            prefill_start_layer=start,
                            n_src=num,
                            **kwargs,
                        ),
                        _peer(n_dst=n_dst),
                        num,
                        n_dst,
                    )
                )


class TestMatchedPpWithDraftRegions(CustomTestCase):
    """Derived property: bootstrap admits a peer at our pp or at 1, so under
    matched pp a decode-side draft buffer makes n_src != n_dst and reaches the
    span. Stage 0 degenerates to the identity and stays correct; every stage
    above it must be rejected by the bound rather than writing at an offset the
    peer does not have."""

    def test_rank0_identity_span_stays_correct(self):
        start, num = _pp_layer_split(NUM_LAYERS, 4)[0]
        self.assertEqual(start, 0)
        self.assertEqual(
            _resolve(
                _StubKVManager(pp_size=4, prefill_start_layer=start, n_src=num),
                _peer(n_dst=num + 1),
                num,
                num + 1,
            ),
            list(range(num)),
        )

    def test_later_ranks_are_rejected(self):
        for start, num in _pp_layer_split(NUM_LAYERS, 4)[1:]:
            with self.subTest(start_layer=start):
                self.assertIsNone(
                    _resolve(
                        _StubKVManager(pp_size=4, prefill_start_layer=start, n_src=num),
                        _peer(n_dst=num + 1),
                        num,
                        num + 1,
                    )
                )


class TestPointerAndIndexViewsAgree(CustomTestCase):
    """Derived property: get_mla_kv_ptrs_with_pp (pointer view) and
    _pp_layer_offset_dst_indices (index view) read the same span, so a change to
    one must not silently diverge from the other."""

    def test_views_select_the_same_destination_entries(self):
        dst_ptrs = [9000 + i for i in range(NUM_LAYERS)]
        for start, num in _pp_layer_split(NUM_LAYERS, 4):
            with self.subTest(start_layer=start):
                manager = _StubKVManager(
                    pp_size=4, prefill_start_layer=start, n_src=num
                )
                src_ptrs = [1000 + i for i in range(num)]

                _, sliced_dst, count = manager.get_mla_kv_ptrs_with_pp(
                    src_ptrs, dst_ptrs
                )
                indices = _resolve(manager, _peer(n_dst=NUM_LAYERS), num, NUM_LAYERS)

                self.assertEqual(count, num)
                self.assertEqual(sliced_dst, [dst_ptrs[j] for j in indices])


if __name__ == "__main__":
    unittest.main()
