"""Unit tests for NIXL KV transfer from a DSA cache layer split prefill, where
each context-parallel rank registers only the layers it owns, to a decode peer
that registers the whole model."""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.layers.cp.utils import get_layer_shard_range
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

NUM_LAYERS = 78
NUM_DRAFT = 1
ITEM_LEN = 576
DRAFT_ITEM_LEN = 640


def _owned_layers(cp_size: int) -> list:
    """(start_layer, num_layers) per CP rank, from the pool's own split."""
    out = []
    for rank in range(cp_size):
        start, end = get_layer_shard_range(rank, cp_size, NUM_LAYERS)
        out.append((start, end - start))
    return out


def _manager(
    *,
    prefill_start_layer: int,
    num_owned: int,
    num_draft: int = 0,
    kv_layer_ids: list = (),
    item_len: int = ITEM_LEN,
) -> NixlKVManager:
    """A layer split prefill rank's manager, without the NIXL agent behind it."""
    manager = object.__new__(NixlKVManager)
    manager.pp_size = 1
    manager.cp_layer_split = True
    manager.is_mla_backend = True
    manager.is_hybrid_mla_backend = False
    manager.src_mem_kind = "VRAM"
    manager.kv_args = SimpleNamespace(
        prefill_start_layer=prefill_start_layer,
        kv_layer_ids=list(kv_layer_ids),
        kv_item_lens=[item_len] * num_owned + [DRAFT_ITEM_LEN] * num_draft,
        num_draft_entries=num_draft,
        mla_compression_ratios=None,
    )
    return manager


def _peer(
    *,
    num_draft: int = 0,
    dst_kv_layer_ids: list = (),
    item_len: int = ITEM_LEN,
    mem_kinds: list = None,
):
    n_dst = NUM_LAYERS + num_draft
    return SimpleNamespace(
        dst_kv_layer_ids=list(dst_kv_layer_ids),
        dst_kv_item_lens=[item_len] * NUM_LAYERS + [DRAFT_ITEM_LEN] * num_draft,
        dst_kv_mem_kinds=mem_kinds or ["VRAM"] * n_dst,
    )


def _resolve(manager, peer):
    return manager._build_transfer_dst_indices(
        peer_info=peer,
        n_src=len(manager.kv_args.kv_item_lens),
        n_dst=len(peer.dst_kv_item_lens),
    )


class TestLayerSplitRanksToWholeModelDecode(CustomTestCase):
    """Bug regression: the pool publishes no layer ids, so every rank above 0
    paired its entries with the decode's by position and wrote its layers into
    the first layers of the peer's pool."""

    def test_ranks_tile_the_decode_pool_exactly_once(self):
        for cp_size in (2, 4, 8):
            covered = []
            for start, num in _owned_layers(cp_size):
                with self.subTest(cp_size=cp_size, start_layer=start):
                    indices = _resolve(
                        _manager(prefill_start_layer=start, num_owned=num),
                        _peer(),
                    )
                    self.assertEqual(indices, list(range(start, start + num)))
                    covered += indices
            self.assertEqual(covered, list(range(NUM_LAYERS)))

    def test_only_the_last_rank_targets_the_draft_entries(self):
        """Layer split sends the draft cache from the last CP rank alone; its
        entries follow the target's on both peers."""
        ranks = _owned_layers(8)
        draft = list(range(NUM_LAYERS, NUM_LAYERS + NUM_DRAFT))
        for start, num in ranks[:-1]:
            with self.subTest(start_layer=start):
                indices = _resolve(
                    _manager(prefill_start_layer=start, num_owned=num),
                    _peer(num_draft=NUM_DRAFT),
                )
                self.assertEqual(indices, list(range(start, start + num)))
        start, num = ranks[-1]
        indices = _resolve(
            _manager(prefill_start_layer=start, num_owned=num, num_draft=NUM_DRAFT),
            _peer(num_draft=NUM_DRAFT),
        )
        self.assertEqual(indices, list(range(start, NUM_LAYERS)) + draft)

    def test_decode_only_draft_entries_are_never_targeted(self):
        """A decode that alone runs speculative decoding registers draft
        buffers no prefill rank fills."""
        start, num = _owned_layers(8)[-1]
        indices = _resolve(
            _manager(prefill_start_layer=start, num_owned=num),
            _peer(num_draft=NUM_DRAFT),
        )
        self.assertEqual(indices, list(range(start, NUM_LAYERS)))

    def test_a_draft_the_decode_lacks_is_rejected(self):
        start, num = _owned_layers(8)[-1]
        with self.assertRaisesRegex(RuntimeError, "do not fit the decode"):
            _resolve(
                _manager(prefill_start_layer=start, num_owned=num, num_draft=NUM_DRAFT),
                _peer(),
            )

    def test_a_draft_of_another_size_is_rejected(self):
        """The last rank's layers must end where the decode's draft begins."""
        start, num = _owned_layers(8)[-1]
        with self.assertRaisesRegex(RuntimeError, "do not fit the decode"):
            _resolve(
                _manager(prefill_start_layer=start, num_owned=num, num_draft=NUM_DRAFT),
                _peer(num_draft=NUM_DRAFT + 1),
            )

    def test_cell_geometry_disagreement_raises(self):
        start, num = _owned_layers(8)[3]
        with self.assertRaisesRegex(RuntimeError, "geometry differs"):
            _resolve(
                _manager(prefill_start_layer=start, num_owned=num),
                _peer(item_len=ITEM_LEN + 80),
            )

    def test_one_sided_layer_ids_are_rejected(self):
        """Without the span, a rank must not fall back to positional pairing."""
        start, num = _owned_layers(8)[3]
        with self.assertRaisesRegex(RuntimeError, "both PD peers or neither"):
            _resolve(
                _manager(prefill_start_layer=start, num_owned=num),
                _peer(dst_kv_layer_ids=range(NUM_LAYERS)),
            )

    def test_layer_ids_on_both_peers_pair_by_id(self):
        start, num = _owned_layers(8)[3]
        indices = _resolve(
            _manager(
                prefill_start_layer=start,
                num_owned=num,
                kv_layer_ids=range(start, start + num),
            ),
            _peer(dst_kv_layer_ids=range(NUM_LAYERS)),
        )
        self.assertEqual(indices, list(range(start, start + num)))


class TestPointerAndIndexViewsAgree(CustomTestCase):
    """Derived property: get_mla_kv_ptrs_with_pp (pointer view, the path state
    and unprepared sends take) and _build_transfer_dst_indices (index view)
    select the same decode entries for every rank, the draft's included."""

    def test_views_select_the_same_destination_entries(self):
        dst_ptrs = [9000 + i for i in range(NUM_LAYERS + NUM_DRAFT)]
        ranks = _owned_layers(8)
        for rank, (start, num) in enumerate(ranks):
            num_draft = NUM_DRAFT if rank == len(ranks) - 1 else 0
            with self.subTest(rank=rank):
                manager = _manager(
                    prefill_start_layer=start, num_owned=num, num_draft=num_draft
                )
                src_ptrs = [100 + i for i in range(num + num_draft)]
                _, ptr_view, _ = manager.get_mla_kv_ptrs_with_pp(src_ptrs, dst_ptrs)
                indices = _resolve(manager, _peer(num_draft=NUM_DRAFT))
                self.assertEqual([dst_ptrs[j] for j in indices], ptr_view)


class TestHostBackedDecodeIsRejected(CustomTestCase):
    """A HiSparse decode keeps its target KV in host memory; the host-backed
    paths pair entries by position, so the pairing must fail at registration
    instead of writing a rank's layers at the wrong offset."""

    def test_host_backed_decode_pool_is_rejected(self):
        start, num = _owned_layers(8)[3]
        manager = _manager(prefill_start_layer=start, num_owned=num)
        for label, mem_kinds in (
            ("host", ["DRAM"] * NUM_LAYERS),
            ("host_with_device_draft", ["DRAM"] * NUM_LAYERS + ["VRAM"] * NUM_DRAFT),
        ):
            with self.subTest(pool=label):
                peer = _peer(num_draft=len(mem_kinds) - NUM_LAYERS, mem_kinds=mem_kinds)
                with self.assertRaisesRegex(
                    NotImplementedError, "layer split with a host-backed"
                ):
                    manager._prepare_payload_xfer(peer)


if __name__ == "__main__":
    unittest.main()
