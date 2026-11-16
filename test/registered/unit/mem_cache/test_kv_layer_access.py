"""A layer's reads and writes through its router's `KVLayer` address the storage, and
write at the ids, that the pool's own per-layer methods do: a plain pool's, the static
window composite's (full and window sides) and the hybrid-linear composite's. The
router's window predicates keep apart the cases backends used to tell apart by pool
class."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    KVWriteLoc,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.frozen_kv_mtp_utils import target_kv_pool_view
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")

_DEV = "cuda"
_SWA = IdSpaceKind.SLIDING_WINDOW


def _plain_pool():
    return MHATokenToKVPool(
        size=32,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=8,
        layer_num=4,
        device=_DEV,
        enable_memory_saver=False,
    )


def _swa_pool(swa_ids=(0, 2), full_ids=(1, 3)):
    return SWAKVPool(
        size=32,
        size_swa=16,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=8,
        swa_attention_layer_ids=list(swa_ids),
        full_attention_layer_ids=list(full_ids),
        device=_DEV,
    )


def _hybrid_linear_pool(**kwargs):
    return HybridLinearKVPool(
        size=32,
        dtype=torch.bfloat16,
        page_size=1,
        head_num=2,
        head_dim=8,
        full_attention_layer_ids=[1, 3],
        device=_DEV,
        mamba_pool=None,
        **kwargs,
    )


_POOLS = {
    "plain": (_plain_pool, (0, 1, 2, 3)),
    "swa": (_swa_pool, (0, 1, 2, 3)),
    "hybrid_linear": (_hybrid_linear_pool, (1, 3)),
}


def _write_loc():
    # Different full and window ids, so a write at the wrong side's ids shows.
    loc = torch.tensor([3, 7, 11], device=_DEV)
    return KVWriteLoc(loc, swa_loc=torch.tensor([2, 5, 9], device=_DEV))


class TestKVLayerAccess(CustomTestCase):
    def test_reads_address_what_the_pool_reads(self):
        for name, (make, layer_ids) in _POOLS.items():
            pool = make()
            self.assertEqual(pool.layer_router.layer_ids(), layer_ids, name)
            for layer_id in layer_ids:
                with self.subTest(pool=name, layer_id=layer_id):
                    kv = pool.layer_router.read(layer_id)
                    self.assertEqual(
                        kv.key().data_ptr(), pool.get_key_buffer(layer_id).data_ptr()
                    )
                    self.assertEqual(
                        kv.value().data_ptr(),
                        pool.get_value_buffer(layer_id).data_ptr(),
                    )
                    self.assertEqual(
                        [t.data_ptr() for t in kv.kv()],
                        [t.data_ptr() for t in pool.get_kv_buffer(layer_id)],
                    )

    def test_writes_land_where_the_pool_writes(self):
        torch.manual_seed(0)
        for name, (make, layer_ids) in _POOLS.items():
            via_pool, via_layer = make(), make()
            for layer_id in layer_ids:
                with self.subTest(pool=name, layer_id=layer_id):
                    layer = SimpleNamespace(layer_id=layer_id)
                    k = torch.randn(3, 2, 8, device=_DEV, dtype=torch.bfloat16)
                    v = torch.randn(3, 2, 8, device=_DEV, dtype=torch.bfloat16)
                    via_pool.set_kv_buffer(layer, _write_loc(), k, v)
                    via_layer.layer_router.layer(layer_id).store(
                        layer, _write_loc(), k, v
                    )
                    for got, want in zip(
                        via_layer.get_kv_buffer(layer_id),
                        via_pool.get_kv_buffer(layer_id),
                    ):
                        self.assertTrue(torch.equal(got, want))

    def test_mla_writes_use_each_sub_pools_locations(self):
        """MLA writes must reach the window or owner slots selected for the layer,
        even when generic, window and explicit owner locations all differ."""
        kwargs = dict(
            page_size=1,
            device=_DEV,
            enable_memory_saver=False,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
        )
        with get_parallel().override(
            dcp_enabled=False, attn_dcp_size=1, attn_dcp_rank=0
        ):
            pool = SWAKVPool(
                size=32,
                size_swa=16,
                page_size=1,
                dtype=torch.bfloat16,
                head_num=1,
                head_dim=6,
                swa_attention_layer_ids=[0, 2],
                full_attention_layer_ids=[1, 3],
                device=_DEV,
                full_kv_pool_class=MLATokenToKVPool,
                swa_kv_pool_class=MLATokenToKVPool,
                full_kv_pool_kwargs=kwargs,
                swa_kv_pool_kwargs=kwargs,
            )
            loc = torch.tensor([11], device=_DEV)
            window_loc = torch.tensor([2], device=_DEV)
            owner_loc = torch.tensor([7], device=_DEV)
            for full_loc in (None, owner_loc):
                for layer_id in (0, 1, 2, 3):
                    with self.subTest(
                        layer_id=layer_id, explicit_owner=full_loc is not None
                    ):
                        kv = pool.layer_router.layer(layer_id)
                        buffer = kv.key()
                        buffer.zero_()
                        nope = torch.full((1, 1, 4), 2.0, dtype=pool.dtype, device=_DEV)
                        rope = torch.full((1, 1, 2), 3.0, dtype=pool.dtype, device=_DEV)
                        layer = SimpleNamespace(layer_id=layer_id)
                        kv.store_mla(
                            layer, KVWriteLoc(loc, window_loc, full_loc), nope, rope
                        )
                        expected = torch.zeros_like(buffer)
                        slot = (
                            2 if layer_id in (0, 2) else (11 if full_loc is None else 7)
                        )
                        expected[slot] = torch.cat((nope, rope), dim=-1)[0]
                        self.assertTrue(torch.equal(buffer, expected))
                        self.assertEqual(layer.layer_id, layer_id)

    def test_hybrid_linear_mla_reaches_its_view_without_rewriting_the_layer(self):
        """The MLA write and read address the full view's layer by its local
        index while `RadixAttention.layer_id` stays the model's layer id."""
        with get_parallel().override(
            dcp_enabled=False, attn_dcp_size=1, attn_dcp_rank=0
        ):
            pool = _hybrid_linear_pool(use_mla=True, kv_lora_rank=4, qk_rope_head_dim=2)
            layer = SimpleNamespace(layer_id=3)
            loc = torch.tensor([5], device=_DEV)
            nope = torch.full((1, 1, 4), 2.0, dtype=torch.bfloat16, device=_DEV)
            rope = torch.full((1, 1, 2), 3.0, dtype=torch.bfloat16, device=_DEV)
            pool.set_mla_kv_buffer(layer, loc, nope, rope)
            self.assertEqual(layer.layer_id, 3)
            written = pool.full_kv_pool.kv_buffer[1][5].flatten().tolist()
            self.assertEqual(written, [2.0] * 4 + [3.0] * 2)
            self.assertEqual(pool.full_kv_pool.kv_buffer[0].abs().sum().item(), 0)
            got_nope, got_rope = pool.get_mla_kv_buffer(layer, loc)
            self.assertTrue(torch.equal(got_nope, nope))
            self.assertTrue(torch.equal(got_rope, rope))

    def test_window_predicates(self):
        """A window sub-pool may exist with no layer routed to it (a full-only
        draft over a hybrid pool): `sub_pool` and `routes_to` must disagree
        there, as the pool-class check and its layer-count variant did."""
        cases = {
            "window layers": (_swa_pool(swa_ids=(0,), full_ids=(1,)), True, True),
            "no window layers": (_swa_pool(swa_ids=(), full_ids=(0,)), True, False),
            "plain": (_plain_pool(), False, False),
        }
        for name, (pool, has_window, routes_window) in cases.items():
            with self.subTest(pool=name):
                router = pool.layer_router
                self.assertEqual(router.sub_pool(_SWA) is not None, has_window)
                self.assertEqual(router.routes_to(_SWA), routes_window)
                kv = router.layer(0)
                self.assertEqual(kv.route.sub_pool == _SWA, routes_window)
        plain = cases["plain"][0].layer_router.layer(1)
        self.assertIs(plain.view, cases["plain"][0])
        self.assertEqual(plain.view_layer_id, 1)

    def test_a_composite_marks_its_owner_and_a_plain_pool_is_plain(self):
        swa = _swa_pool().layer_router
        self.assertEqual(
            {i: swa.layer(i).owner for i in (0, 1, 2, 3)},
            {0: False, 1: True, 2: False, 3: True},
        )
        self.assertFalse(any(swa.layer(i).plain for i in (0, 1, 2, 3)))
        self.assertTrue(_plain_pool().layer_router.layer(2).plain)

    def test_a_backend_follows_a_swapped_pool(self):
        # A frozen-KV MTP draft swaps its backend's pool for the target's for
        # the draft forward; the backend's per-layer reads go to the target's.
        draft, target = _plain_pool(), _plain_pool()
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.token_to_kv_pool = draft
        context = SimpleNamespace(target_token_to_kv_pool=target)
        with target_kv_pool_view(None, context, backend):
            self.assertEqual(
                backend.kv_layers.read(1).key().data_ptr(),
                target.get_key_buffer(1).data_ptr(),
            )
        self.assertEqual(
            backend.kv_layers.read(1).key().data_ptr(),
            draft.get_key_buffer(1).data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
