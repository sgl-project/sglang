"""The layer router addresses the storage a composite pool's own forwarding
addresses, and its window predicates keep apart the cases backends used to tell
apart by pool class."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

_DEV = "cuda"
_SWA = IdSpaceKind.SLIDING_WINDOW


def _swa_pool(swa_ids, full_ids):
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


def _hybrid_linear_pool(full_ids, **kwargs):
    return HybridLinearKVPool(
        size=32,
        dtype=torch.bfloat16,
        page_size=1,
        head_num=2,
        head_dim=8,
        full_attention_layer_ids=list(full_ids),
        device=_DEV,
        mamba_pool=None,
        **kwargs,
    )


class TestKVLayerRouter(CustomTestCase):
    def test_routes_address_what_the_composite_forwards_to(self):
        """Each composite numbers its sub-pools' layers locally (window and
        full sides 0..n-1, a hybrid-linear pool's full layers 0..n-1); a route
        to the wrong side or index reads another layer's KV."""
        pools = {
            "swa": (_swa_pool(swa_ids=(0, 2), full_ids=(1, 3)), (0, 1, 2, 3)),
            "hybrid_linear": (_hybrid_linear_pool(full_ids=(1, 3)), (1, 3)),
        }
        for name, (pool, layer_ids) in pools.items():
            router = pool.layer_router
            self.assertEqual(router.layer_ids(), layer_ids)
            for layer_id in layer_ids:
                with self.subTest(pool=name, layer_id=layer_id):
                    kv = router.read(layer_id)
                    got = kv.view.get_kv_buffer(kv.view_layer_id)
                    want = pool.get_kv_buffer(layer_id)
                    self.assertEqual(
                        [t.data_ptr() for t in got], [t.data_ptr() for t in want]
                    )

    def test_hybrid_linear_mla_reaches_its_view_without_rewriting_the_layer(self):
        """The MLA write and read address the full view's layer by its local
        index while `RadixAttention.layer_id` stays the model's layer id."""
        with get_parallel().override(
            dcp_enabled=False, attn_dcp_size=1, attn_dcp_rank=0
        ):
            self._check_hybrid_linear_mla()

    def _check_hybrid_linear_mla(self):
        pool = _hybrid_linear_pool(
            full_ids=(1, 3), use_mla=True, kv_lora_rank=4, qk_rope_head_dim=2
        )
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
            "plain": (
                MHATokenToKVPool(
                    size=32,
                    page_size=1,
                    dtype=torch.bfloat16,
                    head_num=2,
                    head_dim=8,
                    layer_num=2,
                    device=_DEV,
                    enable_memory_saver=False,
                ),
                False,
                False,
            ),
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


if __name__ == "__main__":
    unittest.main()
