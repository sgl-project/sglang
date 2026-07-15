"""
XPU integration tests for LMCRadixCache (IP mode).

Unlike test_lmcache_connector.py, which drives LMCacheLayerwiseConnector
directly, this test builds a real ReqToTokenPool + MHATokenToKVPool +
TokenToKVPoolAllocator and drives LMCRadixCache itself through the request
lifecycle (match_prefix -> cache_finished_req, then evict a fresh cache to
force a real LMCache load-back through match_prefix again). This exercises
LayerTransferCounter.wait_until, _load_back's slot/offset math, and the
store/load stream synchronization that the connector-level tests never touch.

Usage:
    python3 -m unittest registered.xpu.test_lmcache_radix_cache
"""

import os
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_xpu_ci

# Must be set before lmcache imports. Save prior values so tearDownModule can
# restore them and avoid leaking into other tests in the same process.
_PATCHED_ENV = {
    "LMCACHE_USE_EXPERIMENTAL": "True",
    "LMCACHE_CONFIG_FILE": os.path.join(
        os.path.dirname(__file__), "test_lmcache_connector_config.yaml"
    ),
}
_OLD_ENV = {k: os.environ.get(k) for k in _PATCHED_ENV}
os.environ["LMCACHE_USE_EXPERIMENTAL"] = _PATCHED_ENV["LMCACHE_USE_EXPERIMENTAL"]
os.environ.setdefault("LMCACHE_CONFIG_FILE", _PATCHED_ENV["LMCACHE_CONFIG_FILE"])


def tearDownModule():
    for key, old_value in _OLD_ENV.items():
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


try:
    import lmcache.integration.sglang.sglang_adapter  # noqa: F401
except ImportError:
    raise RuntimeError(
        "LMCache is not installed. Install with: pip install lmcache"
    )

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import LMCRadixCache
from sglang.srt.runtime_context import get_context

XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()

register_xpu_ci(est_time=60, suite="stage-b-test-1-gpu-xpu")


def _make_req(rid, req_pool_idx, token_ids, tree):
    """Fake Req with the fields LMCRadixCache/RadixCache read (mirrors the
    SimpleNamespace pattern in test_swa_eviction_boundary.py)."""
    req = SimpleNamespace(
        rid=rid,
        req_pool_idx=req_pool_idx,
        origin_input_ids=token_ids,
        output_ids=[],
        extra_key=None,
        last_node=tree.root_node,
        cache_protected_len=0,
        priority=0,
        kv_committed_freed=False,
        kv_committed_len=len(token_ids),
    )
    req.pop_committed_kv_cache = lambda: len(token_ids)
    return req


@unittest.skipUnless(XPU_AVAILABLE, "Intel XPU not available")
class TestLMCRadixCacheXPU(unittest.TestCase):
    """Drive LMCRadixCache (IP mode) through match_prefix/cache_finished_req
    with a real KV pool, to cover the code path test_lmcache_connector.py
    (which talks to the connector directly) never exercises."""

    DEVICE = "xpu:0"
    BUFFER_SIZE = 256
    MAX_CONTEXT_LEN = 64
    INPUT_LEN = 16

    @classmethod
    def setUpClass(cls):
        cls.model_config = ModelConfig(model_path="Qwen/Qwen3-4B")
        cls._override = get_context().override_server_args(
            lmcache_config_file=os.environ["LMCACHE_CONFIG_FILE"],
            speculative_eagle_topk=None,
        )
        cls._override.install()

    @classmethod
    def tearDownClass(cls):
        cls._override.restore()

    def _build_tree(self):
        model_config = self.model_config
        kv_pool = MHATokenToKVPool(
            size=self.BUFFER_SIZE,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            layer_num=model_config.num_hidden_layers,
            device=self.DEVICE,
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=self.BUFFER_SIZE,
            dtype=torch.bfloat16,
            device=self.DEVICE,
            kvcache=kv_pool,
            need_sort=False,
        )
        req_to_token_pool = ReqToTokenPool(
            size=8,
            max_context_len=self.MAX_CONTEXT_LEN,
            device=self.DEVICE,
            enable_memory_saver=False,
        )
        tree = LMCRadixCache(
            params=CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
            ),
            model_config=model_config,
            tp_size=1,
            rank=0,
        )
        return tree, allocator, req_to_token_pool, kv_pool

    def test_store_then_load_back_through_match_prefix(self):
        """Full lifecycle: match_prefix (miss) -> cache_finished_req (store to
        LMCache + insert into radix) -> evict the radix entry -> match_prefix
        again must retrieve from LMCache via LayerTransferCounter and restore
        the original KV content, proving _load_back's offset/slot math and
        the load_stream synchronization are correct end-to-end."""
        tree, allocator, req_to_token_pool, kv_pool = self._build_tree()
        try:
            token_ids = torch.randint(
                0, self.model_config.vocab_size, (self.INPUT_LEN,)
            ).tolist()

            # No prior entries: match_prefix should be a full miss.
            miss_res = tree.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
            self.assertEqual(miss_res.device_indices.numel(), 0)

            # Allocate KV slots for the request, write ground-truth K/V, then
            # commit it as a finished request (inserts into radix + stores to
            # LMCache on tree.store_stream).
            req_pool_idx = req_to_token_pool.alloc(
                [SimpleNamespace(req_pool_idx=None, inflight_middle_chunks=0, kv_committed_len=0)]
            )[0]
            kv_slots = allocator.alloc(self.INPUT_LEN)
            self.assertIsNotNone(kv_slots)
            req_to_token_pool.write(
                (req_pool_idx, slice(0, self.INPUT_LEN)), kv_slots
            )

            gt_k = []
            gt_v = []
            for layer_id in range(self.model_config.num_hidden_layers):
                k = torch.randn(
                    self.INPUT_LEN,
                    self.model_config.num_key_value_heads,
                    self.model_config.head_dim,
                    dtype=torch.bfloat16,
                    device=self.DEVICE,
                )
                v = torch.randn_like(k)
                kv_pool.k_buffer[layer_id][kv_slots] = k
                kv_pool.v_buffer[layer_id][kv_slots] = v
                gt_k.append(k.clone())
                gt_v.append(v.clone())

            req = _make_req("req-0", req_pool_idx, token_ids, tree)
            tree.cache_finished_req(req)
            # IP-mode store is async on tree.store_stream; evict()'s
            # synchronize() is what the real scheduler relies on to make the
            # store visible before slots are reused.
            tree.evict(EvictParams(num_tokens=0))

            # Evict everything from the radix tree so the only remaining copy
            # of this KV is inside LMCache, forcing a real load-back on the
            # next match_prefix.
            tree.evict(EvictParams(num_tokens=self.INPUT_LEN))
            self.assertEqual(tree.total_size(), 0)

            for layer_id in range(self.model_config.num_hidden_layers):
                kv_pool.k_buffer[layer_id].zero_()
                kv_pool.v_buffer[layer_id].zero_()

            reload_res = tree.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
            self.assertEqual(reload_res.device_indices.numel(), self.INPUT_LEN)

            new_slots = reload_res.device_indices
            for layer_id in range(self.model_config.num_hidden_layers):
                # get_key_buffer/get_value_buffer (not the raw k_buffer/v_buffer
                # list) is what invokes layer_transfer_counter.wait_until —
                # the real per-layer forward hook this test exists to cover.
                actual_k = kv_pool.get_key_buffer(layer_id)[new_slots]
                actual_v = kv_pool.get_value_buffer(layer_id)[new_slots]
                self.assertTrue(
                    torch.allclose(actual_k, gt_k[layer_id]),
                    f"Layer {layer_id}: K not restored via LMCache load-back",
                )
                self.assertTrue(
                    torch.allclose(actual_v, gt_v[layer_id]),
                    f"Layer {layer_id}: V not restored via LMCache load-back",
                )
        finally:
            tree.lmcache_connector.close()


if __name__ == "__main__":
    unittest.main()
