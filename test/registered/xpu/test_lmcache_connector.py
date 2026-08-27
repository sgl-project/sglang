"""
XPU integration tests for LMCache connector in SGLang.

Tests store/retrieve round-trip on Intel XPU using pure PyTorch ops
(index_copy_, index_select) instead of CUDA lmc_ops kernels.

Uses a single shared connector to avoid LMCacheEngineBuilder singleton
issues (close() does not remove from _instances, so re-creating a
connector returns a dead engine).

Usage:
    python3 -m unittest registered.xpu.test_lmcache_connector
"""

import os
import unittest

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
    from lmcache.integration.sglang.sglang_adapter import (
        LMCacheLayerwiseConnector,
        LoadMetadata,
        StoreMetadata,
    )
except ImportError:
    raise RuntimeError(
        "LMCache is not installed. "
        "Install with: NO_CUDA_EXT=1 pip install -e . --no-build-isolation"
    )

from sglang.srt.configs.model_config import ModelConfig

XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()

register_xpu_ci(est_time=60, suite="stage-b-test-1-gpu-xpu")


@unittest.skipUnless(XPU_AVAILABLE, "Intel XPU not available")
class TestLMCacheXPUConnector(unittest.TestCase):
    """Test LMCache layerwise connector store/retrieve on XPU.

    All tests share a single connector instance to avoid the
    LMCacheEngineBuilder singleton issue where close() does not
    remove the engine from _instances.
    """

    DEVICE = "xpu:0"
    BUFFER_SIZE = 256
    INPUT_LEN = 16

    @classmethod
    def setUpClass(cls):
        cls.model_config = ModelConfig(model_path="Qwen/Qwen3-4B")
        cls.head_num = cls.model_config.num_key_value_heads
        cls.head_dim = cls.model_config.head_dim
        cls.layer_num = cls.model_config.num_hidden_layers
        cls.vocab_size = cls.model_config.vocab_size

        # Shared KV buffers and connector (created once)
        cls.k_buffer = [
            torch.randn(
                cls.BUFFER_SIZE,
                cls.head_num,
                cls.head_dim,
                dtype=torch.bfloat16,
                device=cls.DEVICE,
            )
            for _ in range(cls.layer_num)
        ]
        cls.v_buffer = [
            torch.randn(
                cls.BUFFER_SIZE,
                cls.head_num,
                cls.head_dim,
                dtype=torch.bfloat16,
                device=cls.DEVICE,
            )
            for _ in range(cls.layer_num)
        ]
        cls.connector = LMCacheLayerwiseConnector(
            cls.model_config,
            tp_size=1,
            rank=0,
            k_pool=cls.k_buffer,
            v_pool=cls.v_buffer,
            config_file=os.environ["LMCACHE_CONFIG_FILE"],
        )

    @classmethod
    def tearDownClass(cls):
        cls.connector.close()

    def setUp(self):
        """Re-randomize buffers before each test for isolation."""
        for i in range(self.layer_num):
            self.k_buffer[i].normal_()
            self.v_buffer[i].normal_()

    def _unique_tokens(self, length=None, salt=0):
        """Generate unique token ids unlikely to collide across tests."""
        n = length or self.INPUT_LEN
        base = torch.randint(0, self.vocab_size, (n,))
        return [(t.item() + salt) % self.vocab_size for t in base]

    def test_store_then_retrieve(self):
        """Basic: store KV, clear buffers, retrieve and verify match."""
        token_ids = self._unique_tokens(salt=100)
        kv_indices = torch.randint(0, self.BUFFER_SIZE, (self.INPUT_LEN,))

        # First retrieve should return 0 (cold cache)
        load_meta = LoadMetadata(
            token_ids=token_ids,
            slot_mapping=kv_indices,
            offset=0,
        )
        self.assertEqual(self.connector.start_load_kv(load_meta), 0)

        # Store
        store_meta = StoreMetadata(
            last_node=None,
            token_ids=token_ids,
            kv_indices=kv_indices,
            offset=0,
        )
        self.connector.store_kv(store_meta)
        torch.xpu.synchronize()

        # Save ground truth before clearing
        gt_k = [self.k_buffer[i][kv_indices].clone() for i in range(self.layer_num)]
        gt_v = [self.v_buffer[i][kv_indices].clone() for i in range(self.layer_num)]

        # Clear buffers
        for i in range(self.layer_num):
            self.k_buffer[i].zero_()
            self.v_buffer[i].zero_()

        # Retrieve
        ret = self.connector.start_load_kv(load_meta)
        self.assertEqual(ret, self.INPUT_LEN)

        for i in range(self.layer_num):
            torch.xpu.synchronize()
            self.connector.load_kv_layerwise(i)

        torch.xpu.synchronize()

        # Verify
        for i in range(self.layer_num):
            actual_k = self.k_buffer[i][kv_indices]
            actual_v = self.v_buffer[i][kv_indices]
            self.assertTrue(
                torch.allclose(actual_k, gt_k[i]),
                f"Layer {i}: K mismatch (max diff {(actual_k - gt_k[i]).abs().max():.6f})",
            )
            self.assertTrue(
                torch.allclose(actual_v, gt_v[i]),
                f"Layer {i}: V mismatch (max diff {(actual_v - gt_v[i]).abs().max():.6f})",
            )

    def test_retrieve_cold_cache_returns_zero(self):
        """Retrieve from empty cache should return 0 tokens."""
        token_ids = self._unique_tokens(salt=200)
        kv_indices = torch.randint(0, self.BUFFER_SIZE, (self.INPUT_LEN,))

        load_meta = LoadMetadata(
            token_ids=token_ids,
            slot_mapping=kv_indices,
            offset=0,
        )
        self.assertEqual(self.connector.start_load_kv(load_meta), 0)

    def test_slot_mapping_with_negative_indices(self):
        """slot_mapping may contain -1 for an already-cached prefix.

        The -1 filtering fix must (a) not crash on XPU and (b) leave the -1
        slots untouched while correctly restoring the valid slots.
        """
        # Distinct index for every token so -1 slots and valid slots never
        # alias each other, keeping the untouched/restored assertions exact.
        kv_indices = torch.randperm(self.BUFFER_SIZE)[: self.INPUT_LEN]
        token_ids = self._unique_tokens(salt=400)

        # Store first
        store_meta = StoreMetadata(
            last_node=None,
            token_ids=token_ids,
            kv_indices=kv_indices,
            offset=0,
        )
        self.connector.store_kv(store_meta)
        torch.xpu.synchronize()

        # Build slot_mapping with -1 prefix (simulating already-cached tokens).
        num_cached = 8
        slot_mapping_with_neg = kv_indices.clone()
        slot_mapping_with_neg[:num_cached] = -1
        cached_slots = kv_indices[:num_cached]
        valid_slots = kv_indices[num_cached:]

        # Ground truth for the valid tail (what retrieve must restore).
        gt_k = [self.k_buffer[i][valid_slots].clone() for i in range(self.layer_num)]
        gt_v = [self.v_buffer[i][valid_slots].clone() for i in range(self.layer_num)]

        # Clear buffers: untouched -1 slots stay zero, valid slots get restored.
        for i in range(self.layer_num):
            self.k_buffer[i].zero_()
            self.v_buffer[i].zero_()

        load_meta = LoadMetadata(
            token_ids=token_ids,
            slot_mapping=slot_mapping_with_neg,
            # offset marks how many leading tokens are already cached (the
            # -1 prefix in slot_mapping); this mirrors how lmc_radix_cache's
            # _ip_load_back derives offset from the already-matched prefix.
            offset=num_cached,
        )
        # Should not crash on XPU (the -1 filtering fix is critical here).
        ret = self.connector.start_load_kv(load_meta)
        self.assertEqual(ret, self.INPUT_LEN - num_cached)
        for i in range(self.layer_num):
            torch.xpu.synchronize()
            self.connector.load_kv_layerwise(i)
        torch.xpu.synchronize()

        for i in range(self.layer_num):
            # -1 positions must be untouched (still zero).
            self.assertTrue(
                torch.all(self.k_buffer[i][cached_slots] == 0),
                f"Layer {i}: -1 K slots were written",
            )
            self.assertTrue(
                torch.all(self.v_buffer[i][cached_slots] == 0),
                f"Layer {i}: -1 V slots were written",
            )
            # Valid positions must be restored.
            self.assertTrue(
                torch.allclose(self.k_buffer[i][valid_slots], gt_k[i]),
                f"Layer {i}: valid K slots not restored",
            )
            self.assertTrue(
                torch.allclose(self.v_buffer[i][valid_slots], gt_v[i]),
                f"Layer {i}: valid V slots not restored",
            )

    def test_multiple_store_retrieve_cycles(self):
        """Multiple store/retrieve cycles should not leak or corrupt."""
        for cycle in range(3):
            token_ids = self._unique_tokens(salt=500 + cycle * 1000)
            kv_indices = torch.randint(0, self.BUFFER_SIZE, (self.INPUT_LEN,))

            store_meta = StoreMetadata(
                last_node=None,
                token_ids=token_ids,
                kv_indices=kv_indices,
                offset=0,
            )
            self.connector.store_kv(store_meta)
            torch.xpu.synchronize()

            gt_k = [self.k_buffer[i][kv_indices].clone() for i in range(self.layer_num)]
            gt_v = [self.v_buffer[i][kv_indices].clone() for i in range(self.layer_num)]

            for i in range(self.layer_num):
                self.k_buffer[i].zero_()
                self.v_buffer[i].zero_()

            load_meta = LoadMetadata(
                token_ids=token_ids,
                slot_mapping=kv_indices,
                offset=0,
            )
            ret = self.connector.start_load_kv(load_meta)
            self.assertEqual(
                ret,
                self.INPUT_LEN,
                f"Cycle {cycle}: expected {self.INPUT_LEN}, got {ret}",
            )

            for i in range(self.layer_num):
                torch.xpu.synchronize()
                self.connector.load_kv_layerwise(i)
            torch.xpu.synchronize()

            for i in range(self.layer_num):
                actual_k = self.k_buffer[i][kv_indices]
                actual_v = self.v_buffer[i][kv_indices]
                self.assertTrue(
                    torch.allclose(actual_k, gt_k[i]),
                    f"Cycle {cycle}, Layer {i}: K mismatch",
                )
                self.assertTrue(
                    torch.allclose(actual_v, gt_v[i]),
                    f"Cycle {cycle}, Layer {i}: V mismatch",
                )

    def test_bf16_dtype_preserved(self):
        """Verify bf16 dtype is preserved through store->retrieve cycle.

        TODO: once XPU LMCache supports KV quantization, extend the dtype
        coverage here to the quantized store/retrieve dtypes (e.g. fp8).
        """
        token_ids = self._unique_tokens(salt=600)
        kv_indices = torch.randint(0, self.BUFFER_SIZE, (self.INPUT_LEN,))

        store_meta = StoreMetadata(
            last_node=None,
            token_ids=token_ids,
            kv_indices=kv_indices,
            offset=0,
        )
        self.connector.store_kv(store_meta)
        torch.xpu.synchronize()

        for i in range(self.layer_num):
            self.k_buffer[i].zero_()
            self.v_buffer[i].zero_()

        load_meta = LoadMetadata(
            token_ids=token_ids,
            slot_mapping=kv_indices,
            offset=0,
        )
        ret = self.connector.start_load_kv(load_meta)
        self.assertEqual(ret, self.INPUT_LEN)

        for i in range(self.layer_num):
            torch.xpu.synchronize()
            self.connector.load_kv_layerwise(i)
        torch.xpu.synchronize()

        for i in range(self.layer_num):
            self.assertEqual(self.k_buffer[i].dtype, torch.bfloat16)
            self.assertEqual(self.v_buffer[i].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
