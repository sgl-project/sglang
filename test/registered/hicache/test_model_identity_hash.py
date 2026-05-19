"""Unit tests for model identity hash computation and storage key isolation."""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-only hash computation, runs in milliseconds on any runner
register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

import unittest

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.hiradix_cache import compute_model_identity_hash


class FakeServerArgs:
    """Minimal mock of ServerArgs for testing identity hash computation."""

    def __init__(
        self,
        model_path="meta-llama/Llama-3-8B",
        revision=None,
        dtype="float16",
        quantization=None,
        kv_cache_dtype=None,
    ):
        self.model_path = model_path
        self.revision = revision
        self.dtype = dtype
        self.quantization = quantization
        self.kv_cache_dtype = kv_cache_dtype


class TestComputeModelIdentityHash(unittest.TestCase):
    """Tests for compute_model_identity_hash()."""

    def test_basic_hash_determinism(self):
        args = FakeServerArgs()
        h1 = compute_model_identity_hash(args)
        h2 = compute_model_identity_hash(args)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

    def test_different_models_produce_different_hashes(self):
        args_a = FakeServerArgs(model_path="meta-llama/Llama-3-8B")
        args_b = FakeServerArgs(model_path="Qwen/Qwen2-7B")
        self.assertNotEqual(
            compute_model_identity_hash(args_a),
            compute_model_identity_hash(args_b),
        )

    def test_different_dtype_produces_different_hash(self):
        args_fp16 = FakeServerArgs(dtype="float16")
        args_bf16 = FakeServerArgs(dtype="bfloat16")
        self.assertNotEqual(
            compute_model_identity_hash(args_fp16),
            compute_model_identity_hash(args_bf16),
        )

    def test_different_quantization_produces_different_hash(self):
        args_none = FakeServerArgs(quantization=None)
        args_awq = FakeServerArgs(quantization="awq")
        self.assertNotEqual(
            compute_model_identity_hash(args_none),
            compute_model_identity_hash(args_awq),
        )

    def test_path_normalization(self):
        args_trailing = FakeServerArgs(model_path="/models/llama/")
        args_clean = FakeServerArgs(model_path="/models/llama")
        self.assertEqual(
            compute_model_identity_hash(args_trailing),
            compute_model_identity_hash(args_clean),
        )

    def test_dtype_normalization_case(self):
        args_upper = FakeServerArgs(dtype="Float16")
        args_lower = FakeServerArgs(dtype="float16")
        self.assertEqual(
            compute_model_identity_hash(args_upper),
            compute_model_identity_hash(args_lower),
        )

    def test_dtype_none_equals_auto(self):
        args_none = FakeServerArgs(dtype=None)
        args_auto = FakeServerArgs(dtype="auto")
        self.assertEqual(
            compute_model_identity_hash(args_none),
            compute_model_identity_hash(args_auto),
        )

    def test_kv_cache_dtype_none_equals_auto(self):
        args_none = FakeServerArgs(kv_cache_dtype=None)
        args_auto = FakeServerArgs(kv_cache_dtype="auto")
        self.assertEqual(
            compute_model_identity_hash(args_none),
            compute_model_identity_hash(args_auto),
        )

    def test_empty_model_path(self):
        args = FakeServerArgs(model_path=None)
        h = compute_model_identity_hash(args)
        self.assertEqual(len(h), 16)


class TestHiCacheFileSuffix(unittest.TestCase):
    """Tests for HiCacheFile config_suffix with identity hash."""

    def _make_config(self, identity_hash=None, **kwargs):
        defaults = dict(
            tp_rank=0,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="test-model",
            model_identity_hash=identity_hash,
        )
        defaults.update(kwargs)
        return HiCacheStorageConfig(**defaults)

    def test_suffix_includes_identity_hash(self):
        config = self._make_config(identity_hash="abc123def456")
        backend = HiCacheFile(config, file_path="/tmp/test_hicache_id")
        self.assertIn("abc123def456", backend.config_suffix)

    def test_suffix_without_identity_hash(self):
        config = self._make_config(identity_hash=None)
        backend = HiCacheFile(config, file_path="/tmp/test_hicache_id")
        self.assertIn("test-model", backend.config_suffix)
        self.assertNotIn("None", backend.config_suffix)

    def test_different_hash_different_suffix(self):
        config_a = self._make_config(identity_hash="aaaa")
        config_b = self._make_config(identity_hash="bbbb")
        backend_a = HiCacheFile(config_a, file_path="/tmp/test_hicache_id")
        backend_b = HiCacheFile(config_b, file_path="/tmp/test_hicache_id")
        self.assertNotEqual(backend_a.config_suffix, backend_b.config_suffix)

    def test_pp_suffix_present_when_pp_enabled(self):
        config = self._make_config(identity_hash="hash1", pp_size=4, pp_rank=2)
        backend = HiCacheFile(config, file_path="/tmp/test_hicache_id")
        self.assertIn("4_2", backend.config_suffix)
        self.assertIn("hash1", backend.config_suffix)

    def test_mla_model_omits_tp_from_suffix(self):
        config = self._make_config(
            identity_hash="mlahash", is_mla_model=True, tp_rank=3, tp_size=8
        )
        backend = HiCacheFile(config, file_path="/tmp/test_hicache_id")
        self.assertIn("mlahash", backend.config_suffix)
        self.assertNotIn("3_8", backend.config_suffix)


if __name__ == "__main__":
    unittest.main()
