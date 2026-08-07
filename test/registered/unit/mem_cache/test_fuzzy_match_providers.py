"""Unit tests for fuzzy-match provider integration edges.

Covers the provider factory contract, config validation at construction,
the SemBlend adapter seams (decode/extra_key threading, config key names,
version gate), the adapter->sglang result translation, and donor lock_ref
accounting across multiple concurrent recipients.

The `semblend` package is faked through ``sys.modules`` so none of these
tests require the real dependency.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import threading
import types
import unittest
from array import array
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import (
    FuzzyMatchProvider,
    FuzzyMatchResult,
    create_fuzzy_match_provider,
)
from sglang.srt.mem_cache.fuzzy_match.fuzzy_radix_cache import FuzzyRadixCache
from sglang.srt.mem_cache.fuzzy_match.semantic_embedding import (
    SemanticEmbeddingProvider,
    _adapter_to_sglang_result,
    _version_lt,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.test_utils import CustomTestCase


def _install_fake_semblend(version=None):
    """Fake the `semblend` package via sys.modules.

    Returns ``(patcher, adapter_instances)``: a ``patch.dict`` context manager
    for ``sys.modules`` and the list that collects every adapter the provider
    constructs while the patch is active.
    """
    if version is None:
        version = SemanticEmbeddingProvider._MIN_SEMBLEND_VERSION
    adapter_instances = []

    class FakeConfig:
        @classmethod
        def from_dict(cls, values):
            inst = cls()
            inst.values = dict(values)
            return inst

    class FakeAdapter:
        def __init__(self, config):
            self.config = config
            self.match_calls = []
            adapter_instances.append(self)

        def match(
            self,
            prompt_token_ids,
            already_matched_len,
            *,
            prompt_text=None,
            extra_key=None,
        ):
            self.match_calls.append(
                {
                    "prompt_token_ids": list(prompt_token_ids),
                    "already_matched_len": already_matched_len,
                    "prompt_text": prompt_text,
                    "extra_key": extra_key,
                }
            )
            return None

    fake_semblend = types.ModuleType("semblend")
    fake_semblend.__version__ = version
    config_mod = types.ModuleType("semblend.integration.sglang.config")
    config_mod.SemBlendProviderConfig = FakeConfig
    provider_mod = types.ModuleType("semblend.integration.sglang.provider")
    provider_mod.SemBlendProviderAdapter = FakeAdapter
    modules = {
        "semblend": fake_semblend,
        "semblend.integration": types.ModuleType("semblend.integration"),
        "semblend.integration.sglang": types.ModuleType("semblend.integration.sglang"),
        "semblend.integration.sglang.config": config_mod,
        "semblend.integration.sglang.provider": provider_mod,
    }
    return patch.dict(sys.modules, modules), adapter_instances


class _JoinTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return "decoded:" + ",".join(map(str, token_ids))


class TestCreateFuzzyMatchProvider(CustomTestCase):
    def test_returns_none_when_disabled(self):
        """A config with fuzzy matching disabled must not build a provider;
        a regression here silently turns fuzzy matching always-on."""
        self.assertIsNone(create_fuzzy_match_provider(FuzzyMatchConfig()))

    def test_builds_semantic_embedding_when_enabled(self):
        """The factory must route the default provider name to
        SemanticEmbeddingProvider; a broken lookup leaves the backend
        exact-only despite fuzzy being enabled."""
        patcher, _adapters = _install_fake_semblend()
        with patcher:
            provider = create_fuzzy_match_provider(
                FuzzyMatchConfig(enable_fuzzy_match=True)
            )
        self.assertIsInstance(provider, SemanticEmbeddingProvider)


class TestFuzzyMatchConfig(CustomTestCase):
    def test_validation_runs_at_construction(self):
        """Out-of-range values must be rejected when the struct is built.

        Guards the msgspec.Struct migration: if ``__post_init__`` stops being
        invoked on construction, invalid CLI values pass through silently and
        only fail deep inside the provider at runtime.
        """
        with self.assertRaises(ValueError):
            FuzzyMatchConfig(fuzzy_min_match_length=0)
        with self.assertRaises(ValueError):
            FuzzyMatchConfig(fuzzy_semantic_threshold=1.5)
        with self.assertRaises(ValueError):
            FuzzyMatchConfig(fuzzy_min_reuse_ratio=0.0)
        with self.assertRaises(ValueError):
            FuzzyMatchConfig(fuzzy_match_provider="NotAProvider")


class TestSemanticEmbeddingProvider(CustomTestCase):
    def test_match_decodes_query_suffix_and_passes_extra_key(self):
        """The adapter must receive the decoded *unmatched suffix* text plus
        the caller's extra_key; dropping either breaks semantic lookup or
        leaks matches across extra_key namespaces."""
        patcher, adapters = _install_fake_semblend()
        with patcher:
            provider = SemanticEmbeddingProvider(
                FuzzyMatchConfig(enable_fuzzy_match=True, fuzzy_min_match_length=1)
            )
            provider.match_on_prefix_miss(
                prompt_token_ids=[1, 2, 3, 4],
                already_matched_len=1,
                request=SimpleNamespace(tokenizer=_JoinTokenizer()),
                extra_key="tenant-a",
            )
        call = adapters[-1].match_calls[-1]
        self.assertEqual(call["prompt_token_ids"], [1, 2, 3, 4])
        self.assertEqual(call["already_matched_len"], 1)
        self.assertEqual(call["prompt_text"], "decoded:2,3,4")
        self.assertEqual(call["extra_key"], "tenant-a")

    def test_adapter_config_receives_expected_semblend_keys(self):
        """SemBlend's ``from_dict`` filters unknown keys silently, so every
        tuning field must arrive under exactly the key name SemBlend reads;
        a renamed/dropped key becomes a config flag with no effect."""
        patcher, adapters = _install_fake_semblend()
        config = FuzzyMatchConfig(
            enable_fuzzy_match=True,
            fuzzy_min_match_length=4,
            fuzzy_semantic_threshold=0.75,
            fuzzy_min_reuse_ratio=0.6,
            semantic_max_entries=123,
            fuzzy_block_size=8,
            embedding_model_name="test-embedder",
            model_arch="llama",
        )
        with patcher:
            SemanticEmbeddingProvider(config)
        self.assertEqual(
            adapters[-1].config.values,
            {
                "min_similarity": 0.75,
                "min_reuse_ratio": 0.6,
                "min_match_length": 4,
                "max_entries": 123,
                "block_size": 8,
                "embedding_model_name": "test-embedder",
                "model_arch": "llama",
            },
        )

    def test_semblend_below_minimum_version_raises_import_error(self):
        """An outdated semblend must be rejected at provider construction;
        without the gate, missing integration entrypoints surface as
        confusing AttributeErrors mid-request."""
        patcher, _adapters = _install_fake_semblend(version="0.0.1")
        with patcher:
            with self.assertRaises(ImportError) as caught:
                SemanticEmbeddingProvider(FuzzyMatchConfig(enable_fuzzy_match=True))
        self.assertIn(
            SemanticEmbeddingProvider._MIN_SEMBLEND_VERSION, str(caught.exception)
        )

    def test_version_lt_compares_numerically_not_lexicographically(self):
        """Version components must compare as integers: a string-compare
        rewrite would wrongly reject 0.3.102 as older than 0.3.12."""
        self.assertTrue(_version_lt("0.3.9", "0.3.12"))
        self.assertFalse(_version_lt("0.3.102", "0.3.12"))
        self.assertFalse(_version_lt("0.3.12", "0.3.12"))
        self.assertFalse(_version_lt("0.3.12.dev0", "0.3.12"))
        self.assertTrue(_version_lt("0.3.12", "0.4"))


class TestAdapterResultTranslation(CustomTestCase):
    def test_translation_maps_fields_and_renames_match_entry(self):
        """The semblend-side ``_match_entry`` handle must land on the
        sglang-side ``match_entry`` field, index lists must be coerced to
        int64 tensors, and segment/quality/donor fields must survive the
        copy; a silent drop here loses donor addressing or the provider's
        internal handle."""
        sentinel = object()
        adapter_result = SimpleNamespace(
            cached_token_count=2,
            cached_token_ids=[7, 8],
            prompt_token_count=4,
            kv_cache_indices=[10, 11],
            position_offset=3,
            cached_start_pos=1,
            segments=[
                SimpleNamespace(
                    target_positions=[5, 6],
                    donor_positions=[2, 3],
                    donor_node_id=42,
                    donor_offset=1,
                    length=2,
                    donor_kv_indices=[8, 9],
                    donor_req_id="donor-1",
                    layer_recompute_mask=[True, False],
                )
            ],
            layer_recompute_mask=None,
            quality_signals=SimpleNamespace(
                cosine_similarity=0.9,
                reuse_ratio=0.8,
                confidence_tier="high",
                passed_quality_gate=True,
                rejection_reason=None,
            ),
            donor_last_node_id=42,
            _match_entry=sentinel,
        )

        result = _adapter_to_sglang_result(adapter_result)

        self.assertIs(result.match_entry, sentinel)
        self.assertEqual(result.donor_last_node_id, 42)
        self.assertEqual(result.kv_cache_indices.tolist(), [10, 11])
        self.assertEqual(result.kv_cache_indices.dtype, torch.int64)
        self.assertEqual(result.cached_start_pos, 1)
        segment = result.segments[0]
        self.assertEqual(segment.donor_node_id, 42)
        self.assertEqual(segment.donor_kv_indices.tolist(), [8, 9])
        self.assertEqual(segment.target_positions.tolist(), [5, 6])
        self.assertEqual(result.quality_signals.confidence_tier, "high")
        self.assertTrue(result.quality_signals.passed_quality_gate)


class _DonorScriptedProvider(FuzzyMatchProvider):
    """Provider that answers every miss with one pre-built result."""

    def __init__(self, config, result):
        super().__init__(config)
        self._result = result

    def cache_on_request_finished(
        self,
        request,
        token_ids,
        kv_cache,
        cache_start_pos,
        cache_end_pos,
        radix_tree=None,
    ):
        return False

    def match_on_prefix_miss(
        self, prompt_token_ids, already_matched_len, request=None, extra_key=None
    ):
        return self._result


class _DeviceOnlyAllocator:
    device = "cpu"

    def __init__(self):
        self._next = 1000
        self._lock = threading.Lock()

    def alloc(self, size: int):
        with self._lock:
            start = self._next
            self._next += size
        return torch.arange(start, start + size, dtype=torch.int64)

    def free(self, indices):
        return None


class TestFuzzyDonorLockAccounting(CustomTestCase):
    def test_concurrent_matches_lock_same_donor_node(self):
        """N recipients matching the same donor must each take exactly one
        lock_ref pin (net N, and net zero after release). A regression that
        skips the pin for already-locked donors -- or double-releases -- lets
        LRU eviction free donor KV mid-forward."""
        cache = FuzzyRadixCache(
            params=CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=_DeviceOnlyAllocator(),
                page_size=1,
            )
        )
        donor_insert = cache.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", [1, 2, 3, 4]), extra_key=None),
                value=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            )
        )
        donor = donor_insert.last_device_node

        config = FuzzyMatchConfig(
            enable_fuzzy_match=True,
            fuzzy_min_match_length=1,
            cache_fuzzy_results=False,
            fuzzy_min_suffix_tokens=0,
        )
        scripted = FuzzyMatchResult(
            cached_token_count=2,
            cached_token_ids=[1, 2],
            prompt_token_count=3,
            kv_cache_indices=torch.tensor([10, 11], dtype=torch.int64),
            position_offset=0,
            # Non-aligned (donor positions differ from the zero-length exact
            # anchor) so the match takes the realization path rather than
            # being dropped as position-aligned.
            cached_start_pos=2,
            donor_last_node_id=donor.id,
        )
        cache.init_fuzzy_match(
            config=config,
            provider=_DonorScriptedProvider(config=config, result=scripted),
        )

        reqs = [
            SimpleNamespace(
                rid=f"req-{i}",
                fuzzy_donor_node=None,
                fuzzy_realized_locs=None,
                fuzzy_match_result=None,
            )
            for i in range(8)
        ]

        def run_match(req):
            result = cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(token_ids=array("q", [90, 91, 92]), extra_key=None),
                    req=req,
                )
            )
            self.assertEqual(result.fuzzy_matched_len, 2)
            self.assertIs(req.fuzzy_donor_node, donor)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(run_match, reqs))

        self.assertEqual(donor.lock_ref, len(reqs))

        for req in reqs:
            cache.dec_lock_ref(req.fuzzy_donor_node)
            req.fuzzy_donor_node = None

        self.assertEqual(donor.lock_ref, 0)


if __name__ == "__main__":
    unittest.main()
