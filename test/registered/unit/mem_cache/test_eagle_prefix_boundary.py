"""Convert reusable KV rows to raw-token limits for bigram prefix matching."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import FullComponent, UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_cache(backend, page, *, bigram=True):
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=page,
        is_eagle=bigram,
        tree_components=(ComponentType.FULL,),
        tree_core_backend=backend if backend != "legacy" else None,
    )
    if backend == "legacy":
        return RadixCache(params)
    cache = UnifiedRadixCache(params)
    # A Rust fallback is not evidence that the Rust implementation passed.
    expected = "RustUnifiedTreeCore" if backend == "rust" else "UnifiedTreeCore"
    assert type(cache.tree_core).__name__ == expected
    return cache


def insert(cache, tokens):
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)),
            value=torch.arange(len(tokens), dtype=torch.int64),
        )
    )


def request_match(
    cache,
    tokens,
    *,
    mode="null",
    logprob_start=-1,
    return_logprob=False,
    session=None,
    multi_layer=False,
    embed_override=None,
):
    req = Req(
        "boundary",
        "",
        array("q", tokens),
        SamplingParams(temperature=0),
        return_logprob=return_logprob,
    )
    req.logprob_start_len = logprob_start
    req.session = session
    req.positional_embed_overrides = embed_override
    with (
        patch(
            "sglang.srt.managers.schedule_batch.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode=mode),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.get_spec",
            return_value=SimpleNamespace(enable_multi_layer_eagle=multi_layer),
        ),
    ):
        req.init_next_round_input(cache)
    return req


class TestEaglePrefixBoundary(CustomTestCase):
    def test_execution_and_logprob_caps(self):
        for backend in ("legacy", "python", "rust"):
            for page in (1, 64):
                for bigram in (False, True):
                    cache = make_cache(backend, page, bigram=bigram)
                    raw = list(range(70))
                    insert(cache, raw)
                    # Short inputs and one page boundary; no algorithm-name matrix.
                    lengths = sorted({0, 1, page - 1, page, page + 1, page + 2})
                    for length in lengths:
                        with self.subTest(
                            backend=backend, page=page, bigram=bigram, length=length
                        ):
                            req = request_match(cache, raw[:length])
                            expected = max(0, length - 1) // page * page
                            torch.testing.assert_close(
                                req.prefix_indices, torch.arange(expected)
                            )
                    # Logprob caps below, on, and above the execution/page boundary.
                    cases = (
                        ((4, 0, 0), (4, 1, 1), (4, 2, 2), (4, 4, 3))
                        if page == 1
                        else (
                            (66, 0, 0),
                            (66, 63, 0),
                            (66, 64, 64),
                            (66, 65, 64),
                            (66, 67, 64),
                        )
                    )
                    for length, start, expected in cases:
                        with self.subTest(
                            backend=backend,
                            page=page,
                            bigram=bigram,
                            length=length,
                            logprob_start=start,
                        ):
                            req = request_match(
                                cache,
                                raw[:length],
                                return_logprob=True,
                                logprob_start=start,
                            )
                            torch.testing.assert_close(
                                req.prefix_indices, torch.arange(expected)
                            )

    def test_successor_and_truncated_key(self):
        for backend in ("legacy", "python", "rust"):
            for page in (1, 64):
                cache = make_cache(backend, page)
                raw = list(range(140))
                insert(cache, raw)
                shared_lengths = sorted(
                    {k * page + offset for k in (1, 2) for offset in (-1, 0, 1, 2)}
                    - {0}
                )
                for shared in shared_lengths:
                    for same_successor in (False, True):
                        query = raw.copy()
                        query[shared + int(same_successor)] = -1
                        with self.subTest(
                            backend=backend,
                            page=page,
                            shared=shared,
                            same_successor=same_successor,
                        ):
                            req = request_match(cache, query)
                            expected = (shared - 1 + int(same_successor)) // page * page
                            torch.testing.assert_close(
                                req.prefix_indices, torch.arange(expected)
                            )
                    # A truncated raw key cannot prove its unknown successor.
                    key = RadixKey(array("q", raw), limit=shared)
                    result = cache.match_prefix(MatchPrefixParams(key=key))
                    torch.testing.assert_close(
                        result.device_indices, torch.arange((shared - 1) // page * page)
                    )

    def test_other_request_modes_keep_existing_limits(self):
        raw = list(range(70))
        for backend in ("legacy", "python", "rust"):
            cache = make_cache(backend, 64)
            insert(cache, raw)
            for mode, multi_layer in (
                ("prefill", False),
                ("decode", False),
                ("null", True),
            ):
                with self.subTest(backend=backend, mode=mode, multi_layer=multi_layer):
                    req = request_match(
                        cache, raw[:65], mode=mode, multi_layer=multi_layer
                    )
                    self.assertEqual(len(req.prefix_indices), 0)

    def test_session_keeps_existing_limit(self):
        cache = make_cache("legacy", 64)
        raw = list(range(70))
        insert(cache, raw)
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                session = SimpleNamespace(streaming=streaming)
                with patch.object(
                    cache, "match_prefix", wraps=cache.match_prefix
                ) as match_prefix:
                    req = request_match(cache, raw[:65], session=session)
                self.assertEqual(match_prefix.call_args.args[0].key.limit, 64)
                self.assertEqual(len(req.prefix_indices), 0)

    def test_none_limit_skips_conversion(self):
        cache = make_cache("legacy", 64)
        raw = list(range(70))
        insert(cache, raw)
        with (
            patch.object(
                cache,
                "get_match_key_raw_token_limit",
                wraps=cache.get_match_key_raw_token_limit,
            ) as convert_limit,
            patch.object(
                cache, "match_prefix", wraps=cache.match_prefix
            ) as match_prefix,
        ):
            # Embed overrides select the existing empty-key/None-limit path.
            req = request_match(cache, raw[:65], embed_override=object())
        convert_limit.assert_not_called()
        self.assertIsNone(match_prefix.call_args.args[0].key.limit)
        self.assertEqual(len(req.prefix_indices), 0)

    def test_unsupported_cache_contracts(self):
        class DerivedRadix(RadixCache):
            pass

        params = CacheInitParams(False, None, None, 64, is_eagle=True)
        self.assertEqual(
            DerivedRadix(params).get_match_key_raw_token_limit(max_reusable_kv_rows=64),
            64,
        )

        class DerivedUnified(UnifiedRadixCache):
            pass

        class CustomFull(FullComponent):
            pass

        # Explicit expected limits; this checks opt-in without allocating other layouts.
        full = (ComponentType.FULL,)
        for cls, components, bigram, hicache, full_cls, expected in (
            (UnifiedRadixCache, full, True, False, FullComponent, 65),
            (DerivedUnified, full, True, False, FullComponent, 64),
            (
                UnifiedRadixCache,
                (ComponentType.FULL, ComponentType.SWA),
                True,
                False,
                FullComponent,
                64,
            ),
            (
                UnifiedRadixCache,
                (ComponentType.FULL, ComponentType.MAMBA),
                True,
                False,
                FullComponent,
                64,
            ),
            (
                UnifiedRadixCache,
                (ComponentType.FULL, ComponentType.C128),
                True,
                False,
                FullComponent,
                64,
            ),
            (UnifiedRadixCache, full, False, False, FullComponent, 64),
            (UnifiedRadixCache, full, True, True, FullComponent, 64),
            (UnifiedRadixCache, full, True, False, CustomFull, 64),
        ):
            with self.subTest(
                cls=cls,
                components=components,
                bigram=bigram,
                hicache=hicache,
                full_cls=full_cls,
            ):
                cache = cls.__new__(cls)
                cache.tree_components = components
                cache.tree_core = SimpleNamespace(
                    is_eagle=bigram, enable_hicache=hicache
                )
                cache.components = {ComponentType.FULL: full_cls.__new__(full_cls)}
                self.assertEqual(
                    cache.get_match_key_raw_token_limit(max_reusable_kv_rows=64),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
