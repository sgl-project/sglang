"""Unified file objects retain their byte identity across TP/PP reshaping."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_key_scheme import (
    plan_unified_draft_sidecar,
    plan_unified_kv,
    with_packed_draft,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
    UnifiedPoolChunks,
)
from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _HostPool:
    def __init__(self, layout, page):
        self.page_unified_layout = layout
        self.page_size = layout.page_size
        self.page = page

    def get_data_page(self, offset, flat=False):
        assert offset == 0
        return self.page.flatten() if flat else self.page

    def get_dummy_flat_data_page(self):
        return torch.empty_like(self.page).flatten()

    def set_from_flat_data_page(self, offset, page):
        assert offset == 0
        self.page.copy_(page.reshape_as(self.page))


def _logical_page(draft_layers=0):
    """One page of the whole model: 4 heads x (4 target layers + drafts).

    Every scalar identifies an absolute head/layer/component/token coordinate,
    the same one whether or not draft layers follow.
    """
    full = torch.arange(4 * 6 * 2 * 4 * 4, dtype=torch.int32).reshape(4, 6, 2, 4, 1, 4)
    return full[:, : 4 + draft_layers]


def _rank(tp_size=1, pp_size=1, tp_rank=0, pp_rank=0, draft_layers=0, draft_id="x"):
    """One rank's config, layout and page. ``draft_layers`` packed draft layers
    follow the target layers on the final pipeline stage, as a host pool with
    packed drafts lays them out."""
    full = _logical_page(draft_layers)
    heads, layers = 4 // tp_size, 4 // pp_size
    start_layer = pp_rank * layers
    head_slice = slice(tp_rank * heads, (tp_rank + 1) * heads)
    page = full[head_slice, start_layer : start_layer + layers]
    is_final_stage = pp_rank == pp_size - 1
    if not is_final_stage:
        draft_layers = 0
    page = torch.cat([page, full[head_slice, 4 : 4 + draft_layers]], dim=1).contiguous()
    layout = PageUnifiedLayout(
        page_size=4,
        layer_num=layers + draft_layers,
        head_num=heads,
        head_group_num=heads,
        head_dim=4,
        itemsize=4,
    )
    plan = plan_unified_kv(
        model_id="test/model",
        dtype="int32",
        page_size=layout.page_size,
        rank_replicated=False,
        local_kv_heads=heads,
        attn_tp_rank=tp_rank,
        attn_tp_size=tp_size,
        attn_cp_size=1,
        start_layer=start_layer,
        end_layer=start_layer + layers,
        is_final_stage=pp_rank == pp_size - 1,
        head_group=1,
        layer_partition=2,
    )
    if draft_layers:
        plan = with_packed_draft(
            plan,
            target_layer_num=layers,
            draft_layer_num=draft_layers,
            algorithm="EAGLE",
            draft_id=draft_id,
            dtype="int32",
            row_width=layout.head_dim,
            attn_tp_rank=tp_rank,
            attn_cp_size=1,
            is_final_stage=True,
        )
    config = HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="test/model",
        unified_suffixes=plan.suffixes,
        unified_chunk_spans=[
            layout.chunk_span(layer_range, head_group)
            for layer_range in plan.layer_ranges
            for head_group in range(plan.head_group_num)
        ],
        extra_config={
            "enable_metadata_cache": True,
            "metadata_ttl": -1,
            "max_size": 0,
            "min_free_space": 0,
        },
    )
    return config, layout, page


class TestUnifiedFileChunks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        context = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(
            self.directory.name
        )
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def _backend(self, config, layout=None, page=None):
        backend = HiCacheFile(config)
        if layout is not None:
            pool = _HostPool(layout, page)
            backend.register_mem_pool_host(pool)
            backend.register_mem_host_pool_v2(pool, PoolName.KV)
        return backend

    def _path(self, backend, key, suffix):
        return Path(backend.file_path) / f"{key}_{suffix}.bin"

    def test_reshard_tp_and_pp_in_both_directions(self):
        for writer, reader in (((2, 1), (1, 2)), ((1, 2), (2, 1))):
            with self.subTest(writer=writer, reader=reader):
                key = f"page-{writer}"
                writer_files = set()
                for pp_rank in range(writer[1]):
                    for tp_rank in range(writer[0]):
                        config, layout, page = _rank(*writer, tp_rank, pp_rank)
                        backend = self._backend(config, layout, page)
                        transfer = PoolTransfer(
                            name=PoolName.KV,
                            keys=[key],
                            host_indices=torch.arange(layout.page_size),
                        )
                        self.assertEqual(
                            backend.batch_set_v2([transfer]), {PoolName.KV: [True]}
                        )
                        writer_files.update(backend._get_component_keys(key))
                reader_files = set()
                for pp_rank in range(reader[1]):
                    for tp_rank in range(reader[0]):
                        config, layout, expected = _rank(*reader, tp_rank, pp_rank)
                        target = torch.full_like(expected, -1)
                        backend = self._backend(config, layout, target)
                        self.assertEqual(backend.batch_exists_v2([key]).kv_hit_pages, 1)
                        transfer = PoolTransfer(
                            name=PoolName.KV,
                            keys=[key],
                            host_indices=torch.arange(layout.page_size),
                        )
                        self.assertEqual(
                            backend.batch_get_v2([transfer]), {PoolName.KV: [True]}
                        )
                        torch.testing.assert_close(target, expected, rtol=0, atol=0)
                        reader_files.update(backend._get_component_keys(key))
                self.assertEqual(reader_files, writer_files)

    # Packed draft layers: one more chunk per head group, under the draft
    # digest, appended to the rank's chunk list.

    def _set(self, config, layout, page, key="page"):
        backend = self._backend(config, layout, page)
        transfer = PoolTransfer(
            name=PoolName.KV, keys=[key], host_indices=torch.arange(layout.page_size)
        )
        self.assertEqual(backend.batch_set_v2([transfer]), {PoolName.KV: [True]})
        return backend

    def _get(self, config, layout, expected, key="page"):
        """Read into a page of -1; the page, and whether it was restored."""
        target = torch.full_like(expected, -1)
        backend = self._backend(config, layout, target)
        hit = backend.batch_exists_v2([key]).kv_hit_pages == 1
        transfer = PoolTransfer(
            name=PoolName.KV, keys=[key], host_indices=torch.arange(layout.page_size)
        )
        restored = backend.batch_get_v2([transfer]) == {PoolName.KV: [True]}
        self.assertEqual(hit, restored)
        return backend, target, restored

    def _files(self, backend):
        return {
            path.name: path.read_bytes()
            for path in sorted(Path(backend.file_path).iterdir())
        }

    def test_draft_chunks_reshard_with_the_target_across_tp_and_pp(self):
        for writer, reader in (((2, 1), (1, 2)), ((1, 2), (2, 1)), ((2, 2), (1, 1))):
            with self.subTest(writer=writer, reader=reader):
                key = f"page-{writer}-{reader}"
                for pp_rank in range(writer[1]):
                    for tp_rank in range(writer[0]):
                        rank = _rank(*writer, tp_rank, pp_rank, draft_layers=1)
                        backend = self._set(*rank, key=key)
                        draft_keys = [
                            name
                            for name in backend._get_component_keys(key)
                            if "ukd1-" in name
                        ]
                        # Only the final stage owns draft chunks: one per head.
                        final = pp_rank == writer[1] - 1
                        self.assertEqual(
                            len(draft_keys), 4 // writer[0] if final else 0
                        )
                for pp_rank in range(reader[1]):
                    for tp_rank in range(reader[0]):
                        config, layout, expected = _rank(
                            *reader, tp_rank, pp_rank, draft_layers=1
                        )
                        _, target, restored = self._get(config, layout, expected, key)
                        self.assertTrue(restored)
                        torch.testing.assert_close(target, expected, rtol=0, atol=0)

    def test_each_draft_object_holds_its_head_groups_draft_rows(self):
        # Independently of any reader: a consistent swap of two head groups,
        # or a chunk cut from target rows, would still round-trip.
        logical = _logical_page(draft_layers=1)
        for tp_rank in range(2):
            backend = self._set(*_rank(2, 1, tp_rank, 0, draft_layers=1))
        files = self._files(backend)
        draft = {name: data for name, data in files.items() if "ukd1-" in name}
        self.assertEqual(len(draft), 4)
        self.assertEqual(len(files), 4 * 3)
        for head in range(4):
            (name,) = [n for n in draft if n.endswith(f"_D0-1_H{head}.bin")]
            self.assertEqual(
                draft[name], logical[head, 4:5].contiguous().numpy().tobytes()
            )
        for name, data in files.items():
            if "_L2-4_H3" in name:
                self.assertEqual(data, logical[3, 2:4].contiguous().numpy().tobytes())

    def test_reader_without_draft_kv_reads_the_target_chunks_alone(self):
        writer = self._set(*_rank(draft_layers=1))
        for name in list(self._files(writer)):
            if "ukd1-" in name:
                # Not even present: the reader must not need them.
                (Path(writer.file_path) / name).unlink()
        config, layout, expected = _rank()
        reader, target, restored = self._get(config, layout, expected)
        self.assertTrue(restored)
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
        self.assertFalse(any("ukd1-" in k for k in reader._get_component_keys("page")))

    def test_target_only_page_misses_for_a_draft_reader_until_backfilled(self):
        writer = self._set(*_rank())
        before = self._files(writer)
        config, layout, page = _rank(draft_layers=1)
        _, target, restored = self._get(config, layout, page)
        self.assertFalse(restored)
        self.assertTrue(torch.all(target == -1))
        # The draft reader's own backup adds draft objects and nothing else.
        self._set(config, layout, page)
        after = self._files(writer)
        self.assertEqual({name: after[name] for name in before}, before)
        self.assertEqual(len(after) - len(before), 4)
        self.assertTrue(all("ukd1-" in name for name in set(after) - set(before)))
        self.assertTrue(self._get(config, layout, page)[2])

    def test_another_draft_misses_on_the_draft_chunk_alone(self):
        ours = _rank(draft_layers=1, draft_id="draft-a")
        theirs = _rank(draft_layers=1, draft_id="draft-b")
        self.assertEqual(ours[0].unified_suffixes[:-4], theirs[0].unified_suffixes[:-4])
        self.assertFalse(
            set(ours[0].unified_suffixes[-4:]) & set(theirs[0].unified_suffixes[-4:])
        )
        self._set(*ours)
        _, target, restored = self._get(*theirs)
        self.assertFalse(restored)
        self.assertTrue(torch.all(target == -1))
        self.assertTrue(self._get(*ours)[2])

    def test_missing_or_short_draft_chunk_fails_the_whole_page(self):
        for damage in ("missing", "short"):
            with self.subTest(damage=damage):
                config, layout, page = _rank(draft_layers=1)
                key = f"page-{damage}"
                writer = self._set(config, layout, page, key=key)
                path = self._path(writer, key, config.unified_suffixes[-1])
                if damage == "missing":
                    path.unlink()
                else:
                    path.write_bytes(path.read_bytes()[:-1])
                _, target, restored = self._get(config, layout, page, key)
                self.assertFalse(restored)
                # Nothing of the page reaches the host, target rows included.
                self.assertTrue(torch.all(target == -1))

    def test_key_order_does_not_define_byte_order(self):
        config, layout, page = _rank()
        self.assertNotEqual(
            config.unified_chunk_spans, sorted(config.unified_chunk_spans)
        )
        writer = self._backend(config)
        self.assertTrue(writer.set("page", page))
        permutation = [3, 0, 6, 1, 7, 4, 2, 5]
        reader = self._backend(
            replace(
                config,
                unified_suffixes=[config.unified_suffixes[i] for i in permutation],
                unified_chunk_spans=[
                    config.unified_chunk_spans[i] for i in permutation
                ],
            )
        )
        target = torch.zeros_like(page)
        self.assertIs(reader.get("page", target), target)
        torch.testing.assert_close(target, page, rtol=0, atol=0)

    def test_missing_or_truncated_chunk_is_a_miss_with_cached_metadata(self):
        for truncate in (False, True):
            with self.subTest(truncate=truncate):
                config, _, page = _rank()
                writer = self._backend(config)
                keys = [f"{truncate}-{i}" for i in range(3)]
                self.assertTrue(writer.batch_set(keys, [page] * 3))
                reader = self._backend(config)
                stems = reader._get_component_keys(keys[1])
                self.assertTrue(
                    all(reader.metadata_cache.contains(stem) for stem in stems)
                )
                path = self._path(reader, keys[1], config.unified_suffixes[-1])
                if truncate:
                    path.write_bytes(b"short")
                else:
                    path.unlink()
                self.assertFalse(reader.exists(keys[1]))
                self.assertEqual(reader.batch_exists_v2(keys).kv_hit_pages, 1)
                target = torch.full_like(page, -1)
                self.assertIsNone(reader.get(keys[1], target))
                self.assertTrue(torch.all(target == -1))
                self.assertFalse(reader.metadata_cache.contains(stems[-1]))
                # A later backup repairs the missing or malformed object.
                self.assertTrue(writer.set(keys[1], page))
                self.assertEqual(reader.batch_exists_v2(keys).kv_hit_pages, 3)

    def test_read_failure_does_not_publish_partial_page(self):
        config, _, page = _rank()
        backend = self._backend(config)
        self.assertTrue(backend.set("page", page))
        read_file = backend._get_file
        failed_key = backend._get_component_keys("page")[1]

        def read(key, target):
            return None if key == failed_key else read_file(key, target)

        target = torch.full_like(page, -1)
        with patch.object(backend, "_get_file", side_effect=read):
            self.assertIsNone(backend.get("page", target))
        self.assertTrue(torch.all(target == -1))

    def test_lru_tracks_all_chunks_and_never_acknowledges_partial_page(self):
        config, _, page = _rank()
        config = replace(
            config,
            extra_config={
                **config.extra_config,
                "max_size": page.numel() * page.element_size(),
            },
        )
        writer = self._backend(config)
        self.assertTrue(writer.set("old", page))
        self.assertEqual(len(writer._evictor._lru), len(config.unified_suffixes))
        peer = self._backend(replace(config, attn_cp_rank=1, attn_cp_size=2))
        self.assertEqual(peer._evictor._owned_suffixes, ())
        self.assertEqual(len(peer._evictor._lru), 0)
        self.assertTrue(peer.exists("old"))
        self.assertFalse(peer.set("peer", page))
        self.assertTrue(writer.set("new", page))
        self.assertFalse(peer.exists("old"))
        self.assertTrue(peer.exists("new"))
        too_small = self._backend(
            replace(config, extra_config={**config.extra_config, "max_size": 256})
        )
        self.assertFalse(too_small.set("cannot-fit", page))
        self.assertFalse(too_small.exists("cannot-fit"))

    def test_invalid_spans_and_page_sizes_are_rejected(self):
        config, layout, page = _rank()
        for spans in (None, [], [(0, 10)] * 8, [(1 + i * 256, 256) for i in range(8)]):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                self._backend(replace(config, unified_chunk_spans=spans))
        backend = self._backend(config)
        with self.assertRaisesRegex(ValueError, "page has"):
            backend.set("wrong-size", torch.empty(1, dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "page has"):
            backend.register_mem_pool_host(
                SimpleNamespace(page_unified_layout=SimpleNamespace(bytes_per_page=1))
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            self._backend(replace(config, unified_suffixes=["same"] * 8))

    def test_single_chunk_and_legacy_keys_remain_compatible(self):
        config, _, _ = _rank()
        value = torch.arange(64, dtype=torch.uint8)
        single = replace(
            config, unified_suffixes=["namespace_L0-4_H0"], unified_chunk_spans=None
        )
        writer = self._backend(single)
        self.assertTrue(writer.set("page", value))
        self.assertTrue(self._path(writer, "page", single.unified_suffixes[0]).exists())
        reader = self._backend(replace(single, unified_chunk_spans=[(0, 64)]))
        target = torch.empty_like(value)
        self.assertIs(reader.get("page", target), target)
        torch.testing.assert_close(target, value)
        legacy = self._backend(
            replace(config, unified_suffixes=None, unified_chunk_spans=None)
        )
        self.assertEqual(legacy.config_suffix, "_test-model_0_1")
        self.assertTrue(legacy.set("legacy", value))
        self.assertIs(legacy.get("legacy", target), target)
        torch.testing.assert_close(target, value)


class _PagedHostPool:
    """Several pages; the host index of a page's first token is page * page_size."""

    def __init__(self, layout, pages):
        self.page_unified_layout = layout
        self.page_size = layout.page_size
        self.pages = pages

    def get_data_page(self, offset, flat=False):
        page = self.pages[offset // self.page_size]
        return page.flatten() if flat else page

    def get_dummy_flat_data_page(self):
        return torch.empty_like(self.pages[0]).flatten()

    def set_from_flat_data_page(self, offset, page):
        self.pages[offset // self.page_size].copy_(page.reshape_as(self.pages[0]))


_PAGES = 3


def _draft_logical_pages():
    """The whole draft's pages: 4 heads x 3 layers of 8-wide int16 rows, a
    geometry of its own beside the target's 4 layers of 4-wide int32 rows.
    Every scalar identifies its page/head/layer/component/token coordinate."""
    return torch.arange(_PAGES * 4 * 3 * 2 * 4 * 8, dtype=torch.int16).reshape(
        _PAGES, 4, 3, 2, 4, 1, 8
    )


def _draft_rank(tp_size=1, tp_rank=0, draft_id="x", is_writer=True):
    """One rank's planned chunks, layout and pages of the draft KV pool."""
    heads = 4 // tp_size
    layout = PageUnifiedLayout(
        page_size=4,
        layer_num=3,
        head_num=heads,
        head_group_num=heads,
        head_dim=8,
        itemsize=2,
    )
    target = plan_unified_kv(
        model_id="test/model",
        dtype="int32",
        page_size=4,
        rank_replicated=False,
        local_kv_heads=heads,
        attn_tp_rank=tp_rank,
        attn_tp_size=tp_size,
        attn_cp_size=1,
        start_layer=0,
        end_layer=4,
        is_final_stage=True,
        head_group=1,
        layer_partition=2,
    )
    plan = plan_unified_draft_sidecar(
        target=target.namespace,
        algorithm="DFLASH",
        draft_id=draft_id,
        dtype="int16",
        rank_replicated=False,
        local_kv_heads=heads,
        row_width=8,
        layer_num=3,
        attn_tp_rank=tp_rank,
        attn_tp_size=tp_size,
        attn_cp_size=1,
        head_group=1,
    )
    chunks = UnifiedPoolChunks(
        suffixes=tuple(plan.suffixes),
        spans=tuple(plan.chunk_spans(layout)),
        is_writer=is_writer,
    )
    pages = _draft_logical_pages()[:, tp_rank * heads : (tp_rank + 1) * heads]
    return chunks, layout, pages.contiguous()


class TestUnifiedFileDraftSidecar(unittest.TestCase):
    """A draft KV pool beside the target's: chunks of its own, on its own grid."""

    KEYS = [f"page{i}" for i in range(_PAGES)]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        context = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(
            self.directory.name
        )
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def _backend(self, tp_size=1, tp_rank=0, *, draft=None, kv_pages=None, **config):
        """A rank's backend with its KV pool and, planned by ``draft``, its
        draft pool. Returns the backend and both pools."""
        kv_config, kv_layout, kv_page = _rank(tp_size, 1, tp_rank, 0)
        if kv_pages is None:
            kv_pages = torch.stack([kv_page + 100000 * i for i in range(_PAGES)])
        kv_pool = _PagedHostPool(kv_layout, kv_pages)
        if config:
            kv_config = replace(kv_config, **config)
        backend = HiCacheFile(kv_config)
        backend.register_mem_pool_host(kv_pool)
        backend.register_mem_host_pool_v2(kv_pool, PoolName.KV)
        draft_pool = None
        if draft is not None:
            chunks, layout, pages = draft
            draft_pool = _PagedHostPool(layout, pages)
            # The controller adds the plan before it registers the pool.
            kv_config.unified_pool_chunks[PoolName.DRAFT] = chunks
            backend.register_mem_host_pool_v2(draft_pool, PoolName.DRAFT)
        return backend, kv_pool, draft_pool

    def _transfers(self, *names):
        return [
            PoolTransfer(
                name=name, keys=self.KEYS, host_indices=torch.arange(_PAGES * 4)
            )
            for name in names
        ]

    def _write(self, tp_size=1, **draft):
        backends = []
        for tp_rank in range(tp_size):
            backend, _, _ = self._backend(
                tp_size, tp_rank, draft=_draft_rank(tp_size, tp_rank, **draft)
            )
            self.assertEqual(
                backend.batch_set_v2(self._transfers(PoolName.KV, PoolName.DRAFT)),
                {PoolName.KV: [True] * _PAGES, PoolName.DRAFT: [True] * _PAGES},
            )
            backends.append(backend)
        return backends

    def _files(self):
        return {
            path.name: path.read_bytes()
            for path in sorted(Path(self.directory.name).iterdir())
        }

    def test_each_draft_object_holds_its_head_groups_draft_layers(self):
        self._write(tp_size=2)
        files = self._files()
        draft = {name: data for name, data in files.items() if "ukd1-" in name}
        # 4 head groups a page; the target's 8 chunks a page are as before.
        self.assertEqual(len(draft), _PAGES * 4)
        self.assertEqual(len(files), _PAGES * (4 + 8))
        logical = _draft_logical_pages()
        for page, key in enumerate(self.KEYS):
            for head in range(4):
                (name,) = [
                    n
                    for n in draft
                    if n.startswith(f"{key}_ukd1-") and n.endswith(f"_D0-3_H{head}.bin")
                ]
                self.assertEqual(
                    draft[name], logical[page, head].contiguous().numpy().tobytes()
                )

    def test_target_objects_do_not_change_with_a_draft_pool(self):
        backend, _, _ = self._backend()
        backend.batch_set_v2(self._transfers(PoolName.KV))
        target_only = self._files()
        for name in target_only:
            (Path(self.directory.name) / name).unlink()
        self._write()
        with_draft = self._files()
        self.assertEqual(
            {name: with_draft.get(name) for name in target_only}, target_only
        )
        self.assertTrue(
            all("ukd1-" in name for name in set(with_draft) - set(target_only))
        )

    def test_draft_chunks_reshard_across_tp_sizes(self):
        for writer, reader in ((2, 1), (1, 2)):
            with self.subTest(writer=writer, reader=reader):
                for name in self._files():
                    (Path(self.directory.name) / name).unlink()
                self._write(tp_size=writer)
                for tp_rank in range(reader):
                    chunks, layout, expected = _draft_rank(reader, tp_rank)
                    backend, _, draft_pool = self._backend(
                        reader,
                        tp_rank,
                        draft=(chunks, layout, torch.full_like(expected, -1)),
                    )
                    result = backend.batch_exists_v2(
                        self.KEYS, self._transfers(PoolName.DRAFT)
                    )
                    self.assertEqual(result.kv_hit_pages, _PAGES)
                    self.assertEqual(
                        backend.batch_get_v2(self._transfers(PoolName.DRAFT)),
                        {PoolName.DRAFT: [True] * _PAGES},
                    )
                    torch.testing.assert_close(
                        draft_pool.pages, expected, rtol=0, atol=0
                    )

    def test_a_page_is_a_hit_only_with_every_draft_chunk(self):
        for damage in ("missing", "short"):
            with self.subTest(damage=damage):
                for name in self._files():
                    (Path(self.directory.name) / name).unlink()
                self._write()
                chunks, layout, expected = _draft_rank()
                path = Path(self.directory.name) / f"page1_{chunks.suffixes[-1]}.bin"
                if damage == "missing":
                    path.unlink()
                else:
                    path.write_bytes(path.read_bytes()[:-1])
                backend, _, draft_pool = self._backend(
                    draft=(chunks, layout, torch.full_like(expected, -1))
                )
                result = backend.batch_exists_v2(
                    self.KEYS, self._transfers(PoolName.DRAFT)
                )
                self.assertEqual(result.kv_hit_pages, 1)
                self.assertEqual(result.extra_pool_hit_pages[PoolName.KV], _PAGES)
                self.assertEqual(result.extra_pool_hit_pages[PoolName.DRAFT], 1)
                self.assertEqual(
                    backend.batch_get_v2(self._transfers(PoolName.DRAFT)),
                    {PoolName.DRAFT: [True, False, True]},
                )
                # Nothing of the damaged page reaches the host pool.
                self.assertTrue(torch.all(draft_pool.pages[1] == -1))
                torch.testing.assert_close(
                    draft_pool.pages[0], expected[0], rtol=0, atol=0
                )

    def test_another_draft_misses_on_the_draft_chunks_alone(self):
        self._write(draft_id="draft-a")
        before = self._files()
        chunks, layout, expected = _draft_rank(draft_id="draft-b")
        backend, _, draft_pool = self._backend(
            draft=(chunks, layout, torch.full_like(expected, -1))
        )
        result = backend.batch_exists_v2(self.KEYS, self._transfers(PoolName.DRAFT))
        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.KV], _PAGES)
        self.assertEqual(
            backend.batch_get_v2(self._transfers(PoolName.DRAFT)),
            {PoolName.DRAFT: [False] * _PAGES},
        )
        self.assertTrue(torch.all(draft_pool.pages == -1))
        # Its own backup adds its draft objects and rewrites nothing.
        self._write(draft_id="draft-b")
        after = self._files()
        self.assertEqual({name: after[name] for name in before}, before)
        self.assertEqual(len(after) - len(before), _PAGES * 4)

    def test_a_pool_without_planned_chunks_is_refused_when_it_registers(self):
        chunks, layout, pages = _draft_rank()
        backend, _, _ = self._backend()
        pool = _PagedHostPool(layout, pages)
        for name in (PoolName.DRAFT, PoolName.MAMBA, PoolName.DRAFT_SWA):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(NotImplementedError, "has no unified keys"),
            ):
                backend.register_mem_host_pool_v2(pool, name)
            self.assertNotIn(name, backend.registered_pools)
        with self.assertRaisesRegex(NotImplementedError, "has no unified keys"):
            backend.batch_exists_v2(self.KEYS, self._transfers(PoolName.DRAFT))

    def test_a_plan_belongs_to_the_pool_it_was_made_for(self):
        chunks, layout, pages = _draft_rank()
        backend, _, _ = self._backend(draft=(chunks, layout, pages))
        pool = _PagedHostPool(layout, pages)
        with self.assertRaisesRegex(NotImplementedError, "mamba has no unified keys"):
            backend.register_mem_host_pool_v2(pool, PoolName.MAMBA)
        # Nor may a second pool repeat the names of one that is registered.
        backend._storage_config.unified_pool_chunks[PoolName.DRAFT_INDEXER] = chunks
        with self.assertRaisesRegex(ValueError, "draft_indexer must be nonempty"):
            backend.register_mem_host_pool_v2(pool, PoolName.DRAFT_INDEXER)
        self.assertNotIn(PoolName.DRAFT_INDEXER, backend.registered_pools)
        self.assertNotIn(PoolName.MAMBA, backend.registered_pools)

    def test_planned_chunks_must_name_every_byte_of_the_pools_page(self):
        chunks, layout, pages = _draft_rank()
        cases = {
            "gap": replace(chunks, spans=chunks.spans[:-1] + ((0, 1),)),
            "short": replace(
                chunks, suffixes=chunks.suffixes[:-1], spans=chunks.spans[:-1]
            ),
            "unpaired": replace(chunks, suffixes=chunks.suffixes[:-1]),
            "duplicate": replace(chunks, suffixes=chunks.suffixes[:1] * 4),
        }
        for name, bad in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self._backend(draft=(bad, layout, pages))
        # A target suffix must not come back as a draft's.
        kv_config, _, _ = _rank()
        stolen = replace(
            chunks, suffixes=(kv_config.unified_suffixes[0],) + chunks.suffixes[1:]
        )
        with self.assertRaises(ValueError):
            self._backend(draft=(stolen, layout, pages))

    def test_only_the_planned_writer_owns_and_evicts_draft_files(self):
        cap = {"extra_config": {"max_size": 1 << 20, "min_free_space": 0}}
        writer, _, _ = self._backend(draft=_draft_rank(), **cap)
        chunks = _draft_rank()[0]
        stem = f"page0_{chunks.suffixes[0]}"
        self.assertTrue(writer._evictor._owns(stem))
        self.assertEqual(
            writer.batch_set_v2(self._transfers(PoolName.DRAFT)),
            {PoolName.DRAFT: [True] * _PAGES},
        )
        self.assertEqual(len(writer._evictor._lru), _PAGES * 4)
        # A restarted writer adopts the draft files that are already there,
        # beside the target files it found first, and counts each file once.
        writer.batch_set_v2(self._transfers(PoolName.KV))
        restarted, _, _ = self._backend(draft=_draft_rank(), **cap)
        self.assertEqual(len(restarted._evictor._lru), _PAGES * (4 + 8))
        self.assertEqual(
            restarted._evictor._total_bytes,
            sum(len(data) for data in self._files().values()),
        )
        reader, _, _ = self._backend(draft=_draft_rank(is_writer=False), **cap)
        self.assertFalse(reader._evictor._owns(stem))
        self.assertEqual(
            {name for name in reader._evictor._lru if "ukd1-" in name}, set()
        )

    def _controller(self, backend, kv_pool, config):
        controller = object.__new__(HiCacheController)
        controller.page_size = 4
        controller.storage_backend = backend
        controller.storage_host_pool = kv_pool
        controller.storage_config = config
        controller.page_get_func = controller._generic_page_get
        return controller

    def test_read_publishes_only_pages_restored_in_both_pools(self):
        """The reader's slots hold another prompt's bytes: a draft page that
        does not come back must not be published with them."""
        self._write()
        chunks, layout, expected = _draft_rank()
        (Path(self.directory.name) / f"page1_{chunks.suffixes[0]}.bin").unlink()
        kv_expected = self._backend()[1].pages
        backend, kv_pool, draft_pool = self._backend(
            draft=(chunks, layout, torch.full_like(expected, -1)),
            kv_pages=torch.full_like(kv_expected, -1),
        )
        controller = self._controller(backend, kv_pool, backend._storage_config)
        operation = SimpleNamespace(request_id="r", is_terminated=lambda: False)
        hit_pages = controller._page_transfer_kv_batch(
            operation,
            self.KEYS,
            torch.arange(_PAGES * 4),
            HiCacheStorageExtraInfo(),
            [PoolTransfer(name=PoolName.DRAFT, indices_from_pool=PoolName.KV)],
        )
        self.assertEqual(hit_pages, 1)
        torch.testing.assert_close(kv_pool.pages[0], kv_expected[0], rtol=0, atol=0)
        torch.testing.assert_close(draft_pool.pages[0], expected[0], rtol=0, atol=0)
        self.assertTrue(torch.all(draft_pool.pages[1] == -1))

    def test_a_named_pool_the_backend_does_not_report_publishes_nothing(self):
        self._write()
        backend, kv_pool, _ = self._backend(draft=_draft_rank())
        controller = self._controller(backend, kv_pool, backend._storage_config)
        operation = SimpleNamespace(request_id="r", is_terminated=lambda: False)
        args = (
            operation,
            self.KEYS,
            torch.arange(_PAGES * 4),
            HiCacheStorageExtraInfo(),
            [PoolTransfer(name=PoolName.DRAFT, indices_from_pool=PoolName.KV)],
        )
        self.assertEqual(controller._page_transfer_kv_batch(*args), _PAGES)
        with patch.object(backend, "batch_get_v2", return_value={}):
            self.assertEqual(controller._page_transfer_kv_batch(*args), 0)
        # The same for a pool that has no unified keys.
        legacy = [PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)]
        with patch.object(backend, "batch_get_v2", return_value={}):
            self.assertEqual(controller._page_transfer_kv_batch(*args[:-1], legacy), 0)
        with patch.object(
            backend, "batch_get_v2", return_value={"indexer": [True, True, False]}
        ):
            self.assertEqual(controller._page_transfer_kv_batch(*args[:-1], legacy), 2)
        # Every named pool counts, whichever the backend left out.
        both = args[-1] + legacy
        for reported in ({"draft": [True] * _PAGES}, {"indexer": [True] * _PAGES}):
            with patch.object(backend, "batch_get_v2", return_value=reported):
                self.assertEqual(
                    controller._page_transfer_kv_batch(*args[:-1], both), 0
                )

    def test_rank_suffix_keys_ignore_a_pool_the_backend_does_not_report(self):
        """Without the unified scheme the read path is what it was: only the
        pools a backend reports limit the pages that are published."""
        self._write()
        backend, kv_pool, _ = self._backend(draft=_draft_rank())
        operation = SimpleNamespace(request_id="r", is_terminated=lambda: False)
        indexer = PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)
        swa = PoolTransfer(name=PoolName.SWA, indices_from_pool=PoolName.KV)
        args = (
            operation,
            self.KEYS,
            torch.arange(_PAGES * 4),
            HiCacheStorageExtraInfo(),
        )
        rank_suffix = SimpleNamespace(unified_suffixes=None, unified_pool_chunks={})
        controllers = {
            "rank-suffix": self._controller(backend, kv_pool, rank_suffix),
            # Fixtures of other tests build the controller without a config.
            "no config": self._controller(backend, kv_pool, None),
        }
        del controllers["no config"].storage_config
        cases = (
            ({}, [indexer], _PAGES),
            ({}, [indexer, swa], _PAGES),
            ({"swa": [True] * _PAGES}, [indexer, swa], _PAGES),
            ({"swa": [True, False, False]}, [indexer, swa], 1),
            ({"indexer": [True, True, False]}, [indexer], 2),
        )
        for name, controller in controllers.items():
            for reported, transfers, hit_pages in cases:
                with (
                    self.subTest(name, reported=reported),
                    patch.object(backend, "batch_get_v2", return_value=reported),
                ):
                    self.assertEqual(
                        controller._page_transfer_kv_batch(*args, transfers),
                        hit_pages,
                    )


if __name__ == "__main__":
    unittest.main()
