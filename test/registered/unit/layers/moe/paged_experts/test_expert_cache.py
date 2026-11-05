"""The expert cache (``--paged-experts-enable-disk``): unit tests on CPU, with temporary
directories for the cache. The loading helpers are test_paged_experts.py's."""

import contextlib
import errno
import fcntl
import gc
import itertools
import mmap
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_paged_experts import (
    _EXPERT_CACHE,
    _METHOD,
    CONTRACTS,
    TOP_K,
    E,
    K,
    _copy_row,
    _CpuStore,
    _given_back_early,
    _load,
    _loads,
    _mapped,
    _NoStep,
    _page_events,
    _paged_layer,
    _Repacking,
    _rows,
    _same,
    _w13_layer,
)

from sglang.srt.layers.moe.paged_experts import expert_cache, formats
from sglang.srt.layers.moe.paged_experts.expert_cache import (
    ExpertCache,
    _sampler,
    checked_groups,
    fill_store,
)
from sglang.srt.layers.moe.paged_experts.formats import PostLoad, UnquantizedFormat
from sglang.srt.layers.moe.paged_experts.store import ExpertLoads
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _load_through_loader(method, layer, full):
    """Every expert through the layer's host loader, as views of a mapped file."""
    layer._weight_loader_impl = _copy_row
    for name, t in vars(full).items():
        param = getattr(layer, name)
        for e in range(E):
            param.weight_loader(param, _mapped(t[e]), name, None, e)


def _cached_post_load(layer, base_method, loads, cache, repacks=True, width=K):
    """The post-load step through ``fill_store`` with ``cache``, host store on the CPU; the
    entry is on disk when it returns."""
    step = PostLoad(layer, base_method, loads, repacks, ())
    with (
        patch.object(formats, "HostExpertStore", _CpuStore),
        patch(_EXPERT_CACHE + "HostExpertStore", _CpuStore),
    ):
        store = fill_store(step, cache, width)
    step.finish(store)
    expert_cache.wait_for_writes()
    return store


# With E + 1 experts and groups of K, a cached start checks experts 0..K-1, the last K, K
# sampled from the others (here every second of them: 6, 8, 10, 12) and 1..K; it does not
# check UNCHECKED.
CHECKED_CALLS = 4
UNCHECKED = (5, 7, 9, 11)
PATHS = ("repack", "loaded")  # a step that repacks, and one that pages the loaded rows


def _every_second(others, k):
    return list(others)[1::2][:k]


def _rows_then_page_out_kept(loads, experts, out=None):
    """``ExpertLoads.rows`` that gives the kept experts' pages back after every run (the
    checked experts after the first group: ``fill_store`` before its fix, whose build read
    them from the drive again)."""
    result = _rows(loads, experts, out)
    loads.page_out(sorted(loads.keep))
    return result


class _CacheCase(CustomTestCase):
    def setUp(self):
        expert_cache._said.clear()  # each kind of message is logged once per process
        expert_cache._write_failed = False  # a failed write stops later ones
        expert_cache._prune_other_keys.cache_clear()
        expert_cache._sweep_staging.cache_clear()
        for k in expert_cache._outcomes:
            expert_cache._outcomes[k] = 0
        expert_cache._directory.clear()
        _Repacking.nan_first = _Repacking.validates = _Repacking.marks = False
        sampling = patch.object(_sampler, "sample", side_effect=_every_second)
        sampling.start()
        self.addCleanup(sampling.stop)
        margin = patch.object(expert_cache, "_FREE_MARGIN_BYTES", 0)  # any test disk
        margin.start()
        self.addCleanup(margin.stop)

    def _start(
        self,
        folder,
        path="repack",
        wait=True,
        replay=True,
        num_slots=K,
        width=None,
        num_experts=E + 1,
        created=0.0,
    ):
        """A start through the cache in ``folder``: ``wait``, the loads wait (replays of the
        unchecked experts are counted in ``self.replayed``; ``replay=False`` drops them)."""
        repacks = path == "repack"
        layer = _w13_layer(num_slots, created)
        self.replayed = 0

        def count(e):
            if e in UNCHECKED:
                self.replayed += 1

        loads, values = _loads(
            layer,
            num_experts,
            wait=wait,
            skip=() if replay else UNCHECKED,
            count=count,
            num_slots=num_slots,
        )
        cache = ExpertCache(os.path.join(folder, "layer-0.experts"), {"key": "k"})
        _Repacking.calls = 0
        base = _Repacking() if repacks else _NoStep()
        store = _cached_post_load(
            layer, base, loads, cache, repacks, width or num_slots
        )
        name = "w13_qweight" if repacks else "w13_weight"
        expected = (values * 10 + 1).reshape(num_experts, 2, 3) if repacks else values
        if _Repacking.nan_first:
            expected[0] = float("nan")
        return store.host[name], expected, cache

    def _corrupt(self, entry, expert, path="repack"):
        rows = {"w13_qweight": ((2, 3), torch.float32)}
        if path != "repack":
            rows = {"w13_weight": ((6,), torch.float32)}
        layout = expert_cache.ObjectLayout.of(rows)
        with open(entry.path, "r+b") as f:
            f.seek(expert_cache._HEADER_BYTES + expert * layout.object_bytes)
            f.write(b"\xff" * 4)


class TestExpertCache(_CacheCase):
    """The expert cache keeps a layer's paged rows (a repack's output, or the loaded rows of
    a format that pages them as they are); a later start checks a few groups of experts
    against it and takes the rest from it."""

    def test_a_later_start_reads_the_cache_after_checking_a_few_groups(self):
        """Also with loads that cannot wait (in a table, e.g. --weight-loader-disable-mmap)."""
        for path, wait in itertools.product(PATHS, (True, False)):
            with (
                self.subTest(path=path, wait=wait),
                tempfile.TemporaryDirectory() as folder,
            ):
                built, expected, cache = self._start(folder, path, wait=wait)
                self.assertTrue(_same(built, expected))
                self.assertEqual(os.listdir(folder), ["layer-0.experts"])  # no staging
                if path == "repack":
                    chunks = -(-(E + 1) // K) + 1  # every chunk, and the mixing check
                    self.assertEqual(_Repacking.calls, chunks)
                read, expected, _ = self._start(folder, path, wait=wait)
                self.assertTrue(_same(read, expected))
                self.assertEqual(self.replayed, 0)  # the unchecked experts never load
                if path == "repack":
                    self.assertEqual(_Repacking.calls, CHECKED_CALLS)

    def test_an_entry_the_step_does_not_reproduce_is_rebuilt(self):
        """Expert 1 is in the first group, expert K only in the mixing group, expert E in the
        last group, expert 8 only in the sampled one: each corrupted copy is caught (the
        warning names the group), every expert loads, and the entry is replaced, so the next
        start reads it again; also with loads that cannot wait."""
        caught_in = {1: [0, 1, 2, 3], K: [1, 2, 3, 4], E: [13, 14, 15, 16]}
        caught_in[8] = [6, 8, 10, 12]
        for path, wait in itertools.product(PATHS, (True, False)):
            for expert, group in caught_in.items():
                with (
                    self.subTest(path=path, wait=wait, expert=expert),
                    tempfile.TemporaryDirectory() as folder,
                ):
                    expert_cache._said.clear()
                    _, _, cache = self._start(folder, path, wait=wait)
                    self._corrupt(cache, expert, path)
                    with self.assertLogs(expert_cache.logger, "WARNING") as logs:
                        rebuilt, expected, _ = self._start(folder, path, wait=wait)
                    self.assertIn(
                        f"differs from the checkpoint for experts {group}; rebuilding it",
                        logs.output[0],
                    )
                    self.assertEqual(self.replayed, len(UNCHECKED) if wait else 0)
                    self.assertTrue(_same(rebuilt, expected))
                    read, expected, _ = self._start(folder, path, wait=wait)
                    self.assertEqual(self.replayed, 0)
                    self.assertTrue(_same(read, expected))

    def test_a_rebuild_without_the_waiting_loads_is_wrong(self):
        """Negative control: a stale entry rebuilt without the unchecked experts' loads
        gives other bytes, so the tests above see the loads replay."""
        for path in PATHS:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as folder:
                expert_cache._said.clear()  # the warning is logged once per process
                _, _, cache = self._start(folder, path)
                self._corrupt(cache, 1, path)
                with self.assertLogs(expert_cache.logger, "WARNING"):
                    wrong, expected, _ = self._start(folder, path, replay=False)
                self.assertFalse(_same(wrong, expected))

    def test_experts_outside_the_checked_groups_come_from_the_cache(self):
        """Negative control: a corrupted expert outside the checked groups is served as
        cached. This shows the tests above see the cache's content, and where the check
        ends (until a later start samples that expert)."""
        for path in PATHS:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as folder:
                _, _, cache = self._start(folder, path)
                self._corrupt(cache, UNCHECKED[0], path)
                read, expected, _ = self._start(folder, path)
                others = [e for e in range(E + 1) if e != UNCHECKED[0]]
                self.assertTrue(_same(read[others], expected[others]))
                self.assertFalse(_same(read[UNCHECKED[0]], expected[UNCHECKED[0]]))

    def test_an_entry_that_does_not_fit_is_replaced(self):
        """A file with a broken header (e.g. a truncated copy) is rebuilt and replaced."""
        with tempfile.TemporaryDirectory() as folder:
            _, _, cache = self._start(folder)
            with open(cache.path, "r+b") as f:
                f.write(b"garbage")
            with self.assertLogs(expert_cache.logger, "WARNING") as logs:
                self._start(folder)
            self.assertIn("does not fit", logs.output[0])
            read, expected, _ = self._start(folder)
            self.assertEqual(_Repacking.calls, CHECKED_CALLS)
            self.assertTrue(_same(read, expected))

    def test_a_read_that_fails_midway_rebuilds_every_expert(self):
        """A read that fails after it wrote rows (EIO, a short read) leaves none of them
        behind: every expert is rebuilt, also those of the first group, which ran before the
        read."""

        def fails_midway(fd, layout, host):
            for rows in host.values():
                rows.fill_(-3.0)  # rows written, then the error
            raise OSError(errno.EIO, "Input/output error")

        for path in PATHS:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as folder:
                expert_cache._said.clear()
                self._start(folder, path)
                with (
                    patch.object(expert_cache, "read_experts", fails_midway),
                    self.assertLogs(expert_cache.logger, "WARNING") as logs,
                ):
                    built, expected, _ = self._start(folder, path)
                self.assertIn("cannot read", logs.output[0])
                self.assertTrue(_same(built, expected))

    def test_a_cached_start_gives_the_checked_pages_back_once(self):
        """The checked groups share experts (0..t-1 with 1..t): their checkpoint pages stay
        until every group ran, then each expert's are given back once."""
        paged_out = []
        page_out = ExpertLoads.page_out

        def spy(loads, experts):
            experts = list(experts)
            paged_out.extend(experts)
            return page_out(loads, experts)

        with tempfile.TemporaryDirectory() as folder:
            self._start(folder)
            with patch.object(ExpertLoads, "page_out", spy):
                self._start(folder)
        self.assertEqual(sorted(paged_out), sorted(set(paged_out)))
        checked = {0, 1, 2, 3, 4, 6, 8, 10, 12, 13, 14, 15, 16}
        self.assertEqual(set(paged_out), checked)

    def test_a_first_start_gives_the_checked_pages_back_after_replaying_them(self):
        """A start that writes the entry (no file, or a stale one) replays every expert: the
        checked experts' pages stay until their chunk replays them (chunk 0, which the first
        group gave already, right after the check; experts 1..K after the mixing check), and
        each expert's go once, never before its last read."""
        for path in PATHS:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as folder:
                events = _page_events(lambda: self._start(folder, path))
                self.assertEqual(_given_back_early(events, E + 1), [])
                expert_cache._said.clear()
                entry = SimpleNamespace(path=os.path.join(folder, "layer-0.experts"))
                self._corrupt(entry, E, path)
                with self.assertLogs(expert_cache.logger, "WARNING"):  # stale: rebuilt
                    events = _page_events(lambda: self._start(folder, path))
                self.assertEqual(_given_back_early(events, E + 1), [])

    def test_giving_the_checked_pages_back_before_the_build_is_caught(self):
        """Negative control: the check above fails when the checked experts' pages go as
        soon as the first group ran, before the build replays them (from the drive again)."""
        with (
            patch.object(ExpertLoads, "rows", _rows_then_page_out_kept),
            tempfile.TemporaryDirectory() as folder,
        ):
            events = _page_events(lambda: self._start(folder))
        self.assertNotEqual(_given_back_early(events, E + 1), [])

    def test_nan_in_the_output_is_reproduced_as_bytes(self):
        """A step whose output holds NaN still reads its entry: the check compares bytes."""
        _Repacking.nan_first = True
        with tempfile.TemporaryDirectory() as folder:
            self._start(folder)
            read, expected, _ = self._start(folder)
            self.assertEqual(_Repacking.calls, CHECKED_CALLS)
            self.assertTrue(_same(read, expected))

    def test_groups_narrower_than_k_load_and_pad_with_their_own_experts(self):
        """With a large K (set by hand) the groups stay at the router's top-k: a cached start
        loads the same few experts and pads each group with its own, never with a row it did
        not load (the step rejects those). Negative control: the step does reject one."""
        with tempfile.TemporaryDirectory() as folder:
            self._start(folder, num_slots=16, width=K)  # builds the entry
            _Repacking.validates = True  # rows no load wrote hold -1
            read, expected, _ = self._start(folder, num_slots=16, width=K, created=-1.0)
            self.assertEqual(self.replayed, 0)
            self.assertEqual(_Repacking.calls, CHECKED_CALLS)
            self.assertTrue(_same(read, expected))
        layer = torch.nn.Module()
        layer.register_parameter(
            "w13_weight", torch.nn.Parameter(-torch.ones(K, 6), requires_grad=False)
        )
        with self.assertRaisesRegex(ValueError, "never loaded"):
            _Repacking().process_weights_after_loading(layer)

    def test_the_log_says_what_the_cache_did(self):
        """One line after loading: layers read, rebuilt, written."""
        with tempfile.TemporaryDirectory() as folder:
            self._start(folder)  # written
            self._start(folder)  # read
            _, _, cache = self._start(folder)
            self._corrupt(cache, 1)
            with self.assertLogs(expert_cache.logger, "WARNING"):
                self._start(folder)  # rebuilt, and written
            with self.assertLogs(expert_cache.logger, "INFO") as logs:
                expert_cache.loading_done()
            self.assertEqual(
                logs.output[-1].split(":", 2)[-1],
                f"Paged experts: expert cache {folder}: read 2 layers, rebuilt 1 layer, "
                "wrote 2 layers (0.0 GiB); later starts read them",
            )

    def test_a_cache_that_cannot_be_written_does_not_stop_the_start(self):
        with tempfile.TemporaryDirectory() as folder:
            blocked = os.path.join(folder, "file")
            open(blocked, "w").close()  # the cache directory would have to go here
            with self.assertLogs(expert_cache.logger, "WARNING") as logs:
                built, expected, _ = self._start(os.path.join(blocked, "sub"))
            self.assertTrue(any("cannot write" in line for line in logs.output))
            self.assertTrue(_same(built, expected))

    def test_pruning_other_keys_makes_the_space_for_a_write(self):
        """A disk too full for an entry until the model's directory under an older key goes:
        the write removes it first and succeeds. Control: another model's directory stays,
        and the write that needs its space is skipped."""
        host = {"w": torch.ones(E, 8)}
        nbytes = expert_cache.ObjectLayout.of_host(host).file_bytes(E)
        for owner in ("M", "N"):
            with self.subTest(owner=owner), tempfile.TemporaryDirectory() as root:
                expert_cache._said.clear()
                expert_cache._prune_other_keys.cache_clear()
                expert_cache._write_failed = False
                old = os.path.join(root, f"{owner}-{'a' * 24}")
                os.makedirs(old)
                with open(os.path.join(old, "layer-0-x.experts"), "wb") as f:
                    f.write(b"\0" * 4096)
                entry = ExpertCache(
                    os.path.join(root, f"M-{'b' * 24}", "layer-0-y.experts"), {}
                )

                def usage(path):  # full while the old directory is there
                    return SimpleNamespace(free=0 if os.path.exists(old) else nbytes)

                with (
                    patch.object(expert_cache.shutil, "disk_usage", usage),
                    self.assertLogs(expert_cache.logger, "INFO") as logs,
                ):
                    entry.write(host)
                self.assertEqual(os.path.exists(entry.path), owner == "M")
                self.assertEqual(os.path.exists(old), owner == "N")
                if owner == "N":
                    self.assertIn("cannot write", logs.output[-1])

    def test_a_later_model_load_in_the_process_starts_afresh(self):
        """After a model's last layer (``loading_done``) the writer thread ends, and a write
        that failed in that load does not stop the next load's: it writes with a new one."""

        def writers():
            return [
                t
                for t in threading.enumerate()
                if t.name.startswith("expert-cache-writer")
            ]

        with tempfile.TemporaryDirectory() as folder:
            blocked = os.path.join(folder, "file")
            open(blocked, "w").close()
            with self.assertLogs(expert_cache.logger, "WARNING"):
                self._start(os.path.join(blocked, "sub"))  # cannot write
            self.assertTrue(writers())
            expert_cache.loading_done()
            self.assertEqual(writers(), [])
            self._start(folder)
            self.assertIn("layer-0.experts", os.listdir(folder))
            expert_cache.loading_done()
            self.assertEqual(writers(), [])

    def test_a_filesystem_without_direct_io_reads_and_writes_through_the_page_cache(
        self,
    ):
        """Where O_DIRECT is refused (at open, or only at the first read or write, as on some
        filesystems), the entry is written and read without it."""
        real_open, real_preadv, real_pwrite = os.open, os.preadv, os.pwrite

        def refuse_open(path, flags, *a):
            if flags & expert_cache._O_DIRECT:
                if flags & os.O_CREAT:  # some filesystems create the file, then refuse
                    os.close(real_open(path, flags & ~expert_cache._O_DIRECT, *a))
                raise OSError(errno.EINVAL, "no O_DIRECT")
            return real_open(path, flags, *a)

        def refuse_io(real):
            def io(fd, *a):
                if fcntl.fcntl(fd, fcntl.F_GETFL) & expert_cache._O_DIRECT:
                    raise OSError(errno.EINVAL, "no O_DIRECT")
                return real(fd, *a)

            return io

        for name, patches in (
            ("at open", {"open": refuse_open}),
            (
                "at the first read or write",
                {"preadv": refuse_io(real_preadv), "pwrite": refuse_io(real_pwrite)},
            ),
        ):
            with self.subTest(name), tempfile.TemporaryDirectory() as folder:
                expert_cache._said.clear()
                with contextlib.ExitStack() as stack:
                    for attr, fn in patches.items():
                        stack.enter_context(patch.object(expert_cache.os, attr, fn))
                    self._start(folder)
                    read, expected, _ = self._start(folder)
                self.assertEqual(_Repacking.calls, CHECKED_CALLS)
                self.assertTrue(_same(read, expected))

    def test_round_trip_keeps_every_tensor(self):
        """Rows of 16 KiB, 6-byte and empty tensors survive a write and a read, also in
        batches smaller than the layer, read straight into page-aligned rows of whole pages
        and through the bounce buffer otherwise; an entry with another key is not read."""
        host = {
            "w13_qweight": torch.randint(0, 1 << 30, (E, 32, 128), dtype=torch.int32),
            "w13_scales": torch.randn(E, 3).bfloat16(),
            "w2_g_idx": torch.empty(E, 0, dtype=torch.int32),
        }
        for aligned in (False, True):
            with (
                self.subTest(aligned=aligned),
                tempfile.TemporaryDirectory() as folder,
                patch.object(expert_cache, "_BATCH_BYTES", 3 * 20480),
            ):
                cache = ExpertCache(
                    os.path.join(folder, "layer-0.experts"), {"key": "k"}
                )
                cache.write(host)
                back = {k: torch.zeros_like(v) for k, v in host.items()}
                if aligned:  # page-aligned tables, as a pinned store's
                    back["w13_qweight"] = _mapped(back["w13_qweight"])
                self.assertTrue(cache.read(back))
                for name in host:
                    self.assertTrue(_same(back[name], host[name]))
                other = ExpertCache(cache.path, {"key": "other"})
                self.assertFalse(other.read(back))

    def test_a_layer_file_under_another_key_is_replaced(self):
        """A layer's entry written under a new layer key removes its file under the old one
        (the directory is the model's); other layers' files stay."""
        host = {"w": torch.ones(E, 8)}
        with tempfile.TemporaryDirectory() as folder:
            old = ExpertCache(
                os.path.join(folder, "model.layers.3.mlp.experts-aaa.experts"), {}
            )
            kept = ExpertCache(
                os.path.join(folder, "model.layers.30.mlp.experts-bbb.experts"), {}
            )
            new = ExpertCache(
                os.path.join(folder, "model.layers.3.mlp.experts-ccc.experts"), {}
            )
            old.write(host)
            kept.write(host)
            new.write(host)
            self.assertEqual(
                sorted(os.listdir(folder)),
                sorted(os.path.basename(c.path) for c in (kept, new)),
            )

    def test_layers_of_different_formats_share_the_models_directory(self):
        """A model whose MoE layers use different methods (e.g. bf16 layers in an NVFP4
        checkpoint) keeps one directory, so a layer's write never prunes another's. Control:
        another checkpoint is another directory."""
        rows = {"w": ((4,), torch.float32)}
        model = expert_cache._Model("Qwen--Q", "id")
        entries = []
        with patch.object(expert_cache, "_model", return_value=model):
            for i, (fmt, base) in enumerate(
                ((formats.GptqFormat(), object()), (UnquantizedFormat(), _NoStep()))
            ):
                layer = SimpleNamespace(layer_name=f"model.layers.{i}.mlp.experts")
                entries.append(
                    ExpertCache.for_layer(layer, fmt, None, base, rows, E, i == 0)
                )
        with patch.object(
            expert_cache, "_model", return_value=expert_cache._Model("Qwen--Q", "other")
        ):
            layer = SimpleNamespace(layer_name="model.layers.0.mlp.experts")
            other = ExpertCache.for_layer(
                layer, formats.GptqFormat(), None, object(), rows, E, True
            )
        directories = {os.path.dirname(e.path) for e in entries}
        self.assertEqual(len(directories), 1)
        self.assertIn("model.layers.0.mlp.experts", entries[0].path)  # named for humans
        self.assertIn("Qwen--Q-", entries[0].path)
        self.assertNotEqual(entries[0].path, entries[1].path)
        self.assertNotIn(os.path.dirname(other.path), directories)

    def test_the_key_follows_the_source_files(self):
        """Editing a file the step runs changes the key; the declared files and directories
        must exist."""
        with tempfile.TemporaryDirectory() as folder:
            module = os.path.join(folder, "pe_cache_probe_mod.py")
            with open(module, "w") as f:
                f.write("X = 1\n")
            with patch("sys.path", [folder, *__import__("sys").path]):
                expert_cache._sources.cache_clear()
                before = expert_cache._sources((), ("pe_cache_probe_mod",))
                with open(module, "w") as f:
                    f.write("X = 2\n")
                expert_cache._file_digest.cache_clear()
                expert_cache._sources.cache_clear()
                after = expert_cache._sources((), ("pe_cache_probe_mod",))
                self.assertNotEqual(before, after)
                with self.assertRaisesRegex(RuntimeError, "not found"):
                    expert_cache._sources((), ("pe_cache_no_such_module",))
                expert_cache._sources.cache_clear()
                with (
                    patch.object(
                        expert_cache, "_SOURCE_DIRS", ("kernels/no_such_dir",)
                    ),
                    self.assertRaisesRegex(RuntimeError, "no_such_dir are not found"),
                ):
                    expert_cache._sources((), ())
        expert_cache._sources.cache_clear()
        # Every declared module and file exists (a typo would turn the cache off).
        self.assertTrue(expert_cache._sources((), expert_cache._REPACK_SOURCES))

    def test_every_package_in_the_key_is_found_where_it_is_installed(self):
        """A misspelled distribution name would drop silently out of the key."""
        import importlib.util
        from importlib.metadata import version

        modules = {
            "torch": "torch",
            "triton": "triton",
            "sglang-kernel": "sgl_kernel",
            "flashinfer-python": "flashinfer",
        }
        self.assertEqual(set(modules), set(expert_cache._PACKAGES))
        for name, module in modules.items():
            with self.subTest(name=name):
                if importlib.util.find_spec(module) is None:
                    self.skipTest(f"{module} is not installed")
                self.assertTrue(version(name))

    def test_entries_are_named_for_humans(self):
        label = expert_cache._model_label
        self.assertEqual(
            label("Qwen/Qwen3-30B-A3B-GPTQ-Int4"), "Qwen--Qwen3-30B-A3B-GPTQ-Int4"
        )
        with tempfile.TemporaryDirectory() as folder:
            snapshot = os.path.join(
                folder, "models--Qwen--Qwen3-Next", "snapshots", "8fb2"
            )
            os.makedirs(snapshot)
            self.assertEqual(label(snapshot), "Qwen--Qwen3-Next")
            # Local folders: the name and a hash of the path, so that another folder of
            # the same name is another model.
            twin = os.path.join(folder, "a", os.path.basename(folder))
            os.makedirs(twin)
            self.assertTrue(label(folder).startswith(os.path.basename(folder) + "-"))
            self.assertNotEqual(label(folder), label(twin))
        # Long ids are cut with a hash: two that share the start are other models.
        long_a, long_b = "org/" + "x" * 90 + "-a", "org/" + "x" * 90 + "-b"
        self.assertNotEqual(label(long_a), label(long_b))
        self.assertLessEqual(len(label(long_a)), 80)
        self.assertTrue(label(long_a).startswith("org--xxx"))


class TestOneSlotTableAtATime(_CacheCase):
    """A repacking step's rows are views of the layer's whole K-slot table (on the GPU when
    serving): when the step runs again, no rows an earlier run returned may still be alive,
    or a second table sits at peak VRAM. Checked for a build, a cached start's check and a
    build without the cache."""

    @contextlib.contextmanager
    def _tracked(self, hold=None):
        """Records, at each run after the first, whether rows an earlier run returned are
        alive; ``hold`` (a list) keeps the first run's rows referenced, as a control."""
        run, earlier, self.alive = PostLoad.run, [], []

        def tracked(step, experts, out=None):
            if earlier:
                gc.collect()
                self.alive.append(any(ref() is not None for ref in earlier))
            rows = run(step, experts, out)
            if out is None:
                if hold is not None and not hold:
                    hold.append(rows)
                earlier.extend(weakref.ref(t) for t in rows.values())
            return rows

        with patch.object(PostLoad, "run", tracked):
            yield

    def _runs_alive(self, folder, width, hold=None):
        with self._tracked(hold):
            self._start(folder, num_slots=K, width=width)
        self.assertTrue(self.alive)  # the step ran more than once
        return any(self.alive)

    def test_no_rows_outlive_their_use(self):
        with tempfile.TemporaryDirectory() as d:
            # No entry, groups narrower than K: built anew, chunk by chunk.
            self.assertFalse(self._runs_alive(d, TOP_K))
            # The entry is read: the check runs its other groups.
            self.assertFalse(self._runs_alive(d, TOP_K))
        with tempfile.TemporaryDirectory() as d:
            # No entry, the first group is chunk 0: copied once, not run again.
            self.assertFalse(self._runs_alive(d, K))

    def test_no_rows_outlive_their_use_without_the_cache(self):
        layer = _w13_layer(K, 0.0)
        loads, _ = _loads(
            layer, E + 1, wait=True, skip=(), count=lambda e: None, num_slots=K
        )
        step = PostLoad(layer, _Repacking(), loads, True, ())
        with self._tracked(), patch.object(formats, "HostExpertStore", _CpuStore):
            fill_store(step, None, K)
        self.assertTrue(self.alive)
        self.assertFalse(any(self.alive))

    def test_the_tracking_sees_rows_still_held(self):
        # Control: rows the test itself keeps are reported alive.
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(self._runs_alive(d, TOP_K, hold=[]))


class TestCheckedGroups(CustomTestCase):
    def test_checked_groups_cover_every_expert_over_starts(self):
        """Every start checks experts 0..t-1, the last t and 1..t, and t more sampled per start
        from the others, so that over starts every expert is checked; t is the width (the
        router's top-k), whatever K."""
        num_experts, width, seen = E + 1, 3, set()
        for _ in range(200):  # an expert missed 200 times: p < 1e-9
            groups = checked_groups(num_experts, width)
            self.assertEqual(len(groups), 4)
            self.assertEqual(
                (groups[0], groups[1], groups[-1]),
                ([0, 1, 2], [14, 15, 16], [1, 2, 3]),
            )
            self.assertEqual(len(groups[2]), width)
            self.assertTrue(set(groups[2]).isdisjoint({0, 1, 2, 3, 14, 15, 16}))
            for group in groups:
                seen.update(group)
        self.assertEqual(seen, set(range(num_experts)))
        self.assertEqual(checked_groups(K, K), [list(range(K))])
        self.assertEqual(
            checked_groups(K + 1, K), [[0, 1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4]]
        )


class TestThroughTheMethod(_CacheCase):
    def test_a_layer_that_fits_its_slots_has_no_entry(self):
        method, layer, _ = _paged_layer(load=False)
        method.num_resident = E
        with patch(
            _METHOD + "get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(paged_experts_enable_disk=True)
            ),
        ):
            self.assertIsNone(method._expert_cache(layer))

    def test_groups_are_the_routers_top_k_wide_without_one_on_the_layer(self):
        """A model that routes through a separate TopK module (e.g. OLMoE) leaves the layer's
        top_k unset: the checked groups are as wide as the model's top-k, not K. Control: a
        layer that has its own."""
        for top_k, router_top_k in ((None, TOP_K), (TOP_K, K)):
            with (
                self.subTest(top_k=top_k),
                tempfile.TemporaryDirectory() as folder,
                patch.object(expert_cache, "cache_root", return_value=folder),
                patch.object(
                    expert_cache, "_model", return_value=expert_cache._Model("M", "id")
                ),
                patch.object(
                    expert_cache, "checked_groups", wraps=expert_cache.checked_groups
                ) as groups,
            ):
                method, layer, full = _paged_layer(load=False, top_k=top_k)
                layer.layer_name = "model.layers.0.mlp.experts"
                _load_through_loader(method, layer, full)
                _load(method, layer, disk=True, router_top_k=router_top_k)
                groups.assert_called_once_with(E, TOP_K)
                expert_cache.loading_done()

    def test_a_start_through_the_method_writes_and_reads_its_entry(self):
        """End to end, for a repacking and a non-repacking runner: the host loader's loads
        wait, the post-load step builds the entry once the checkpoint is loaded, the next
        start reads it and serves the same experts. Negative control: without the flag, no
        entry, and the log line after loading names the flag."""
        for contract in CONTRACTS:
            for enabled in (True, False):
                with (
                    self.subTest(contract=type(contract).__name__, enabled=enabled),
                    tempfile.TemporaryDirectory() as folder,
                    patch.object(expert_cache, "cache_root", return_value=folder),
                    patch.object(
                        expert_cache,
                        "_model",
                        return_value=expert_cache._Model("M", "id"),
                    ),
                ):
                    for start in range(2):
                        method, layer, full = _paged_layer(contract, load=False)
                        layer.layer_name = "model.layers.0.mlp.experts"
                        _load_through_loader(method, layer, full)
                        _load(method, layer, disk=enabled)
                        expert_cache.wait_for_writes()
                        read = expert_cache._outcomes["read"]
                        with self.assertLogs(expert_cache.logger, "INFO") as logs:
                            expert_cache.logger.info("start")  # a log either way
                            expert_cache.loading_done()
                        hinted = any("enable-disk" in line for line in logs.output)
                        self.assertEqual(hinted, not enabled)
                        for name in ("w13_weight", "w2_weight"):
                            self.assertTrue(
                                torch.equal(
                                    method.store.host[name], getattr(full, name)
                                )
                            )
                    files = [f for _, _, fs in os.walk(folder) for f in fs]
                    self.assertEqual(len(files), 1 if enabled else 0)
                    self.assertEqual(read, 1 if enabled else 0)  # the second start's

    def test_the_hint_names_the_flag_once_with_the_total(self):
        """Without the flag, one line after loading names it, with what the cache would keep
        (every named layer's size) and where; not for a model the cache could not key (a
        loader that does not report its files), nor for unnamed layers."""
        rows = {"w": ((1024,), torch.float32)}
        named = SimpleNamespace(layer_name="model.layers.0.mlp.experts")
        model = expert_cache._Model("M", "id")
        size = expert_cache.ObjectLayout.of(rows).file_bytes(E)
        for layer, found, hinted in (
            (named, model, True),
            (named, None, False),
            (SimpleNamespace(), model, False),
        ):
            with (
                self.subTest(found=found, named=hasattr(layer, "layer_name")),
                patch.object(expert_cache, "_model", return_value=found),
                patch.object(
                    expert_cache, "cache_root", return_value="/c/paged_experts"
                ),
            ):
                for _ in range(3):
                    expert_cache.hint_disk(layer, rows, E, True)
                with self.assertLogs(expert_cache.logger, "INFO") as logs:
                    expert_cache.logger.info("start")  # a log either way
                    expert_cache.loading_done()
                lines = [line for line in logs.output if "enable-disk" in line]
                self.assertEqual(len(lines), 1 if hinted else 0)
                if hinted:
                    self.assertIn(
                        f"from /c/paged_experts ({3 * size / (1 << 30):.1f} GiB) instead "
                        "of loading and repacking them",
                        lines[0],
                    )


class TestCheckpointKey(_CacheCase):
    """The model's directory is keyed by the checkpoint files the loader read."""

    def setUp(self):
        super().setUp()
        from sglang.srt.model_loader import loader

        self.loader = loader
        expert_cache._models.clear()
        self.addCleanup(expert_cache._models.clear)
        files = patch.dict(loader._checkpoint_files, clear=True)
        files.start()
        self.addCleanup(files.stop)

    def _model(self, model_path="org/model", revision=None):
        expert_cache._models.clear()
        config = SimpleNamespace(model_path=model_path, revision=revision)
        with patch(
            "sglang.srt.runtime_context.process_model_config", return_value=config
        ):
            return expert_cache._model()

    def _checkpoint(self, folder):
        weights = os.path.join(folder, "model.safetensors")
        config = os.path.join(folder, "config.json")
        for path, data in ((weights, b"w" * 64), (config, b"{}")):
            with open(path, "wb") as f:
                f.write(data)
        return weights, config

    def test_a_hub_id_gets_its_cache_on_the_first_start(self):
        """A model given by its hub id and found in --download-dir (or downloaded by this
        start) is keyed by the files the loader read; the label is the id. Negative control:
        a loader that did not report its files gives no key."""
        with tempfile.TemporaryDirectory() as folder:
            weights, _ = self._checkpoint(folder)
            self.assertIsNone(self._model())
            self.loader._checkpoint_files[("org/model", None)] = (folder, (weights,))
            model = self._model()
            self.assertEqual(model.label, "org--model")
            self.assertIsNone(self._model(revision="other"))

    def test_the_key_follows_the_weight_files_and_the_json_files(self):
        """A weight file's or a ``*.json``'s size or modification time changes the key.
        Negative control: another file in the folder (a README) does not."""
        with tempfile.TemporaryDirectory() as folder:
            weights, config = self._checkpoint(folder)
            self.loader._checkpoint_files[("org/model", None)] = (folder, (weights,))
            keys = [self._model().identity]
            with open(os.path.join(folder, "README.md"), "w") as f:
                f.write("about")
            self.assertEqual(self._model().identity, keys[0])
            for path in (weights, config):
                with open(path, "ab") as f:
                    f.write(b"x")  # size
                keys.append(self._model().identity)
                st = os.stat(path)
                os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # time
                keys.append(self._model().identity)
            self.assertEqual(len(set(keys)), 5)

    def test_an_hf_snapshot_is_keyed_by_its_blobs(self):
        """A hub snapshot's files are links to blobs: the key follows the blob a link points
        to (a new revision's blob of the same size and time is another key)."""
        with tempfile.TemporaryDirectory() as root:
            blobs = os.path.join(root, "blobs")
            snapshot = os.path.join(root, "snapshots", "abc")
            os.makedirs(blobs)
            os.makedirs(snapshot)
            for name in ("a", "b"):
                with open(os.path.join(blobs, name), "wb") as f:
                    f.write(b"w" * 64)
                os.utime(os.path.join(blobs, name), ns=(10**18, 10**18))
            link = os.path.join(snapshot, "model.safetensors")
            os.symlink(os.path.join(blobs, "a"), link)
            self.loader._checkpoint_files[("org/model", None)] = (snapshot, (link,))
            before = self._model().identity
            os.unlink(link)
            os.symlink(os.path.join(blobs, "b"), link)
            self.assertNotEqual(self._model().identity, before)

    def test_a_broken_link_disables_the_cache(self):
        """A checkpoint file that cannot be read (a broken link) leaves the layer without an
        entry, with a warning; the start goes on."""
        with tempfile.TemporaryDirectory() as folder:
            link = os.path.join(folder, "model.safetensors")
            os.symlink(os.path.join(folder, "missing"), link)
            self.loader._checkpoint_files[("org/model", None)] = (folder, (link,))
            config = SimpleNamespace(model_path="org/model", revision=None)
            layer = SimpleNamespace(layer_name="model.layers.0.mlp.experts")
            with (
                patch(
                    "sglang.srt.runtime_context.process_model_config",
                    return_value=config,
                ),
                self.assertLogs(expert_cache.logger, "WARNING") as logs,
            ):
                entry = ExpertCache.for_layer(
                    layer, UnquantizedFormat(), None, _NoStep(), {}, E, False
                )
            self.assertIsNone(entry)
            self.assertIn("no expert cache", logs.output[0])

    def test_the_default_loader_reports_the_files_it_read(self):
        """DefaultModelLoader records the folder and files of each weight source it reads.
        Negative control: before it read them, there is nothing."""
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat

        loader = self.loader
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "pytorch_model.bin")
            torch.save({"w": torch.ones(2)}, path)
            source = loader.DefaultModelLoader.Source(folder, None)
            self.assertIsNone(loader.loaded_checkpoint_files(folder, None))
            default = loader.DefaultModelLoader(
                LoadConfig(
                    load_format=LoadFormat.PT,
                    model_loader_extra_config={"enable_multithread_load": False},
                )
            )
            default._get_weights_iterator(
                source,
                resolved_source=loader.DefaultModelLoader.ResolvedSource(
                    source, folder, (path,), False
                ),
            )
            self.assertEqual(
                loader.loaded_checkpoint_files(folder, None), (folder, (path,))
            )


class TestFailures(_CacheCase):
    """A write or read that fails midway leaves nothing behind and only costs time."""

    def _staging(self, folder):
        return [f for f in os.listdir(folder) if f.startswith(".staging-")]

    def test_a_write_that_fails_midway_leaves_no_staging_file(self):
        """The space reservation (ENOSPC), the write of the experts (ENOSPC or short), and
        the rename failing: no staging file is left, the warning says why, the next start
        writes the entry. Negative control: a write that is interrupted (the process killed:
        no cleanup runs) leaves one."""
        real_pwrite = os.pwrite
        header = expert_cache._HEADER_BYTES

        def nospace(*a, **k):
            raise OSError(errno.ENOSPC, "No space left on device")

        def short_write(fd, data, offset):  # the header is whole, the experts not
            n = real_pwrite(fd, data, offset)
            return n if offset < header else n - 1

        cases = (
            ("fallocate", "posix_fallocate", nospace, "No space left"),
            ("write", "write_experts", nospace, "No space left"),
            ("short write", "pwrite", short_write, "short write"),
            ("rename", "replace", nospace, "No space left"),
        )
        for name, attr, fn, why in cases:
            with self.subTest(name), tempfile.TemporaryDirectory() as folder:
                expert_cache._said.clear()
                expert_cache._write_failed = False
                target = expert_cache if attr == "write_experts" else expert_cache.os
                with (
                    patch.object(target, attr, fn),
                    self.assertLogs(expert_cache.logger, "WARNING") as logs,
                ):
                    built, expected, _ = self._start(folder)
                self.assertTrue(_same(built, expected))
                self.assertIn("cannot write", logs.output[-1])
                self.assertIn(why, logs.output[-1])
                self.assertEqual(self._staging(folder), [])
                expert_cache._write_failed = False
                self._start(folder)
                self.assertEqual(os.listdir(folder), ["layer-0.experts"])
        with tempfile.TemporaryDirectory() as folder:

            def killed(*a, **k):
                raise KeyboardInterrupt  # not an error the write handles

            with (
                patch.object(expert_cache, "write_experts", killed),
                self.assertRaises(KeyboardInterrupt),
            ):
                self._start(folder)
            self.assertEqual(len(self._staging(folder)), 1)

    def test_a_short_read_rebuilds_the_entry(self):
        """A file that ends early (or a read that returns less) is rebuilt and replaced."""
        real_preadv = os.preadv
        with tempfile.TemporaryDirectory() as folder:
            self._start(folder)

            def short(fd, buffers, offset):
                n = real_preadv(fd, buffers, offset)
                return n - 1 if offset >= expert_cache._HEADER_BYTES else n

            with (
                patch.object(expert_cache.os, "preadv", short),
                self.assertLogs(expert_cache.logger, "WARNING") as logs,
            ):
                built, expected, _ = self._start(folder)
            self.assertIn("short read", logs.output[0])
            self.assertTrue(_same(built, expected))
            read, expected, _ = self._start(folder)
            self.assertEqual(self.replayed, 0)
            self.assertTrue(_same(read, expected))

    def test_the_sweep_removes_the_staging_files_of_gone_writers(self):
        """Staging files are named for their writer: those of a process of this PID
        namespace that is gone are removed at once, a live one's stay; of another namespace
        (another container), only those older than an hour are removed."""
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        mine = expert_cache._pid_namespace()
        names = {
            "gone": f".staging-{mine}-{dead.pid}-x-layer.experts",
            "live": f".staging-{mine}-{os.getpid()}-x-layer.experts",
            "other": ".staging-1-1-x-layer.experts",
            "other-old": ".staging-2-1-x-layer.experts",
            "unnamed-old": ".staging-abc-layer.experts",
        }
        with tempfile.TemporaryDirectory() as folder:
            for key, name in names.items():
                path = os.path.join(folder, name)
                open(path, "w").close()
                if key.endswith("-old"):
                    old = time.time() - 2 * expert_cache._STALE_STAGING_S
                    os.utime(path, (old, old))
            expert_cache._sweep_staging(folder)
            left = sorted(os.listdir(folder))
        self.assertEqual(left, sorted([names["live"], names["other"]]))
        self.assertTrue(
            expert_cache._staging_prefix().startswith(f".staging-{mine}-{os.getpid()}-")
        )


class TestReadAlignment(CustomTestCase):
    def test_rows_aligned_for_direct_io_are_read_straight(self):
        """Rows at 4 KiB-aligned addresses that are whole 4 KiB blocks go straight into the
        store, also where the page size is 64 KiB (as on some aarch64 kernels). Negative
        control: rows 2 KiB off that alignment go through the bounce buffer."""
        rows = 2048  # floats: 8 KiB per expert
        source = {"w": torch.randn(E, rows)}
        layout = expert_cache.ObjectLayout.of_host(source)
        real_mmap = mmap.mmap
        for shift, straight in ((0, True), (512, False)):
            with self.subTest(shift=shift), tempfile.TemporaryFile() as f:
                expert_cache.write_experts(f.fileno(), layout, source)
                base = _mapped(torch.zeros(E * rows + 1024))
                host = {"w": base[shift : shift + E * rows].view(E, rows)}
                sizes = []

                def recording(fd, size, *a, **k):
                    sizes.append(size)
                    return real_mmap(fd, size, *a, **k)

                fake = SimpleNamespace(PAGESIZE=65536, mmap=recording)
                with patch.object(expert_cache, "mmap", fake):
                    expert_cache.read_experts(f.fileno(), layout, host)
                self.assertTrue(_same(host["w"], source["w"]))
                self.assertEqual(all(s == 1 for s in sizes), straight)


class TestLoadingDone(_CacheCase):
    def test_a_failed_load_leaves_nothing_behind(self):
        """model_runner calls ``loading_done`` after every model load, also one that failed
        before its last layer: the writer thread ends and the failed write, the counts and
        the said messages are reset, so the next load writes again. Control: until then they
        are there."""

        def writers():
            return [
                t
                for t in threading.enumerate()
                if t.name.startswith("expert-cache-writer")
            ]

        with tempfile.TemporaryDirectory() as folder:
            blocked = os.path.join(folder, "file")
            open(blocked, "w").close()
            with self.assertLogs(expert_cache.logger, "WARNING"):
                self._start(os.path.join(blocked, "sub"))  # cannot write
            self.assertTrue(writers())
            self.assertTrue(expert_cache._write_failed)
            self.assertTrue(expert_cache._said)
            self.assertEqual(expert_cache._outcomes["failed"], 1)
            with (
                patch.object(expert_cache, "_log_outcomes", side_effect=OSError("log")),
                self.assertRaises(OSError),
            ):
                expert_cache.loading_done()  # resets even when the log line fails
            self.assertEqual(writers(), [])
            self.assertFalse(expert_cache._write_failed)
            self.assertFalse(expert_cache._said)
            self.assertFalse(any(expert_cache._outcomes.values()))
            self._start(folder)
            self.assertIn("layer-0.experts", os.listdir(folder))

    def test_the_paged_method_does_not_count_layers(self):
        """The cache finishes when model_runner says the load is done, not when a count of
        layers meets (a layer that failed, or one created outside the count, left the
        writer and the counts behind)."""
        import importlib.util

        from sglang.srt.layers.moe.paged_experts import method as method_module

        self.assertFalse(hasattr(method_module, "_layers"))
        runner = importlib.util.find_spec("sglang.srt.model_executor.model_runner")
        with open(runner.origin) as f:
            self.assertIn("expert_cache.loading_done()", f.read())


class TestFlag(CustomTestCase):
    def test_enable_disk_requires_paged_experts(self):
        """--paged-experts-enable-disk alone is refused when the MoE arguments resolve.
        Control: with --enable-paged-experts the check passes (to the next step, stubbed)."""
        from sglang.srt.arg_groups import moe_hook
        from sglang.srt.server_args import ServerArgs

        class _Next(Exception):
            pass

        with (
            patch("sglang.srt.arg_groups.serving_hook.get_device", return_value="cuda"),
            patch.object(moe_hook, "run_post_process_pass", side_effect=_Next),
        ):
            args = ServerArgs(model_path="dummy", paged_experts_enable_disk=True)
            with self.assertRaisesRegex(
                ValueError, "paged-experts-enable-disk requires --enable-paged-experts"
            ):
                moe_hook.handle_moe_kernel_config(args)
            args = ServerArgs(
                model_path="dummy",
                enable_paged_experts=True,
                paged_experts_enable_disk=True,
            )
            with self.assertRaises(_Next):
                moe_hook.handle_moe_kernel_config(args)


if __name__ == "__main__":
    unittest.main()
