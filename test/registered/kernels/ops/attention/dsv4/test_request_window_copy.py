"""The request window's per-layer copies against the indexing they replace.

``RequestWindow.buffer`` gathers each request's SWA history into the workspace
and ``commit`` writes the step's tokens back and tags them, on paged buffers that
hold a page of data rows followed by a page of scale rows. Each check runs the
window end to end and compares both buffers byte for byte and the tags with a
reference built from plain tensor indexing. The sink row, which several masked
tokens may write in any order, is excluded.
"""

import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.mem_cache.dsv41_request_window import RequestWindow, window_layout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

WINDOW = 8
CAPACITY = 16
NUM_SLOTS = 4
LAYERS = 2


def _pool_factory(layout, page_size):
    def make(size, layers):
        pages = -(-size // page_size)
        g = torch.Generator(device="cuda").manual_seed(size * 31 + layers)
        bufs = [
            torch.randint(
                0,
                256,
                (pages, layout.page_bytes(page_size)),
                dtype=torch.uint8,
                device="cuda",
                generator=g,
            )
            for _ in range(layers)
        ]
        return SimpleNamespace(kv_buffer=bufs, kv_layout=layout, size=pages * page_size)

    return make


def _rows(loc, layout, page_size):
    """Page and byte indices of each token's data and scale rows."""
    loc = loc.long()[:, None]
    page, slot = loc // page_size, loc % page_size
    data = slot * layout.data_bytes + torch.arange(layout.data_bytes, device=loc.device)
    scale = (
        page_size * layout.data_bytes
        + slot * layout.scale_bytes
        + torch.arange(layout.scale_bytes, device=loc.device)
    )
    return page, data, scale


def _copy(src, dst, src_loc, dst_loc, layout, page_size):
    sp, sd, ss = _rows(src_loc, layout, page_size)
    dp, dd, ds = _rows(dst_loc, layout, page_size)
    dst[dp, dd] = src[sp, sd]
    dst[dp, ds] = src[sp, ss]


def _reference(window, lw, layer):
    """Expected state, workspace and tags after buffer(layer) and commit(layer)."""
    layout, page_size, cap = window.state.kv_layout, window.page_size, window.capacity
    state = window.state.kv_buffer[layer].clone()
    workspace = window.workspace.kv_buffer[0].clone()
    tags = window.tags[layer].clone()
    src = torch.where(
        lw.history_valid, lw.history_req * cap + lw.history_pos % cap, window.zero_row
    )
    _copy(state, workspace, src, lw.history_loc, layout, page_size)
    dst = torch.where(lw.commit_mask, lw.req * cap + lw.pos % cap, window.sink_row)
    _copy(workspace, state, lw.write_loc, dst, layout, page_size)
    tags[dst] = lw.pos
    return state, workspace, tags


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRequestWindowCopy(CustomTestCase):
    def _window(self, kv_layout, page_size):
        return RequestWindow(
            _pool_factory(kv_layout, page_size),
            num_slots=NUM_SLOTS,
            layers=LAYERS,
            page_size=page_size,
            capacity=CAPACITY,
            workspace_rows=1024,
        )

    def _layout(self, window, req, pos):
        return window_layout(
            torch.tensor(req, device="cuda"),
            torch.tensor(pos, device="cuda"),
            window=WINDOW,
            capacity=window.capacity,
            num_groups=len(set(req)),
        )

    def _check(self, window, layer, expected):
        state, workspace, tags = expected
        sink = window.sink_row
        keep = torch.ones_like(tags, dtype=torch.bool)
        keep[sink] = False
        self.assertTrue(torch.equal(window.tags[layer][keep], tags[keep]))
        page, data, scale = _rows(
            torch.tensor([sink], device="cuda"),
            window.state.kv_layout,
            window.page_size,
        )
        keep = torch.ones_like(state, dtype=torch.bool)
        keep[page, data] = False
        keep[page, scale] = False
        self.assertTrue(torch.equal(window.state.kv_buffer[layer][keep], state[keep]))
        self.assertTrue(torch.equal(window.workspace.kv_buffer[0], workspace))

    def test_eager_matches_indexing(self):
        for kv_layout in KVLayout:
            for page_size in (16, 64):
                with self.subTest(layout=kv_layout.value, page_size=page_size):
                    window = self._window(kv_layout, page_size)
                    # Request 0 adds three tokens; request 2's history starts
                    # before position 0 and is partly masked; request 1 adds more
                    # tokens than the window holds, so its oldest goes to the sink.
                    longest = window.capacity + 1
                    req = [0, 0, 0, 2, 3] + [1] * longest
                    pos = [9, 10, 11, 3, 40] + list(range(5, 5 + longest))
                    lw = self._layout(window, req, pos)
                    window.activate(lw)
                    window.tags.fill_(-1)
                    valid = lw.history_valid
                    loc = (
                        lw.history_req * window.capacity
                        + lw.history_pos % window.capacity
                    )
                    window.tags[:, loc[valid]] = lw.history_pos[valid]
                    self.assertFalse(bool(lw.commit_mask.all()))
                    for layer in range(LAYERS):
                        expected = _reference(window, lw, layer)
                        window.commit(layer)
                        self._check(window, layer, expected)

    def test_strided_request_inputs(self):
        # The kernels read req and pos by address, so a strided view (here every
        # other element, with an out-of-range request between them) must not
        # reach them as is.
        def strided(values, filler):
            t = torch.tensor(values, device="cuda")
            return torch.stack([t, torch.full_like(t, filler)], dim=1).flatten()[::2]

        for kv_layout in KVLayout:
            with self.subTest(layout=kv_layout.value):
                window = self._window(kv_layout, page_size=16)
                req = strided([0, 0, 1, 3], filler=99)
                pos = strided([9, 10, 4, 30], filler=0)
                self.assertFalse(req.is_contiguous())
                lw = window_layout(
                    req, pos, window=WINDOW, capacity=window.capacity, num_groups=3
                )
                self.assertTrue(lw.req.is_contiguous() and lw.pos.is_contiguous())
                window.activate(lw)
                window.tags.fill_(-1)
                valid = lw.history_valid
                loc = (
                    lw.history_req * window.capacity + lw.history_pos % window.capacity
                )
                window.tags[:, loc[valid]] = lw.history_pos[valid]
                expected = _reference(window, lw, 0)
                window.commit(0)
                torch.cuda.synchronize()
                self._check(window, 0, expected)

    def test_graph_replay_follows_refreshed_layout(self):
        for kv_layout in KVLayout:
            with self.subTest(layout=kv_layout.value):
                window = self._window(kv_layout, page_size=16)
                captured = self._layout(window, [0, 1, 1], [20, 7, 8])
                window.activate(captured)
                window.initialize_dummy_history()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    window.commit(0)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                window.prepared = None
                with torch.cuda.graph(graph):
                    window.commit(0)

                # Replay refreshes the captured layout in place (WindowLayout.copy_),
                # here with history that is partly masked (request 3 starts at
                # position 3). Fresh state bytes and a poisoned workspace make the
                # replayed gather and commit visible.
                refreshed = self._layout(window, [3, 2, 2], [3, 12, 13])
                self.assertFalse(bool(refreshed.history_valid.all()))
                captured.copy_(refreshed)
                state = window.state.kv_buffer[0]
                state.copy_(torch.randint_like(state, 0, 256))
                window.workspace.kv_buffer[0].fill_(0xA5)
                expected = _reference(window, captured, 0)
                graph.replay()
                torch.cuda.synchronize()
                self._check(window, 0, expected)


def window_with_history():
    window = RequestWindow(
        _pool_factory(KVLayout.V4, 16),
        num_slots=NUM_SLOTS,
        layers=LAYERS,
        page_size=16,
        capacity=CAPACITY,
        workspace_rows=1024,
    )
    layout = window_layout(
        torch.tensor([0, 1, 1], device="cuda"),
        torch.tensor([20, 7, 8], device="cuda"),
        window=WINDOW,
        capacity=window.capacity,
        num_groups=2,
    )
    window.activate(layout)
    window.initialize_dummy_history()
    return window, layout


def drop_history_row(window, layout, layer):
    """Untag one history row the layout gathers for ``layer``."""
    valid = layout.history_valid.nonzero().flatten()
    row = (layout.history_req * window.capacity + layout.history_pos % window.capacity)[
        valid[0]
    ]
    window.tags[layer, row] = -1


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRequestWindowHistoryCheck(CustomTestCase):
    """The eager history check runs once per forward and covers every layer."""

    def _commit_checks(self, window, layers):
        """Commit ``layers`` and return the condition of each history assert."""
        with mock.patch.object(torch, "_assert_async") as assert_async:
            for layer in layers:
                window.commit(layer)
        calls = assert_async.call_args_list
        for call in calls:
            self.assertIn("SWA history is missing", call.args[1])
        return [bool(call.args[0]) for call in calls]

    def test_valid_history_is_checked_once_for_all_layers(self):
        window, _ = window_with_history()
        self.assertEqual(self._commit_checks(window, range(LAYERS)), [True])

    def test_check_does_not_wait_for_the_gpu(self):
        window, _ = window_with_history()
        mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for layer in range(LAYERS):
                window.commit(layer)
        finally:
            torch.cuda.set_sync_debug_mode(mode)
        torch.cuda.synchronize()

    def test_first_commit_checks_every_layer(self):
        window, layout = window_with_history()
        drop_history_row(window, layout, LAYERS - 1)
        self.assertEqual(self._commit_checks(window, [0]), [False])

    def test_checks_again_after_a_new_layout(self):
        window, layout = window_with_history()
        self.assertEqual(self._commit_checks(window, [0]), [True])
        window.activate(
            window_layout(
                layout.req.clone(),
                layout.pos + 1,
                window=WINDOW,
                capacity=window.capacity,
                num_groups=2,
            )
        )
        window.tags[1].fill_(-1)
        self.assertEqual(self._commit_checks(window, [0]), [False])

    def test_missing_history_fails_on_the_device(self):
        # A failed device assert leaves the CUDA context unusable, so it runs in
        # a child process.
        child = (
            "import importlib.util, torch\n"
            f"spec = importlib.util.spec_from_file_location('t', {__file__!r})\n"
            "t = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(t)\n"
            "window, layout = t.window_with_history()\n"
            "t.drop_history_row(window, layout, t.LAYERS - 1)\n"
            "window.commit(0)\n"
            "torch.cuda.synchronize()\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, text=True, timeout=300
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("device-side assert", result.stderr)


if __name__ == "__main__":
    unittest.main()
