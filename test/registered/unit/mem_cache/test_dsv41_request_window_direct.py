"""Direct (in-ring) decode / verify windows see and leave the same SWA state as the workspace path."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.mem_cache.dsv41_request_window import (
    RequestWindow,
    copy_packed_tokens,
    direct_window_layout,
    window_layout,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE, LAYOUT, WINDOW = 16, KVLayout.V4, 128


def make_pool(size, layers):
    pages = (size + PAGE - 1) // PAGE
    return SimpleNamespace(
        kv_buffer=[
            torch.zeros(
                pages, LAYOUT.page_bytes(PAGE), dtype=torch.uint8, device="cuda"
            )
            for _ in range(layers)
        ],
        kv_layout=LAYOUT,
        size=pages * PAGE,
    )


def make_window(num_slots, layers, max_draft):
    return RequestWindow(
        make_pool,
        num_slots=num_slots,
        layers=layers,
        page_size=PAGE,
        capacity=WINDOW + max_draft,
        workspace_rows=(num_slots + 1) * WINDOW + 64,
    )


def gather(buf, indices):
    valid = indices >= 0
    src = indices[valid].to(torch.int64)
    out = make_pool(max(1, src.numel()), 1).kv_buffer[0]
    dst = torch.arange(src.numel(), device="cuda")
    copy_packed_tokens(buf, out, src, dst, page_size=PAGE, layout=LAYOUT)
    return out, valid


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRequestWindowDirect(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _seed_history(self, win, committed):
        for buf in win.state.kv_buffer:
            buf.copy_(
                torch.randint(0, 256, buf.shape, dtype=torch.uint8, device="cuda")
            )
        for slot, length in committed.items():
            pos = torch.arange(max(0, length - win.capacity), length, device="cuda")
            win.tags[:, slot * win.capacity + pos % win.capacity] = pos

    def _run(self, win, layout, new_rows):
        win.activate(layout)
        seen = []
        n = layout.pos.numel()
        for layer, new in enumerate(new_rows):
            buf = win.buffer(layer)
            copy_packed_tokens(
                new,
                buf,
                torch.arange(n, device="cuda"),
                layout.write_loc,
                page_size=PAGE,
                layout=LAYOUT,
            )
            seen.append(gather(buf, layout.indices))
            win.commit(layer)
        return seen

    def _check(self, rows_per_request, floor_back=None):
        layers, slots, max_draft = 3, 5, 5
        committed = {0: 700, 2: 37, 3: 129, 4: 3}
        order = [2, 0, 4, 3]
        req = torch.tensor(
            [s for s in order for _ in range(rows_per_request)], device="cuda"
        )
        pos = torch.tensor(
            [committed[s] + i for s in order for i in range(rows_per_request)],
            device="cuda",
        )
        # A floor is per request (its replay start), shared by all of its rows.
        floor = (
            None
            if floor_back is None
            else torch.tensor(
                [
                    max(0, committed[s] - floor_back)
                    for s in order
                    for _ in range(rows_per_request)
                ],
                device="cuda",
            )
        )

        ref = make_window(slots, layers, max_draft)
        self._seed_history(ref, committed)
        new = make_window(slots, layers, max_draft)
        for a, b in zip(ref.state.kv_buffer, new.state.kv_buffer):
            b.copy_(a)
        new.tags.copy_(ref.tags)
        new_rows = [
            make_pool(pos.numel(), 1).kv_buffer[0].random_(0, 256)
            for _ in range(layers)
        ]

        old_layout = window_layout(
            req, pos, capacity=ref.capacity, floor=floor, num_groups=len(order)
        )
        direct = direct_window_layout(req, pos, capacity=new.capacity, floor=floor)
        self.assertTrue(direct.direct)
        torch.testing.assert_close(direct.lengths, old_layout.lengths, rtol=0, atol=0)

        seen_ref = self._run(ref, old_layout, new_rows)
        seen_new = self._run(new, direct, new_rows)
        for (a, va), (b, vb) in zip(seen_ref, seen_new):
            torch.testing.assert_close(va, vb, rtol=0, atol=0)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(ref.state.kv_buffer, new.state.kv_buffer):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(ref.tags, new.tags, rtol=0, atol=0)

    def test_decode_matches_workspace_path(self):
        self._check(rows_per_request=1)

    def test_verify_matches_workspace_path(self):
        self._check(rows_per_request=5)

    def test_floor_matches_workspace_path(self):
        self._check(rows_per_request=5, floor_back=50)

    def test_direct_buffer_is_the_ring(self):
        win = make_window(2, 2, 5)
        layout = direct_window_layout(
            torch.tensor([1], device="cuda"),
            torch.tensor([200], device="cuda"),
            capacity=win.capacity,
        )
        win.activate(layout)
        self.assertEqual(win.buffer(1).data_ptr(), win.state.kv_buffer[1].data_ptr())

    def test_reused_layout_retags_every_forward(self):
        win = make_window(2, 3, 5)
        req = torch.tensor([1], device="cuda")
        pos = torch.tensor([10], device="cuda")
        layout = direct_window_layout(req, pos.clone(), capacity=win.capacity)
        win.activate(layout)
        for step in range(3):
            # A graph bucket refreshes its layout in place, keeping the object.
            layout.copy_(direct_window_layout(req, pos + step, capacity=win.capacity))
            for layer in range(3):
                win.commit(layer)
            row = 1 * win.capacity + (10 + step) % win.capacity
            self.assertEqual(win.tags[:, row].tolist(), [10 + step] * 3)


if __name__ == "__main__":
    unittest.main()
