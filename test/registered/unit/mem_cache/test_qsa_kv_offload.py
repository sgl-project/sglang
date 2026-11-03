"""QSA K/V host offload: layout, pool sizing and the reference resolve.

The resolve test drives ``QSAKVHostOffload`` the way the QSA backend does
(extend, then decode or chain-verify forwards that store their new K/V and
resolve the selected positions) and checks after every forward that each
selected position's device row holds exactly the K/V of the logical slot
``req_to_token`` maps it to, with the hot cache far smaller than the contexts,
radix remaps of resident blocks and rejected speculative tokens.
"""

import random
import unittest

import torch

from sglang.srt.mem_cache.qsa_kv_offload import (
    QSAKVHostOffload,
    QSAKVOffloadLayout,
    compute_offload_layout,
    offload_max_total_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

CR = 4


class TestOffloadLayout(CustomTestCase):
    def test_regions_scratch_and_page_alignment(self):
        layout = compute_offload_layout(
            compress_ratio=CR,
            token_topk=2048,
            hot_blocks=4096,
            num_request_slots=65,
            rows_per_request=1,
            max_extend_tokens=8192,
            page_size=64,
        )
        self.assertEqual(layout.region_rows, (4096 + 2) * CR)
        self.assertEqual(layout.scratch_base, 65 * (4096 + 2) * CR)
        self.assertGreaterEqual(layout.scratch_rows, 8192 + 2 * 64)
        self.assertEqual(layout.device_rows % 64, 0)

    def test_refuses_hot_cache_below_one_selection(self):
        with self.assertRaisesRegex(ValueError, "below 1536"):
            compute_offload_layout(
                compress_ratio=CR,
                token_topk=2048,
                hot_blocks=1535,
                num_request_slots=9,
                rows_per_request=3,
                max_extend_tokens=1024,
                page_size=64,
            )

    def test_refuses_more_rows_than_compress_ratio(self):
        with self.assertRaisesRegex(ValueError, "query rows per request"):
            compute_offload_layout(
                compress_ratio=CR,
                token_topk=2048,
                hot_blocks=4096,
                num_request_slots=9,
                rows_per_request=CR + 1,
                max_extend_tokens=1024,
                page_size=64,
            )


class TestOffloadSizing(CustomTestCase):
    def test_gpu_bound(self):
        # 100 B/token stay on the GPU; the host allows far more.
        tokens = offload_max_total_tokens(
            available_bytes=10**9,
            cell_bytes=1100,
            offloaded_bytes_per_token=1000,
            device_rows=1000 - 64,
            page_size=64,
            host_size_gb=1000,
        )
        self.assertEqual(tokens, (10**9 - 1000 * 1000) // 100 // 64 * 64)

    def test_host_bound(self):
        tokens = offload_max_total_tokens(
            available_bytes=10**12,
            cell_bytes=1100,
            offloaded_bytes_per_token=1000,
            device_rows=0,
            page_size=64,
            host_size_gb=1,
        )
        self.assertEqual(tokens, (10**9 // 1000 - 64) // 64 * 64)

    def test_hot_cache_larger_than_budget(self):
        with self.assertRaisesRegex(ValueError, "exceed"):
            offload_max_total_tokens(
                available_bytes=10**6,
                cell_bytes=1100,
                offloaded_bytes_per_token=1000,
                device_rows=10**4,
                page_size=64,
                host_size_gb=1,
            )


class _OffloadSim:
    """Logical K/V pool plus an offload, driven like the QSA backend drives it."""

    def __init__(
        self, *, rows_per_request, seed, device="cpu", pin=False, hot_blocks=None
    ):
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.device = device
        self.layers, self.heads, self.dim = 2, 2, 8
        self.block_topk = 16
        self.q = rows_per_request
        self.slots = 6  # request slot 0 is padding
        self.width = 1024
        self.pool_size = 8192
        self.layout = compute_offload_layout(
            compress_ratio=CR,
            token_topk=self.block_topk * CR,
            hot_blocks=self.q * self.block_topk + 3,
            num_request_slots=self.slots,
            rows_per_request=self.q,
            max_extend_tokens=self.width,
            page_size=64,
        )
        if hot_blocks is not None:  # below the minimum compute_offload_layout allows
            self.layout = QSAKVOffloadLayout(
                compress_ratio=CR,
                hot_blocks=hot_blocks,
                num_request_slots=self.slots,
                scratch_rows=self.layout.scratch_rows,
            )
        shape = (self.layout.device_rows + 64, self.heads, self.dim)
        self.dev_k = [torch.zeros(shape, device=device) for _ in range(self.layers)]
        self.dev_v = [torch.zeros(shape, device=device) for _ in range(self.layers)]
        self.offload = QSAKVHostOffload(
            layout=self.layout,
            k_buffers=self.dev_k,
            v_buffers=self.dev_v,
            host_rows=self.pool_size + 64,
            pin_memory=pin,
        )
        ref_shape = (self.layers, self.pool_size + 64, self.heads, self.dim)
        self.ref_k = torch.zeros(ref_shape, device=device)
        self.ref_v = torch.zeros(ref_shape, device=device)
        self.req_to_token = torch.zeros(
            (self.slots, self.width), dtype=torch.int32, device=device
        )
        self.free = list(range(64, self.pool_size + 64))
        self.rng.shuffle(self.free)
        self.lengths = {}

    def _alloc(self, n):
        out, self.free = self.free[:n], self.free[n:]
        return torch.tensor(out, dtype=torch.int64, device=self.device)

    def _write(self, slots, rows):
        """New K/V of ``slots``: device rows, write-through, reference pool."""
        for layer in range(self.layers):
            k = torch.randn(slots.numel(), self.heads, self.dim, device=self.device)
            v = torch.randn_like(k)
            self.dev_k[layer][rows] = k
            self.dev_v[layer][rows] = v
            self.offload.write_through(layer, rows, slots)
            self.ref_k[layer][slots] = k
            self.ref_v[layer][slots] = v

    def extend(self, req, length):
        slots = self._alloc(length)
        self.req_to_token[req, :length] = slots.to(torch.int32)
        self.lengths[req] = length
        self.offload.begin_forward(
            self.req_to_token,
            reset_req_pool_indices=torch.tensor([req], device=self.device),
        )
        self._write(slots, self.offload.extend_rows(length))

    def remap(self, req):
        """Radix dedup: map complete blocks before the ring to other slots."""
        first_block = self.lengths[req] // CR
        if first_block < 2:
            return
        b = self.rng.randrange(first_block - 1)
        cols = torch.arange(b * CR, (b + 1) * CR, device=self.device)
        slots = self._alloc(CR)
        self.req_to_token[req, cols] = slots.to(torch.int32)
        for layer in range(self.layers):
            self.ref_k[layer][slots] = torch.randn_like(self.ref_k[layer][slots])
            self.ref_v[layer][slots] = torch.randn_like(self.ref_v[layer][slots])
            self.offload.host_k[layer][slots.cpu()] = self.ref_k[layer][slots].cpu()
            self.offload.host_v[layer][slots.cpu()] = self.ref_v[layer][slots].cpu()

    def positions(self, qpos):
        """Expanded QSA selection of one query row (valid positions first)."""
        out = [-1] * (self.block_topk * CR + CR - 1)
        tail_start = (qpos + 1) // CR * CR
        blocks = self.rng.sample(
            range(tail_start // CR), min(self.block_topk, tail_start // CR)
        )
        chosen = [b * CR + i for b in blocks for i in range(CR)]
        chosen += list(range(tail_start, qpos + 1))
        out[: len(chosen)] = chosen
        return out

    def step(self):
        """One decode / chain-verify forward over every request."""
        reqs = sorted(self.lengths)
        row_req, row_seq, positions = [], [], []
        new_slots = []
        for req in reqs:
            first = self.lengths[req]
            slots = self._alloc(self.q)
            self.req_to_token[req, first : first + self.q] = slots.to(torch.int32)
            new_slots.append(slots)
            for i in range(self.q):
                row_req.append(req)
                row_seq.append(first + i + 1)
                positions.append(self.positions(first + i))
        row_req = torch.tensor(row_req, device=self.device)
        row_seq = torch.tensor(row_seq, dtype=torch.int32, device=self.device)
        positions = torch.tensor(positions, dtype=torch.int32, device=self.device)
        self.offload.begin_forward(self.req_to_token)
        self._write(torch.cat(new_slots), self.offload.new_token_rows(row_seq, row_req))
        for layer in range(self.layers):
            rows = self.offload.resolve_rows(layer, positions, row_seq, row_req, self.q)
            self._check(layer, rows, positions, row_seq, row_req)
        for req in reqs:  # accept a prefix of the verify rows; free the rest
            accepted = self.rng.randrange(1, self.q + 1)
            first = self.lengths[req]
            rejected = self.req_to_token[req, first + accepted : first + self.q]
            self.free += rejected.tolist()
            self.lengths[req] = first + accepted

    def _check(self, layer, rows, positions, row_seq, row_req):
        pos = positions.long()
        valid = (pos >= 0) & (pos < row_seq[:, None].long())
        assert torch.equal(rows >= 0, valid)
        slots = self.req_to_token[row_req[:, None], pos.clamp(min=0)].long()
        got_k = self.dev_k[layer][rows.long()[valid]]
        got_v = self.dev_v[layer][rows.long()[valid]]
        assert torch.equal(got_k, self.ref_k[layer][slots[valid]])
        assert torch.equal(got_v, self.ref_v[layer][slots[valid]])


class TestOffloadResolve(CustomTestCase):
    def _run(self, rows_per_request, seed):
        sim = _OffloadSim(rows_per_request=rows_per_request, seed=seed)
        for req in range(1, sim.slots):
            sim.extend(req, sim.rng.randrange(200, 400))
        for step in range(60):
            if step % 7 == 3:
                sim.remap(sim.rng.randrange(1, sim.slots))
            if step == 30:  # a request slot is reused by a new request
                sim.extend(2, sim.rng.randrange(200, 400))
            sim.step()

    def test_decode(self):
        self._run(rows_per_request=1, seed=1)

    def test_chain_verify_with_rejections(self):
        for q in (2, 3, CR):
            with self.subTest(rows_per_request=q):
                self._run(rows_per_request=q, seed=10 + q)

    def test_selection_larger_than_hot_cache_raises(self):
        sim = _OffloadSim(rows_per_request=1, seed=3, hot_blocks=4)
        sim.extend(1, 300)
        with self.assertRaisesRegex(RuntimeError, "cannot hold"):
            sim.step()


if __name__ == "__main__":
    unittest.main()
