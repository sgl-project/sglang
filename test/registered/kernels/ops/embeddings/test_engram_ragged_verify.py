"""Ragged Engram must hash the same prefixes as dense verify and commit only accepted tokens."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.engram import EngramHasher
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def make_hasher(device, n=4):
    h = EngramHasher.__new__(EngramHasher)
    nn.Module.__init__(h)
    h.max_ngram_size = n
    h.pad_id = 0
    h.image_token_id = 127
    h.register_buffer("token_map", (torch.arange(128, device=device) * 7) % 128)
    h.register_buffer(
        "multipliers", torch.arange(1, 2 * n + 1, device=device).reshape(2, n) * 7919
    )
    h.register_buffer(
        "primes",
        torch.tensor([101, 103], device=device).expand(2, n - 1, 2).contiguous(),
    )
    h.register_buffer(
        "offsets",
        torch.arange((n - 1) * 2, device=device).expand(2, -1).contiguous() * 107,
    )
    h.init_history(12, device)
    h.history.copy_(
        torch.arange(h.history.numel(), device=device).reshape_as(h.history) % 97
    )
    return h


def batch(slots, positions, qo=None):
    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        req_pool_indices=slots,
        positions=positions,
        spec_info=SimpleNamespace(
            draft_token_num=6,
            ragged_verify_layout=None
            if qo is None
            else SimpleNamespace(qo_indptr_device=qo),
        ),
        out_cache_loc=torch.ones_like(positions),
    )


def scalar_reference(h, ids, positions, slots, lens):
    # Derive each request's full prefix independently; no row-search or dense
    # implementation is used as the correctness oracle.
    history = h.history.cpu().tolist()
    token_map = h.token_map.cpu().tolist()
    mult = h.multipliers.cpu().tolist()
    primes, offsets = h.primes.cpu().tolist(), h.offsets.cpu().tolist()
    ids, positions, slots = (
        ids.cpu().tolist(),
        positions.cpu().tolist(),
        slots.cpu().tolist(),
    )
    result, start = [], 0
    for slot, length in zip(slots, lens):
        prefix = history[slot] + ids[start : start + length]
        for j in range(length):
            comps, blocked = [], False
            for shift in range(h.max_ngram_size):
                token = prefix[h.max_ngram_size - 1 + j - shift]
                blocked |= positions[start + j] < shift or token == h.image_token_id
                comps.append(h.pad_id if blocked else token_map[token])
            layers = []
            for layer in range(2):
                cols = []
                for order in range(2, h.max_ngram_size + 1):
                    hashed = 0
                    for shift in range(order):
                        hashed ^= comps[shift] * mult[layer][shift]
                    cols.extend(
                        hashed % primes[layer][order - 2][head]
                        + offsets[layer][(order - 2) * 2 + head]
                        for head in range(2)
                    )
                layers.append(cols)
            result.append(layers)
        start += length
    return torch.tensor(result, dtype=torch.int64, device=h.history.device)


class TestEngramRaggedVerify(CustomTestCase):
    def run_cases(self, device):
        for n in (4, 8):
            h = make_hasher(device, n)
            for lens in ([1, 2, 3, 4, 5, 6], [0, 6, 0, 1, 0, 4], [1] * 6, [6] * 6):
                slots = torch.tensor([4, 2, 7, 1, 6, 3], device=device)
                dense = (torch.arange(36, device=device).reshape(6, 6) * 3 + 11) % 127
                dense[2, 1] = 127  # image token resets the lookback
                packed = torch.cat([r[:length] for r, length in zip(dense, lens)])
                positions = torch.cat(
                    [
                        torch.arange(length, device=device) + prefix
                        for length, prefix in zip(lens, [0, 1, 11, 24, 31, 90])
                    ]
                )
                qo = torch.tensor(
                    [0] + list(torch.tensor(lens).cumsum(0).tolist()),
                    dtype=torch.int32,
                    device=device,
                )
                before = h.history.clone()
                # Leave an uncovered graph-padding tail after the final row.
                ids = torch.cat([packed, packed.new_zeros(9)])
                pos = torch.cat([positions, positions.new_zeros(9)])
                actual = h(ids, batch(slots, pos, qo))
                expected = scalar_reference(h, packed, positions, slots, lens)
                torch.testing.assert_close(
                    actual[: len(packed)], expected, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual[len(packed) :], h.offsets.expand(9, -1, -1), rtol=0, atol=0
                )
                torch.testing.assert_close(h.history, before, rtol=0, atol=0)
                self.assertEqual(tuple(actual.shape), (len(packed) + 9, 2, (n - 1) * 2))
                dense_positions = torch.cat(
                    [
                        torch.arange(6, device=device) + prefix
                        for prefix in [0, 1, 11, 24, 31, 90]
                    ]
                )
                static = h(dense.flatten(), batch(slots, dense_positions))
                torch.testing.assert_close(
                    static,
                    scalar_reference(
                        h, dense.flatten(), dense_positions, slots, [6] * 6
                    ),
                    rtol=0,
                    atol=0,
                )

    def test_cpu_mixed_lengths(self):
        self.run_cases("cpu")

    def test_all_empty_rows_produce_only_padding_without_committing_history(self):
        for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
            h = make_hasher(device)
            slots = torch.tensor([4, 2, 7], device=device)
            ids = torch.full((9,), 127, device=device)
            positions = torch.arange(9, device=device) + 31
            qo = torch.zeros(4, dtype=torch.int32, device=device)
            before = h.history.clone()
            actual = h(ids, batch(slots, positions, qo))
            torch.testing.assert_close(actual, h.offsets.expand(9, -1, -1))
            torch.testing.assert_close(h.history, before)

    def test_reported_42_token_eight_request_geometry(self):
        lens = [6, 6, 5, 5, 5, 5, 5, 5]
        for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
            h = make_hasher(device)
            slots = torch.arange(8, device=device)
            ids = torch.arange(42, device=device) + 11
            positions = torch.cat(
                [torch.arange(length, device=device) + 31 for length in lens]
            )
            qo = torch.tensor(
                [0] + list(torch.tensor(lens).cumsum(0).tolist()), device=device
            )
            actual = h(ids, batch(slots, positions, qo))
            expected = scalar_reference(h, ids, positions, slots, lens)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_mixed_lengths(self):
        self.run_cases("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_row_search_power_of_two_boundaries(self):
        generator = torch.Generator().manual_seed(11)
        h = make_hasher("cuda", 8)
        for bs in (1, 2, 7, 8, 9, 31, 32, 33, 64, 65):
            lens = torch.randint(0, 7, (bs,), generator=generator).tolist()
            lens[0] = 1
            slots = torch.arange(bs, device="cuda") % 12
            ids = torch.arange(sum(lens), device="cuda") % 127
            positions = torch.cat(
                [torch.arange(length, device="cuda") + 31 for length in lens]
            )
            qo = torch.tensor(
                [0] + list(torch.tensor(lens).cumsum(0).tolist()), device="cuda"
            )
            actual = h(ids, batch(slots, positions, qo))
            torch.testing.assert_close(
                actual,
                scalar_reference(h, ids, positions, slots, lens),
                rtol=0,
                atol=0,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_graph_replay_changed_layout_and_history(self):
        h = make_hasher("cuda", 8)
        slots = torch.tensor([4, 2, 7, 1, 6, 3], device="cuda")
        ids = torch.arange(36, device="cuda") + 11
        positions = torch.arange(36, device="cuda") + 16
        qo = torch.arange(7, device="cuda", dtype=torch.int32) * 6
        fb = batch(slots, positions, qo)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                h(ids, fb)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = h(ids, fb)
        for lens in ([1, 2, 3, 4, 5, 6], [6, 0, 1, 0, 2, 0], [6] * 6, [1] * 6):
            qo.copy_(
                torch.tensor(
                    [0] + list(torch.tensor(lens).cumsum(0).tolist()), device="cuda"
                )
            )
            h.history.add_(1)
            expected = scalar_reference(
                h, ids[: sum(lens)], positions[: sum(lens)], slots, lens
            )
            graph.replay()
            torch.testing.assert_close(output[: sum(lens)], expected, rtol=0, atol=0)
            torch.testing.assert_close(
                output[sum(lens) :],
                h.offsets.expand(36 - sum(lens), -1, -1),
                rtol=0,
                atol=0,
            )

    def test_rejection_commit_and_slot_reuse(self):
        for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
            h = make_hasher(device, 8)
            slots = torch.tensor([4, 2, 7, 1, 6, 3], device=device)
            ids = torch.arange(36, device=device, dtype=torch.int32).reshape(6, 6) + 70
            commit = torch.tensor([1, 2, 3, 4, 5, 6], device=device, dtype=torch.int32)
            old = h.history.clone()
            h.commit_after_verify(ids, slots, commit)
            for row, (slot, count) in enumerate(zip(slots.tolist(), commit.tolist())):
                expected = torch.cat([old[slot], ids[row, :count]])[-7:]
                torch.testing.assert_close(h.history[slot], expected, rtol=0, atol=0)
            # A reused request slot gets the new prompt history from PD seeding.
            h.history[4].fill_(9)
            h.commit_after_verify(ids[:1], slots[:1], commit[:1])
            torch.testing.assert_close(
                h.history[4],
                torch.tensor([9] * 6 + [70], device=device, dtype=torch.int32),
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
