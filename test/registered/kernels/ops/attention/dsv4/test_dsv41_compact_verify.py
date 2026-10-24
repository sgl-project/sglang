"""DeepSeek-V4.1 per-token request lookups under compact (packed) target verify."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.dsv41_sparse import token_req_indices
from sglang.srt.layers.engram import EngramHasher
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

WIDTH = 6  # gamma + 1
IMAGE_TOKEN = 127
DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def make_hasher(device, n=4, num_slots=16):
    h = EngramHasher.__new__(EngramHasher)
    nn.Module.__init__(h)
    h.max_ngram_size = n
    h.pad_id = 0
    h.image_token_id = IMAGE_TOKEN
    g = torch.Generator().manual_seed(n)
    h.register_buffer("token_map", ((torch.arange(128) * 7) % 128).to(device))
    h.register_buffer("multipliers", torch.randint(1, 1 << 20, (2, n), generator=g))
    h.register_buffer("primes", torch.randint(1000, 5000, (2, n - 1, 2), generator=g))
    h.register_buffer("offsets", torch.arange((n - 1) * 2).expand(2, -1) * 5000)
    h.to(device)
    h.init_history(num_slots, device)
    h.history.copy_(torch.randint(1, 127, h.history.shape, generator=g))
    return h


def verify_batch(slots, positions, layout, draft_token_num=WIDTH):
    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        req_pool_indices=slots,
        positions=positions,
        spec_info=SimpleNamespace(
            draft_token_num=draft_token_num, ragged_verify_layout=layout
        ),
        out_cache_loc=None if positions is None else torch.ones_like(positions),
    )


def packed_layout(lens, num_tokens, device):
    return RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=torch.tensor(lens, dtype=torch.int32, device=device),
        graph_num_tokens=num_tokens,
    )


def packed_inputs(lens, num_tokens, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    total = sum(lens)
    ids = torch.randint(0, IMAGE_TOKEN, (num_tokens,), generator=g)
    if total > 3:
        ids[total // 2] = IMAGE_TOKEN  # an image token blocks the lookback
    starts = torch.randint(0, 40, (len(lens),), generator=g)
    starts[0] = 1  # a short prefix blocks the older shifts
    positions = torch.zeros(num_tokens, dtype=torch.int64)
    positions[:total] = torch.cat(
        [torch.arange(n) + s for n, s in zip(lens, starts.tolist())]
    )
    return ids.to(device), positions.to(device)


class TestEngramCompactVerify(CustomTestCase):
    def check_against_dense(self, h, lens, num_tokens, device):
        slots = torch.arange(len(lens), device=device) * 2 + 1
        ids, positions = packed_inputs(lens, num_tokens, device)
        history = h.history.clone()
        layout = packed_layout(lens, num_tokens, device)
        out = h(ids, verify_batch(slots, positions, layout))
        self.assertEqual(out.shape[0], num_tokens)
        start = 0
        for i, n in enumerate(lens):
            if n == 0:
                continue
            # Oracle: the request verified alone on the dense path.
            run = slice(start, start + n)
            dense = h(
                ids[run],
                verify_batch(slots[i : i + 1], positions[run], None, draft_token_num=n),
            )
            torch.testing.assert_close(out[run], dense, rtol=0, atol=0)
            start += n
        torch.testing.assert_close(h.history, history, rtol=0, atol=0)

    def test_matches_per_request_dense_verify(self):
        for device in DEVICES:
            for n in (4, 8):
                h = make_hasher(device, n)
                with self.subTest(device=device, n=n):
                    # Capture geometry from #39173: 42 tokens over 8 slots.
                    self.check_against_dense(h, [6, 6, 5, 5, 5, 5, 5, 5], 42, device)
                    # Empty rows and a graph-padding tail.
                    self.check_against_dense(h, [3, 0, 6, 1, 0, 2], 18, device)
                    # One request in a 3-token graph.
                    self.check_against_dense(h, [3, 0, 0], 3, device)

    def test_uniform_layout_matches_dense_path(self):
        for device in DEVICES:
            h = make_hasher(device)
            slots = torch.tensor([3, 1, 4], device=device)
            ids, positions = packed_inputs([WIDTH] * 3, 3 * WIDTH, device)
            layout = packed_layout([WIDTH] * 3, 3 * WIDTH, device)
            torch.testing.assert_close(
                h(ids, verify_batch(slots, positions, layout)),
                h(ids, verify_batch(slots, positions, None)),
                rtol=0,
                atol=0,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_graph_replay_follows_layout(self):
        h = make_hasher("cuda", 8)
        num_tokens, num_slots = 12, 8
        slots = torch.arange(num_slots, device="cuda") + 2
        ids = torch.zeros(num_tokens, dtype=torch.int64, device="cuda")
        positions = torch.zeros(num_tokens, dtype=torch.int64, device="cuda")
        layout = packed_layout([2, 2, 2, 2, 1, 1, 1, 1], num_tokens, "cuda")
        batch = verify_batch(slots, positions, layout)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            h(ids, batch), token_req_indices(batch, num_tokens=num_tokens)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = h(ids, batch)
            req = token_req_indices(batch, num_tokens=num_tokens)
        for seed, lens in enumerate(
            ([6, 1, 0, 0, 0, 0, 0, 0], [1] * 8, [0, 3, 0, 2, 0, 0, 1, 0])
        ):
            live_ids, live_positions = packed_inputs(lens, num_tokens, "cuda", seed)
            live = packed_layout(lens, num_tokens, "cuda")
            ids.copy_(live_ids)
            positions.copy_(live_positions)
            layout.qo_indptr_device.copy_(live.qo_indptr_device)
            graph.replay()
            eager_batch = verify_batch(slots, positions, live)
            total = sum(lens)
            torch.testing.assert_close(
                out[:total], h(ids, eager_batch)[:total], rtol=0, atol=0
            )
            torch.testing.assert_close(
                req, token_req_indices(eager_batch, num_tokens=num_tokens)
            )


class TestTokenReqIndicesCompactVerify(CustomTestCase):
    def test_rows_and_padding_tail(self):
        slots = torch.tensor([7, 5, 9, 4])
        batch = verify_batch(slots, None, packed_layout([2, 0, 3, 0], 8, "cpu"))
        # Empty rows own no tokens; the graph-padding tail maps to pool slot 0,
        # like the padded rows of a dense verify graph.
        self.assertEqual(
            token_req_indices(batch, num_tokens=8).tolist(), [7, 7, 9, 9, 9, 0, 0, 0]
        )

    def test_uniform_layout_matches_dense_path(self):
        slots = torch.tensor([7, 5])
        uniform = verify_batch(slots, None, packed_layout([WIDTH] * 2, 12, "cpu"))
        dense = verify_batch(slots, None, None)
        torch.testing.assert_close(
            token_req_indices(uniform, num_tokens=12),
            token_req_indices(dense, num_tokens=12),
        )


class TestCapturedVariantsReadStagedLayout(CustomTestCase):
    """DeepSeek-V4.1 captures each verify token tier once per attention variant.
    Replay stages the live lengths into the tier's registered layout, and both
    lookups must see them from the graphs of every variant."""

    def test_every_variant_maps_tokens_to_live_requests(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        num_tokens, num_slots = 12, 8
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.ragged_verify_mode = True
        runner.max_bs = num_slots
        runner.captured_req_width = WIDTH
        runner.capture_num_tokens = [WIDTH, 2 * WIDTH, 4 * WIDTH, 8 * WIDTH]
        runner.device = "cpu"
        runner._captured_ragged_layouts = {}
        with envs.SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE.override(False):
            captured = {
                variant: runner._capture_ragged_verify_layout(num_tokens)
                for variant in ("candidate_unfiltered", "candidate_filtered")
            }

        # Live step: request slot 7 verifies 6 tokens and slot 9 verifies 1.
        lens = [6, 1]
        total = sum(lens)
        live = packed_layout(lens, num_tokens, "cpu")
        runner._stage_ragged_verify_layout(live, num_tokens)
        slots = torch.tensor([7, 9] + [0] * (num_slots - len(lens)))
        h = make_hasher("cpu", 8)
        ids, positions = packed_inputs(lens, num_tokens, "cpu")
        eager = h(ids, verify_batch(slots[: len(lens)], positions, live))
        for variant, layout in captured.items():
            with self.subTest(variant=variant):
                batch = verify_batch(slots, positions, layout)
                self.assertEqual(
                    token_req_indices(batch, num_tokens=num_tokens)[:total].tolist(),
                    [7] * 6 + [9],
                )
                torch.testing.assert_close(
                    h(ids, batch)[:total], eager[:total], rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
