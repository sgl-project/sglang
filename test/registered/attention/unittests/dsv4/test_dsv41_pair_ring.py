"""Ratio-2 pending-pair ring: one fixed-shape routine for decode, extend,
target-verify and draft-extend.

Tokens compress in pairs (2k, 2k+1), so an even token has to survive until its
odd partner is projected -- possibly in a later batch, after a chunk boundary or
after a rejected speculative draft is regenerated. The partner comes from the
previous row of the batch when that row is present and from the request's
position ring otherwise; the ring is addressed by position % ring_size, so it
wraps, and slots left behind by an earlier occupant of the request slot must
never be mistaken for live state.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

HEAD_DIM = 16
NUM_REQ_SLOTS = 4


class _Pairing:
    """Drives `_low_ratio_pair_partners` batch by batch and keeps the oracle: the
    newest (kv, score) fed in for every (request, position) so far."""

    def __init__(self, ring_size, *, num_draft_tokens=0):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )
        from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool

        self.state = CompressStatePool(
            size=NUM_REQ_SLOTS * ring_size,
            ring_size=ring_size,
            overlap=False,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
            enable_memory_saver=False,
            ratio=2,
        )
        self.seen = {}
        self.step = 0
        self.backend = object.__new__(DeepseekV4AttnBackend)
        self.backend.token_to_kv_pool = SimpleNamespace(
            get_attention_compress_states=lambda layer_id: self.state
        )
        self.backend.extend_seq_lens_buffer = torch.full(
            (NUM_REQ_SLOTS,), num_draft_tokens, dtype=torch.int32, device="cuda"
        )

    def run(self, batch, *, mode="extend", pads=()):
        """`batch` is [(req, [positions...]), ...] in token order; returns the flat
        token list and the partner kv / score of every row."""
        flat = [(r, p) for r, ps in batch for p in ps]
        num_tokens = len(flat)
        self.step += 1
        # Distinct per (step, row), so a stale slot can never pass for a live one.
        rows = torch.arange(num_tokens, dtype=torch.float32, device="cuda")
        kv = (rows + 1000.0 * self.step).unsqueeze(-1).repeat(1, HEAD_DIM)
        score = torch.zeros_like(kv)
        pad = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")
        for i in pads:
            pad[i] = True
        partner_kv, partner_score = self.backend._low_ratio_pair_partners(
            layer_id=0,
            kv=kv,
            score=score,
            pos=torch.tensor([p for _, p in flat], dtype=torch.int64, device="cuda"),
            pad=pad,
            req=torch.tensor([r for r, p in flat], dtype=torch.int64, device="cuda"),
        )
        for i, (r, p) in enumerate(flat):
            if i not in pads:
                self.seen[(r, p)] = kv[i]
        return flat, partner_kv, partner_score

    def check_odd_partners(self, case, flat, partner_kv, pads=()):
        """Every odd position must be handed the newest value of the position
        before it. Even positions complete no group and are not checked."""
        for i, (r, p) in enumerate(flat):
            if p % 2 == 0 or i in pads:
                continue
            expected = self.seen.get((r, p - 1))
            case.assertIsNotNone(expected, f"no source recorded for {(r, p)}")
            torch.testing.assert_close(partner_kv[i], expected, msg=f"{(r, p)}")


@unittest.skipUnless(torch.cuda.is_available(), "allocates a device state pool")
class TestPairRing(CustomTestCase):
    def test_prefill_graph_replays_changed_chunks_without_host_sync(self):
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import token_req_indices

        pairing = _Pairing(ring_size=8)
        state = pairing.state.kv_score_buffer.kv_score
        initial = state.clone()
        kv = torch.randn(7, HEAD_DIM, device="cuda")
        score = torch.randn_like(kv)
        pos = torch.arange(7, device="cuda", dtype=torch.int64)
        pad = torch.zeros(7, device="cuda", dtype=torch.bool)
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 2], device="cuda"),
            extend_seq_lens=torch.tensor([4, 3], device="cuda"),
            forward_mode=SimpleNamespace(
                is_decode=lambda: False,
                is_target_verify=lambda: False,
                is_extend=lambda: True,
            ),
        )

        def run():
            req = token_req_indices(batch, num_tokens=pos.numel())
            return pairing.backend._low_ratio_pair_partners(
                layer_id=0, kv=kv, score=score, req=req, pos=pos, pad=pad
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = run()
        state.copy_(initial)
        history = {}
        for step in range(6):
            # Request 2 alternates between an even and an odd chunk boundary.
            positions = list(range(step * 4, step * 4 + 4)) + list(
                range(step * 3, step * 3 + 3)
            )
            pos.copy_(torch.tensor(positions, device="cuda"))
            kv.normal_()
            score.normal_()
            for row, (req, position) in enumerate(zip([1] * 4 + [2] * 3, positions)):
                history[req, position] = (kv[row].clone(), score[row].clone())
            graph.replay()
            for row, (req, position) in enumerate(zip([1] * 4 + [2] * 3, positions)):
                if position % 2:
                    expected = history[req, position - 1]
                    for actual, wanted in zip(got, expected):
                        torch.testing.assert_close(actual[row], wanted, rtol=0, atol=0)

    def test_decode_steps_wrap_the_ring(self):
        """One token per request per step, so every pair straddles two batches and
        every odd position is served by the ring, wrap after wrap."""
        pairing = _Pairing(ring_size=2)
        offsets = {0: 0, 2: 6, 3: 12}
        for step in range(24):
            flat, partner_kv, _ = pairing.run(
                [(r, [base + step]) for r, base in offsets.items()], mode="decode"
            )
            pairing.check_odd_partners(self, flat, partner_kv)

    def test_chunk_boundary_splits_a_pair(self):
        """An extend chunk ending on an even position parks it; the next chunk
        starts on the odd partner and has to take it back out of the ring."""
        pairing = _Pairing(ring_size=2)
        for positions in ([0, 1, 2, 3, 4], [5, 6], [7, 8, 9, 10], [11]):
            flat, partner_kv, _ = pairing.run([(1, positions)])
            pairing.check_odd_partners(self, flat, partner_kv)

    def test_stale_ring_content_is_ignored_at_an_even_start(self):
        """A recycled request slot still holds the previous occupant's pair state.
        A chunk starting at an even (page-aligned) position pairs entirely inside
        itself, so no completed pair may draw on those rows."""
        pairing = _Pairing(ring_size=2)
        state = pairing.state.kv_score_buffer.kv_score
        state[0:2] = -7.0  # request 0 owns rows [0, ring_size)

        flat, partner_kv, partner_score = pairing.run([(0, [256, 257, 258, 259])])
        pairing.check_odd_partners(self, flat, partner_kv)
        odd = torch.tensor([p % 2 == 1 for _, p in flat], device="cuda")
        self.assertFalse(bool((partner_kv[odd] == -7.0).any()))
        self.assertFalse(bool((partner_score[odd] == -7.0).any()))
        # The trailing pair replaced both stale rows, so nothing outlives the chunk.
        self.assertFalse(bool((state[0:2] == -7.0).any()))

    def test_regenerated_draft_position_replaces_the_rejected_one(self):
        """A verify batch writes its whole optimistic tail into the ring. After a
        partial accept the rejected positions come back with new values, and a
        later batch must pair against those, not against the rejected drafts."""
        num_draft_tokens = 4
        ring_size = 1 << (num_draft_tokens + 1).bit_length()
        pairing = _Pairing(ring_size, num_draft_tokens=num_draft_tokens)

        # Prefill 0..7, so both requests enter the first verify at an even position.
        flat, partner_kv, _ = pairing.run([(0, list(range(8))), (1, list(range(8)))])
        pairing.check_odd_partners(self, flat, partner_kv)

        start = {0: 8, 1: 8}
        # The third round starts on an odd position whose partner was a rejected
        # draft of the first round and was regenerated by the second.
        for accepted in (2, 1, 3, 2):
            flat, partner_kv, _ = pairing.run(
                [
                    (r, list(range(start[r], start[r] + num_draft_tokens)))
                    for r in (0, 1)
                ],
                mode="verify",
            )
            pairing.check_odd_partners(self, flat, partner_kv)
            for r in (0, 1):
                for p in range(start[r] + accepted, start[r] + num_draft_tokens):
                    pairing.seen.pop((r, p))
                start[r] += accepted

    def test_pad_rows_touch_only_the_sentinel(self):
        """Padded graph rows carry a live req_pool_idx and out_loc 0; they must
        leave that request's ring alone and read the empty row."""
        pairing = _Pairing(ring_size=2)
        pairing.run([(0, [0, 1, 2])])  # request 0 parks position 2 in slot 0
        before = pairing.state.kv_score_buffer.kv_score.clone()

        flat, partner_kv, partner_score = pairing.run(
            [(0, [3]), (0, [0]), (0, [0])], mode="decode", pads=(1, 2)
        )
        pairing.check_odd_partners(self, flat, partner_kv, pads=(1, 2))
        rows = pairing.state.kv_score_buffer.kv_score
        torch.testing.assert_close(rows[0], before[0])
        torch.testing.assert_close(rows[-1], before[-1])
        self.assertTrue(bool(torch.isinf(partner_score[1]).all()))


if __name__ == "__main__":
    unittest.main()
