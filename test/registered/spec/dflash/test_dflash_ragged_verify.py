"""DFLASH on the device: under confidence-scheduled verify, sampling acceptance
stops at the planned verify length and the lagged confidence relay resolves the
same copy on every rank; under simulated acceptance, the plain head's sampled
drafts get fresh noise on every CUDA-graph replay."""

import types
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import overlap_utils
from sglang.srt.speculative import bench_drafter
from sglang.srt.speculative import dflash_worker_v2 as worker_mod
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")


def _selector_accept(candidates, target_tokens, cutoff_verify_lens):
    """Selector sampling acceptance with a one-hot draft q on each drawn token
    and a target that puts all of its mass on target_tokens, so every
    acceptance and bonus is determined."""
    bs, block = candidates.shape
    vocab = 32
    other = (candidates[:, 1:] + 1) % vocab
    candidate_ids = torch.stack((candidates[:, 1:], other), dim=-1)
    q_rows = torch.zeros(candidate_ids.shape, device=DEVICE)
    q_rows[..., 0] = 1.0
    target_logits = torch.zeros(bs * block, vocab, device=DEVICE)
    target_logits.scatter_(1, target_tokens.reshape(-1, 1), 100.0)

    worker = types.SimpleNamespace(
        _selector_sample=(candidate_ids, q_rows),
        _draft_probs_buf=None,
        _block_verification=False,
        _tp_sync=types.SimpleNamespace(sync=lambda site, tensor: tensor),
        block_size=block,
    )
    worker._selector_sampling_accept = types.MethodType(
        worker_mod.DFlashWorkerV2._selector_sampling_accept, worker
    )
    return worker_mod.DFlashWorkerV2._accept_block(
        worker,
        candidates=candidates,
        next_token_logits=target_logits,
        sampling_info=types.SimpleNamespace(
            temperatures=torch.ones(bs, 1, device=DEVICE),
            need_top_k_sampling=False,
            need_top_p_sampling=False,
        ),
        draft_input=None,
        prefix_lens=torch.zeros(bs, dtype=torch.int64, device=DEVICE),
        bs=bs,
        cutoff_verify_lens=cutoff_verify_lens,
    )


class TestDFlashRaggedVerify(CustomTestCase):
    def test_selector_sampling_accept_stops_at_planned_verify_len(self):
        # Rows 0 and 2 draft the target's tokens; row 1 misses its second draft.
        candidates = torch.tensor(
            [[9, 1, 2, 3, 4], [9, 5, 6, 7, 8], [9, 10, 11, 12, 13]], device=DEVICE
        )
        target_tokens = torch.tensor(
            [[1, 2, 3, 4, 20], [5, 21, 7, 8, 22], [10, 11, 12, 13, 23]],
            device=DEVICE,
        )

        full = _selector_accept(candidates, target_tokens, None)
        self.assertEqual(full[0].tolist(), [4, 1, 4])
        self.assertEqual(full[2].tolist(), [20, 21, 23])

        accept_len, commit_lens, bonus, out_tokens, _, cap_trim_lens = _selector_accept(
            candidates,
            target_tokens,
            torch.tensor([3, 5, 1], dtype=torch.int32, device=DEVICE),
        )
        # Each request accepts at most verify_len - 1 drafts and takes its bonus
        # from its last verified row; the cutoff trims the rest.
        self.assertEqual(accept_len.tolist(), [2, 1, 0])
        self.assertEqual(bonus.tolist(), [3, 21, 10])
        self.assertEqual(cap_trim_lens.tolist(), [2, 0, 4])
        self.assertEqual(commit_lens.tolist(), [3, 2, 1])
        self.assertEqual(out_tokens[0, :3].tolist(), [1, 2, 3])

    def test_relay_resolves_the_lagged_copy_while_it_is_in_flight(self):
        lag = overlap_utils.CONFIDENCE_RELAY_RING_LAG
        relay = overlap_utils.ConfidenceRelay(
            device=DEVICE,
            req_pool_size=4,
            pool=types.SimpleNamespace(
                req_generation=torch.zeros(4, dtype=torch.int64)
            ),
        )
        side = torch.cuda.Stream()
        indices = torch.tensor([1, 2], device=DEVICE)
        ready = None
        for step in range(lag):
            relay.scatter(indices, torch.full((2, 3), step + 1.0, device=DEVICE))
            ready = torch.cuda.Event()
            ready.record()
            # Hold the ring copy in flight behind a busy side stream.
            with torch.cuda.stream(side):
                torch.cuda._sleep(100_000_000)
            relay.issue_ring_copy(stream=side, publish_ready=ready)

        batch = types.SimpleNamespace(
            spec_info=types.SimpleNamespace(future_indices=torch.tensor([0])),
            req_pool_indices_cpu=torch.tensor([1, 2]),
        )
        # The copy of the first step is still queued: resolving waits for it
        # rather than returning nothing, and the next scatter did not overwrite
        # the buffer it reads.
        resolved = relay.resolve(batch, stream=side, publish_ready=ready)
        self.assertIsNotNone(resolved)
        torch.testing.assert_close(resolved.confidence, torch.full((2, 3), 1.0))
        torch.cuda.synchronize()


class TestPlainHeadSampling(CustomTestCase):
    def test_graph_replays_draw_fresh_noise(self):
        torch.manual_seed(0)
        block, hidden_size, vocab = 4, 8, 40
        weight = torch.randn(vocab, hidden_size, device=DEVICE)
        hidden = torch.randn(2 * block, hidden_size, device=DEVICE)
        noise = bench_drafter.BenchHeadNoise(
            key=7, output_multiplier=1.0, softcap=None, vocab_limit=vocab
        )
        sampler = worker_mod._DflashDraftSampler(
            weight=weight,
            block_size=block,
            num_org=vocab,
            org_vocab_start=0,
            max_bs=2,
            bench_noise=noise,
            bench_padded_width=vocab,
        )
        greedy = types.SimpleNamespace(
            temperatures=torch.zeros(2, 1, device=DEVICE),
            top_ks=torch.ones(2, dtype=torch.int32, device=DEVICE),
        )
        with envs.SGLANG_BENCH_ACC_DRAFT_TEMPERATURE.override("1000"):
            bench_drafter.bench_drafter_temperature.cache_clear()
            sampler.stage_sampling_params(bs=2, sampling_info=greedy)
        bench_drafter.bench_drafter_temperature.cache_clear()
        sampler(hidden)  # compile the kernels before capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sampler(hidden)
        sampler.bench_offset.zero_()
        rows = hidden.view(2, block, hidden_size)[:, 1:, :].reshape(-1, hidden_size)
        logits = (rows @ weight.T).cpu()
        n = rows.shape[0]
        for step in range(2):
            graph.replay()
            expected = bench_drafter.gumbel_argmax(
                logits,
                noise=noise,
                temperatures=torch.full((n,), 1000.0),
                offset=torch.tensor([step], dtype=torch.int64),
                step=bench_drafter.GRAPH_STEP,
                vocab_start=0,
            )
            self.assertTrue(torch.equal(sampler.out[:n].cpu(), expected))
        self.assertEqual(int(sampler.bench_offset), 2)


if __name__ == "__main__":
    unittest.main()
