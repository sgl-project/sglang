"""DFLASH sparse top-k verify (SGLANG_OPT_DFLASH_SPARSE_VERIFY) against the dense
selector verify, through the worker's own support and accept methods."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.activation.softcap import gathered_softcap_logits
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative import dflash_worker_v2 as worker_mod
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# A vocab divisible by 8 TP ranks, Gemma-style softcap 20, 8 verify tokens, selector
# top_k 16, generation-config top_k 64.
VOCAB = 202048
BLOCK = 8
SELECTOR_TOP_K = 16
SOFTCAP = 20.0
GRAPH_BUCKET_BS = 4


class _SimulatedTpGroup:
    """The tp_group surface the support reads, for ``world`` ranks simulated on one
    GPU; the all-gather is rank-major, as NCCL's all_gather_into_tensor."""

    def __init__(self, world, rank, packed):
        self.world_size = world
        self.rank_in_group = rank
        self._packed = packed

    def all_gather_into_tensor(self, output, input):
        self._packed[self.rank_in_group] = input.clone()
        if len(self._packed) == self.world_size:
            output.copy_(torch.cat([self._packed[r] for r in range(self.world_size)]))
        else:
            output.zero_()


def _worker(*, block_verification=False):
    worker = worker_mod.DFlashWorkerV2.__new__(worker_mod.DFlashWorkerV2)
    worker._sparse_verify = True
    worker._selector_sample = (None, None)
    worker._block_verification = block_verification
    worker._draft_probs_buf = None
    worker._target_logits_processor = SimpleNamespace(
        vocab_size=VOCAB, final_logit_softcapping=SOFTCAP
    )
    return worker


def _make_case(*, bs, top_ks, temperature, seed, scale=3.0, noise=0.3, peak=0.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = bs * BLOCK
    raw = torch.randn(rows, VOCAB, device="cuda", generator=gen) * scale
    if peak:
        spikes = torch.randint(0, VOCAB, (rows,), device="cuda", generator=gen)
        raw[torch.arange(rows, device="cuda"), spikes] += peak
    # Graph replay hands verify the first bs * block rows of a larger bucket's
    # buffer; bf16-rounded values put ties at the top-k boundary.
    bucket = torch.zeros(GRAPH_BUCKET_BS * BLOCK, VOCAB, device="cuda")
    bucket[:rows] = raw.bfloat16().float()
    raw = bucket[:rows]
    capped = (torch.tanh(raw / SOFTCAP) * SOFTCAP).view(bs, BLOCK, VOCAB)
    q_ids = torch.empty(bs, BLOCK - 1, SELECTOR_TOP_K, dtype=torch.int64, device="cuda")
    q_rows = torch.empty(bs, BLOCK - 1, SELECTOR_TOP_K, device="cuda")
    candidates = torch.empty(bs, BLOCK, dtype=torch.int64, device="cuda")
    candidates[:, 0] = torch.randint(0, VOCAB, (bs,), device="cuda", generator=gen)
    for slot in range(BLOCK - 1):
        noisy = (
            capped[:, slot]
            + torch.randn(bs, VOCAB, device="cuda", generator=gen) * noise
        )
        vals, ids = torch.topk(noisy, SELECTOR_TOP_K, dim=-1)
        q = torch.softmax(vals * 1.3, dim=-1)
        q_ids[:, slot] = ids
        q_rows[:, slot] = q
        pick = torch.multinomial(q, 1, generator=gen).squeeze(1)
        candidates[:, slot + 1] = ids.gather(1, pick[:, None]).squeeze(1)
    sampling_info = SamplingBatchInfo(
        temperatures=torch.full((bs, 1), temperature, device="cuda"),
        top_ps=torch.ones(bs, device="cuda"),
        top_ks=torch.tensor(top_ks, dtype=torch.int32, device="cuda"),
        min_ps=torch.zeros(bs, device="cuda"),
        is_all_greedy=False,
        is_any_greedy=1 in top_ks,
        need_top_p_sampling=False,
        need_top_k_sampling=True,
        need_min_p_sampling=False,
        vocab_size=VOCAB,
    )
    draft_input = SimpleNamespace(
        max_top_k=max(top_ks),
        uniform_top_k_value=top_ks[0] if len(set(top_ks)) == 1 else None,
    )
    return raw, candidates, q_ids, q_rows, sampling_info, draft_input


def _rank_shards(raw, world):
    """Each rank's logits shard in its own contiguous lm_head output buffer; verify
    reads the first bs * block rows of a larger graph bucket's buffer."""
    rows = raw.shape[0]
    shard = VOCAB // world
    shards = []
    for rank in range(world):
        bucket = torch.zeros(GRAPH_BUCKET_BS * BLOCK, shard, device=raw.device)
        bucket[:rows] = raw[:, rank * shard : (rank + 1) * shard]
        shards.append(bucket[:rows])
    return shards


def _support_on_every_rank(*, raw, world, sampling_info, draft_input):
    shards = _rank_shards(raw, world)
    packed = {}
    # Pass 1 records every rank's packed shard; pass 2 gathers them all.
    for _ in range(2):
        supports = []
        for rank in range(world):
            parallel = SimpleNamespace(
                tp_group=_SimulatedTpGroup(world, rank, packed), tp_rank=rank
            )
            with mock.patch.object(worker_mod, "get_parallel", return_value=parallel):
                supports.append(
                    _worker()._sparse_verify_support(
                        batch=SimpleNamespace(has_grammar=False, return_logprob=False),
                        sampling_info=sampling_info,
                        draft_input=draft_input,
                        logits_shard=shards[rank],
                    )
                )
    return supports


def _dense_full_logits(raw, world):
    """finish_deferred_gather: the rank-major gather of the shards, softcapped."""
    shard = VOCAB // world
    gathered = torch.stack(
        [raw[:, r * shard : (r + 1) * shard] for r in range(world)]
    ).contiguous()
    full = torch.empty(raw.shape[0], VOCAB, dtype=torch.float32, device=raw.device)
    return gathered_softcap_logits(gathered, SOFTCAP, full)


def _check_parity(*, world, bs, top_ks, temperature, trials, seed, **case_kw):
    stats = dict(all_accepted=0, rejected=0)
    for t in range(trials):
        raw, cands, q_ids, q_rows, sampling_info, draft_input = _make_case(
            bs=bs, top_ks=top_ks, temperature=temperature, seed=seed + t, **case_kw
        )
        supports = _support_on_every_rank(
            raw=raw, world=world, sampling_info=sampling_info, draft_input=draft_input
        )
        for other in supports[1:]:
            for a, b in zip(supports[0], other):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        sup_vals, _, argmax = supports[0]
        full = _dense_full_logits(raw, world)
        k = draft_input.max_top_k
        # The merged support is the top-k of the full softcapped rows, sorted desc.
        torch.testing.assert_close(
            sup_vals[:, :k], torch.topk(full, k, dim=-1).values, rtol=0, atol=0
        )
        torch.testing.assert_close(argmax, torch.argmax(full, dim=-1), rtol=0, atol=0)

        accept_kw = dict(
            candidates=cands,
            candidate_ids=q_ids,
            q_rows=q_rows,
            sampling_info=sampling_info,
            draft_input=draft_input,
        )
        torch.cuda.manual_seed(1000 + t)
        dense = _worker()._selector_sampling_accept(next_token_logits=full, **accept_kw)
        dense_probe = torch.rand(4, device="cuda")
        torch.cuda.manual_seed(1000 + t)
        sparse = _worker()._selector_sparse_accept(support=supports[0], **accept_kw)
        sparse_probe = torch.rand(4, device="cuda")
        msg = f"world={world} bs={bs} top_ks={top_ks} T={temperature} trial={t}"
        assert torch.equal(dense[0], sparse[0]), msg
        assert torch.equal(dense[1], sparse[1]), msg
        # Same draws in the same order: the sampler's RNG stream continues identically.
        assert torch.equal(dense_probe, sparse_probe), msg
        stats["all_accepted"] += int((dense[0] == BLOCK - 1).sum())
        stats["rejected"] += int((dense[0] < BLOCK - 1).sum())
    return stats


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("bs", [1, 2, 3, 4])
@pytest.mark.parametrize("temperature", [0.6, 1.0, 1.3])
def test_top_k_64_matches_the_dense_verify(world, bs, temperature):
    stats = _check_parity(
        world=world,
        bs=bs,
        top_ks=[64] * bs,
        temperature=temperature,
        trials=6,
        seed=17 * world + 101 * bs,
    )
    assert stats["rejected"] > 0


def test_long_accepts_sample_the_bonus_from_the_last_row():
    stats = _check_parity(
        world=8,
        bs=4,
        top_ks=[64] * 4,
        temperature=1.0,
        trials=40,
        seed=7,
        scale=1.0,
        noise=0.01,
        peak=10.0,
    )
    assert stats["all_accepted"] > 0 and stats["rejected"] > 0


def test_mixed_top_k_and_greedy_rows_match_the_dense_verify():
    # Non-uniform top_k masks ranks per row; top_k 1 is a greedy row.
    _check_parity(
        world=8,
        bs=4,
        top_ks=[64, 20, 1, 64],
        temperature=0.7,
        trials=30,
        seed=3,
        scale=2.0,
        noise=0.02,
        peak=6.0,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
