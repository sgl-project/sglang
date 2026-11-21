"""DFLASH prompt lookup on the device: the token table, its pinned staging and
the hit ring under CUDA, and looked-up blocks through both sampling accepts."""

import types
import unittest
from array import array

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative import dflash_worker_v2 as worker_mod
from sglang.srt.speculative.dflash_prompt_lookup import (
    DFlashPromptLookup,
    PromptLookupConfig,
)
from sglang.srt.speculative.dflash_utils import (
    compute_dflash_sampling_correct_drafts_and_bonus,
)
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")
BLOCK = 8
VOCAB = 64
# Each request copies its own pattern over and over: a 16-token match per step.
PATTERNS = [list(range(10, 30)), list(range(30, 50))]


def _sampling_info(bs, top_k):
    return types.SimpleNamespace(
        temperatures=torch.ones(bs, 1, device=DEVICE),
        top_ks=torch.full((bs,), top_k, dtype=torch.int32, device=DEVICE),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        has_custom_logit_processor=False,
        step_penalties=None,
        grammar_mask=None,
        logit_bias=None,
        sampling_mask_batch_indices=None,
    )


def _setup(top_k):
    pool = ReqToTokenPool(
        size=8, max_context_len=1, device="cpu", enable_memory_saver=False
    )
    histories = [pattern * 2 for pattern in PATTERNS]
    reqs = [
        Req(f"r{i}", "", array("q", history), SamplingParams())
        for i, history in enumerate(histories)
    ]
    pool.alloc(reqs)
    lookup = DFlashPromptLookup(
        config=PromptLookupConfig(),
        vocab_size=VOCAB,
        max_rows=8,
        positions=256,
        device=DEVICE,
        log_rank=False,
    )
    return pool, reqs, histories, lookup, _sampling_info(len(reqs), top_k)


def _propose(pool, reqs, histories, lookup, sampling_info, selector_sample=None):
    """One decode step in which request i has committed histories[i][:-1] and
    carries the bonus token histories[i][-1]."""
    prefix_lens = torch.tensor([len(h) - 1 for h in histories])
    batch = ScheduleBatch(
        reqs=reqs,
        seq_lens=prefix_lens.to(DEVICE),
        seq_lens_cpu=prefix_lens,
        sampling_info=sampling_info,
    )
    draft_input = make_draft_input_v2(
        bonus_tokens=torch.tensor([h[-1] for h in histories], device=DEVICE),
        new_seq_lens=prefix_lens.to(DEVICE),
    )
    draft_input.nxt_kv_lens_cpu = (prefix_lens + 2 * BLOCK).to(torch.int32)
    drafts = torch.full((len(reqs), BLOCK), VOCAB - 1, device=DEVICE)
    selector_sample = lookup.propose(
        batch=batch,
        draft_input=draft_input,
        draft_tokens=drafts,
        selector_sample=selector_sample,
        req_generation=pool.req_generation,
    )
    return batch, draft_input, drafts, selector_sample


def _continuations(histories, length):
    return [
        [pattern[(len(h) + j) % len(pattern)] for j in range(length)]
        for h, pattern in zip(histories, PATTERNS)
    ]


class TestDFlashPromptLookupDevice(CustomTestCase):
    def test_decode_steps_propose_from_the_device_table(self):
        """Rows filled through pinned staging and kept current on the device propose
        each step's continuation, while the hit ring's events pace the lookups."""
        pool, reqs, histories, lookup, sampling_info = _setup(top_k=1)
        for _ in range(6):
            batch, _, drafts, _ = _propose(pool, reqs, histories, lookup, sampling_info)
            self.assertEqual(
                drafts[:, 1:].tolist(), _continuations(histories, BLOCK - 1)
            )
            # Every drafted token commits, then the target's bonus.
            committed = _continuations(histories, BLOCK)
            lookup.record(
                prefix_lens=batch.seq_lens,
                out_tokens=torch.tensor(committed, device=DEVICE),
            )
            histories = [h + c for h, c in zip(histories, committed)]

    def test_point_mass_blocks_through_both_sampling_accepts(self):
        """The target takes a looked-up token where it puts all its mass on it and
        replaces it where it puts none, through the selector accept and through the
        drafter-only accept alike."""
        pool, reqs, histories, lookup, sampling_info = _setup(top_k=20)
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        gamma, top_k = BLOCK - 1, 4
        selector_sample = (
            torch.randint(
                0, VOCAB, (2, gamma, top_k), device=DEVICE, generator=generator
            ),
            torch.rand(2, gamma, top_k, device=DEVICE, generator=generator),
        )
        batch, draft_input, drafts, selector_sample = _propose(
            pool, reqs, histories, lookup, sampling_info, selector_sample
        )
        self.assertEqual(drafts[:, 1:].tolist(), _continuations(histories, gamma))

        # Request 0's target follows its whole block; request 1's replaces its
        # third looked-up token.
        target_tokens = torch.cat(
            (drafts[:, 1:], torch.full((2, 1), VOCAB - 1, device=DEVICE)), dim=1
        )
        target_tokens[1, 2] = VOCAB - 2
        target_logits = torch.zeros(2 * BLOCK, VOCAB, device=DEVICE)
        target_logits.scatter_(1, target_tokens.reshape(-1, 1), 100.0)

        worker = types.SimpleNamespace(
            _selector_sample=selector_sample,
            _draft_probs_buf=None,
            _block_verification=False,
            _tp_sync=types.SimpleNamespace(sync=lambda site, tensor: tensor),
            block_size=BLOCK,
        )
        worker._selector_sampling_accept = types.MethodType(
            worker_mod.DFlashWorkerV2._selector_sampling_accept, worker
        )
        accept_len, _, bonus, _, _, _ = worker_mod.DFlashWorkerV2._accept_block(
            worker,
            candidates=drafts,
            next_token_logits=target_logits,
            sampling_info=sampling_info,
            draft_input=draft_input,
            prefix_lens=batch.seq_lens,
            bs=2,
        )
        self.assertEqual(accept_len.tolist(), [gamma, 2])
        self.assertEqual(bonus.tolist(), [VOCAB - 1, VOCAB - 2])

        accept_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
            candidates=drafts,
            next_token_logits=target_logits,
            sampling_info=sampling_info,
            threshold_single=1.0,
            threshold_acc=1.0,
        )
        self.assertEqual(accept_len.tolist(), [gamma, 2])
        self.assertEqual(bonus.tolist(), [VOCAB - 1, VOCAB - 2])


if __name__ == "__main__":
    unittest.main()
