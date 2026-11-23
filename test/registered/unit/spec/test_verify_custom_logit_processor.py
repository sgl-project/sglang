"""Speculative verify runs a request's custom logit processor on each row's own
history: the committed output followed by the row's draft prefix. A processor
that does not declare it runs on the first row only, and its request commits
one token per round."""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.sampling.custom_logit_processor import (
    ROW_PREFIX_KEY,
    CustomLogitProcessor,
)
from sglang.srt.sampling.sampling_batch_info import ProcessorEntry, SamplingBatchInfo
from sglang.srt.sampling.sampling_pipeline import VerifyRoute, VerifyStep
from sglang.srt.speculative import eagle_utils
from sglang.srt.speculative.eagle_worker_common import accept_draft_tree
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    needs_verify_tree,
    verify_row_prefixes,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

VOCAB = 32
ROOT = 1


class _CountUp(CustomLogitProcessor):
    """Forces the token after the last one the row follows, so every verify row
    of a draft path 2, 3 agrees only if it sees its own draft prefix."""

    supports_row_prefix = True

    def __call__(self, logits, custom_param_list):
        for i, params in enumerate(custom_param_list):
            req = params["__req__"]
            last = [*req.output_ids, *params.get(ROW_PREFIX_KEY, ())][-1]
            logits[i, :] = -float("inf")
            logits[i, last + 1] = 0.0
        return logits


class _CountUpWithoutRowPrefix(_CountUp):
    supports_row_prefix = False


def _chain():
    # root -> 2 -> 3; the target logits favor 2, 3, then the bonus 4
    return dict(
        tokens=[ROOT, 2, 3],
        next_token=[1, 2, -1],
        next_sibling=[-1, -1, -1],
        argmax=[2, 3, 4],
    )


def _tree():
    # root -> {2 -> 3, 5}; the target logits favor 2, 3, then the bonus 4
    return dict(
        tokens=[ROOT, 2, 5, 3],
        next_token=[1, 3, -1, -1],
        next_sibling=[-1, 2, -1, -1],
        argmax=[2, 3, 9, 4],
    )


def _host_tree(layout, bs):
    return GrammarTree.from_host(
        torch.tensor([layout["next_token"]] * bs),
        torch.tensor([layout["next_sibling"]] * bs),
        torch.tensor([layout["tokens"]] * bs),
    )


def _sampling_info(processor, *, greedy, device):
    # Request 0 runs the processor after its committed output; request 1 has none.
    req = SimpleNamespace(output_ids=[5, ROOT])
    return SamplingBatchInfo(
        temperatures=torch.ones((2, 1), device=device),
        top_ps=torch.ones(2, device=device),
        top_ks=torch.full((2,), VOCAB, dtype=torch.int32, device=device),
        min_ps=torch.zeros(2, device=device),
        is_all_greedy=greedy,
        is_any_greedy=greedy,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=VOCAB,
        has_custom_logit_processor=True,
        custom_params=[{"__req__": req}, None],
        custom_logit_processor={
            0: ProcessorEntry(
                processor=processor,
                rows=[0],
                indices=torch.tensor([0], device=device),
            )
        },
        device=device,
    )


def _batch(processor, *, greedy, device):
    return SimpleNamespace(
        device=device,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([7, 11], device=device),
        sampling_info=_sampling_info(processor, greedy=greedy, device=device),
        req_pool_indices=torch.tensor([3, 5], device=device),
        return_logprob=False,
        has_grammar=False,
    )


def _bound_verify(batch, layout):
    verify = VerifyStep(
        batch=batch,
        route=VerifyRoute(adjusts_logits=True, need_truncated_sampling=False),
        watermark=None,
        watermark_full_mask=True,
    )
    verify.bind_tree(_host_tree(layout, 2), logits_output=None, barrier=None)
    return verify


class TestRowPrefixes(CustomTestCase):
    def test_chain_row_j_gets_drafts_one_to_j(self):
        """DFlash and DSpark verify a chain whose column 0 is the committed token."""
        chain = torch.tensor([[ROOT, 2, 3, 4], [ROOT, 7, 8, 9]])
        self.assertEqual(
            verify_row_prefixes(*GrammarTree.from_linear_chain(chain).resolve()),
            [[], [2], [2, 3], [2, 3, 4], [], [7], [7, 8], [7, 8, 9]],
        )

    def test_a_custom_processor_alone_needs_the_host_tree(self):
        info = _sampling_info(_CountUp(), greedy=True, device="cpu")
        plain = SimpleNamespace(has_custom_logit_processor=False)
        for has_grammar, sampling_info, want in [
            (False, info, True),
            (False, plain, False),
            (True, plain, True),
            (False, None, False),
        ]:
            batch = SimpleNamespace(
                has_grammar=has_grammar, sampling_info=sampling_info
            )
            self.assertEqual(needs_verify_tree(batch), want)


class TestOneTokenCap(CustomTestCase):
    def _verify(self, first_row_only):
        verify = VerifyStep(
            batch=None,
            route=VerifyRoute(adjusts_logits=True, need_truncated_sampling=False),
            watermark=None,
            watermark_full_mask=True,
        )
        verify.first_row_only_reqs = torch.tensor(first_row_only)
        return verify

    def test_chain_accept_keeps_the_first_token(self):
        """Request 0 accepted draft 1 (commits it), request 1 accepted none
        (commits its first-row bonus), request 2 is not capped."""
        out_tokens = torch.tensor([[21, 22, 23], [31, 0, 0], [41, 42, 43]])
        commit_lens, bonus, new_seq_lens = self._verify([0, 1]).cap_chain_accept(
            out_tokens=out_tokens,
            commit_lens=torch.tensor([3, 1, 3], dtype=torch.int32),
            bonus=torch.tensor([23, 31, 43]),
            prefix_lens=torch.tensor([10, 20, 30]),
            new_seq_lens=None,
        )
        self.assertEqual(commit_lens.tolist(), [1, 1, 3])
        self.assertEqual(commit_lens.dtype, torch.int32)
        self.assertEqual(bonus.tolist(), [21, 31, 43])
        self.assertEqual(new_seq_lens.tolist(), [11, 21, 33])

    def test_nothing_capped_returns_the_inputs(self):
        verify = self._verify([])
        verify.first_row_only_reqs = None
        commit_lens, bonus = torch.tensor([3]), torch.tensor([7])
        out = verify.cap_chain_accept(
            out_tokens=torch.tensor([[5, 6, 7]]),
            commit_lens=commit_lens,
            bonus=bonus,
            prefix_lens=torch.tensor([10]),
            new_seq_lens=None,
        )
        self.assertIs(out[0], commit_lens)
        self.assertIs(out[1], bonus)
        self.assertIsNone(out[2])


class TestCustomLogitProcessorSync(CustomTestCase):
    """Under overlap the last result is processed after the next launch, so a
    verify of the next batch would read req.output_ids a round late."""

    def _batch(self, *, has_custom, algorithm, has_grammar=False):
        return SimpleNamespace(
            sampling_info=SimpleNamespace(has_custom_logit_processor=has_custom),
            spec_algorithm=algorithm,
            has_grammar=has_grammar,
        )

    def test_spec_batches_with_a_processor_sync(self):
        for name, has_custom in itertools.product(
            ("NONE", "EAGLE", "NGRAM", "DFLASH", "DSPARK"), (False, True)
        ):
            algorithm = SpeculativeAlgorithm.from_string(name)
            batch = self._batch(has_custom=has_custom, algorithm=algorithm)
            with self.subTest(algorithm=name, has_custom=has_custom):
                self.assertEqual(
                    ScheduleBatch.custom_logit_processor_needs_sync(batch),
                    has_custom and not algorithm.is_none(),
                )

    def test_the_scheduler_processes_the_last_result_first(self):
        batch = self._batch(
            has_custom=True, algorithm=SpeculativeAlgorithm.from_string("EAGLE")
        )
        batch.grammar_needs_sync = lambda: False
        batch.custom_logit_processor_needs_sync = lambda: (
            ScheduleBatch.custom_logit_processor_needs_sync(batch)
        )
        with envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(False):
            self.assertTrue(
                Scheduler.should_process_last_batch_early(SimpleNamespace(), batch)
            )


def _verify_input(layout, *, kind, bs):
    n = len(layout["tokens"])
    tensors = dict(
        draft_token=torch.tensor(layout["tokens"] * bs, device="cuda"),
        retrieve_index=torch.arange(bs * n, device="cuda").view(bs, n),
        retrieve_next_token=torch.tensor([layout["next_token"]] * bs, device="cuda"),
        retrieve_next_sibling=torch.tensor(
            [layout["next_sibling"]] * bs, device="cuda"
        ),
        custom_mask=torch.ones(bs * n * n, dtype=torch.bool, device="cuda"),
        positions=None,
    )
    if kind == "ngram":
        return NgramVerifyInput(draft_token_num=n, **tensors)
    return SimpleNamespace(
        draft_token_num=n,
        max_tree_depth=3,
        tree_topk=1 if kind == "eagle-chain" else 2,
        draft_probs=None,
        **tensors,
    )


def _logits(layout, bs, *, peak):
    n = len(layout["tokens"])
    logits = torch.zeros((bs * n, VOCAB), device="cuda")
    for b in range(bs):
        for i, token in enumerate(layout["argmax"]):
            logits[b * n + i, token] = peak
    return logits


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestTreeVerifyCustomLogitProcessor(CustomTestCase):
    def test_processor_sees_each_row_prefix_or_commits_one_token(self):
        group = SimpleNamespace(world_size=1)
        parallel = SimpleNamespace(tp_group=group, attn_tp_group=group)
        spec = SimpleNamespace(
            speculative_use_rejection_sampling=False,
            speculative_use_block_verification=False,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        )
        cases = [
            ("eagle-chain", _chain(), False),
            # The NV mega runtime's post-verify carries no custom processors, so
            # such a batch stays on eagle_sample.
            ("eagle-chain", _chain(), True),
            ("eagle-tree", _tree(), False),
            ("ngram", _tree(), False),
        ]
        # A row-prefix processor accepts the whole path; without the declaration,
        # its request keeps the first token only.
        processors = [(_CountUp(), [2, 3, 4]), (_CountUpWithoutRowPrefix(), [2])]
        for (kind, layout, mega), ((processor, accept), greedy) in itertools.product(
            cases, itertools.product(processors, (True, False))
        ):
            with (
                self.subTest(
                    kind=kind,
                    mega=mega,
                    greedy=greedy,
                    processor=type(processor).__name__,
                ),
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(mega),
                patch.object(eagle_utils, "get_spec", return_value=spec),
                patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            ):
                batch = _batch(processor, greedy=greedy, device="cuda")
                logits_output = SimpleNamespace(
                    # peaked so that temperature sampling is deterministic
                    next_token_logits=_logits(layout, 2, peak=50.0),
                    next_token_logits_vocab_shard=None,
                )
                result = accept_draft_tree(
                    batch,
                    verify=_bound_verify(batch, layout),
                    verify_input=_verify_input(layout, kind=kind, bs=2),
                    logits_output=logits_output,
                    speculative_runtime_hooks=None,
                )
                tokens = result.accept_tokens()
                self.assertEqual(
                    [
                        tokens[b, : result.accept_lens[b]].tolist()
                        for b in range(len(result.accept_lens))
                    ],
                    [accept, [2, 3, 4]],
                )


if __name__ == "__main__":
    unittest.main()
