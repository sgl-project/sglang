"""Speculative verify runs a request's custom logit processor on each row's own
history: the committed output followed by the row's draft prefix. A processor
that does not declare it runs on the first row only, and its request commits
one token per round. Under overlap, the committed tokens result processing has
not appended to req.output_ids yet lead every row prefix."""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
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
    HostVerifyTree,
    needs_verify_tree,
    verify_row_prefixes,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

VOCAB = 32
ROOT = 1


class _Req:
    """A request stub; hashable like Req, which the barrier's tokens key on."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


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
    return HostVerifyTree.from_host(
        torch.tensor([layout["next_token"]] * bs),
        torch.tensor([layout["next_sibling"]] * bs),
        torch.tensor([layout["tokens"]] * bs),
    )


def _sampling_info(processor, *, greedy, device):
    # Request 0 runs the processor after its committed output; request 1 has none.
    req = _Req(output_ids=[5, ROOT])
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
    info = _sampling_info(processor, greedy=greedy, device=device)
    return SimpleNamespace(
        reqs=[info.custom_params[0]["__req__"], _Req()],
        device=device,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([7, 11], device=device),
        sampling_info=info,
        req_pool_indices=torch.tensor([3, 5], device=device),
        return_logprob=False,
        has_grammar=False,
    )


def _bound_verify(batch, layout, barrier=None):
    verify = VerifyStep(
        batch=batch,
        route=VerifyRoute(adjusts_logits=True, need_truncated_sampling=False),
        watermark=None,
        watermark_full_mask=True,
    )
    verify.build_row_inputs(None, tree=_host_tree(layout, 2), barrier=barrier)
    return verify


class TestRowPrefixes(CustomTestCase):
    def test_chain_row_j_gets_drafts_one_to_j(self):
        """DFlash and DSpark verify a chain whose column 0 is the committed token."""
        chain = torch.tensor([[ROOT, 2, 3, 4], [ROOT, 7, 8, 9]])
        self.assertEqual(
            verify_row_prefixes(
                *HostVerifyTree.from_linear_chain(chain).resolve(),
                pending_tokens=[[], [6, ROOT]],
            ),
            [
                [],
                [2],
                [2, 3],
                [2, 3, 4],
                [6, 1],
                [6, 1, 7],
                [6, 1, 7, 8],
                [6, 1, 7, 8, 9],
            ],
        )

    def test_tree_rows_get_the_pending_tokens_then_their_path(self):
        tree = _host_tree(_tree(), 2).resolve()
        self.assertEqual(
            verify_row_prefixes(*tree, pending_tokens=[[ROOT], []]),
            [[1], [1, 2], [1, 5], [1, 2, 3], [], [2], [5], [2, 3]],
        )

    def test_the_barrier_supplies_tokens_output_ids_lacks(self):
        """Under overlap req.output_ids misses the previous step's tokens; the
        barrier's tokens for them lead the row prefixes, so the processor reads
        the same history as after result processing."""
        batch = _batch(_CountUp(), greedy=True, device="cpu")
        req = batch.reqs[0]
        req.output_ids = [5]
        calls = []

        def barrier():
            calls.append(None)
            return {req: [ROOT]}

        verify = _bound_verify(batch, _chain(), barrier=barrier)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            verify.row_prefixes, [[ROOT], [ROOT, 2], [ROOT, 2, 3], [], [2], [2, 3]]
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
    verify of the next batch would read req.output_ids a round late, unless
    the worker's barrier hands it the missing tokens in the row prefixes."""

    def _batch(self, processor, *, algorithm, has_grammar=False):
        entries = {} if processor is None else {0: SimpleNamespace(processor=processor)}
        return SimpleNamespace(
            sampling_info=SimpleNamespace(
                has_custom_logit_processor=processor is not None,
                custom_logit_processor=entries,
            ),
            spec_algorithm=algorithm,
            has_grammar=has_grammar,
        )

    def test_only_a_processor_the_barrier_cannot_serve_syncs(self):
        # NGRAM verifies without the barrier; every other spec algorithm here
        # has it, and a processor without row prefixes does not read it.
        for name, processor in itertools.product(
            ("NONE", "EAGLE", "NGRAM", "DFLASH", "DSPARK"),
            (None, _CountUp(), _CountUpWithoutRowPrefix()),
        ):
            algorithm = SpeculativeAlgorithm.from_string(name)
            batch = self._batch(processor, algorithm=algorithm)
            want = (
                processor is not None
                and not algorithm.is_none()
                and (name == "NGRAM" or not processor.supports_row_prefix)
            )
            with self.subTest(algorithm=name, processor=type(processor).__name__):
                self.assertEqual(
                    ScheduleBatch.custom_logit_processor_needs_sync(batch), want
                )

    def test_the_scheduler_processes_the_last_result_first(self):
        batch = self._batch(
            _CountUpWithoutRowPrefix(),
            algorithm=SpeculativeAlgorithm.from_string("EAGLE"),
        )
        batch.grammar_needs_sync = lambda: False
        batch.custom_logit_processor_needs_sync = lambda: (
            ScheduleBatch.custom_logit_processor_needs_sync(batch)
        )
        with envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.override(False):
            self.assertTrue(
                Scheduler.should_process_last_batch_early(SimpleNamespace(), batch)
            )


class TestPendingTokens(CustomTestCase):
    """The barrier lists, per custom-processor request, the tokens processing
    the queued result will append to req.output_ids."""

    def _req(self, *, custom=True, grammar=None):
        return _Req(custom_logit_processor="proc" if custom else None, grammar=grammar)

    def _pending(self, result, batch, left=()):
        processor = SimpleNamespace(still_in_batch=lambda req: req not in left)
        return SchedulerBatchResultProcessor.pending_output_tokens(
            processor, result, batch
        )

    def _batch(self, reqs, *, decode, chunked_req=None):
        return SimpleNamespace(
            reqs=reqs,
            chunked_req=chunked_req,
            forward_mode=SimpleNamespace(
                is_decode=lambda: decode, is_extend=lambda: not decode
            ),
        )

    def test_decode_lists_each_accept_run(self):
        plain, with_grammar, no_custom, left = (
            self._req(),
            self._req(grammar="fsm"),
            self._req(custom=False),
            self._req(),
        )
        result = SimpleNamespace(
            copy_done=None,
            next_token_ids=torch.tensor([1, 2, 0, 3, 4, 5, 6, 0, 0, 7, 8, 0]),
            accept_lens=torch.tensor([2, 3, 1, 2]),
            speculative_output_stride=3,
            # advance_grammar_fsm cut the grammar request's run at termination
            grammar_retained_tokens=[None, [3], None, None],
        )
        batch = self._batch([plain, with_grammar, no_custom, left], decode=True)
        self.assertEqual(
            self._pending(result, batch, left=(left,)),
            {plain: [1, 2], with_grammar: [3]},
        )

    def test_extend_lists_the_prefill_token_except_a_middle_chunk(self):
        done, chunked = self._req(), self._req()
        result = SimpleNamespace(copy_done=None, next_token_ids=torch.tensor([9, 4]))
        batch = self._batch([done, chunked], decode=False, chunked_req=chunked)
        self.assertEqual(self._pending(result, batch), {done: [9]})

    def test_a_batch_without_processors_reads_nothing(self):
        batch = self._batch([self._req(custom=False)], decode=True)
        self.assertEqual(self._pending(SimpleNamespace(), batch), {})

    def test_the_barrier_advances_the_grammar_then_merges_queued_results(self):
        req = self._req()
        calls = []
        processor = SimpleNamespace(
            advance_grammar_fsm=lambda result, batch: calls.append(("fsm", result)),
            pending_output_tokens=lambda result, batch: (
                calls.append(("pending", result)) or {req: [result]}
            ),
        )
        scheduler = SimpleNamespace(
            result_queue=[("b1", 1), ("b2", 2)], batch_result_processor=processor
        )
        self.assertEqual(Scheduler._resolve_pending_results(scheduler), {req: [1, 2]})
        self.assertEqual(
            calls, [("fsm", 1), ("pending", 1), ("fsm", 2), ("pending", 2)]
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
                    _bound_verify(batch, layout),
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
