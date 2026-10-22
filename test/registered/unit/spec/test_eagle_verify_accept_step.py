"""A speculative algorithm can supply the accept step of an EAGLE verify - CPU only.

``run_eagle_verify(..., accept_step=...)`` runs the target verify forward as
usual, then hands a ``VerifyAcceptInput`` (the forward's outputs and the
batch's lengths and slots) to the accept step instead of running sglang's
sampling and acceptance, commits the linear-attention states of the path the
step accepted and returns the step's tokens in the ``GenerationBatchResult``.
Without one, sglang's accept code runs.

The verify preparation, the target forward and the state commit are stubbed;
only the hand-off around the forward is under test.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative import eagle_worker_common as common
from sglang.srt.speculative.eagle_worker_common import (
    VerifyAcceptInput,
    VerifyAcceptResult,
    run_eagle_verify,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _DefaultAcceptRan(Exception):
    pass


def _verify(accept_step=None, kv_cache=None, **batch_overrides):
    """run_eagle_verify on a stub batch; returns (result, batch, logits output, commits)."""
    forward_batch = SimpleNamespace(name="verify forward batch")
    logits_output = SimpleNamespace(
        next_token_logits=torch.zeros(16, 4),
        next_token_ids=None,
        hidden_states=torch.zeros(16, 8),
    )
    target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            watermark_state=None, attn_backend=SimpleNamespace(verify_mask=None)
        ),
        forward_batch_generation=mock.Mock(
            return_value=SimpleNamespace(
                logits_output=logits_output,
                routed_experts_output="routed",
                indexer_topk_output="indexer",
            )
        ),
    )
    batch = SimpleNamespace(
        spec_info=SimpleNamespace(name="verify input"),
        seq_lens=torch.tensor([7]),
        req_pool_indices=torch.tensor([2]),
        input_ids=None,
        out_cache_loc=torch.arange(16),
        has_grammar=False,
        return_logprob=False,
        hisparse_coordinator=None,
        sampling_info=SimpleNamespace(
            is_all_greedy=True,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            has_watermark_candidates=False,
        ),
    )
    for name, value in batch_overrides.items():
        target = batch.sampling_info if hasattr(batch.sampling_info, name) else batch
        setattr(target, name, value)
    allocator = SimpleNamespace(get_kvcache=lambda: kv_cache or SimpleNamespace())
    commits = []
    with (
        mock.patch.object(common, "record_stream_for_v2_verify"),
        mock.patch.object(common, "record_stream_each"),
        mock.patch.object(
            common, "eagle_prepare_for_verify", return_value=(forward_batch, True)
        ),
        mock.patch.object(common, "eagle_sample", side_effect=_DefaultAcceptRan),
        mock.patch.object(
            common,
            "commit_mamba_states_after_verify",
            side_effect=lambda *args: commits.append(args),
        ),
    ):
        result = run_eagle_verify(
            batch,
            target_worker=target_worker,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=allocator,
            plan_stream=None,
            plan_stream_ctx=contextlib.nullcontext(),
            topk=4,
            num_draft_tokens=16,
            device="cpu",
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
            accept_step=accept_step,
        )
    return result, batch, logits_output, commits


class TestVerifyAcceptStep(CustomTestCase):
    def test_accept_step_replaces_the_default_accept_code(self):
        compacted = torch.ones(16, 8)
        accepted = VerifyAcceptResult(
            next_token_ids=torch.tensor([11, 12, 13]),
            accept_lens=torch.tensor([3], dtype=torch.int32),
            accept_index=torch.tensor([[0, 1, 5]]),
            new_seq_lens=torch.tensor([10]),
            bonus_tokens=torch.tensor([13], dtype=torch.int32),
            hidden_states=compacted,
        )
        inputs = []

        def accept_step(inp):
            inputs.append(inp)
            return accepted

        result, batch, logits_output, commits = _verify(accept_step)
        (inp,) = inputs
        self.assertIsInstance(inp, VerifyAcceptInput)
        self.assertIs(inp.verify_input, batch.spec_info)
        self.assertIs(inp.next_token_logits, logits_output.next_token_logits)
        self.assertIsNone(inp.next_token_ids)
        self.assertIs(inp.seq_lens, batch.seq_lens)
        self.assertIs(inp.req_pool_indices, batch.req_pool_indices)
        self.assertIs(inp.out_cache_loc, batch.out_cache_loc)
        self.assertIs(inp.can_run_cuda_graph, True)
        # sglang commits the states of the step's path and keeps its rows.
        self.assertEqual(len(commits), 1)
        self.assertIs(commits[0][1], batch)
        self.assertIs(commits[0][2], accepted.accept_lens)
        self.assertIs(commits[0][3], accepted.accept_index)
        self.assertIs(logits_output.hidden_states, compacted)
        self.assertIs(result.logits_output, logits_output)
        self.assertIs(result.next_token_ids, accepted.next_token_ids)
        self.assertIs(result.accept_lens, accepted.accept_lens)
        self.assertIs(result.accept_index, accepted.accept_index)
        self.assertIs(result.new_seq_lens, accepted.new_seq_lens)
        self.assertIs(result.next_draft_input.bonus_tokens, accepted.bonus_tokens)
        self.assertTrue(result.can_run_cuda_graph)
        self.assertEqual(result.speculative_num_draft_tokens, 16)
        self.assertEqual(result.routed_experts_output, "routed")
        self.assertEqual(result.indexer_topk_output, "indexer")
        self.assertEqual(len(result.extra_keep_alive_refs), 1)

    def test_accept_step_refuses_batches_it_cannot_serve(self):
        def accept_step(inp):
            raise AssertionError("accept_step ran")

        c128 = SimpleNamespace(clear_unaccepted_c128_draft_states=lambda *a: None)
        for kwargs, word in (
            (dict(has_grammar=True), "grammar"),
            (dict(return_logprob=True), "logprobs"),
            (dict(is_all_greedy=False), "non-greedy"),
            (dict(acc_scaling_penalties=torch.ones(1)), "penalties"),
            (dict(logit_bias=torch.zeros(1, 4)), "logit bias"),
            (dict(has_watermark_candidates=True), "watermark"),
            (dict(hisparse_coordinator=object()), "sparse"),
            (dict(kv_cache=c128), "c128"),
        ):
            with self.subTest(word), self.assertRaisesRegex(ValueError, word):
                _verify(accept_step, **kwargs)
        # The default accept code serves every batch.
        with self.assertRaises(_DefaultAcceptRan):
            _verify(None, return_logprob=True)

    def test_without_accept_step_sglang_accepts(self):
        with self.assertRaises(_DefaultAcceptRan):
            _verify()


if __name__ == "__main__":
    unittest.main()
