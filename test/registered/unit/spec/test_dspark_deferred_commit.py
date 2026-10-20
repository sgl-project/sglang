"""SGLANG_DSPARK_DEFER_CONTEXT_KV: the staged draft context-KV commit replays exactly the
injection the static verify graph would have run, once, and is written before any
draft forward that would not write it itself (eager, a graph without the fork, or a
graph whose rows miss staged ones)."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspark_components.dspark_deferred_commit import (
    DeferredContextCommit,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    CommitInjectCtx,
    DsparkVerifyEpilogue,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MAX_BS, STRIDE, HIDDEN, VOCAB = 4, 3, 8, 32


class _Injector:
    swa_ring_size = None

    def __init__(self):
        self.calls = []

    def inject_target_hidden(self, **kw):
        self.calls.append(
            {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()}
        )


def _commit(injector):
    return DeferredContextCommit(
        max_bs=MAX_BS,
        stride=STRIDE,
        hidden_width=HIDDEN,
        hidden_dtype=torch.float32,
        device="cpu",
        kv_injector=injector,
        with_state_slot=False,
    )


def _staged(bs):
    n = bs * STRIDE
    return dict(
        target_hidden=torch.randn(n, HIDDEN),
        cache_loc=torch.arange(100, 100 + n),
        positions=torch.arange(n) + 7,
        commit_lens=torch.tensor([2, 0, 3][:bs], dtype=torch.int32),
        state_slot=None,
    )


class TestDeferredCommit(CustomTestCase):
    def test_write_replays_the_inline_injection_once(self):
        injector = _Injector()
        commit = _commit(injector)
        x = _staged(3)
        commit.stage(**x)
        commit.note_static_verify(3)
        commit.flush()
        (call,) = injector.calls
        self.assertTrue(torch.equal(call["target_hidden"][:9], x["target_hidden"]))
        self.assertTrue(torch.equal(call["cache_loc"][:9], x["cache_loc"]))
        self.assertTrue(torch.equal(call["positions"][:9], x["positions"]))
        self.assertTrue(torch.equal(call["commit_lens"][:3], x["commit_lens"]))
        # Requests past the staged ones write nothing.
        self.assertTrue(bool((call["commit_lens"][3:] == 0).all()))
        self.assertEqual(tuple(call["cache_loc_2d"].shape), (MAX_BS, STRIDE))
        # Disarmed and no longer pending: a second flush writes nothing.
        self.assertFalse(commit.pending)
        self.assertTrue(bool((commit.commit_lens == 0).all()))
        commit.flush()
        self.assertEqual(len(injector.calls), 1)

    def test_flush_before_every_draft_forward_that_would_miss_it(self):
        for graph_bs, forked, staged, flushes in (
            (None, set(), 2, True),  # eager draft
            (4, set(), 2, True),  # graph captured without the fork
            (2, {2}, 3, True),  # graph rows miss staged requests
            (4, {4}, 3, False),  # the graph writes it
        ):
            with self.subTest(graph_bs=graph_bs, forked=forked, staged=staged):
                injector = _Injector()
                commit = _commit(injector)
                commit._forked_bs = set(forked)
                commit.stage(**_staged(3))
                commit.note_static_verify(staged)
                commit.before_draft_forward(graph_bs=graph_bs)
                self.assertEqual(len(injector.calls), int(flushes))
                commit.after_draft_forward(graph_ran=graph_bs is not None)
                self.assertFalse(commit.pending)


class TestEpilogueStagesInsteadOfInjecting(CustomTestCase):
    def test_static_epilogue_stages(self):
        injector = _Injector()
        pool = SimpleNamespace(set_swa_key_buffer_radix_fused_norm_rope=None)
        epilogue = DsparkVerifyEpilogue(
            max_bs=MAX_BS,
            verify_num_draft_tokens=STRIDE,
            device="cpu",
            tp_sync=SimpleNamespace(sync=lambda site, values: values),
            commit_ctx=CommitInjectCtx(
                draft_model=None,
                block_pos_offsets=torch.arange(STRIDE),
                resolve_pool=lambda: pool,
                resolve_req_to_token=lambda: None,
                kv_injector=injector,
            ),
            defer_context_kv=True,
        )
        bs = 2
        epilogue.begin_static_step(bs, armed=True)
        torch.manual_seed(0)
        out = SimpleNamespace(
            next_token_logits=torch.randn(bs * STRIDE, VOCAB),
            hidden_states=torch.randn(bs * STRIDE, HIDDEN),
        )
        fb = SimpleNamespace(
            batch_size=bs,
            input_ids=torch.randint(0, VOCAB, (bs * STRIDE,)),
            seq_lens=torch.tensor([5, 9]),
            out_cache_loc=torch.arange(bs * STRIDE) + 50,
            positions=torch.arange(bs * STRIDE),
            req_pool_indices=torch.tensor([1, 2]),
        )
        epilogue._static_epilogue(out, fb)
        self.assertEqual(injector.calls, [])
        commit = epilogue.deferred_commit
        self.assertIsNotNone(commit)
        self.assertTrue(torch.equal(commit.hidden[: bs * STRIDE], out.hidden_states))
        self.assertTrue(
            torch.equal(
                commit.commit_lens[:bs],
                epilogue.read_accept(bs).commit_lens.to(torch.int32),
            )
        )
        epilogue.note_static_verify(bs)
        commit.flush()
        self.assertEqual(len(injector.calls), 1)


if __name__ == "__main__":
    unittest.main()
