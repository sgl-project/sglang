"""SGLANG_DSPARK_DEFER_CONTEXT_KV: the staged draft context-KV stores replay exactly the
stores the static verify graph would have run (same projected KV, same locations from
the same commit lengths), once, and run before any draft forward that would not run
them itself (eager, a graph without the fork, or a graph whose rows miss staged ones);
a static epilogue whose graph forked no projection keeps the inline commit."""

import os
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
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

MAX_BS, STRIDE, KV_WIDTH, VOCAB, HIDDEN = 4, 3, 8, 32, 6


class _Injector:
    swa_ring_size = None

    def __init__(self):
        self.stores = []
        self.inline = []
        pool = SimpleNamespace(set_swa_key_buffer_radix_fused_norm_rope=None)
        self.draft_model_runner = SimpleNamespace(token_to_kv_pool=pool)
        self.draft_model = SimpleNamespace(store_commit_kv=self._store)

    def committed_swa_loc(self, *, cache_loc, cache_loc_2d, commit_lens, **_):
        col = torch.arange(cache_loc_2d.shape[1]).view(1, -1)
        committed = (col < commit_lens.long().view(-1, 1)).reshape(-1)
        return torch.where(committed, cache_loc, torch.full_like(cache_loc, -1))

    def _store(self, *, kvs, swa_loc, positions, pool):
        self.stores.append(
            dict(
                kvs=[k.clone() for k in kvs],
                swa_loc=swa_loc.clone(),
                positions=positions.clone(),
            )
        )

    def inject_target_hidden(self, **kw):
        self.inline.append(kw)


def _commit(injector):
    commit = DeferredContextCommit(
        max_bs=MAX_BS, stride=STRIDE, device="cpu", kv_injector=injector
    )
    # What the forked projection leaves: per-stage KV of the verify rows.
    commit.kvs = [torch.randn(MAX_BS * STRIDE, KV_WIDTH) for _ in range(3)]
    return commit


def _stage(commit, bs):
    n = bs * STRIDE
    lens = torch.tensor([2, 0, 3][:bs], dtype=torch.int32)
    commit.stage(
        cache_loc=torch.arange(100, 100 + n),
        positions=torch.arange(n) + 7,
        commit_lens=lens,
        state_slot=None,
    )
    return lens


class TestDeferredCommit(CustomTestCase):
    def test_write_runs_the_inline_stores_once(self):
        injector = _Injector()
        commit = _commit(injector)
        lens = _stage(commit, 3)
        commit.note_static_verify(3)
        commit.flush()
        (store,) = injector.stores
        n = MAX_BS * STRIDE
        for got, kv in zip(store["kvs"], commit.kvs):
            self.assertTrue(torch.equal(got, kv[:n]))
        loc = commit.cache_loc.clone()
        expect = injector.committed_swa_loc(
            cache_loc=loc,
            cache_loc_2d=loc.view(MAX_BS, STRIDE),
            commit_lens=torch.cat([lens, torch.zeros(1, dtype=torch.int32)]),
        )
        self.assertTrue(torch.equal(store["swa_loc"], expect))
        # Requests past the staged ones store nothing.
        self.assertTrue(bool((store["swa_loc"][9:] == -1).all()))
        self.assertFalse(commit.pending)
        self.assertTrue(bool((commit.commit_lens == 0).all()))
        commit.flush()
        self.assertEqual(len(injector.stores), 1)

    def test_flush_before_every_draft_forward_that_would_miss_it(self):
        for graph_bs, forked, staged, flushes in (
            (None, set(), 2, True),  # eager draft
            (4, set(), 2, True),  # graph captured without the fork
            (2, {2}, 3, True),  # graph rows miss staged requests
            (4, {4}, 3, False),  # the graph runs the stores
        ):
            with self.subTest(graph_bs=graph_bs, forked=forked, staged=staged):
                injector = _Injector()
                commit = _commit(injector)
                commit._forked_bs = set(forked)
                _stage(commit, 3)
                commit.note_static_verify(staged)
                commit.before_draft_forward(graph_bs=graph_bs)
                self.assertEqual(len(injector.stores), int(flushes))
                commit.after_draft_forward(graph_ran=graph_bs is not None)
                self.assertFalse(commit.pending)


class TestEpilogueStagesOnlyAfterItsProjection(CustomTestCase):
    def _run(self, projected: bool):
        injector = _Injector()
        commit = _commit(injector)
        commit.join_projection = lambda: projected
        epilogue = DsparkVerifyEpilogue(
            max_bs=MAX_BS,
            verify_num_draft_tokens=STRIDE,
            device="cpu",
            tp_sync=SimpleNamespace(sync=lambda site, values: values),
            commit_ctx=CommitInjectCtx(
                draft_model=None,
                block_pos_offsets=torch.arange(STRIDE),
                resolve_pool=lambda: injector.draft_model_runner.token_to_kv_pool,
                resolve_req_to_token=lambda: None,
                kv_injector=injector,
            ),
            deferred_commit=commit,
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
        return injector, commit, epilogue, bs

    def test_projected_graph_stages_the_stores(self):
        injector, commit, epilogue, bs = self._run(projected=True)
        self.assertEqual(injector.inline, [])
        self.assertTrue(
            torch.equal(
                commit.commit_lens[:bs],
                epilogue.read_accept(bs).commit_lens.to(torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(commit.cache_loc[: bs * STRIDE], torch.arange(6) + 50)
        )

    def test_graph_without_projection_keeps_the_inline_commit(self):
        injector, commit, _, _ = self._run(projected=False)
        self.assertEqual(len(injector.inline), 1)
        self.assertTrue(bool((commit.commit_lens == 0).all()))


_TRITON_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("cpu" if os.environ.get("TRITON_INTERPRET") == "1" else None)
)


@unittest.skipIf(_TRITON_DEVICE is None, "needs CUDA or TRITON_INTERPRET=1")
class TestOneLaunchStage(CustomTestCase):
    def test_one_launch_writes_what_the_copies_write(self):
        """The staging kernel leaves every staged buffer equal to the copies and the
        zero tail, with and without state slots, at bs 1 and a partial bs."""
        from sglang.kernels.ops.speculative.dspark.stage_commit import stage_commit

        dev, max_bs, stride = _TRITON_DEVICE, 4, 6
        for bs, with_slot in ((1, False), (3, True)):
            with self.subTest(bs=bs, with_slot=with_slot):
                n = bs * stride
                src = dict(
                    cache_loc=torch.arange(n, device=dev) + 100,
                    positions=torch.arange(n, device=dev) + 7,
                    commit_lens=torch.arange(1, bs + 1, dtype=torch.int32, device=dev),
                    state_slot=torch.arange(n, device=dev) + 3 if with_slot else None,
                )
                rows = torch.full((max_bs * stride,), -1, dtype=torch.int64, device=dev)
                got = dict(
                    cache_loc=rows.clone(),
                    positions=rows.clone(),
                    commit_lens=torch.full((max_bs,), 9, dtype=torch.int32, device=dev),
                    state_slot=rows.clone() if with_slot else None,
                )
                stage_commit(
                    cache_loc_dst=got["cache_loc"],
                    positions_dst=got["positions"],
                    commit_lens_dst=got["commit_lens"],
                    state_slot_dst=got["state_slot"],
                    rows=n,
                    **src,
                )
                for name, value in src.items():
                    if value is None:
                        continue
                    want = got[name].new_full(got[name].shape, -1)
                    if name == "commit_lens":
                        want.zero_()
                    want[: value.numel()] = value
                    self.assertTrue(torch.equal(got[name], want), name)


if __name__ == "__main__":
    unittest.main()
