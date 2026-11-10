"""SGLANG_DSPARK_DEFER_CONTEXT_KV: the staged draft KV store writes what the static
verify would have, once, and only the rows of the requests the draft carries."""

import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspark_components.dspark_deferred_draft_kv import (
    DeferredDraftKVStore,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    CommitInjectCtx,
    DsparkVerifyEpilogue,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.scheduler_test_utils import FakeReq
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

MAX_BS, STRIDE, KV_WIDTH, VOCAB, HIDDEN = 4, 3, 8, 32, 6

_TRITON_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("cpu" if os.environ.get("TRITON_INTERPRET") == "1" else None)
)


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


def _deferred_draft_kv(injector, device="cpu"):
    deferred_draft_kv = DeferredDraftKVStore(
        max_bs=MAX_BS, stride=STRIDE, device=device, kv_injector=injector
    )
    # What the forked projection leaves: per-stage KV of the verify rows.
    deferred_draft_kv.kvs = [
        torch.randn(MAX_BS * STRIDE, KV_WIDTH, device=device) for _ in range(3)
    ]
    return deferred_draft_kv


def _staged_req():
    return FakeReq(retraction_count=0)


def _stage(deferred_draft_kv, bs):
    n = bs * STRIDE
    lens = torch.tensor([2, 1, 3][:bs], dtype=torch.int32)
    deferred_draft_kv.stage(
        cache_loc=torch.arange(100, 100 + n),
        positions=torch.arange(n) + 7,
        commit_lens=lens,
        state_slot=None,
    )
    return lens


class TestDeferredCommit(CustomTestCase):
    def test_write_runs_the_inline_stores_once(self):
        injector = _Injector()
        deferred_draft_kv = _deferred_draft_kv(injector)
        lens = _stage(deferred_draft_kv, 3)
        reqs = [_staged_req() for _ in range(3)]
        deferred_draft_kv.note_staged(reqs)
        deferred_draft_kv.before_draft_forward(graph_bs=None, reqs=reqs)
        (store,) = injector.stores
        n = MAX_BS * STRIDE
        for got, kv in zip(store["kvs"], deferred_draft_kv.kvs):
            self.assertTrue(torch.equal(got, kv[:n]))
        loc = deferred_draft_kv.cache_loc.clone()
        expect = injector.committed_swa_loc(
            cache_loc=loc,
            cache_loc_2d=loc.view(MAX_BS, STRIDE),
            commit_lens=torch.cat([lens, torch.zeros(1, dtype=torch.int32)]),
        )
        self.assertTrue(torch.equal(store["swa_loc"], expect))
        # Requests past the staged ones store nothing.
        self.assertTrue(bool((store["swa_loc"][9:] == -1).all()))
        self.assertFalse(deferred_draft_kv.pending)
        self.assertTrue(bool((deferred_draft_kv.commit_lens == 0).all()))
        deferred_draft_kv.before_draft_forward(graph_bs=None, reqs=reqs)
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
                deferred_draft_kv = _deferred_draft_kv(injector)
                deferred_draft_kv._forked_bs = set(forked)
                _stage(deferred_draft_kv, 3)
                reqs = [_staged_req() for _ in range(staged)]
                deferred_draft_kv.note_staged(reqs)
                deferred_draft_kv.before_draft_forward(graph_bs=graph_bs, reqs=reqs)
                self.assertEqual(len(injector.stores), int(flushes))
                self.assertFalse(deferred_draft_kv.pending)

    def test_only_rows_of_the_requests_the_draft_carries_are_written(self):
        """Left-out or retracted rows stay unwritten, in any draft batch order."""
        for case in ("left out", "readmitted"):
            for graph_bs in (None, MAX_BS):
                with self.subTest(case=case, graph_bs=graph_bs):
                    a, b, c = (_staged_req() for _ in range(3))
                    injector = _Injector()
                    deferred_draft_kv = _deferred_draft_kv(injector)
                    deferred_draft_kv._forked_bs = {MAX_BS}
                    _stage(deferred_draft_kv, 3)
                    deferred_draft_kv.note_staged([a, b, c])
                    draft_reqs = [c, a]
                    if case == "readmitted":
                        b.retraction_count += 1
                        draft_reqs.append(b)
                    deferred_draft_kv.before_draft_forward(
                        graph_bs=graph_bs, reqs=draft_reqs
                    )
                    if graph_bs is None:
                        (store,) = injector.stores
                        rows = store["swa_loc"][: 3 * STRIDE].view(3, STRIDE)
                        self.assertEqual((rows != -1).sum(dim=1).tolist(), [2, 0, 3])
                    else:
                        # What the draft graph reads when it writes the staging.
                        self.assertEqual(
                            deferred_draft_kv.commit_lens.tolist()[:3], [2, 0, 3]
                        )


class TestEpilogueStagesOnlyAfterItsProjection(CustomTestCase):
    def _run(self, projected: bool, *, bs=2, ring=False, device="cpu"):
        injector = _Injector()
        if ring:
            injector.swa_ring_size = 8
        deferred_draft_kv = _deferred_draft_kv(injector, device)
        deferred_draft_kv.join_projection = lambda: projected
        if ring:
            deferred_draft_kv._one_launch_stage = True
        epilogue = DsparkVerifyEpilogue(
            max_bs=MAX_BS,
            verify_num_draft_tokens=STRIDE,
            device=device,
            tp_sync=SimpleNamespace(sync=lambda site, values: values),
            commit_ctx=CommitInjectCtx(
                draft_model=None,
                block_pos_offsets=torch.arange(STRIDE),
                resolve_pool=lambda: injector.draft_model_runner.token_to_kv_pool,
                resolve_req_to_token=lambda: None,
                kv_injector=injector,
            ),
            deferred_draft_kv=deferred_draft_kv,
        )
        epilogue.begin_static_step(bs, armed=True)
        torch.manual_seed(0)
        out = SimpleNamespace(
            next_token_logits=torch.randn(bs * STRIDE, VOCAB, device=device),
            hidden_states=torch.randn(bs * STRIDE, HIDDEN, device=device),
        )
        # A graph's request slots are the head of a max_bs buffer.
        req_pool_indices = torch.tensor([1, 2, 7, 7], device=device)[:bs]
        fb = SimpleNamespace(
            batch_size=bs,
            input_ids=torch.randint(0, VOCAB, (bs * STRIDE,), device=device),
            seq_lens=torch.tensor([5, 9], device=device)[:bs],
            out_cache_loc=torch.arange(bs * STRIDE, device=device) + 50,
            positions=torch.arange(bs * STRIDE, device=device),
            req_pool_indices=req_pool_indices,
        )
        epilogue._static_epilogue(out, fb)
        return injector, deferred_draft_kv, epilogue, bs

    def test_projected_graph_stages_the_stores(self):
        injector, deferred_draft_kv, epilogue, bs = self._run(projected=True)
        self.assertEqual(injector.inline, [])
        self.assertTrue(
            torch.equal(
                deferred_draft_kv.commit_lens[:bs],
                epilogue.read_accept(bs).commit_lens.to(torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                deferred_draft_kv.cache_loc[: bs * STRIDE], torch.arange(6) + 50
            )
        )

    def test_graph_without_projection_keeps_the_inline_commit(self):
        injector, deferred_draft_kv, _, _ = self._run(projected=False)
        self.assertEqual(len(injector.inline), 1)
        self.assertTrue(bool((deferred_draft_kv.commit_lens == 0).all()))

    @unittest.skipIf(_TRITON_DEVICE is None, "needs CUDA or TRITON_INTERPRET=1")
    def test_one_launch_stage_writes_each_request_slot_into_its_rows(self):
        """With a draft SWA ring every staged row carries its own request's slot,
        also at bs 1."""
        for bs in (1, 2):
            with self.subTest(bs=bs):
                _, deferred_draft_kv, _, _ = self._run(
                    projected=True, bs=bs, ring=True, device=_TRITON_DEVICE
                )
                want = torch.tensor([1, 2], device=_TRITON_DEVICE)[:bs]
                want = want.repeat_interleave(STRIDE)
                self.assertTrue(
                    torch.equal(deferred_draft_kv.state_slot[: bs * STRIDE], want)
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
