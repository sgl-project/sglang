"""The DSpark static verify epilogue, captured as its own CUDA graph per verify graph.

A folded (all-greedy) step replays the verify graph and then the epilogue graph; its
accept buffers and committed draft KV must equal, bit for bit, the epilogue recorded
inside the verify graph (the layout before the split). A sampled step replays only the
verify graph, which must no longer write either.

Requires a GPU: the epilogue is captured and replayed as CUDA graphs.
"""

import contextlib
import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_capture_stream,
    get_or_create_global_graph_memory_pool,
)
from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    CommitInjectCtx,
    DsparkVerifyEpilogue,
    StaticVerifyIO,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")

STRIDE, VOCAB, HIDDEN, KV_DIM = 5, 1024, 64, 32
# The runner's captured verify batch sizes, largest first, and its graph variants per
# size (DSv4.1 target verify captures two candidate-indexer variants).
CAPTURE_BS = (4, 2)
VARIANTS = ("candidate_unfiltered", "candidate_filtered")
MAX_BS = max(CAPTURE_BS)
NUM_SLOTS = 256
SENTINEL = -7


@contextlib.contextmanager
def _local_graph_capture(stream):
    """parallel_state.graph_capture on one GPU: no TP group, no communicators."""
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        yield SimpleNamespace(stream=stream)


class _RecordingTpSync:
    def __init__(self):
        self.calls = []

    def sync(self, site, values):
        self.calls.append((site, torch.cuda.is_current_stream_capturing()))
        return values


class _KvInjector:
    """The draft KV commit's shape: a GEMM, then a row store gated by commit_lens."""

    swa_ring_size = None

    def __init__(self, weight):
        self.weight = weight
        # Row NUM_SLOTS takes the uncommitted tokens; the tests compare the rest.
        self.kv = torch.zeros((NUM_SLOTS + 1, KV_DIM), device="cuda")

    def inject_target_hidden(
        self,
        *,
        target_hidden,
        cache_loc,
        cache_loc_2d,
        positions,
        commit_lens,
        state_slot,
    ):
        kv = target_hidden @ self.weight + positions.unsqueeze(1).float()
        col = torch.arange(cache_loc_2d.shape[1], device=cache_loc.device)
        committed = (col < commit_lens.long().unsqueeze(1)).reshape(-1)
        self.kv[torch.where(committed, cache_loc, NUM_SLOTS)] = kv

    @property
    def committed_kv(self):
        return self.kv[:NUM_SLOTS]


def _epilogue(weight, simulated_correct_len):
    tp_sync = _RecordingTpSync()
    epilogue = DsparkVerifyEpilogue(
        max_bs=MAX_BS,
        verify_num_draft_tokens=STRIDE,
        device="cuda",
        tp_sync=tp_sync,
        commit_ctx=CommitInjectCtx(
            draft_model=None,
            block_pos_offsets=torch.arange(STRIDE, device="cuda"),
            # folds_commit: an MLA draft pool.
            resolve_pool=lambda: SimpleNamespace(
                set_swa_key_buffer_radix_fused_norm_rope=None
            ),
            resolve_req_to_token=lambda: None,
            kv_injector=_KvInjector(weight),
        ),
        fused_argmax=True,
        simulated_correct_len=simulated_correct_len,
    )
    return epilogue, tp_sync


class _VerifyRunner:
    """The static verify graph runner's buffers and capture/replay, minus the model:
    the "forward" copies this step's logits and hidden states into the outputs. The
    logits buffer is shared by every graph; each variant owns its hidden states."""

    ragged_verify_mode = False
    model_runner = SimpleNamespace(is_draft_worker=False)

    def __init__(self):
        rows = MAX_BS * STRIDE

        def zeros(*shape, dtype=torch.int64):
            return torch.zeros(shape, dtype=dtype, device="cuda")

        self.input_ids = zeros(rows)
        self.positions = zeros(rows)
        self.out_cache_loc = zeros(rows)
        self.seq_lens = zeros(MAX_BS)
        self.req_pool_indices = zeros(MAX_BS)
        self.logits = zeros(rows, VOCAB, dtype=torch.float32)
        self.hidden = {v: zeros(rows, HIDDEN, dtype=torch.float32) for v in VARIANTS}
        self.step_logits = torch.zeros_like(self.logits)
        self.step_hidden = torch.zeros((rows, HIDDEN), device="cuda")

    def forward_batch(self, bs):
        rows = bs * STRIDE
        return SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=self.input_ids[:rows],
            positions=self.positions[:rows],
            out_cache_loc=self.out_cache_loc[:rows],
            seq_lens=self.seq_lens[:bs],
            req_pool_indices=self.req_pool_indices[:bs],
        )

    def out(self, bs, variant):
        rows = bs * STRIDE
        return LogitsProcessorOutput(
            next_token_logits=self.logits[:rows],
            hidden_states=self.hidden[variant][:rows],
        )

    def capture(self, tail_hook):
        """One graph per size and variant: the forward, then ``tail_hook``
        (capture_tail_hooks), after two eager warmups, on the shared capture
        stream and graph pool."""
        graphs = {}
        pool = get_or_create_global_graph_memory_pool(torch.cuda)
        stream = get_or_create_global_graph_capture_stream()
        for bs in CAPTURE_BS:
            for variant in VARIANTS:
                forward_batch, out = self.forward_batch(bs), self.out(bs, variant)

                def run_once():
                    out.next_token_logits.copy_(self.step_logits[: bs * STRIDE])
                    out.hidden_states.copy_(self.step_hidden[: bs * STRIDE])
                    tail_hook(self, out, forward_batch, bs * STRIDE)

                for _ in range(2):
                    run_once()
                graph = graphs[bs, variant] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    run_once()
        return graphs

    def replay(self, graphs, bs, variant):
        """Replay the padded size's graph; return its outputs cut to the real rows."""
        graphs[min(b for b in CAPTURE_BS if b >= bs), variant].replay()
        return self.out(bs, variant)

    def load_step(self, bs, generator):
        """This step's inputs; padded rows are zero, as the runner fills them."""
        rows = bs * STRIDE

        def randint(high, shape):
            return torch.randint(
                0, high, shape, generator=generator, device="cuda", dtype=torch.int64
            )

        for buf in (self.input_ids, self.positions, self.out_cache_loc, self.seq_lens):
            buf.zero_()
        candidates = randint(VOCAB, (bs, STRIDE))
        seq_lens = 16 + randint(1000, (bs,))
        self.input_ids[:rows] = candidates.view(-1)
        self.seq_lens[:bs] = seq_lens
        self.positions[:rows] = (
            seq_lens.unsqueeze(1) + torch.arange(STRIDE, device="cuda")
        ).view(-1)
        self.out_cache_loc[:rows] = torch.randperm(
            NUM_SLOTS, generator=generator, device="cuda"
        )[:rows]
        self.req_pool_indices.copy_(torch.arange(MAX_BS, device="cuda"))
        self.step_logits.normal_(generator=generator)
        self.step_hidden.normal_(generator=generator)
        # Request r matches its first r % STRIDE drafts, so accept lengths vary.
        predict = randint(VOCAB, (bs, STRIDE))
        for r in range(bs):
            matched = r % STRIDE
            predict[r, :matched] = candidates[r, 1 : 1 + matched]
        self.step_logits[:rows].scatter_(1, predict.view(-1, 1), 100.0)


def _snapshot(epilogue, bs):
    accept = epilogue.read_accept(bs)
    return {f: getattr(accept, f).clone() for f in AcceptOuts.__struct_fields__}


def _fill_accept(epilogue, value):
    accept = epilogue.read_accept(MAX_BS)
    for f in AcceptOuts.__struct_fields__:
        getattr(accept, f).fill_(value)


def _capture_in_graph(runner, epilogue):
    """The layout before the split, which ROCm keeps: the epilogue recorded inside
    every verify graph."""
    with mock.patch.object(dspark_verify, "_is_cuda", False):
        return runner.capture(epilogue.capture_hook)


class TestStaticEpilogueGraph(CustomTestCase):
    def _assert_equal(self, a, b, what):
        self.assertTrue(torch.equal(a, b), what)

    def _assert_same_static_io(self, new, ref):
        # The sampling-accept region reads static_io: the same entries as before.
        self.assertEqual(sorted(new.static_io), sorted(ref.static_io))
        for bs, io in ref.static_io.items():
            for field in StaticVerifyIO.__struct_fields__:
                self.assertEqual(
                    getattr(new.static_io[bs], field).data_ptr(),
                    getattr(io, field).data_ptr(),
                    field,
                )

    def _run_parity(self, simulated):
        torch.manual_seed(0)
        weight = torch.randn(HIDDEN, KV_DIM, device="cuda")

        def simulated_correct_len():
            if simulated is None:
                return None
            return torch.full((1,), simulated, dtype=torch.int64, device="cuda")

        runner = _VerifyRunner()
        # Before the split: the epilogue recorded inside every verify graph.
        ref, _ = _epilogue(weight, simulated_correct_len())
        ref_graphs = _capture_in_graph(runner, ref)
        # After: the verify graph records its buffers; the epilogue is its own graph.
        new, new_sync = _epilogue(weight, simulated_correct_len())
        new_graphs = runner.capture(new.capture_hook)
        self.assertEqual(new_sync.calls, [], "the verify graph still broadcasts")
        with mock.patch.object(dspark_verify, "graph_capture", _local_graph_capture):
            self.assertEqual(new.capture_static_graphs(), sorted(CAPTURE_BS))
        site = SpecTpSyncSite.DSPARK_ACCEPT_GRAPH
        # Per verify graph: the warmup's 3 accept-site broadcasts, then the captured 3.
        self.assertEqual(
            new_sync.calls,
            ([(site, False)] * 3 + [(site, True)] * 3)
            * (len(CAPTURE_BS) * len(VARIANTS)),
        )
        self._assert_same_static_io(new, ref)

        ref_kv, new_kv = ref.commit_ctx.kv_injector, new.commit_ctx.kv_injector
        generator = torch.Generator(device="cuda").manual_seed(1)
        # Exact and padded sizes; the padded rows must not commit.
        for bs in (2, 3, 1, 4):
            for variant in VARIANTS:
                with self.subTest(bs=bs, variant=variant, step="folded"):
                    runner.load_step(bs, generator)
                    ref.begin_static_step(bs, True)
                    runner.replay(ref_graphs, bs, variant)
                    expected = _snapshot(ref, bs)

                    new.begin_static_step(bs, True)
                    out = runner.replay(new_graphs, bs, variant)
                    self.assertTrue(new.replay_static_epilogue(bs, out))
                    actual = _snapshot(new, bs)
                    for field, value in expected.items():
                        self._assert_equal(actual[field], value, field)
                    self._assert_equal(new_kv.committed_kv, ref_kv.committed_kv, "kv")
                    if simulated is None and bs >= 3:
                        # Not vacuous: rows accept different lengths.
                        correct_lens = set(expected["correct_len"].tolist())
                        self.assertGreater(len(correct_lens), 1)
                    self.assertGreater(int((new_kv.committed_kv != 0).any(1).sum()), 0)

                with self.subTest(bs=bs, variant=variant, step="sampled"):
                    # The eager accept serves this step: only the verify graph replays.
                    runner.load_step(bs, generator)
                    for epilogue in (ref, new):
                        epilogue.begin_static_step(bs, False)
                        _fill_accept(epilogue, SENTINEL)
                    kv_before = new_kv.kv.clone()
                    runner.replay(new_graphs, bs, variant)
                    for field, value in _snapshot(new, MAX_BS).items():
                        self.assertTrue(bool((value == SENTINEL).all()), field)
                    self._assert_equal(new_kv.kv, kv_before, "kv")
                    # Before the split the same step ran the accept and dropped it.
                    runner.replay(ref_graphs, bs, variant)
                    bonus = ref.read_accept(bs).bonus
                    self.assertFalse(bool((bonus == SENTINEL).all()))
                    self._assert_equal(ref_kv.committed_kv, new_kv.committed_kv, "kv")

        # The target graphs captured again after startup on new buffers (a weight
        # update with recapture_cuda_graph), twice. The epilogue graphs read the old
        # buffers, so every static step takes the eager accept; static_io follows the
        # new verify graphs as before the split; nothing piles up or keeps the old
        # verify outputs alive.
        startup_graphs = dict(new._epilogue_graphs)
        startup_outs = [weakref.ref(rec[1]) for rec in new._epilogue_records.values()]
        for _ in range(2):
            runner = _VerifyRunner()
            _capture_in_graph(runner, ref)
            runner.capture(new.capture_hook)
            self.assertEqual(new._epilogue_records, {})
            self.assertEqual(new._epilogue_graphs, {})
            self.assertEqual(new._retired_epilogue_graphs, startup_graphs)
            self._assert_same_static_io(new, ref)
            for bs in range(1, MAX_BS + 1):
                for variant in VARIANTS:
                    out = runner.out(bs, variant)
                    self.assertFalse(new.replay_static_epilogue(bs, out))
        gc.collect()
        self.assertEqual(len(startup_outs), len(CAPTURE_BS) * len(VARIANTS))
        self.assertTrue(all(out_ref() is None for out_ref in startup_outs))

    def test_folded_step_matches_the_in_graph_epilogue(self):
        for simulated in (None, 2):
            with self.subTest(simulated=simulated):
                self._run_parity(simulated)

    def test_unknown_buffers_accept_eagerly(self):
        runner = _VerifyRunner()
        epilogue, _ = _epilogue(torch.randn(HIDDEN, KV_DIM, device="cuda"), None)
        variant = VARIANTS[0]
        out4, out2 = runner.out(4, variant), runner.out(2, variant)
        epilogue._record_epilogue(out4, runner.forward_batch(4))
        epilogue._record_epilogue(out2, runner.forward_batch(2))
        # The same outputs recorded again with other input buffers, for size 2 and
        # for size 4's other variant.
        other = runner.forward_batch(2)
        other.input_ids = other.input_ids.clone()
        epilogue._record_epilogue(out2, other)
        out4_other = runner.out(4, VARIANTS[1])
        epilogue._record_epilogue(out4_other, runner.forward_batch(4))
        other4 = runner.forward_batch(4)
        other4.input_ids = other4.input_ids.clone()
        epilogue._record_epilogue(out4_other, other4)
        with mock.patch.object(dspark_verify, "graph_capture", _local_graph_capture):
            self.assertEqual(epilogue.capture_static_graphs(), [4])
        # static_io keeps the last verify graph recorded per size, as the in-graph
        # epilogue does, although only size 4's first variant got an epilogue graph.
        io4, io2 = epilogue.static_io[4], epilogue.static_io[2]
        self.assertEqual(io4.hidden.data_ptr(), out4_other.hidden_states.data_ptr())
        self.assertEqual(io4.candidates.data_ptr(), other4.input_ids.data_ptr())
        self.assertEqual(io2.candidates.data_ptr(), other.input_ids.data_ptr())
        # Sizes 1-2 replay the size-2 verify graph, which has no epilogue graph.
        self.assertFalse(epilogue.replay_static_epilogue(1, runner.out(1, variant)))
        self.assertFalse(epilogue.replay_static_epilogue(2, out2))
        self.assertTrue(epilogue.replay_static_epilogue(3, runner.out(3, variant)))
        # Outputs the epilogue graphs do not read: a variant with no epilogue graph,
        # or a backend that copies the outputs out of the graph.
        self.assertFalse(epilogue.replay_static_epilogue(3, runner.out(3, VARIANTS[1])))
        copied = runner.out(3, variant)
        copied.next_token_logits = copied.next_token_logits.clone()
        self.assertFalse(epilogue.replay_static_epilogue(3, copied))

    def test_deferred_commit_and_rocm_keep_the_epilogue_in_the_verify_graph(self):
        # SGLANG_DSPARK_DEFER_CONTEXT_KV forks a projection in the verify forward
        # that the epilogue joins; ROCm keeps the layout the split was not tested on.
        for case in ("deferred_commit", "rocm"):
            with (
                self.subTest(case=case),
                mock.patch.object(dspark_verify, "_is_cuda", case != "rocm"),
            ):
                runner = _VerifyRunner()
                epilogue, tp_sync = _epilogue(
                    torch.randn(HIDDEN, KV_DIM, device="cuda"), None
                )
                if case == "deferred_commit":
                    epilogue.deferred_commit = SimpleNamespace(
                        join_projection=lambda: False
                    )
                graphs = runner.capture(epilogue.capture_hook)
                self.assertIn((SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, True), tp_sync.calls)
                self.assertEqual(sorted(epilogue.static_io), sorted(CAPTURE_BS))
                self.assertEqual(epilogue.capture_static_graphs(), [])
                runner.load_step(3, torch.Generator(device="cuda").manual_seed(1))
                epilogue.begin_static_step(3, True)
                _fill_accept(epilogue, SENTINEL)
                out = runner.replay(graphs, 3, VARIANTS[0])
                bonus = epilogue.read_accept(3).bonus
                self.assertFalse(bool((bonus == SENTINEL).all()))
                self.assertTrue(epilogue.replay_static_epilogue(3, out))


class TestWorkerReplaysOnlyFoldedSteps(CustomTestCase):
    def test_only_steps_that_take_the_accept_replay_the_epilogue(self):
        calls, hit = [], [True]

        def replay_static_epilogue(bs, out):
            calls.append(bs)
            return hit[0]

        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._verify_epilogue = SimpleNamespace(
            replay_static_epilogue=replay_static_epilogue
        )

        def step(fold_eligible, can_run_cuda_graph=True):
            target_verify = SimpleNamespace(
                can_run_cuda_graph=can_run_cuda_graph, logits_output=None
            )
            return worker._replay_static_epilogue(
                bs=3, fold_eligible=fold_eligible, target_verify=target_verify
            )

        # A sampled step, or a verify forward that ran eagerly: no epilogue replay.
        self.assertFalse(step(False))
        self.assertFalse(step(True, can_run_cuda_graph=False))
        self.assertEqual(calls, [])
        # An all-greedy step replays it; with no epilogue graph the step takes the
        # eager accept (folded_accept is False).
        self.assertTrue(step(True))
        hit[0] = False
        self.assertFalse(step(True))
        self.assertEqual(calls, [3, 3])


if __name__ == "__main__":
    unittest.main()
