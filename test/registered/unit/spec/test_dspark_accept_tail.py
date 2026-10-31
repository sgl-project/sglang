import unittest
from types import SimpleNamespace

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_sampling_accept_graph import (
    GraphSamplingInfo,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")

BS, STRIDE, VOCAB = 3, 5, 50


class _RecordingTpSync:
    def __init__(self):
        self.calls = []

    def sync(self, site, values):
        self.calls.append((site, values.clone()))
        return values


def _greedy_inputs():
    torch.manual_seed(0)
    candidates = torch.randint(0, VOCAB, (BS, STRIDE))
    target_predict = torch.randint(0, VOCAB, (BS, STRIDE))
    # Rows match the drafts for 0, 2 and all positions.
    target_predict[1, :2] = candidates[1, 1:3]
    target_predict[2, :-1] = candidates[2, 1:]
    logits = torch.randn(BS * STRIDE, VOCAB)
    logits.scatter_(1, target_predict.view(-1, 1), 100.0)
    prefix_lens = torch.tensor([7, 11, 13], dtype=torch.int64)
    return candidates, logits, prefix_lens


def _eager_accept(
    *,
    candidates,
    logits,
    prefix_lens,
    verify_lens,
    tp_sync,
    simulated_correct_len,
    scored_lens=None,
    draft_logits=None,
    sampling_info=None,
):
    bs = candidates.shape[0]
    draft_tokens = candidates[:, 1:].contiguous()
    target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type="test")
            )
        )
    )
    executor = TargetVerifyExecutor(
        target_worker=target_worker,
        gamma=STRIDE - 1,
        verify_num_draft_tokens=STRIDE,
        model_runner=None,
        kv_injector=None,
        tp_sync=tp_sync,
        simulated_correct_len=simulated_correct_len,
    )
    return executor.accept_and_finalize(
        folded_accept=False,
        bs=bs,
        verify_ids_2d=candidates,
        target_logits=logits,
        draft_block=DraftBlockResult(
            draft_tokens=draft_tokens,
            corrected_logits=draft_logits,
            greedy_mask=torch.full(
                (bs,), draft_logits is None, dtype=torch.bool, device=candidates.device
            ),
            temperatures=torch.ones(bs, device=candidates.device),
            greedy_draft=draft_logits is None,
        ),
        sampling_info=sampling_info,
        draft_input=None,
        layout=None
        if verify_lens is None
        else SimpleNamespace(verify_lens=verify_lens),
        prefix_lens=prefix_lens,
        draft_tokens=draft_tokens,
        scored_lens=scored_lens,
    )


def _expected_greedy_accept(candidates, target_predict, verify_lens, scored_lens):
    """Contiguous draft matches over the rows that carry real target logits,
    capped to verify_len - 1; cap_trim is what the target saw past the cap."""
    correct, trim, bonus = [], [], []
    for r in range(candidates.shape[0]):
        rows = STRIDE if scored_lens is None else int(scored_lens[r])
        raw = 0
        while (
            raw < min(rows, STRIDE - 1)
            and candidates[r, raw + 1] == target_predict[r, raw]
        ):
            raw += 1
        capped = min(raw, int(verify_lens[r]) - 1)
        correct.append(capped)
        trim.append(raw - capped)
        bonus.append(int(target_predict[r, capped]))
    return correct, trim, bonus


def _compact_verify_inputs(candidates, target_predict, verify_lens, num_rows):
    """The compact target rows (num_rows long, unused tail left as garbage) and
    the strided logits run_compact rebuilds from them (zero padding rows)."""
    bs = candidates.shape[0]
    compact_logits = torch.randn(num_rows, VOCAB)
    compact_ids = torch.randint(0, VOCAB, (num_rows,))
    strided_logits = torch.zeros(bs * STRIDE, VOCAB)
    i = 0
    for r in range(bs):
        for w in range(int(verify_lens[r])):
            compact_logits[i] = torch.randn(VOCAB)
            compact_logits[i, target_predict[r, w]] = 100.0
            compact_ids[i] = candidates[r, w]
            strided_logits[r * STRIDE + w] = compact_logits[i]
            i += 1
    return compact_logits, compact_ids, strided_logits


class TestAcceptTailParity(CustomTestCase):
    def _run(self, simulated_correct_len, verify_lens):
        candidates, logits, prefix_lens = _greedy_inputs()
        draft_tokens = candidates[:, 1:].contiguous()

        graph_sync = _RecordingTpSync()
        epilogue = DsparkVerifyEpilogue(
            max_bs=BS + 1,
            verify_num_draft_tokens=STRIDE,
            device="cpu",
            tp_sync=graph_sync,
            simulated_correct_len=simulated_correct_len,
        )
        epilogue._accept(
            candidates=candidates,
            logits=logits,
            draft_tokens=draft_tokens,
            seq_lens=prefix_lens,
            cutoff_verify_lens=verify_lens,
        )
        graph = epilogue.read_accept(BS)

        eager_sync = _RecordingTpSync()
        eager = _eager_accept(
            candidates=candidates,
            logits=logits,
            prefix_lens=prefix_lens,
            verify_lens=verify_lens,
            tp_sync=eager_sync,
            simulated_correct_len=simulated_correct_len,
        )

        got, want = msgspec.structs.asdict(graph), msgspec.structs.asdict(eager)
        for field, value in want.items():
            self.assertTrue(
                torch.equal(got[field].cpu().long(), value.cpu().long()), field
            )
        self.assertEqual(
            [site for site, _ in graph_sync.calls],
            [SpecTpSyncSite.DSPARK_ACCEPT_GRAPH] * 3,
        )
        self.assertEqual(
            [site for site, _ in eager_sync.calls],
            [SpecTpSyncSite.DSPARK_ACCEPT_GREEDY] * 3,
        )
        for (_, graph_values), (_, eager_values) in zip(
            graph_sync.calls, eager_sync.calls
        ):
            self.assertTrue(torch.equal(graph_values, eager_values))

    def test_graph_accept_matches_eager_greedy(self):
        # Row 1 has a 2-token budget, so its 2 correct drafts are capped to 1.
        ragged = torch.tensor([STRIDE, 2, 3], dtype=torch.int64)
        for simulated in (None, 1):
            for verify_lens in (None, ragged):
                with self.subTest(simulated=simulated, ragged=verify_lens is not None):
                    simulated_correct_len = (
                        None if simulated is None else torch.tensor([simulated])
                    )
                    self._run(simulated_correct_len, verify_lens)


@unittest.skipUnless(torch.cuda.is_available(), "compact scatter is Triton-only")
class TestCompactAcceptTail(CustomTestCase):
    """Compact verify scores only verify_len rows per request; the rows past them
    are zero padding whose argmax is token 0. No accept path may count a draft
    past the scored rows, while cap-accept (all rows real) keeps its full count."""

    def _epilogue(self, bs):
        # The post-forward graph reads block accept lengths from this epilogue buffer.
        with envs.SGLANG_OPT_DSV41_POST_FORWARD_CUDA_GRAPH.override(True):
            return DsparkVerifyEpilogue(
                max_bs=bs + 1,
                verify_num_draft_tokens=STRIDE,
                device="cuda",
                tp_sync=_RecordingTpSync(),
            )

    def _assert_block_accept(
        self, epilogue, bs, candidates, target_predict, verify_lens
    ):
        correct, trim, _ = _expected_greedy_accept(
            candidates, target_predict, verify_lens, verify_lens
        )
        self.assertEqual(
            epilogue.read_block_accept_lens(bs).tolist(),
            [c + 1 + t for c, t in zip(correct, trim)],
        )

    def _assert_accept(self, accept, candidates, target_predict, verify_lens, scored):
        correct, trim, bonus = _expected_greedy_accept(
            candidates, target_predict, verify_lens, scored
        )
        self.assertEqual(accept.correct_len.tolist(), correct)
        self.assertEqual(accept.cap_trim_lens.tolist(), trim)
        self.assertEqual(accept.bonus.tolist(), bonus)
        self.assertEqual(accept.commit_lens.tolist(), [c + 1 for c in correct])

    def _eager(self, candidates, logits, verify_lens, scored_lens):
        bs = candidates.shape[0]
        return _eager_accept(
            candidates=candidates.cuda(),
            logits=logits.cuda(),
            prefix_lens=torch.arange(7, 7 + bs, device="cuda"),
            verify_lens=verify_lens.cuda(),
            tp_sync=_RecordingTpSync(),
            simulated_correct_len=None,
            scored_lens=None if scored_lens is None else scored_lens.cuda(),
        )

    def test_padding_rows_never_count_as_accepted_drafts(self):
        torch.manual_seed(0)
        bs = 5
        verify_lens = torch.tensor([2, 3, STRIDE, 2, 3], dtype=torch.int64)
        candidates = torch.randint(1, VOCAB, (bs, STRIDE))
        target_predict = torch.randint(1, VOCAB, (bs, STRIDE))
        # Row 0: d1 accepted, then the target predicts token 0 on its last scored row.
        target_predict[0, 0] = candidates[0, 1]
        target_predict[0, 1] = 0
        # Row 1: the last scored row also accepts the first cut draft d3.
        target_predict[1, :3] = candidates[1, 1:4]
        # Row 3: d2 is the accepted cut draft; the real drafts after it are token 0
        # and match the padding rows unless the scored rows bound the count.
        candidates[3, 3:] = 0
        target_predict[3, :] = torch.tensor([*candidates[3, 1:3].tolist(), 0, 0, 1])
        # Row 4: d1 is rejected, so its matching tail must not count at all.
        candidates[4, 4] = 0
        target_predict[4, 0] = candidates[4, 1] % (VOCAB - 1) + 1
        target_predict[4, 1:4] = candidates[4, 2:5]
        num_rows = int(verify_lens.sum())
        compact_logits, compact_ids, strided_logits = _compact_verify_inputs(
            candidates, target_predict, verify_lens, num_rows
        )

        epilogue = self._epilogue(bs)
        epilogue.begin_step(verify_lens.cuda(), armed=False)
        epilogue.draft_tokens_buf[: bs * (STRIDE - 1)].copy_(
            candidates[:, 1:].reshape(-1)
        )
        epilogue(
            compact_logits=compact_logits.cuda(),
            compact_hidden=torch.zeros(num_rows, 4, device="cuda"),
            input_ids=compact_ids.cuda(),
            seq_lens=torch.arange(7, 7 + bs, device="cuda"),
            req_pool_indices=torch.arange(bs, device="cuda"),
            bs=bs,
        )
        graph = epilogue.read_accept(bs)
        eager = self._eager(candidates, strided_logits, verify_lens, verify_lens)

        self.assertEqual(graph.cap_trim_lens.tolist(), [0, 1, 0, 1, 0])
        self._assert_block_accept(epilogue, bs, candidates, target_predict, verify_lens)
        for accept in (graph, eager):
            self._assert_accept(
                accept, candidates, target_predict, verify_lens, verify_lens
            )

        # Cap-accept scores every row, so row 3's real token-0 matches do count.
        dense_logits = torch.randn(bs * STRIDE, VOCAB)
        dense_logits.scatter_(1, target_predict.view(-1, 1), 100.0)
        cap_accept = self._eager(candidates, dense_logits, verify_lens, None)
        self.assertEqual(cap_accept.cap_trim_lens.tolist()[3], 3)
        self._assert_accept(cap_accept, candidates, target_predict, verify_lens, None)

    def test_replayed_graph_reads_each_steps_drafts_and_cutoffs(self):
        torch.manual_seed(1)
        bs, num_rows = 4, 4 * STRIDE
        epilogue = self._epilogue(bs)
        compact_logits = torch.zeros(num_rows, VOCAB, device="cuda")
        compact_ids = torch.zeros(num_rows, dtype=torch.int64, device="cuda")
        call = dict(
            compact_logits=compact_logits,
            compact_hidden=torch.zeros(num_rows, 4, device="cuda"),
            input_ids=compact_ids,
            seq_lens=torch.arange(7, 7 + bs, device="cuda"),
            req_pool_indices=torch.arange(bs, device="cuda"),
            bs=bs,
        )

        def load_step():
            # A 3-token alphabet makes long matching chains and token-0 drafts common.
            candidates = torch.randint(0, 3, (bs, STRIDE))
            target_predict = torch.where(
                torch.rand(bs, STRIDE) < 0.75,
                torch.roll(candidates, -1, dims=1),
                torch.randint(0, 3, (bs, STRIDE)),
            )
            verify_lens = torch.randint(1, STRIDE + 1, (bs,))
            step_logits, step_ids, strided = _compact_verify_inputs(
                candidates, target_predict, verify_lens, num_rows
            )
            compact_logits.copy_(step_logits)
            compact_ids.copy_(step_ids)
            epilogue.begin_step(verify_lens.cuda(), armed=False)
            epilogue.draft_tokens_buf[: bs * (STRIDE - 1)].copy_(
                candidates[:, 1:].reshape(-1)
            )
            return candidates, target_predict, verify_lens, strided

        load_step()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            epilogue(**call)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            epilogue(**call)

        for step in range(8):
            with self.subTest(step=step):
                candidates, target_predict, verify_lens, strided = load_step()
                graph.replay()
                torch.cuda.synchronize()
                replayed = epilogue.read_accept(bs)
                self._assert_accept(
                    replayed, candidates, target_predict, verify_lens, verify_lens
                )
                self._assert_block_accept(
                    epilogue, bs, candidates, target_predict, verify_lens
                )
                eager = self._eager(candidates, strided, verify_lens, verify_lens)
                for field, value in msgspec.structs.asdict(eager).items():
                    self.assertTrue(
                        torch.equal(
                            getattr(replayed, field).cpu().long(), value.cpu().long()
                        ),
                        field,
                    )

    def test_sampled_accept_bounds_only_the_trim(self):
        """Rejection sampling also scans the padding rows: a zero row's softmax
        equals a uniform draft's, so every padded draft is accepted. The scored
        rows must bound that trim and leave the accept and bonus untouched."""
        torch.manual_seed(2)
        bs, gamma = 4, STRIDE - 1
        verify_lens = torch.tensor([2, 3, 3, STRIDE], dtype=torch.int64)
        candidates = torch.randint(1, VOCAB, (bs, STRIDE))
        # One-hot scored rows (p ~ 1): rows 0, 1 and 3 accept every scored draft.
        target_token = torch.roll(candidates, -1, dims=1)
        # Row 2 rejects d1, so its bonus is the residual's only token.
        target_token[2, 0] = candidates[2, 1] % (VOCAB - 1) + 1
        logits = torch.zeros(bs * STRIDE, VOCAB)
        for r in range(bs):
            for w in range(int(verify_lens[r])):
                logits[r * STRIDE + w, target_token[r, w]] = 100.0

        def accept(scored_lens):
            torch.manual_seed(3)
            return self._eager_sampled(candidates, logits, verify_lens, scored_lens)

        unbounded, bounded = accept(None), accept(verify_lens)
        self.assertEqual(unbounded.cap_trim_lens.tolist(), [3, 2, 0, 0])
        self.assertEqual(bounded.cap_trim_lens.tolist(), [1, 1, 0, 0])
        self.assertEqual(bounded.correct_len.tolist(), [1, 2, 0, gamma])
        self.assertEqual(
            bounded.bonus.tolist(),
            [
                int(candidates[0, 2]),
                int(candidates[1, 3]),
                int(target_token[2, 0]),
                int(target_token[3, gamma]),
            ],
        )
        for field in ("correct_len", "bonus", "commit_lens", "out_tokens"):
            self.assertTrue(
                torch.equal(getattr(bounded, field), getattr(unbounded, field)), field
            )

    def _eager_sampled(self, candidates, logits, verify_lens, scored_lens):
        bs = candidates.shape[0]
        return _eager_accept(
            candidates=candidates.cuda(),
            logits=logits.cuda(),
            prefix_lens=torch.arange(7, 7 + bs, device="cuda"),
            verify_lens=verify_lens.cuda(),
            tp_sync=_RecordingTpSync(),
            simulated_correct_len=None,
            scored_lens=None if scored_lens is None else scored_lens.cuda(),
            # Zero draft logits: q = 1 / VOCAB for every drafted token.
            draft_logits=torch.zeros(bs, STRIDE - 1, VOCAB, device="cuda"),
            sampling_info=GraphSamplingInfo(
                temperatures=torch.ones(bs, 1, device="cuda")
            ),
        )


class _NoSiteTpSync(_RecordingTpSync):
    def enabled(self, site):
        return False


class _AllSitesTpSync(_RecordingTpSync):
    def enabled(self, site):
        return True


def _verify_out(logits):
    # A static verify graph's output always carries the target hidden states.
    return SimpleNamespace(
        next_token_logits=logits,
        hidden_states=logits.new_zeros(logits.shape[0], 4),
    )


def _verify_batch(candidates, prefix_lens):
    rows = candidates.numel()
    return SimpleNamespace(
        batch_size=BS,
        input_ids=candidates.view(-1),
        seq_lens=prefix_lens,
        req_pool_indices=torch.arange(BS, device=candidates.device),
        out_cache_loc=candidates.new_zeros(rows),
        positions=candidates.new_zeros(rows),
    )


class TestFusedStaticAcceptTail(CustomTestCase):
    """SGLANG_DSPARK_FUSED_ACCEPT_TAIL: the one-launch accept tail (the Triton kernel on
    CUDA) writes what the static epilogue's separate ops write."""

    def _static_accept(self, *, fused, simulated, tp_sync, device="cpu"):
        candidates, logits, prefix_lens = (t.to(device) for t in _greedy_inputs())
        epilogue = DsparkVerifyEpilogue(
            max_bs=BS + 1,
            verify_num_draft_tokens=STRIDE,
            device=device,
            tp_sync=tp_sync,
            simulated_correct_len=simulated,
            fused_accept_tail=fused,
        )
        epilogue._static_epilogue(
            _verify_out(logits), _verify_batch(candidates, prefix_lens)
        )
        return epilogue

    def test_fused_matches_separate_ops(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for simulated in (None, torch.tensor([1], device=device)):
                with self.subTest(device=device, simulated=simulated is not None):
                    kw = dict(simulated=simulated, device=device)
                    fused = self._static_accept(
                        fused=True, tp_sync=_NoSiteTpSync(), **kw
                    )
                    self.assertTrue(fused._serves_accept_tail(BS))
                    separate = self._static_accept(
                        fused=False, tp_sync=_NoSiteTpSync(), **kw
                    )
                    got = msgspec.structs.asdict(fused.read_accept(BS))
                    ref = msgspec.structs.asdict(separate.read_accept(BS))
                    for field, value in ref.items():
                        self.assertTrue(torch.equal(got[field], value), field)

    def test_accept_site_broadcast_keeps_separate_ops(self):
        # Rank 0's results are broadcast between the ops; one kernel would skip it.
        sync = _AllSitesTpSync()
        epilogue = self._static_accept(fused=True, simulated=None, tp_sync=sync)
        self.assertFalse(epilogue._serves_accept_tail(BS))
        self.assertEqual(
            [site for site, _ in sync.calls], [SpecTpSyncSite.DSPARK_ACCEPT_GRAPH] * 3
        )

    def test_accept_site_broadcast_keeps_separate_ops_under_simulate_fold(self):
        # A simulated run keeps the separate ops that real serving runs here.
        epilogue = DsparkVerifyEpilogue(
            max_bs=BS + 1,
            verify_num_draft_tokens=STRIDE,
            device="cpu",
            tp_sync=_AllSitesTpSync(),
            simulated_correct_len=torch.tensor([1]),
            simulate_fold=True,
            fused_accept_tail=True,
        )
        self.assertTrue(epilogue.simulate_fold)
        self.assertFalse(epilogue._serves_accept_tail(BS))


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestSimBonusSharedLength(CustomTestCase):
    def test_a_stride_0_view_of_the_shared_length_reads_like_a_full_tensor(self):
        """The simulated bonus takes the one shared length as expand(bs) of its [1]
        buffer (no copy in the graph); the partials equal those of a [bs] tensor."""
        from sglang.kernels.ops.speculative.dspark.simulated_bonus import (
            simulated_bonus_partials_graph,
        )

        bs, rows = 8, 6
        shared = torch.tensor([2], dtype=torch.int64, device="cuda")
        args = dict(
            target_logits=torch.randn(bs * rows, 9000, device="cuda"),
            temperatures=torch.ones(bs, device="cuda"),
            seed=torch.tensor([77], dtype=torch.int32, device="cuda"),
            bs=bs,
            rows_per_request=rows,
        )
        view = simulated_bonus_partials_graph(correct_len=shared.expand(bs), **args)
        full = simulated_bonus_partials_graph(correct_len=shared.repeat(bs), **args)
        self.assertTrue(torch.equal(view[0], full[0]))
        self.assertTrue(torch.equal(view[1], full[1]))


if __name__ == "__main__":
    unittest.main()
