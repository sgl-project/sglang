import unittest
from types import SimpleNamespace

import msgspec
import torch

from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

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
            tp_sync=eager_sync,
            simulated_correct_len=simulated_correct_len,
        )
        eager = executor.accept_and_finalize(
            folded_accept=False,
            bs=BS,
            verify_ids_2d=candidates,
            target_logits=logits,
            draft_block=DraftBlockResult(
                draft_tokens=draft_tokens,
                corrected_logits=None,
                greedy_mask=torch.ones(BS, dtype=torch.bool),
                temperatures=torch.ones(BS),
                greedy_draft=True,
            ),
            sampling_info=None,
            draft_input=None,
            layout=None
            if verify_lens is None
            else SimpleNamespace(verify_lens=verify_lens),
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
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


class _NoSiteTpSync(_RecordingTpSync):
    def enabled(self, site):
        return False


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
        class _AllSites(_RecordingTpSync):
            def enabled(self, site):
                return True

        sync = _AllSites()
        epilogue = self._static_accept(fused=True, simulated=None, tp_sync=sync)
        self.assertFalse(epilogue._serves_accept_tail(BS))
        self.assertEqual(
            [site for site, _ in sync.calls], [SpecTpSyncSite.DSPARK_ACCEPT_GRAPH] * 3
        )


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
