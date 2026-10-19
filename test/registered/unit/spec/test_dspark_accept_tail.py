import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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

        for field in eager.__struct_fields__:
            self.assertTrue(
                torch.equal(
                    getattr(graph, field).cpu().long(),
                    getattr(eager, field).cpu().long(),
                ),
                field,
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


if __name__ == "__main__":
    unittest.main()
