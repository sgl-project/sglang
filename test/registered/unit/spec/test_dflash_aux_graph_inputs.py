import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


class TestDFlashAuxGraphInputs(unittest.TestCase):
    def test_shared_target_output_and_eager_fallback(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                self._check_replay(overlap)

    def _check_replay(self, overlap):
        hidden = torch.zeros(8, 16, device="cuda")
        other = torch.zeros_like(hidden)
        cache = torch.empty_like(hidden)
        loc = torch.arange(8, device="cuda")
        worker = object.__new__(DFlashWorkerV2)
        runner = SimpleNamespace(buffers=SimpleNamespace(aux_hidden_states=hidden))
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(decode_cuda_graph_runner=runner)
        )
        worker.device = hidden.device
        worker.draft_model = SimpleNamespace(fc_tp_columns=None)
        worker._use_aux_graph = True
        worker._megakernel_aux = None
        worker._batched_small_copy = False
        worker._aux_graphs = {}
        worker._share_aux_hidden = not overlap
        worker._aux_overlap = overlap
        worker._aux_pending = False
        worker._aux_stream = worker._aux_done = None

        def append(*, target_hidden, cache_loc, **kwargs):
            cache.index_copy_(0, cache_loc, target_hidden)

        worker._append_target_hidden_to_draft_kv_by_loc = append
        for value, (source, rows) in enumerate(
            [(hidden, 4), (hidden, 8), (hidden, 4), (other, 4), (hidden, 4)]
        ):
            worker._join_aux()
            cache.fill_(-1)
            source.fill_(value)
            worker._append_target_hidden_decode(
                target_hidden=source[:rows],
                cache_loc=loc[:rows],
                cache_loc_2d=loc[:rows].view(1, rows),
                positions=loc[:rows],
                commit_lens=torch.tensor([rows], device="cuda", dtype=torch.int32),
            )
            worker._join_aux()
            self.assertTrue((cache[:rows] == value).all().item())
            self.assertTrue((cache[rows:] == -1).all().item())
            entry = worker._aux_graphs[((rows, 16), (1, rows))]
            self.assertEqual(
                entry[0][0].data_ptr() == hidden.data_ptr(), not overlap and value < 3
            )


if __name__ == "__main__":
    unittest.main()
