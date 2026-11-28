"""Which mixed batches DFLASH verifies inside the extend forward."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.cp import base as cp_base
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative import dflash_worker_v2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashMixedVerifyGate(unittest.TestCase):
    def test_context_parallel_takes_the_plain_mixed_step(self):
        """A context-parallel forward runs the stock attention, which has no tree
        mask, so a mixed batch under CP must not carry verify rows; the running
        requests take the plain mixed step instead."""
        worker = object.__new__(DFlashWorkerV2)
        worker._target_logits_processor = object()
        worker._is_domino = False
        worker.lilicorr = None
        worker._need_mamba_verify_commit = False
        worker._verify_planner = None
        worker._runtime = None
        worker.draft_kv_ring = None
        worker.model_runner = SimpleNamespace(prefill_cuda_graph_runner=None)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.MIXED,
            decoding_reqs=[object()],
            sampling_info=None,
            has_grammar=False,
            return_logprob=False,
            return_hidden_states=False,
            seq_lens_cpu=torch.tensor([64, 8]),
            hisparse_coordinator=None,
        )
        parallel = SimpleNamespace(
            attn_dp_enabled=False, pp_group=SimpleNamespace(world_size=1)
        )
        with (
            envs.SGLANG_ENABLE_DFLASH_MIXED_VERIFY.override(True),
            patch.object(dflash_worker_v2, "get_parallel", return_value=parallel),
        ):
            for strategy, runs in ((False, True), (object(), False)):
                with (
                    self.subTest(cp=strategy is not False),
                    patch.object(cp_base, "_STRATEGY", strategy),
                ):
                    self.assertEqual(worker._runs_mixed_verify(batch), runs)


if __name__ == "__main__":
    unittest.main()
