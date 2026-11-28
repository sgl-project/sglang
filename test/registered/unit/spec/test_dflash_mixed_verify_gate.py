"""Which mixed batches DFLASH verifies inside the extend forward."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.cp import base as cp_base
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative import dflash_utils, dflash_worker_v2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashMixedVerifyGate(unittest.TestCase):
    def test_context_parallel_takes_the_plain_mixed_step(self):
        """A context-parallel forward runs the stock attention, which has no tree
        mask, so a mixed batch under CP must not carry verify rows; the running
        requests take the plain mixed step, and the scheduler budgets one row
        each for them instead of a verify block."""
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
        spec = SimpleNamespace(
            speculative_dflash_mixed_verify=True, speculative_num_draft_tokens=8
        )
        with (
            patch.object(dflash_utils, "get_parallel", return_value=parallel),
            patch.object(dflash_utils, "get_spec", return_value=spec),
        ):
            for strategy, runs, rows in ((False, True, 8), (object(), False, 1)):
                # The configuration part is read once per process.
                dflash_utils.mixed_verify_configured.cache_clear()
                self.addCleanup(dflash_utils.mixed_verify_configured.cache_clear)
                with (
                    self.subTest(cp=strategy is not False),
                    patch.object(cp_base, "_STRATEGY", strategy),
                ):
                    self.assertEqual(worker._runs_mixed_verify(batch), runs)
                    self.assertEqual(
                        dflash_utils.mixed_decode_tokens_per_request(is_dflash=True),
                        rows,
                    )

    def test_decode_batches_skip_the_configuration_check(self):
        """Every DFLASH decode step asks the gate; a non-mixed batch answers from
        its forward mode alone."""
        worker = object.__new__(DFlashWorkerV2)
        configured = Mock(return_value=True)
        with patch.object(dflash_worker_v2, "mixed_verify_configured", configured):
            batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
            self.assertFalse(worker._runs_mixed_verify(batch))
        configured.assert_not_called()


if __name__ == "__main__":
    unittest.main()
