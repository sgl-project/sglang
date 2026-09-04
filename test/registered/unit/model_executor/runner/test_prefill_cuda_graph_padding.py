import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    prefill_graph_tolerates_sum_len,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPrefillCudaGraphPadding(CustomTestCase):
    def _make_runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = Backend.TC_PIECEWISE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [4, 16]
        runner.max_num_tokens = 16
        return runner

    def _make_forward_batch(self, num_tokens):
        return SimpleNamespace(
            batch_size=1,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=None,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            input_ids=list(range(num_tokens)),
            extend_prefix_lens_cpu=[0],
        )

    def test_rejects_more_than_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertFalse(runner.can_run_graph(self._make_forward_batch(5)))

    def test_accepts_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertTrue(runner.can_run_graph(self._make_forward_batch(8)))

    def test_replay_snapshot_uses_padded_token_count(self):
        runner = self._make_runner()
        runner.use_captured_attn_metadata = False
        attn_backend = mock.Mock()
        runner.model_runner = SimpleNamespace(attn_backend=attn_backend)
        forward_batch = self._make_forward_batch(8)
        static_forward_batch = self._make_forward_batch(16)

        runner._prepare_forward_metadata_for_replay(
            forward_batch,
            static_forward_batch,
            num_tokens=16,
        )

        attn_backend.init_forward_metadata.assert_called_once_with(forward_batch)
        attn_backend.prepare_prefill_shared_read_snapshot.assert_called_once_with(
            forward_batch, num_qo_tokens=16
        )

    def _megamoe_no_prefill_cp(self):
        return (
            mock.patch(
                "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                return_value=MoeA2ABackend.MEGAMOE,
            ),
            mock.patch(
                "sglang.srt.layers.attention.dsa.utils.is_dsa_enable_prefill_cp",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.layers.utils.cp_utils.is_mla_prefill_cp_enabled",
                return_value=False,
            ),
        )

    def test_megamoe_sparse_dp_falls_back_to_eager(self):
        # Regression (#37561): with MegaMoE a 0-token DP rank stayed IDLE/eager
        # under SUM_LEN while the busy rank replayed the graph, and the two
        # collective sequences deadlocked. The sparse batch must keep MAX_LEN
        # and take the eager fallback like every other a2a backend.
        runner = self._make_runner()
        a2a, dsa_cp, mla_cp = self._megamoe_no_prefill_cp()
        with a2a, dsa_cp, mla_cp:
            self.assertFalse(prefill_graph_tolerates_sum_len([8, 0]))
            self.assertTrue(
                runner._has_inactive_dp_rank(
                    SimpleNamespace(global_num_tokens_cpu=[8, 0])
                )
            )

    def test_megamoe_all_ranks_busy_keeps_per_rank_buckets(self):
        runner = self._make_runner()
        a2a, dsa_cp, mla_cp = self._megamoe_no_prefill_cp()
        with a2a, dsa_cp, mla_cp:
            self.assertTrue(prefill_graph_tolerates_sum_len([8, 16]))
            self.assertFalse(
                runner._has_inactive_dp_rank(
                    SimpleNamespace(global_num_tokens_cpu=[8, 16])
                )
            )


if __name__ == "__main__":
    unittest.main()
