import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_context, get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDecoderTailRouting(CustomTestCase):
    def setUp(self):
        super().setUp()
        config = get_context().override_server_args(
            model_path="dummy", tp_size=2, attn_dp_size=2, ep_size=2
        )
        config.install()
        self.addCleanup(config.restore)
        moe = get_flags().moe.override(a2a_backend=MoeA2ABackend.MEGAMOE)
        moe.__enter__()
        self.addCleanup(moe.__exit__, None, None, None)

        self.backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
        self.backend.token_to_kv_pool = SimpleNamespace(request_window=None)
        self.backend._bcg_tail_num_token_non_padded = None
        core = SimpleNamespace(low_ratios=())
        self.backend.forward_metadata = SimpleNamespace(core_attn_metadata=core)
        self.tail = SimpleNamespace(real_rows_count=128, cp_metadata=None)
        self.backend.tail_forward_metadata = SimpleNamespace(
            core_attn_metadata=core, late_layer_tail=self.tail
        )
        self.batch = ForwardBatch.__new__(ForwardBatch)
        self.batch.attn_cp_metadata = None
        self.batch.global_num_token_non_padded_cpu = 16384
        self.batch.global_num_token_non_padded = torch.tensor(16384, dtype=torch.int32)
        self.batch.num_token_non_padded = self.batch.global_num_token_non_padded

    def test_padded_tail_rows_are_not_routed_and_replays_reuse_the_bound(self):
        """A graph-sized tail must route only live rows on every replay, while
        earlier layers and subsequent batches retain their full-prefill bound."""
        original = self.batch.global_num_token_non_padded
        saved = self.backend.enter_late_layer_tail(self.batch)
        graph_bound = self.batch.moe_num_token_non_padded()
        self.assertEqual(int(graph_bound), 128)
        self.backend.exit_late_layer_tail(saved, self.batch)
        self.assertIs(self.batch.global_num_token_non_padded, original)
        self.assertIs(self.batch.num_token_non_padded, original)
        self.assertEqual(int(original), 16384)
        self.assertEqual(self.batch.global_num_token_non_padded_cpu, 16384)

        self.tail.real_rows_count = 7
        saved = self.backend.enter_late_layer_tail(self.batch)
        self.assertIs(self.batch.moe_num_token_non_padded(), graph_bound)
        self.assertEqual(int(graph_bound), 7)
        self.backend.exit_late_layer_tail(saved, self.batch)
        self.assertIs(self.batch.num_token_non_padded, original)

    def test_idle_peer_keeps_zero_routing_bound(self):
        """DP graph padding may fabricate rows, but an idle peer must not
        dispatch any of those rows to experts."""
        self.batch.global_num_token_non_padded.zero_()
        saved = self.backend.enter_late_layer_tail(self.batch)
        self.assertEqual(int(self.batch.moe_num_token_non_padded()), 0)
        self.backend.exit_late_layer_tail(saved, self.batch)
        self.assertEqual(int(self.batch.num_token_non_padded), 0)


if __name__ == "__main__":
    unittest.main()
