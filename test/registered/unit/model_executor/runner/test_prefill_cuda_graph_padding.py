import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.model_executor.runner.prefill_cuda_graph_runner as runner_module
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor import forward_batch_info
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    prefill_graph_tolerates_sum_len,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestPrefillCudaGraphPadding(CustomTestCase):
    def _make_runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = Backend.TC_PIECEWISE
        runner.has_mha_companion_layers = False
        runner.prefer_eager_mixed_prefill = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [4, 16]
        runner.max_context_size = None
        runner.max_num_tokens = 16
        return runner

    def _make_forward_batch(self, num_tokens, mode=ForwardMode.EXTEND):
        return SimpleNamespace(
            batch_size=1,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=None,
            forward_mode=mode,
            global_forward_mode=None,
            _original_forward_mode=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            input_ids=list(range(num_tokens)),
            extend_prefix_lens_cpu=[0],
            seq_lens_cpu=torch.tensor([num_tokens], dtype=torch.int64),
            seq_lens=torch.tensor([num_tokens], dtype=torch.int64),
        )

    def test_rejects_more_than_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertFalse(runner.can_run_graph(self._make_forward_batch(5)))

    def test_accepts_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertTrue(runner.can_run_graph(self._make_forward_batch(8)))

    def test_mixed_batch_uses_scoped_runner_policy(self):
        runner = self._make_runner()
        batch = self._make_forward_batch(8, mode=ForwardMode.MIXED)

        self.assertTrue(runner.can_run_graph(batch))
        runner.prefer_eager_mixed_prefill = True
        self.assertFalse(runner.can_run_graph(batch))

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
            shape_key=ShapeKey(size=16),
        )

        attn_backend.init_forward_metadata.assert_called_once_with(forward_batch)
        attn_backend.prepare_prefill_shared_read_snapshot.assert_called_once_with(
            forward_batch, num_qo_tokens=16
        )

    def test_full_replay_pads_request_slot_cpu_mirror(self):
        for batch_size, has_mirror in ((2, True), (4, True), (2, False)):
            with self.subTest(batch_size=batch_size, has_mirror=has_mirror):
                runner = self._make_runner()
                runner._is_full_backend = True
                runner._capture_req_slots = 4
                runner._full_cg_seq_lens_cpu = torch.full((4,), -1)
                slots = torch.tensor([7, 2, 9, 5])[:batch_size]
                seq_lens = torch.arange(1, batch_size + 1)
                static_slots = torch.zeros(4, dtype=slots.dtype)
                static_slots[:batch_size].copy_(slots)
                static_lens = torch.zeros(4, dtype=seq_lens.dtype)
                static_lens[:batch_size].copy_(seq_lens)
                runner._prefill_static_buffers = {
                    "req_pool_indices": static_slots,
                    "seq_lens": static_lens,
                    "extend_seq_lens": static_lens.clone(),
                    "extend_prefix_lens": torch.zeros(4, dtype=torch.int64),
                }
                attn_backend = mock.Mock()
                runner.model_runner = SimpleNamespace(attn_backend=attn_backend)
                batch = ForwardBatch(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=batch_size,
                    input_ids=torch.arange(batch_size),
                    req_pool_indices=slots,
                    req_pool_indices_cpu=slots if has_mirror else None,
                    seq_lens=seq_lens,
                    seq_lens_cpu=seq_lens,
                    out_cache_loc=torch.arange(batch_size),
                    seq_lens_sum=int(seq_lens.sum()),
                )

                runner._prepare_forward_metadata_for_replay(
                    batch, batch, shape_key=ShapeKey(size=16)
                )

                attn_backend.init_forward_metadata_out_graph.assert_called_once()
                padded = attn_backend.init_forward_metadata_out_graph.call_args.args[0]
                self.assertIsInstance(padded, ForwardBatch)
                self.assertIsNot(padded, batch)
                self.assertEqual(padded.batch_size, 4)
                torch.testing.assert_close(padded.seq_lens_cpu, static_lens)
                torch.testing.assert_close(padded.req_pool_indices, static_slots)
                if has_mirror:
                    torch.testing.assert_close(
                        padded.req_pool_indices_cpu, static_slots
                    )
                    self.assertEqual(padded.req_pool_indices_cpu.device.type, "cpu")
                    self.assertIs(batch.req_pool_indices_cpu, slots)
                else:
                    self.assertIsNone(padded.req_pool_indices_cpu)
                self.assertEqual(batch.batch_size, batch_size)
                torch.testing.assert_close(batch.seq_lens_cpu, seq_lens)
                attn_backend.init_forward_metadata.assert_not_called()

    def _megamoe_no_prefill_cp(self, graph_has_dp_gather=False):
        return (
            mock.patch.object(
                forward_batch_info,
                "get_flags",
                return_value=SimpleNamespace(
                    dp=SimpleNamespace(prefill_graph_has_dp_gather=graph_has_dp_gather)
                ),
            ),
            mock.patch(
                "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                return_value=MoeA2ABackend.MEGAMOE,
            ),
            mock.patch(
                "sglang.srt.layers.attention.dsa.utils.is_dsa_enable_prefill_cp",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.layers.cp.utils.is_mla_cp_enabled",
                return_value=False,
            ),
        )

    def test_megamoe_idle_rank_without_graph_gather_keeps_sum_len(self):
        # Without a DP gather in the graph, an eager idle rank matches the
        # replaying peers, so the sparse batch keeps per-rank buckets.
        runner = self._make_runner()
        flags, a2a, dsa_cp, mla_cp = self._megamoe_no_prefill_cp()
        with flags, a2a, dsa_cp, mla_cp:
            self.assertTrue(prefill_graph_tolerates_sum_len())
            self.assertFalse(
                runner._has_inactive_dp_rank(
                    SimpleNamespace(global_num_tokens_cpu=[8, 0])
                )
            )

    def test_megamoe_all_ranks_busy_keeps_per_rank_buckets(self):
        runner = self._make_runner()
        flags, a2a, dsa_cp, mla_cp = self._megamoe_no_prefill_cp()
        with flags, a2a, dsa_cp, mla_cp:
            self.assertTrue(prefill_graph_tolerates_sum_len())
            self.assertFalse(
                runner._has_inactive_dp_rank(
                    SimpleNamespace(global_num_tokens_cpu=[8, 16])
                )
            )

    def test_megamoe_graph_with_dp_gather_forces_shared_bucket(self):
        # A DP gather captured in the graph has fixed MAX_LEN geometry; per-rank
        # buckets or an eager idle rank would deadlock its all_gather.
        runner = self._make_runner()
        flags, a2a, dsa_cp, mla_cp = self._megamoe_no_prefill_cp(
            graph_has_dp_gather=True
        )
        with flags, a2a, dsa_cp, mla_cp:
            self.assertFalse(prefill_graph_tolerates_sum_len())
            self.assertTrue(
                runner._has_inactive_dp_rank(
                    SimpleNamespace(global_num_tokens_cpu=[8, 0])
                )
            )

    def test_rejects_context_above_fixed_maximum(self):
        runner = self._make_runner()
        runner.max_context_size = 700

        much_shorter = self._make_forward_batch(4)
        much_shorter.seq_lens_cpu.fill_(200)
        self.assertTrue(runner.can_run_graph(much_shorter))

        uncovered = self._make_forward_batch(4)
        uncovered.seq_lens_cpu.fill_(701)
        self.assertFalse(runner.can_run_graph(uncovered))

    def test_unsupported_path_ignores_max_context_size(self):
        runner = self._make_runner()
        runner.max_context_size = 1024

        with self.assertLogs(runner_module.logger, level="WARNING") as logs:
            runner._ignore_max_context_size("test path")

        self.assertIsNone(runner.max_context_size)
        self.assertIn("fixed metadata extent", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
