import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDPAttnSchedulerMetadata(CustomTestCase):
    def test_skip_all_gather_policy(self):
        with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False):
            self.assertTrue(dp_attn.should_skip_scheduler_all_gather(dp_size=1))
            self.assertFalse(dp_attn.should_skip_scheduler_all_gather(dp_size=2))
        with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(True):
            self.assertTrue(dp_attn.should_skip_scheduler_all_gather(dp_size=2))

    def test_dp1_skip_preserves_local_tbo_metadata(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=lambda: 4,
        )
        tbo_preparer = Mock()
        tbo_preparer.prepare_all_gather.return_value = (
            True,
            ForwardMode.DECODE.value,
        )
        tbo_preparer.compute_output.return_value = (2, ForwardMode.DECODE)

        with (
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            patch.object(dp_attn, "TboDPAttentionPreparer", return_value=tbo_preparer),
            patch.object(dp_attn, "world_dp_gather_enabled", return_value=False),
            patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
            patch.object(dp_attn.MLPSyncBatchInfo, "all_gather") as all_gather,
        ):
            result = dp_attn.prepare_mlp_sync_batch_raw(
                batch,
                model_runner=SimpleNamespace(
                    prefill_cuda_graph_runner=None,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    model_config=object(),
                ),
                dp_size=1,
                attn_tp_size=4,
                attn_cp_size=1,
                tp_group=SimpleNamespace(
                    device_group=object(), device="cpu", cpu_group=object()
                ),
                get_idle_batch=Mock(
                    side_effect=AssertionError("DP1 must not emit idle batch")
                ),
                disable_cuda_graph=False,
                require_mlp_tp_gather=False,
                disable_overlap_schedule=True,
                offload_tags=set(),
            )

        all_gather.assert_not_called()
        self.assertEqual(result.global_num_tokens, [4])
        self.assertEqual(result.tbo_split_seq_index, 2)
        self.assertEqual(result.global_forward_mode, ForwardMode.DECODE)
        self.assertEqual(result.recv_skipper_forward_mode, ForwardMode.DECODE)
        self.assertEqual(
            tbo_preparer.compute_output.call_args.args[0].tolist(),
            [[1, ForwardMode.DECODE.value]],
        )


class TestDecodeToExtendConversionVote(CustomTestCase):
    """A decode batch votes for the prefill graph only when its 1-token extend
    view can represent every row. Beam requests cannot: the converted batch
    takes the prefill result path, which commits them per-req instead of
    through the batch decode fold, and member rows carry no req."""

    def _vote(self, *, beam):
        runner = Mock(spec=dp_attn.PrefillCudaGraphRunner)
        runner.enable_lora = False
        runner.can_replay_locally.return_value = True
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=lambda: 2,
            return_logprob=False,
            has_grammar=False,
            reqs=[
                SimpleNamespace(beam_group=Mock() if beam else None),
                SimpleNamespace(beam_group=None),
            ],
        )
        with (
            patch.object(
                dp_attn, "get_moe_a2a_backend", return_value=Mock(is_none=lambda: True)
            ),
            patch.object(dp_attn, "uses_ssm_state", return_value=False),
            patch.object(
                dp_attn,
                "get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
            patch.object(
                dp_attn,
                "get_exec",
                return_value=SimpleNamespace(
                    overlap=SimpleNamespace(enable_two_batch_overlap=False)
                ),
            ),
            patch.object(dp_attn, "get_cp_strategy", return_value=None),
        ):
            return dp_attn._local_prefill_cuda_graph_vote(
                local_batch=batch,
                prefill_graph_runner=runner,
                coordinated_prefill=True,
                breakable_prefill=True,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                model_config=object(),
            )

    def test_plain_decode_batch_votes_for_conversion(self):
        self.assertTrue(self._vote(beam=False))

    def test_beam_request_blocks_conversion(self):
        self.assertFalse(self._vote(beam=True))


if __name__ == "__main__":
    unittest.main()
