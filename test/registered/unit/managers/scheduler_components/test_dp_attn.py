import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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


def _sync_info(**overrides):
    kwargs = dict(
        dp_size=2,
        tp_size=1,
        cp_size=1,
        num_tokens=1,
        num_tokens_for_logprob=1,
        can_run_decode_cuda_graph=True,
        can_run_prefill_cuda_graph=True,
        is_extend_in_batch=False,
        local_can_run_tbo=True,
        local_forward_mode=ForwardMode.DECODE.value,
    )
    kwargs.update(overrides)
    return dp_attn.MLPSyncBatchInfo(**kwargs)


class TestDPMaxSeqLenSync(CustomTestCase):
    """Decode graphs keyed by context length must be chosen from the longest
    sequence of the whole attention-DP group, so the scheduler sync carries it."""

    def test_local_tensor_carries_max_seq_len(self):
        info = _sync_info(max_seq_len=37)
        self.assertEqual(int(info._get_local_tensor(device="cpu")[8]), 37)
        # an inactive rank contributes no length
        self.assertEqual(int(info._get_fallback_tensor(device="cpu")[8]), 0)
        info.finalize_local()
        self.assertEqual(info.global_max_seq_len, 37)

    def test_all_gather_takes_group_max(self):
        rows = {
            0: _sync_info(max_seq_len=100),
            1: _sync_info(num_tokens=0, max_seq_len=0),
        }

        def fake_all_gather(out, local, group=None):
            width = local.numel()
            for rank, info in rows.items():
                out[rank * width : (rank + 1) * width] = info._get_local_tensor(
                    device=local.device, dtype=local.dtype
                )

        tp_group = SimpleNamespace(active_ranks_cpu=torch.ones(2, dtype=torch.int64))
        with (
            patch.object(
                dp_attn.torch.distributed, "all_gather_into_tensor", fake_all_gather
            ),
            patch.object(dp_attn, "get_tp_group", return_value=tp_group),
        ):
            for rank, info in rows.items():
                info.all_gather(device="cpu", group=object())
                self.assertEqual(info.global_max_seq_len, 100, f"rank {rank}")
                self.assertEqual(info.global_num_tokens, [1, 0])

    def test_gathered_batch_gets_group_max(self):
        info = _sync_info(max_seq_len=5)
        info.global_num_tokens = [1, 0]
        info.global_num_tokens_for_logprob = [1, 0]
        info.global_max_seq_len = 100
        info.tbo_split_seq_index = None
        info.global_forward_mode = ForwardMode.DECODE
        idle_batch = SimpleNamespace()
        dp_attn._update_gather_batch(idle_batch, info, require_mlp_tp_gather=True)
        self.assertEqual(idle_batch.dp_max_seq_len, 100)

    def test_local_max_seq_len_sources(self):
        self.assertEqual(
            dp_attn._local_max_seq_len(
                SimpleNamespace(seq_lens_cpu=torch.tensor([3, 41, 7]))
            ),
            41,
        )
        self.assertEqual(
            dp_attn._local_max_seq_len(
                SimpleNamespace(
                    seq_lens_cpu=torch.empty(0, dtype=torch.int64), seq_lens=None
                )
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
