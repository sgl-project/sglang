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

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDPAttnSchedulerMetadata(CustomTestCase):
    def test_scheduler_counts_remain_global_and_owned_without_mlp_gather(self):
        for counts in ([4, 0], [0, 4], [4] * 8, [0] * 8):
            with self.subTest(counts=counts):
                batch = SimpleNamespace()
                info = dp_attn.MLPSyncBatchInfo(
                    num_dp_ranks=len(counts),
                    tp_size=1,
                    cp_size=1,
                    num_tokens=counts[0],
                    num_tokens_for_logprob=counts[0],
                    can_run_decode_cuda_graph=False,
                    can_run_draft_cuda_graph=False,
                    can_run_prefill_cuda_graph=False,
                    is_extend_in_batch=False,
                    local_can_run_tbo=False,
                    local_forward_mode=ForwardMode.IDLE.value,
                    global_num_tokens=list(counts),
                    global_num_tokens_for_logprob=list(counts),
                )
                dp_attn._update_gather_batch(batch, info, require_mlp_tp_gather=False)
                self.assertEqual(batch.global_num_tokens, [counts[0]])
                self.assertEqual(batch.scheduler_global_num_tokens, counts)
                info.global_num_tokens[0] += 1
                self.assertEqual(batch.scheduler_global_num_tokens, counts)

    def test_pdmux_peer_only_and_all_idle_participation(self):
        for counts, mode in (
            ([0, 4], ForwardMode.DECODE),
            ([0, 7], ForwardMode.EXTEND),
            ([0] * 7 + [2], ForwardMode.DECODE),
            ([0] * 8, ForwardMode.IDLE),
        ):
            with self.subTest(counts=counts, mode=mode):
                idle = SimpleNamespace(forward_mode=ForwardMode.IDLE, spec_info=None)
                get_idle = Mock(return_value=idle)
                tbo = Mock()
                tbo.prepare_all_gather.return_value = (False, ForwardMode.IDLE.value)
                tbo.compute_output.return_value = (None, None)

                def gather(info, **kwargs):
                    info.global_num_tokens = list(counts)
                    info.global_num_tokens_for_logprob = list(counts)
                    info.tp0_info_cpu = torch.zeros((len(counts), 9), dtype=torch.int64)
                    for rank, tokens in enumerate(counts):
                        info.tp0_info_cpu[rank, 5] = (
                            mode if tokens else ForwardMode.IDLE
                        ).value
                    info.is_extend_in_batch = mode.is_extend()

                with (
                    patch.object(
                        dp_attn,
                        "get_parallel",
                        return_value=SimpleNamespace(
                            num_dp_ranks=len(counts),
                            attn_tp_size=8 // len(counts),
                            attn_cp_size=1,
                            tp_group=SimpleNamespace(
                                device="cpu", device_group=object()
                            ),
                        ),
                    ),
                    patch.object(dp_attn, "TboDPAttentionPreparer", return_value=tbo),
                    patch.object(
                        dp_attn, "world_dp_gather_enabled", return_value=False
                    ),
                    patch.object(
                        dp_attn, "check_cuda_graph_backend", return_value=False
                    ),
                    patch.object(dp_attn.MLPSyncBatchInfo, "all_gather", gather),
                ):
                    result = dp_attn.prepare_mlp_sync_batch_raw(
                        None,
                        model_runner=SimpleNamespace(
                            prefill_cuda_graph_runner=None,
                            spec_algorithm=SpeculativeAlgorithm.NONE,
                            model_config=object(),
                        ),
                        get_idle_batch=get_idle,
                        disable_cuda_graph=False,
                        require_mlp_tp_gather=False,
                        disable_overlap_schedule=True,
                        offload_tags=set(),
                    )
                if any(counts):
                    self.assertIs(result, idle)
                    self.assertEqual(result.scheduler_global_num_tokens, counts)
                    self.assertEqual(result.global_num_tokens, [0])
                else:
                    self.assertIsNone(result)
                    get_idle.assert_not_called()

    def test_skip_all_gather_policy(self):
        with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False):
            self.assertTrue(dp_attn.should_skip_scheduler_all_gather(num_dp_ranks=1))
            self.assertFalse(dp_attn.should_skip_scheduler_all_gather(num_dp_ranks=2))
        with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(True):
            self.assertTrue(dp_attn.should_skip_scheduler_all_gather(num_dp_ranks=2))

    def test_dp1_skip_preserves_local_tbo_metadata(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=lambda: 4,
            spec_info=None,
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
            patch.object(
                dp_attn,
                "get_parallel",
                return_value=SimpleNamespace(
                    num_dp_ranks=1,
                    attn_tp_size=4,
                    attn_cp_size=1,
                    tp_group=SimpleNamespace(
                        device_group=object(), device="cpu", cpu_group=object()
                    ),
                ),
            ),
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
        runner.max_context_size = None
        runner.can_replay_locally.return_value = True
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=lambda: 2,
            return_logprob=False,
            has_grammar=False,
            multimodal_inputs=None,
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


class TestPrefillCudaGraphVote(CustomTestCase):
    def _vote(self, mode, *, mm=None):
        runner_cls = dp_attn.PrefillCudaGraphRunner
        runner = Mock(spec=runner_cls)
        runner.enable_lora = False
        runner.max_context_size = None
        runner._qwen_bcg_hc_sidechannel = True
        runner.can_replay_locally.side_effect = lambda **kwargs: (
            not kwargs["contains_mm_inputs"]
            or runner_cls.can_replay_locally(self=runner, **kwargs)
        )
        batch = SimpleNamespace(
            forward_mode=mode,
            extend_num_tokens=4,
            multimodal_inputs=mm,
            input_embeds=None,
            replace_embeds=None,
            prefix_lens=[1, 1],
            return_logprob=False,
            batch_size=lambda: 2,
        )
        vote = dp_attn._local_prefill_cuda_graph_vote(
            local_batch=batch,
            prefill_graph_runner=runner,
            coordinated_prefill=True,
            breakable_prefill=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            model_config=object(),
        )
        return vote, runner

    def test_extend_batch_votes_for_prefill_graph(self):
        vote, runner = self._vote(ForwardMode.EXTEND)

        self.assertTrue(vote)
        runner.can_replay_locally.assert_called_once()
        self.assertFalse(runner.can_replay_locally.call_args.kwargs["is_mixed"])

    def test_mixed_batch_delegates_to_runner_policy(self):
        vote, runner = self._vote(ForwardMode.MIXED)

        self.assertTrue(vote)
        runner.can_replay_locally.assert_called_once()
        self.assertTrue(runner.can_replay_locally.call_args.kwargs["is_mixed"])

    @patch.object(dp_attn, "get_parallel")
    @patch.object(dp_attn, "all_gather_single")
    def test_multimodal_ranks_vote_eager(self, gather, parallel):
        infos = [
            dp_attn.MLPSyncBatchInfo(
                num_dp_ranks=2,
                tp_size=1,
                cp_size=1,
                num_tokens=4,
                num_tokens_for_logprob=1,
                can_run_decode_cuda_graph=False,
                can_run_draft_cuda_graph=False,
                can_run_prefill_cuda_graph=self._vote(ForwardMode.EXTEND, mm=mm)[0],
                is_extend_in_batch=True,
                local_can_run_tbo=False,
                local_forward_mode=ForwardMode.EXTEND.value,
            )
            for mm in ([SimpleNamespace(contains_mm_input=lambda: True)], None)
        ]
        values = torch.cat([i._get_local_tensor(device="cpu") for i in infos])
        gather.side_effect = lambda output, *a, **kw: output.copy_(values)
        parallel.return_value.tp_group.active_ranks_cpu = torch.ones(2)
        for info in infos:
            info.all_gather(device="cpu", group=None)
            self.assertFalse(info.can_run_prefill_cuda_graph)


if __name__ == "__main__":
    unittest.main()
