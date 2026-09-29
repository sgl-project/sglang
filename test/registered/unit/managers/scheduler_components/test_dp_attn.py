import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.cuda_graph_config import Backend  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (  # noqa: E402
    PrefillCudaGraphRunner,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDPAttnSchedulerMetadata(CustomTestCase):
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
    def _vote(self, mode):
        runner = Mock(spec=dp_attn.PrefillCudaGraphRunner)
        runner.enable_lora = False
        runner.max_context_size = None
        runner.can_replay_locally.return_value = True
        batch = SimpleNamespace(
            forward_mode=mode,
            extend_num_tokens=4,
            multimodal_inputs=None,
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

    def test_multimodal_rank_votes_every_rank_eager(self):
        image = SimpleNamespace(
            contains_mm_input=lambda: True,
            contains_image_inputs=lambda: True,
            contains_video_inputs=lambda: False,
            contains_audio_inputs=lambda: False,
        )
        for arch in (
            "Qwen4ExpForConditionalGeneration",
            "Qwen4ExpForCausalLMMTP",
        ):
            with self.subTest(arch=arch):
                runner = object.__new__(PrefillCudaGraphRunner)
                runner.prefill_backend_name = Backend.BREAKABLE
                runner._qwen_bcg_hc_sidechannel = arch.endswith("ConditionalGeneration")
                runner._qwen_bcg_pad_mtp_embeds = not runner._qwen_bcg_hc_sidechannel
                runner.model_runner = SimpleNamespace(
                    model_config=SimpleNamespace(
                        hf_config=SimpleNamespace(architectures=[arch]),
                        enable_multimodal=None,
                    )
                )
                runner._is_full_backend = False
                runner.enable_lora = False
                runner.has_mha_companion_layers = False
                runner._capture_chunked_prefix = False
                runner.max_context_size = None
                runner.max_num_tokens = 8
                runner.capture_num_tokens = [8]
                runner.capture_hidden_mode = CaptureHiddenMode.NULL
                batches = [
                    ForwardBatch(
                        forward_mode=ForwardMode.EXTEND,
                        batch_size=1,
                        input_ids=None,
                        req_pool_indices=None,
                        seq_lens=None,
                        seq_lens_sum=8,
                        out_cache_loc=None,
                        mm_inputs=items,
                    )
                    for items in ([image], [None])
                ]
                infos = []
                for batch in batches:
                    local = SimpleNamespace(
                        forward_mode=ForwardMode.EXTEND,
                        batch_size=lambda: 1,
                        extend_num_tokens=8,
                        input_embeds=None,
                        replace_embeds=None,
                        prefix_lens=[0],
                        return_logprob=False,
                        multimodal_inputs=batch.mm_inputs,
                    )
                    vote = dp_attn._local_prefill_cuda_graph_vote(
                        local_batch=local,
                        prefill_graph_runner=runner,
                        coordinated_prefill=True,
                        breakable_prefill=True,
                        spec_algorithm=SpeculativeAlgorithm.NONE,
                        model_config=runner.model_runner.model_config,
                    )
                    infos.append(
                        dp_attn.MLPSyncBatchInfo(
                            dp_size=2,
                            tp_size=1,
                            cp_size=1,
                            num_tokens=8,
                            num_tokens_for_logprob=1,
                            can_run_decode_cuda_graph=False,
                            can_run_draft_cuda_graph=False,
                            can_run_prefill_cuda_graph=vote,
                            is_extend_in_batch=True,
                            local_can_run_tbo=False,
                            local_forward_mode=ForwardMode.EXTEND.value,
                        )
                    )
                self.assertEqual(
                    [info.can_run_prefill_cuda_graph for info in infos], [False, True]
                )
                gathered = torch.stack(
                    [info._get_local_tensor(device="cpu") for info in infos]
                )
                with (
                    patch.object(
                        dp_attn,
                        "all_gather_single",
                        side_effect=lambda output, *args, **kwargs: output.copy_(
                            gathered.flatten()
                        ),
                    ),
                    patch.object(
                        dp_attn,
                        "get_parallel",
                        return_value=SimpleNamespace(
                            tp_group=SimpleNamespace(active_ranks_cpu=torch.ones(2))
                        ),
                    ),
                ):
                    for info, batch in zip(infos, batches):
                        info.all_gather(device="cpu", group=None)
                        self.assertFalse(info.can_run_prefill_cuda_graph)
                        batch.global_num_tokens_cpu = info.global_num_tokens
                        batch.can_run_dp_prefill_cuda_graph = (
                            info.can_run_prefill_cuda_graph
                        )
                        self.assertFalse(runner.can_run_graph(forward_batch=batch))


if __name__ == "__main__":
    unittest.main()
