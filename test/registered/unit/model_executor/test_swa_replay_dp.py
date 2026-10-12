"""Late-layer DP geometry must be restored before logits/DSpark processing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v41_features
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_dp_global_num_tokens,
    get_global_dp_buffer_len,
    get_local_dp_buffer_len,
    set_dp_buffer_len_from_batch,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.swa_replay_dp import (
    enter_late_layer_dp,
    exit_late_layer_dp,
    late_layer_dp_counts,
)
from sglang.srt.runtime_context import get_context, get_parallel, override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSWAReplayDP(CustomTestCase):
    @override_platform(is_cuda=True, is_hip=False)
    def test_validation_allows_only_supported_dp_geometry(self):
        for backend, tp, dp, attention in (
            ("megamoe", 2, 2, "flashmla"),
            ("none", 2, 2, "flashmla"),
            ("megamoe", 4, 2, "flashmla"),
            ("deepep", 2, 2, "flashmla"),
            ("megamoe", 2, 2, "trtllm"),
        ):
            with self.subTest(backend=backend, tp=tp, dp=dp, attention=attention):
                args = ServerArgs(
                    model_path="dummy",
                    cuda_graph_config=CudaGraphConfig(
                        prefill=PhaseConfig(backend=Backend.DISABLED)
                    ),
                    tp_size=tp,
                    attn_dp_size=dp,
                    moe_a2a_backend=backend,
                    dsv4_attn_backend=attention,
                    enable_decoder_swa_bounded_replay=True,
                )
                args._model_config = SimpleNamespace(
                    hf_config=SimpleNamespace(model_type="deepseek_v41")
                )
                if (
                    tp == dp
                    and backend in ("none", "megamoe")
                    and attention == "flashmla"
                ):
                    validate_deepseek_v41_features(args)
                else:
                    with self.assertRaisesRegex(ValueError, "DP attention|TRT-LLM"):
                        validate_deepseek_v41_features(args)

    def test_decode_peer_stays_eager_on_a_prefill_step(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        graph = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        graph.require_mlp_sync = True
        graph.model_runner = SimpleNamespace(is_draft_worker=False)
        with get_context().override_server_args(enable_decoder_swa_bounded_replay=True):
            self.assertFalse(
                graph.can_run_graph(SimpleNamespace(is_extend_in_batch=True))
            )

    def test_tail_geometry_and_idle_mask_are_transactional(self):
        for rank, real_rows in ((0, 1024), (1, 0)):
            with (
                self.subTest(rank=rank),
                get_context().override_server_args(tp_size=2, attn_dp_size=2),
                get_parallel().override(attn_dp_rank=rank, attn_tp_size=1),
            ):
                fb = ForwardBatch(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=1,
                    input_ids=torch.zeros(1024, dtype=torch.long),
                    req_pool_indices=torch.tensor([0]),
                    seq_lens=torch.tensor([1024]),
                    out_cache_loc=torch.arange(1024),
                    seq_lens_sum=1024,
                    global_num_tokens_cpu=[1024, 1024],
                    global_num_tokens_gpu=torch.tensor([1024, 1024]),
                    global_num_tokens_padded_cpu=[1024, 1024],
                    global_num_tokens_for_logprob_cpu=[1024, 0],
                    global_dp_buffer_len=2048,
                    dp_padding_mode=DpPaddingMode.MAX_LEN,
                    dp_local_start_pos=torch.tensor(rank * 1024),
                    dp_local_num_tokens=torch.tensor(1024),
                    global_num_token_non_padded=torch.tensor(real_rows),
                    global_num_token_non_padded_cpu=real_rows,
                    num_token_non_padded=torch.tensor(real_rows),
                )
                set_dp_buffer_len_from_batch(fb)
                original = dict(vars(fb))
                saved = enter_late_layer_dp(fb, [128, 0])
                self.assertEqual(get_dp_global_num_tokens(), [128, 0])
                self.assertEqual(get_global_dp_buffer_len(), 128)
                self.assertEqual(get_local_dp_buffer_len(), 128 if rank == 0 else 0)
                self.assertEqual(
                    fb.num_token_non_padded.item(), 128 if rank == 0 else 0
                )
                self.assertEqual(fb.global_num_tokens_for_logprob_cpu, [128, 0])
                self.assertIsNone(fb.dp_local_start_pos)
                exit_late_layer_dp(fb, saved)
                for name, value in original.items():
                    self.assertIs(getattr(fb, name), value, name)
                self.assertEqual(get_dp_global_num_tokens(), [1024, 1024])
                self.assertEqual(get_local_dp_buffer_len(), 1024)

    def test_every_peer_publishes_its_own_late_rows(self):
        class Collective:
            def all_gather_into_tensor(self, output, local):
                self.local = local.item()
                output.copy_(torch.tensor([128, 5]))

        collective = Collective()
        fb = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=0,
            input_ids=torch.empty(0),
            req_pool_indices=torch.empty(0),
            seq_lens=torch.empty(0),
            out_cache_loc=torch.empty(0),
            seq_lens_sum=0,
            global_num_tokens_cpu=[1024, 5],
            global_num_tokens_gpu=torch.tensor([1024, 5]),
        )
        with (
            get_parallel().override(tp_group=collective),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
        ):
            self.assertEqual(late_layer_dp_counts(fb, 5), [128, 5])
            self.assertEqual(collective.local, 5)
            fb.global_num_tokens_padded_cpu = [128, 5]
            self.assertIsNone(late_layer_dp_counts(fb, 5))


if __name__ == "__main__":
    unittest.main()
