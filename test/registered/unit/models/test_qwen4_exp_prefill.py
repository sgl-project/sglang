import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.managers.scheduler_components import dp_attn
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.srt.models import qwen4_exp
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_batch(lengths, mm_inputs=None):
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(lengths),
        input_ids=torch.arange(1, 9),
        req_pool_indices=torch.arange(1, len(lengths) + 1),
        seq_lens=torch.tensor(lengths),
        seq_lens_sum=sum(lengths),
        out_cache_loc=torch.arange(1, 9),
        extend_seq_lens=torch.tensor(lengths),
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens_cpu=[0] * len(lengths),
        mm_inputs=mm_inputs,
    )


class TestQwen4ExpPrefill(unittest.TestCase):
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
                batches = [make_batch([8], [image]), make_batch([8], [None])]
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
                        self.assertFalse(runner.can_run_graph(batch))

    def test_ple_live_layout_and_state_at_layer_two(self):
        context = SimpleNamespace(forward_batch=make_batch([8]))
        history = torch.zeros((3, 2), dtype=torch.long)
        conv_state = torch.zeros((3, 1, 9))
        pool = SimpleNamespace(
            get_mamba_indices=lambda indices: indices,
            get_ngram_context=lambda indices: history[indices],
            set_ngram_context=lambda indices, values: history.index_copy_(
                0, indices, values
            ),
            short_conv_layer_cache=lambda layer_id: conv_state,
        )
        layouts = []

        class PLE:
            layer_id = 2
            conv_channels = 1
            short_conv_dilation = 3
            short_conv_state_len = 9
            conv1d = SimpleNamespace(weight=torch.ones((1, 1, 4)))

            def start_prefetch(self, *, batch, forward_batch):
                self.prefetched = batch

            def __call__(self, *, hidden_states, forward_batch, batch):
                assert self.prefetched is batch
                layouts.append(batch.state_indices[batch.req_indices].tolist())
                output = qwen4_exp.Qwen4ExpPLELayer._short_conv(
                    self=self,
                    x=hidden_states[: batch.processed_tokens],
                    forward_batch=forward_batch,
                    batch=batch,
                )
                return qwen4_exp._pad_token_rows(
                    x=output, total_tokens=batch.physical_tokens
                )

        ple = PLE()
        capture = SimpleNamespace(
            _end_current_segment=Mock(),
            _begin_new_segment=Mock(),
            _barrier_fn=None,
            cuda_graph=SimpleNamespace(_break_fns=[]),
        )
        hidden = torch.arange(1, 9, dtype=torch.float32).view(8, 1)
        with (
            patch.object(qwen4_exp, "get_req_to_token_pool", return_value=pool),
            patch.object(
                qwen4_exp, "get_tc_piecewise_forward_context", return_value=context
            ),
        ):
            token = bcg._current_capture_var.set(capture)
            try:
                batch = qwen4_exp._breakable_prepare_ple_batch(
                    input_ids=context.forward_batch.input_ids,
                    ngram_size=3,
                    ngram_eos_token_id=0,
                )
                qwen4_exp._breakable_ple_prefetch(ple=ple, batch=batch)
                output = qwen4_exp._breakable_ple_forward(
                    ple=ple, hidden_states=hidden, batch=batch
                )
                qwen4_exp._breakable_commit_ple_batch(batch=batch)
            finally:
                bcg._current_capture_var.reset(token)
            self.assertEqual(layouts[-1], [1] * 8)
            context.forward_batch = make_batch([4, 4])
            history.zero_()
            conv_state.zero_()
            conv_state[1].fill_(10)
            conv_state[2].fill_(20)
            expected = torch.cat(
                [
                    F.silu(
                        F.conv1d(
                            torch.cat(
                                [
                                    conv_state[i : i + 1],
                                    hidden[4 * (i - 1) : 4 * i].T.unsqueeze(0),
                                ],
                                dim=-1,
                            ),
                            ple.conv1d.weight,
                            dilation=3,
                        )
                    ).view(4, 1)
                    for i in (1, 2)
                ]
            )
            for replay in capture.cuda_graph._break_fns:
                replay()
            self.assertEqual(layouts[-1], [1] * 4 + [2] * 4)
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(history[1:], torch.tensor([[3, 4], [7, 8]]))
            torch.testing.assert_close(conv_state[1, 0, -4:], hidden[:4, 0])
            torch.testing.assert_close(conv_state[2, 0, -4:], hidden[4:, 0])


if __name__ == "__main__":
    unittest.main()
