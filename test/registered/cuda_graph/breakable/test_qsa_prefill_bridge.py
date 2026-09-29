import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.srt.models import qwen4_exp
from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestQSAPrefillBridge(unittest.TestCase):
    def make_runner(self, draft=False):
        runner = object.__new__(PrefillCudaGraphRunner)
        runner.backend = object.__new__(BreakableCudaGraphBackend)
        runner._is_full_backend = False
        runner._qwen_bcg_mtp_embeddings = None
        runner._qwen_bcg_hc_sidechannel = not draft
        runner._qwen_bcg_pad_mtp_embeds = draft
        runner.capture_num_tokens = [128]
        runner.model_runner = SimpleNamespace(
            is_draft_worker=draft,
            spec_algorithm=SimpleNamespace(is_speculative=lambda: True),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=[
                        "Qwen4ExpForCausalLMMTP"
                        if draft
                        else "Qwen4ExpForConditionalGeneration"
                    ]
                )
            ),
        )
        return runner

    def test_live_length_bridge(self):
        context = SimpleNamespace(forward_batch=SimpleNamespace(rows=8, prefix=0))
        seen = []

        class Layer:
            def __init__(self):
                self._qsa_prefill_topk_bridge = None

            def _compute_qsa_topk_indices_eager(
                self, hidden_states, positions, forward_batch, **kwargs
            ):
                seen.append(forward_batch)
                return (
                    hidden_states[: forward_batch.rows, :2] + forward_batch.prefix
                ).to(torch.int32)

        layer = Layer()
        x = torch.zeros((8, 2), device="cuda")
        positions = torch.arange(8, device="cuda")
        result = torch.empty_like(x, dtype=torch.int32)
        graph = BreakableCUDAGraph()
        with patch.object(
            qwen4_exp, "get_tc_piecewise_forward_context", return_value=context
        ):
            qwen4_exp._breakable_qsa_indexer(
                layer=layer, hidden_states=x, positions=positions
            )
            torch.cuda.synchronize()
            with BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()):
                result.copy_(
                    qwen4_exp._breakable_qsa_indexer(
                        layer=layer, hidden_states=x + 1, positions=positions
                    )
                )
            for rows, prefix in ((3, 17), (8, 4)):
                context.forward_batch = SimpleNamespace(rows=rows, prefix=prefix)
                x.fill_(2)
                graph.replay()
                torch.cuda.synchronize()
                expected = torch.full_like(result, -1)
                expected[:rows].fill_(3 + prefix)
                torch.testing.assert_close(result, expected)
                self.assertIs(seen[-1], context.forward_batch)

    def test_hc_replay_and_embedding_bucket_boundaries(self):
        runner = self.make_runner()
        runner.layer_model = SimpleNamespace()
        x = torch.ones((128, 4), device="cuda")
        graph = BreakableCUDAGraph()
        torch.cuda.synchronize()
        with BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()):
            runner.layer_model.last_hc_hidden_states = (x + 2).repeat(1, 4)
            captured = runner._pack_qwen_bcg_hc_output(x + 1)
        for raw in (122, 128):
            with self.subTest(raw=raw):
                x.fill_(raw)
                runner.layer_model.last_hc_hidden_states = torch.zeros(
                    (1, 16), device="cuda"
                )
                graph.replay()
                hidden = runner._restore_qwen_bcg_hc_output(captured)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    runner.layer_model.last_hc_hidden_states,
                    torch.full((128, 16), raw + 2.0, device="cuda"),
                )
                runner.raw_num_tokens = raw
                output = runner._trim_logits_output(
                    LogitsProcessorOutput(
                        next_token_logits=None, hidden_states=hidden, mm_input_embeds=x
                    )
                )
                self.assertEqual(output.hidden_states.shape[0], raw)
                padded = runner._pad_qwen_bcg_mtp_embeddings(
                    live=output.mm_input_embeds,
                    raw_num_tokens=raw,
                    static_num_tokens=128,
                )
                torch.testing.assert_close(padded[:raw], x[:raw])
                torch.testing.assert_close(padded[raw:], torch.zeros_like(padded[raw:]))
                torch.testing.assert_close(x, torch.full_like(x, raw))
                if raw == 128:
                    self.assertIs(padded, output.mm_input_embeds)

    def test_pure_text_draft_embedding_fallback(self):
        target, draft = self.make_runner(), self.make_runner(draft=True)
        target.raw_num_tokens = 122
        output = target._trim_logits_output(
            LogitsProcessorOutput(
                next_token_logits=None,
                hidden_states=torch.ones((128, 16), device="cuda"),
                mm_input_embeds=None,
            )
        )
        self.assertIsNone(output.mm_input_embeds)
        draft.layer_model = SimpleNamespace(forward=lambda: None)
        draft._input_embeds_arg_idx = None
        draft.buffer_registry = SimpleNamespace(has_slot=lambda name: False)
        draft._prefill_forward_context = lambda *args, **kwargs: nullcontext()
        layer = SimpleNamespace(
            model=SimpleNamespace(
                embed_tokens=torch.nn.Embedding(128, 4, device="cuda")
            )
        )
        batch = SimpleNamespace(mm_input_embeds=output.mm_input_embeds)
        ids = torch.arange(128, device="cuda")
        static = SimpleNamespace(
            input_ids=ids,
            positions=ids,
            forward_mode=ForwardMode.EXTEND,
            contains_mm_inputs=lambda: False,
        )

        def forward(input_ids, positions, padded_batch, **kwargs):
            self.assertIsNone(padded_batch.mm_input_embeds)
            return Qwen4ExpForCausalLMMTP._prepare_input_embeds(
                self=layer,
                input_ids=input_ids,
                forward_batch=padded_batch,
                input_embeds=None,
            )

        draft.model_runner.model = SimpleNamespace(forward=forward)
        actual = draft._execute_body_capture(
            forward_batch=batch,
            static_forward_batch=static,
            static_num_tokens=128,
            raw_num_tokens=122,
            shape_key=None,
        )
        torch.testing.assert_close(actual, layer.model.embed_tokens(ids))
        self.assertEqual(actual.shape[0], 128)
        self.assertEqual(output.hidden_states.shape[0], 122)


def make_batch(lengths):
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
    )


class TestPLEPrefillLayout(unittest.TestCase):
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
            reference_inputs = torch.cat(
                (conv_state[1:], hidden.view(2, 4, 1).transpose(1, 2)), dim=-1
            )
            expected = (
                F.silu(
                    F.conv1d(
                        input=reference_inputs, weight=ple.conv1d.weight, dilation=3
                    )
                )
                .transpose(1, 2)
                .reshape(8, 1)
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
