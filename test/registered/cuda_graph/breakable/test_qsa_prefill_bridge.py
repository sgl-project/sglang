"""QSA-specific metadata and MTP side channels across breakable prefill replay."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
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
from sglang.srt.models import qwen4_exp
from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestQSAPrefillBridge(unittest.TestCase):
    def make_runner(self, draft=False):
        runner = object.__new__(PrefillCudaGraphRunner)
        runner.backend = object.__new__(BreakableCudaGraphBackend)
        runner._is_full_backend = False
        runner._qwen_bcg_mtp_embeddings = None
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
                self, hidden, positions, batch, **kwargs
            ):
                seen.append(batch)
                return (hidden[: batch.rows, :2] + batch.prefix).to(torch.int32)

        layer = Layer()
        x = torch.zeros((8, 2), device="cuda")
        positions = torch.arange(8, device="cuda")
        result = torch.empty_like(x, dtype=torch.int32)
        graph = BreakableCUDAGraph()
        with patch.object(
            qwen4_exp, "get_tc_piecewise_forward_context", return_value=context
        ):
            qwen4_exp._breakable_qsa_indexer(layer, x, positions)
            torch.cuda.synchronize()
            with BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()):
                result.copy_(qwen4_exp._breakable_qsa_indexer(layer, x + 1, positions))
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
                    output.mm_input_embeds, raw, 128
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
                layer, input_ids, padded_batch, None
            )

        draft.model_runner.model = SimpleNamespace(forward=forward)
        actual = draft._execute_body_capture(batch, static, 128, 122, None)
        torch.testing.assert_close(actual, layer.model.embed_tokens(ids))
        self.assertEqual(actual.shape[0], 128)
        self.assertEqual(output.hidden_states.shape[0], 122)


if __name__ == "__main__":
    unittest.main()
