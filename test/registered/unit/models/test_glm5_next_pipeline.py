import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.models import glm5_next
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _RecordingLayer(torch.nn.Module):
    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        residual,
        zero_allocator,
        gemm_output_zero_allocator,
        prev_topk_indices=None,
    ):
        self.received_residual = residual
        return hidden_states + 1, residual, None


class _RecordingNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, residual=None):
        self.calls += 1
        self.received_residual = residual
        if residual is None:
            return hidden_states * 2
        return (hidden_states + residual) * 2, residual


class TestGlm5NextPipeline(CustomTestCase):
    def setUp(self):
        super().setUp()
        # Avoid the process-global expert recorder; tensor allocation remains real.
        graph_backend = patch.object(
            glm5_next, "check_cuda_graph_backend", return_value=True
        )
        graph_backend.start()
        self.addCleanup(graph_backend.stop)
        self.hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        self.residual = torch.full_like(self.hidden, 3)

    @staticmethod
    def _make_model(*, mhc, first=False, last=False):
        # Exercise the production forward without constructing GPU decoder layers.
        model = glm5_next.Glm5NextModel.__new__(glm5_next.Glm5NextModel)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(mhc=mhc)
        model.pp_group = SimpleNamespace(is_first_rank=first, is_last_rank=last)
        model.start_layer, model.end_layer = 0, 1
        model.first_k_dense_replace = 1
        model.layers = torch.nn.ModuleList([_RecordingLayer()])
        model.norm = _RecordingNorm()
        model.embed_tokens = torch.nn.Embedding.from_pretrained(
            torch.arange(16, dtype=torch.float32).reshape(2, 8)
        )
        model.layers_to_capture = []
        model.dflash_capture = False
        model.enable_a2a_moe = False
        return model

    @staticmethod
    def _forward(model, proxy=None, *, idle=False, input_embeds=None):
        input_ids = torch.arange(0 if idle else 2)
        batch = SimpleNamespace(
            can_run_tbo=False,
            forward_mode=ForwardMode.IDLE if idle else ForwardMode.EXTEND,
        )
        return model(
            input_ids,
            positions=input_ids,
            forward_batch=batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=proxy,
        )

    def test_mhc_first_stage_sends_only_hidden_states(self):
        model = self._make_model(mhc=True, first=True)

        output = self._forward(model)

        self.assertIsInstance(output, PPProxyTensors)
        self.assertEqual(set(output.tensors), {"hidden_states"})
        torch.testing.assert_close(output["hidden_states"], self.hidden + 1)
        self.assertIsNone(model.layers[0].received_residual)
        self.assertEqual(model.norm.calls, 0)

    def test_mhc_middle_stage_forwards_hidden_states_without_residual(self):
        model = self._make_model(mhc=True)

        output = self._forward(model, PPProxyTensors({"hidden_states": self.hidden}))

        self.assertEqual(set(output.tensors), {"hidden_states"})
        torch.testing.assert_close(output["hidden_states"], self.hidden + 1)
        self.assertIsNone(model.layers[0].received_residual)
        self.assertEqual(model.norm.calls, 0)

    def test_mhc_last_stage_normalizes_without_residual(self):
        model = self._make_model(mhc=True, last=True)

        output = self._forward(model, PPProxyTensors({"hidden_states": self.hidden}))

        torch.testing.assert_close(output, (self.hidden + 1) * 2)
        self.assertIsNone(model.layers[0].received_residual)
        self.assertIsNone(model.norm.received_residual)
        self.assertEqual(model.norm.calls, 1)

    def test_plain_middle_stage_preserves_residual(self):
        model = self._make_model(mhc=False)
        proxy = PPProxyTensors(
            {"hidden_states": self.hidden, "residual": self.residual}
        )

        output = self._forward(model, proxy)

        self.assertEqual(set(output.tensors), {"hidden_states", "residual"})
        self.assertIs(output["residual"], self.residual)
        self.assertIs(model.layers[0].received_residual, self.residual)
        torch.testing.assert_close(output["hidden_states"], self.hidden + 1)

    def test_plain_last_stage_normalizes_with_residual(self):
        model = self._make_model(mhc=False, last=True)
        proxy = PPProxyTensors(
            {"hidden_states": self.hidden, "residual": self.residual}
        )

        output = self._forward(model, proxy)

        torch.testing.assert_close(output, (self.hidden + 1 + self.residual) * 2)
        self.assertIs(model.norm.received_residual, self.residual)

    def test_plain_stage_requires_residual_field(self):
        for last in (False, True):
            with self.subTest(last=last):
                model = self._make_model(mhc=False, last=last)
                with self.assertRaisesRegex(KeyError, "residual"):
                    self._forward(model, PPProxyTensors({"hidden_states": self.hidden}))

    def test_single_stage_keeps_embedding_and_input_embed_paths(self):
        for mhc in (False, True):
            for use_input_embeds in (False, True):
                with self.subTest(mhc=mhc, input_embeds=use_input_embeds):
                    model = self._make_model(mhc=mhc, first=True, last=True)
                    embeds = self.hidden + 5 if use_input_embeds else None

                    output = self._forward(model, input_embeds=embeds)

                    expected = embeds if use_input_embeds else self.hidden
                    torch.testing.assert_close(output, (expected + 1) * 2)
                    self.assertIsNone(model.layers[0].received_residual)
                    self.assertEqual(model.norm.calls, 1)

    def test_mhc_idle_pipeline_skips_final_norm(self):
        for last in (False, True):
            with self.subTest(last=last):
                model = self._make_model(mhc=True, last=last)
                empty = self.hidden[:0]

                output = self._forward(
                    model, PPProxyTensors({"hidden_states": empty}), idle=True
                )

                if not last:
                    self.assertEqual(set(output.tensors), {"hidden_states"})
                    output = output["hidden_states"]
                torch.testing.assert_close(output, empty)
                self.assertEqual(model.norm.calls, 0)


if __name__ == "__main__":
    unittest.main()
