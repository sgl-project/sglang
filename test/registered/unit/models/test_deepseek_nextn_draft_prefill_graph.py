"""NextN draft prefill CUDA graph: the `layers` view and the replay-time seed publish."""

import contextlib
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    set_tc_piecewise_forward_context,
)
from sglang.srt.models import deepseek_nextn
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeShare:
    def __init__(self, topk_indices):
        self.topk_indices = topk_indices
        self.published = 0

    def publish(self):
        self.published += 1


def _nextn(*, exposes_decoder_as_layers):
    model = deepseek_nextn.DeepseekModelNextN.__new__(deepseek_nextn.DeepseekModelNextN)
    nn.Module.__init__(model)
    model.decoder = nn.Linear(2, 2)
    model.exposes_decoder_as_layers = exposes_decoder_as_layers
    return model


@contextlib.contextmanager
def _graph_scope(*, breakable, prefill_context):
    with contextlib.ExitStack() as stack:
        if breakable:
            stack.enter_context(enable_breakable_cuda_graph())
        if prefill_context:
            stack.enter_context(
                set_tc_piecewise_forward_context(
                    forward_batch=object(),
                    attention_layers=[],
                    quant_config=None,
                    moe_layers=[],
                    moe_fusions=[],
                )
            )
        yield


class TestNextNLayersView(CustomTestCase):
    def test_layers_hidden_when_off(self):
        # Prefill graph setup skips models without `layers`.
        self.assertFalse(hasattr(_nextn(exposes_decoder_as_layers=False), "layers"))

    def test_layers_view_stays_out_of_the_state_dict(self):
        model = _nextn(exposes_decoder_as_layers=True)
        self.assertEqual(len(model.layers), 1)
        self.assertIs(model.layers[0], model.decoder)
        self.assertEqual(sorted(model.state_dict()), ["decoder.bias", "decoder.weight"])

    def test_text_only_graph_declines_multimodal_batches(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._qwen_bcg_hc_sidechannel = False
        runner._qwen_bcg_mtp_draft = False
        runner._layer_model_text_only = (
            deepseek_nextn.DeepseekModelNextN.prefill_cuda_graph_text_only
        )
        self.assertFalse(
            runner.can_replay_locally(
                batch_size=1,
                num_tokens=64,
                input_embeds=None,
                replace_embeds=None,
                prefix_lens=None,
                is_target_verify=False,
                capture_hidden_mode=None,
                return_logprob=False,
                contains_mm_inputs=True,
            )
        )


class TestSeedPublishGate(CustomTestCase):
    """The replay-time seed publish reads the live batch from the tc_piecewise
    context, which only the prefill graph runner installs; every other scope
    (eager, the decode breakable backend, switch off, non-DSA drafts) must
    publish directly."""

    def _path(self, *, exposes, topk, breakable, prefill_context):
        breaks = []
        share = _FakeShare(topk)
        with patch.object(
            deepseek_nextn, "_publish_index_topk_share_at_replay", breaks.append
        ):
            with _graph_scope(breakable=breakable, prefill_context=prefill_context):
                _nextn(exposes_decoder_as_layers=exposes)._publish_index_topk_share(
                    share
                )
        return {(1, 0): "break", (0, 1): "publish"}[(len(breaks), share.published)]

    def test_paths(self):
        topk = torch.zeros(4, 8, dtype=torch.int32)
        cases = [
            ("eager", True, topk, False, False, "publish"),
            ("decode breakable backend", True, topk, True, False, "publish"),
            ("prefill graph", True, topk, True, True, "break"),
            ("prefill graph, switch off", False, topk, True, True, "publish"),
            ("prefill graph, non-DSA draft", True, None, True, True, "publish"),
        ]
        for name, exposes, tk, breakable, ctx, want in cases:
            with self.subTest(name):
                self.assertEqual(
                    self._path(
                        exposes=exposes,
                        topk=tk,
                        breakable=breakable,
                        prefill_context=ctx,
                    ),
                    want,
                )


if __name__ == "__main__":
    unittest.main()
