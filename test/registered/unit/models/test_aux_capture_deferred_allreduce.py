import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers import communicator as comm
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.bailing_moe import BailingMoEModel
from sglang.srt.models.bailing_moe_v3 import BailingMoELinearModel
from sglang.srt.models.glm4_moe import Glm4MoeModel
from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteModel
from sglang.srt.models.glm5_next import Glm5NextModel
from sglang.srt.models.gpt_oss import GptOssModel
from sglang.srt.models.laguna import LagunaModel
from sglang.srt.models.qwen3_vl_moe import Qwen3MoeLLMModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

NUM_LAYERS = 4
MODELS = (
    BailingMoEModel,
    BailingMoELinearModel,
    Glm4MoeModel,
    Glm4MoeLiteModel,
    Glm5NextModel,
    GptOssModel,
    LagunaModel,
    Qwen3MoeLLMModel,
)


def all_reduce(hidden_states):
    return hidden_states * 2


# The group a deferred FFN output owes its sum over.
GROUP = SimpleNamespace(all_reduce=all_reduce)


class DeferringLayer(nn.Module):
    def __init__(self, defer, return_topk=False):
        super().__init__()
        self.return_topk = return_topk
        self.layer_communicator = comm.LayerCommunicator.__new__(comm.LayerCommunicator)
        self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer = (
            lambda batch: defer
        )
        self.layer_communicator.should_defer_ffn_reduction = lambda batch: defer
        self.layer_communicator.is_last_layer = False
        self.layer_communicator._postprocess_scatters_to_local_tokens = False
        self.layer_communicator.ffn_reduction_group = lambda: GROUP
        self.layer_communicator.should_use_reduce_scatter = lambda batch: False
        self.layer_communicator.postprocess_layer = lambda hidden, residual, batch: (
            all_reduce(hidden),
            residual,
        )

    def forward(
        self,
        positions=None,
        hidden_states=None,
        forward_batch=None,
        residual=None,
        *args,
        **kwargs,
    ):
        hidden_states = comm.reduce_output(hidden_states)
        if residual is None:
            residual = hidden_states.clone()
        else:
            # later norms mutate the residual; captured snapshots must stay intact
            residual.add_(hidden_states)
        with self.layer_communicator.ffn_exit(forward_batch) as ffn_exit:
            partial = torch.full_like(hidden_states, 0.5)
        hidden_states, residual = ffn_exit.finish(partial, residual)
        if self.return_topk:
            return hidden_states, residual, None
        return hidden_states, residual


class SumNorm(nn.Module):
    def forward(self, hidden_states, residual=None, **kwargs):
        if residual is None:
            return hidden_states
        return hidden_states + residual, residual


def build_model(model_cls, *, defer, capture):
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.start_layer = 0
    model.end_layer = NUM_LAYERS
    model.first_k_dense_replace = 0
    model.layers_to_capture = [1, 2, 3] if capture else []
    model.capture_aux_hidden_states = capture
    model.use_hf_deepstack_order = False
    model.dflash_capture = capture
    model.enable_a2a_moe = False
    model.config = SimpleNamespace(mhc=False)
    if model_cls is BailingMoELinearModel:
        # Bailing-v3 captures after the layer; other models use boundary indices
        model.layers_to_capture = [0, 1, 2] if capture else []
    model.layers = nn.ModuleList(
        DeferringLayer(
            defer and i < NUM_LAYERS - 1, return_topk=model_cls is Glm5NextModel
        )
        for i in range(NUM_LAYERS)
    )
    model.norm = SumNorm()
    return model


class TestAuxCaptureDeferredAllreduce(CustomTestCase):
    def test_capture_matches_eager_reduction(self):
        inputs = torch.tensor([[0.25, -0.5, 0.75, 1.0], [1.5, 2.0, -1.0, 0.0]])
        batch = SimpleNamespace(
            can_run_tbo=False,
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=SimpleNamespace(need_capture=lambda: True),
        )
        for model_cls in MODELS:
            for defer in (False, True):
                for capture in (False, True):
                    with (
                        self.subTest(
                            model=model_cls.__name__, defer=defer, capture=capture
                        ),
                        patch.object(
                            GROUP, "all_reduce", side_effect=all_reduce
                        ) as reduce,
                    ):
                        model = build_model(model_cls, defer=defer, capture=capture)
                        result = model(
                            input_ids=None,
                            positions=None,
                            forward_batch=batch,
                            input_embeds=inputs.clone(),
                        )
                        output, snapshots = result if capture else (result, [])
                        torch.testing.assert_close(output, inputs + NUM_LAYERS)
                        self.assertEqual(len(snapshots), 3 if capture else 0)
                        for boundary, snapshot in enumerate(snapshots, 1):
                            torch.testing.assert_close(snapshot, inputs + boundary)
                        self.assertEqual(
                            reduce.call_count, NUM_LAYERS - 1 if defer else 0
                        )


if __name__ == "__main__":
    unittest.main()
