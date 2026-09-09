import inspect
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
    Qwen4ExpVLModel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ple = None

    def forward(self, *, hidden_states, residual, **kwargs):
        return hidden_states + 1, residual


class _FailEmbedding(nn.Module):
    def forward(self, input_ids):
        raise AssertionError("a non-first PP stage must not run token embedding")


class _FinalMixer(nn.Module):
    def mix(self, hidden_states):
        return hidden_states.unflatten(-1, (2, 3)).mean(dim=-2), None


class TestQwen4ExpPipelineParallel(CustomTestCase):
    @staticmethod
    def _make_model(*, is_first_rank: bool, is_last_rank: bool):
        model = Qwen4ExpModel.__new__(Qwen4ExpModel)
        nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(
            is_first_rank=is_first_rank,
            is_last_rank=is_last_rank,
        )
        model.embed_tokens = _FailEmbedding()
        model.layers = nn.ModuleList([_FakeLayer()])
        model._start_layer = 0
        model._end_layer = 1
        model.has_ple = False
        model.hyper_connection_mixer = (
            _FinalMixer() if is_last_rank else PPMissingLayer()
        )
        return model

    @staticmethod
    def _forward_batch():
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
        )

    def test_public_forward_declares_pp_proxy_tensors(self):
        signature = inspect.signature(Qwen4ExpForConditionalGeneration.forward)

        self.assertIn("pp_proxy_tensors", signature.parameters)

    def test_nonfirst_stage_does_not_allocate_token_embedding(self):
        model = Qwen4ExpModel.__new__(Qwen4ExpModel)
        nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(is_first_rank=False)

        embedding = model._build_embed_tokens(SimpleNamespace())

        self.assertIsInstance(embedding, PPMissingLayer)

    def test_middle_stage_forwards_hyper_connection_stream(self):
        model = self._make_model(is_first_rank=False, is_last_rank=False)
        hidden_states = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        residual = torch.zeros_like(hidden_states)
        proxy = PPProxyTensors(
            {
                "hidden_states": hidden_states,
                "residual": residual,
            }
        )

        with patch(
            "sglang.srt.models.qwen4_exp.get_global_expert_distribution_recorder"
        ) as recorder:
            recorder.return_value.with_current_layer.return_value = nullcontext()
            output = model(
                input_ids=torch.zeros(2, dtype=torch.long),
                positions=torch.arange(2),
                forward_batch=self._forward_batch(),
                pp_proxy_tensors=proxy,
            )

        self.assertIsInstance(output, PPProxyTensors)
        torch.testing.assert_close(output["hidden_states"], hidden_states + 1)
        torch.testing.assert_close(output["residual"], residual)

    def test_last_stage_contracts_only_after_local_layers(self):
        model = self._make_model(is_first_rank=False, is_last_rank=True)
        hidden_states = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        residual = torch.zeros_like(hidden_states)

        with patch(
            "sglang.srt.models.qwen4_exp.get_global_expert_distribution_recorder"
        ) as recorder:
            recorder.return_value.with_current_layer.return_value = nullcontext()
            output, hc_hidden_states = model(
                input_ids=torch.zeros(2, dtype=torch.long),
                positions=torch.arange(2),
                forward_batch=self._forward_batch(),
                pp_proxy_tensors=PPProxyTensors(
                    {
                        "hidden_states": hidden_states,
                        "residual": residual,
                    }
                ),
            )

        torch.testing.assert_close(hc_hidden_states, hidden_states + 1)
        torch.testing.assert_close(
            output,
            (hidden_states + 1).unflatten(-1, (2, 3)).mean(dim=-2),
        )

    def test_vl_model_threads_pp_proxy_to_language_model(self):
        model = Qwen4ExpVLModel.__new__(Qwen4ExpVLModel)
        nn.Module.__init__(model)
        model.last_hc_hidden_states = None
        proxy = PPProxyTensors(
            {
                "hidden_states": torch.ones(2, 6),
                "residual": torch.zeros(2, 6),
            }
        )
        forward_batch = SimpleNamespace(input_ids=torch.ones(2, dtype=torch.long))

        with patch.object(Qwen4ExpModel, "forward", return_value=proxy) as forward:
            output = model(
                input_ids=torch.ones(2, dtype=torch.long),
                positions=torch.arange(2),
                forward_batch=forward_batch,
                pp_proxy_tensors=proxy,
            )

        self.assertIs(output, proxy)
        self.assertIs(forward.call_args.kwargs["pp_proxy_tensors"], proxy)

    def test_nonlocal_ple_and_final_mixer_weights_are_skipped(self):
        model = Qwen4ExpForConditionalGeneration.__new__(
            Qwen4ExpForConditionalGeneration
        )
        nn.Module.__init__(model)
        language_model = nn.Module()
        language_model.start_layer = 12
        language_model.end_layer = 24
        model.model = language_model
        model.config = SimpleNamespace(
            num_experts=None,
            tie_word_embeddings=False,
            encoder_only=False,
            split_ngram_parts=128,
        )
        model.quant_config = None
        model.language_model_only = True
        model.pp_group = SimpleNamespace(is_last_rank=False)

        loaded = model.load_weights(
            [
                (
                    "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                    torch.ones(1),
                ),
                (
                    "model.hyper_connection_mixer.hc_norm.weight",
                    torch.ones(1),
                ),
            ]
        )
        self.assertEqual(loaded, set())

    def test_ple_request_state_is_allocated_only_on_owning_stage(self):
        config = Qwen4ExpTextConfig(
            hidden_size=16,
            hc_count=2,
            ple_layer_ids=[2],
            ngram_size=3,
            ple_conv_kernel_size=4,
            eos_token_id=1,
        )
        configurator = KVCacheConfigurator.__new__(KVCacheConfigurator)
        configurator.mambaish_config = config

        configurator.layer_info = SimpleNamespace(start_layer=0, end_layer=12)
        owner_kwargs = configurator._get_ple_req_pool_kwargs()
        configurator.layer_info = SimpleNamespace(start_layer=12, end_layer=24)
        nonowner_kwargs = configurator._get_ple_req_pool_kwargs()

        self.assertEqual(owner_kwargs["short_conv_layer_ids"], [1])
        self.assertEqual(owner_kwargs["ngram_context_len"], 2)
        self.assertEqual(nonowner_kwargs["short_conv_layer_ids"], [])
        self.assertIsNone(nonowner_kwargs["short_conv_state_shape"])
        self.assertEqual(nonowner_kwargs["ngram_context_len"], 0)


if __name__ == "__main__":
    unittest.main()
