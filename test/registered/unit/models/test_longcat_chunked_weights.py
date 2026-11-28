"""CPU regressions for LongCat fused weights split across update buckets."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.models.longcat_flash import LongcatFlashForCausalLM
from sglang.srt.models.longcat_flash_nextn import LongcatFlashForCausalLMNextN
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MODEL_CLASSES = (LongcatFlashForCausalLM, LongcatFlashForCausalLMNextN)


def _loader(model_class, *, quantized=False, suffix="weight"):
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(n_routed_experts=0, q_lora_rank=2)
    model.quant_config = SimpleNamespace(get_name=lambda: "awq") if quantized else None
    model.use_ngram_embedding = False
    # Inference-only postprocessing is outside this CPU weight-copy fixture.
    model.post_load_weights = lambda **kwargs: None
    if model_class is LongcatFlashForCausalLMNextN:
        source = "model.mtp.layers.0.self_attn"
        destination = "model.decoder.self_attn"
    else:
        source = destination = "model.layers.0.self_attn.0"
    target = model
    for part in (destination + ".fused_qkv_a_proj_with_mqa").split("."):
        child = nn.Module()
        target.add_module(part, child)
        target = child
    parameter = nn.Parameter(torch.zeros(3, 3), requires_grad=False)
    target.register_parameter(suffix, parameter)
    return model, parameter, source, destination


def _updater(model):
    return WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=SimpleNamespace(dtype=torch.float32),
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=lambda **kwargs: None,
        recapture_cuda_graph=None,
        get_model_runner=lambda: None,
    )


class TestLongcatChunkedWeights(CustomTestCase):
    def test_split_pair_survives_sender_buffer_reuse(self):
        """Both loaders must pair separate buckets using the original tensor values."""
        for model_class in MODEL_CLASSES:
            for quantized in (False, True):
                for suffix in ("weight", "weight_scale_inv"):
                    for reverse in (False, True):
                        with self.subTest(
                            model=model_class.__name__,
                            quantized=quantized,
                            suffix=suffix,
                            reverse=reverse,
                        ):
                            model, target, source, _ = _loader(
                                model_class, quantized=quantized, suffix=suffix
                            )
                            axis = 1 if quantized else 0
                            q = torch.ones((3, 2) if quantized else (2, 3))
                            kv = torch.full((3, 1) if quantized else (1, 3), 2.0)
                            expected = torch.cat((q, kv), dim=axis)
                            halves = [
                                (f"{source}.q_a_proj.{suffix}", q),
                                (f"{source}.kv_a_proj_with_mqa.{suffix}", kv),
                            ]
                            if reverse:
                                halves.reverse()
                            model.load_weights([halves[0]])
                            torch.testing.assert_close(target, torch.zeros(3, 3))
                            halves[0][1].fill_(9)
                            model.load_weights([halves[1]])
                            torch.testing.assert_close(target, expected)
                            with patch.object(
                                DefaultModelLoader, "postprocess_weights"
                            ):
                                self.assertIsNone(
                                    _updater(model).end_weight_update(
                                        run_post_load=False
                                    )
                                )

    def test_end_update_reports_and_discards_an_unpaired_half(self):
        """An incomplete bucket sequence must fail instead of reporting a successful update."""
        for model_class in MODEL_CLASSES:
            with self.subTest(model=model_class.__name__):
                model, target, source, destination = _loader(model_class)
                model.load_weights([(f"{source}.q_a_proj.weight", torch.ones(2, 3))])
                # Exercise discovery through a wrapper as well as the loader hook.
                wrapper = nn.Module()
                wrapper.add_module("language_model", model)
                updater = _updater(wrapper)
                with patch.object(DefaultModelLoader, "postprocess_weights"):
                    error = updater.end_weight_update(run_post_load=False)
                    self.assertIsNotNone(error)
                    self.assertIn(f"{destination}.q_a_proj.weight", error)
                    self.assertIsNone(updater.end_weight_update(run_post_load=False))
                torch.testing.assert_close(target, torch.zeros(3, 3))

    def test_begin_update_cannot_pair_with_a_previous_load(self):
        """Starting a new session discards old halves before accepting new buckets."""
        for model_class in MODEL_CLASSES:
            with self.subTest(model=model_class.__name__):
                model, target, source, _ = _loader(model_class)
                q_name = f"{source}.q_a_proj.weight"
                kv_name = f"{source}.kv_a_proj_with_mqa.weight"
                model.load_weights([(q_name, torch.ones(2, 3))])
                with patch.object(DefaultModelLoader, "restore_weights_before_loading"):
                    _updater(model).begin_weight_update()
                model.load_weights([(kv_name, torch.full((1, 3), 2.0))])
                torch.testing.assert_close(target, torch.zeros(3, 3))
                model.load_weights([(q_name, torch.full((2, 3), 3.0))])
                expected = torch.tensor([[3.0] * 3, [3.0] * 3, [2.0] * 3])
                torch.testing.assert_close(target, expected)


if __name__ == "__main__":
    unittest.main()
