"""Online updates must reject model-owned derived buffers before writing."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _unsupported_derived_weight_cache_error,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class TestDerivedWeightCache(CustomTestCase):
    def test_nested_model_cache_rejects_updates(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False, device="cpu"),
            torch.nn.Sequential(torch.nn.Module()),
        )
        model[1][0]._derived_weight_cache_error = "derived scales require restart"
        original = model[0].weight.detach().clone()
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
        )
        with patch(
            "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
            return_value=SimpleNamespace(weight_cache_mode="off"),
        ):
            weights = [("0.weight", torch.zeros_like(original))]
            operations = (
                lambda: updater.update_weights_from_disk("unused", "auto"),
                lambda: updater.load_weights_from_distributed(weights),
                lambda: updater.update_weights_from_tensor(
                    weights, load_format="direct"
                ),
                lambda: updater.update_weights_from_ipc(SimpleNamespace()),
            )
            for source, update in zip(
                ("disk", "distributed", "tensor", "ipc"), operations
            ):
                with self.subTest(source=source):
                    self.assertEqual(
                        update(), (False, "derived scales require restart")
                    )
                    self.assertTrue(torch.equal(model[0].weight, original))


class TestNativeDSparkWeightCompatibility(CustomTestCase):
    @staticmethod
    def native_class():
        from sglang.srt.models.ling_v3_dspark import LingV3ForCausalLMDSparkMLA

        return LingV3ForCausalLMDSparkMLA

    def test_fp32_head_preserves_quantized_target_scales(self):
        cls = self.native_class()
        hidden = torch.ones(1, 2, dtype=torch.bfloat16)
        for quantized in (False, True):
            for fp32_enabled in (False, True):
                with self.subTest(quantized=quantized, fp32=fp32_enabled):
                    model = cls.__new__(cls)
                    nn.Module.__init__(model)
                    model._use_fp32_lm_head = fp32_enabled
                    head = model.lm_head = nn.Linear(
                        2, 3, bias=False, dtype=torch.bfloat16
                    )
                    head.weight.data.fill_(1)
                    head.quant_method = None
                    if quantized:
                        head.weight.data = head.weight.data.to(torch.float8_e4m3fn)
                        head.quant_method = SimpleNamespace(
                            apply=Mock(
                                side_effect=lambda layer, x, bias: (
                                    x.float() @ (layer.weight.float() * 2).T
                                )
                            )
                        )
                    with (
                        get_context().override_server_args(),
                        patch(
                            "sglang.srt.models.ling_v3_dspark.gather_and_crop_vocab",
                            side_effect=lambda logits, head: logits,
                        ),
                    ):
                        logits, features = model.compute_base_logits(hidden)
                    expected = torch.full((1, 3), 4.0 if quantized else 2.0)
                    torch.testing.assert_close(logits.float(), expected)
                    self.assertIs(features, hidden)
                    if quantized:
                        head.quant_method.apply.assert_called_once()
                    elif fp32_enabled:
                        self.assertEqual(logits.dtype, torch.float32)

    def test_attached_native_reload_rejects_before_any_parameter_write(self):
        cls = self.native_class()
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(q_lora_rank=None, num_experts=1)
        model.is_bundled = False
        model.confidence_head = None
        model.target_proj = nn.Linear(2, 2, bias=False)
        model.target_proj.weight.data.fill_(1)
        model.attach_shared_modules(
            embed_tokens=nn.Embedding(3, 2), lm_head=nn.Linear(2, 3, bias=False)
        )
        target = nn.Module()
        target.embed_tokens, target.lm_head = model.embed_tokens, model.lm_head
        self.assertEqual(
            _unsupported_derived_weight_cache_error(target),
            model._derived_weight_cache_error,
        )
        before = {
            name: value.detach().clone() for name, value in model.named_parameters()
        }
        with self.assertRaisesRegex(ValueError, "restart|Restart"):
            model.load_weights([("target_proj.weight", torch.zeros(2, 2))])
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, before[name])

    def test_direct_update_error_only_blocks_the_direct_format(self):
        model = torch.nn.Sequential(torch.nn.Module())
        model[0]._direct_weight_update_error = "no direct"
        loaded = []
        model.load_weights = loaded.append
        stub = SimpleNamespace(
            get_model=lambda: model,
            device="cpu",
            custom_weight_loaders={},
            _assert_weight_cache_inactive=lambda _: None,
        )
        update = WeightUpdater.update_weights_from_tensor
        self.assertEqual(update(stub, [], load_format="direct"), (False, "no direct"))
        with get_parallel().override(tp_rank=0):
            self.assertTrue(update(stub, [], load_format=None)[0])
        self.assertEqual(loaded, [[]])


if __name__ == "__main__":
    unittest.main()
