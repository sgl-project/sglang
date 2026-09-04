import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.vocab_parallel_embedding as embedding
import sglang.srt.models.bailing_moe_linear as bailing
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _quant_config(targets, ignore=(), block=False):
    weights = {
        "num_bits": 8,
        "type": "float" if block else "int",
        "strategy": "block" if block else "channel",
        "symmetric": True,
        "dynamic": False,
    }
    if block:
        weights["block_structure"] = [128, 128]
    return CompressedTensorsConfig.from_config(
        {
            "format": "float-quantized" if block else "int-quantized",
            "quant_method": "compressed-tensors",
            "ignore": list(ignore),
            "config_groups": {
                "head": {
                    "targets": list(targets),
                    "weights": weights,
                    "input_activations": {
                        "num_bits": 8,
                        "type": weights["type"],
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                    },
                }
            },
        }
    )


class TestBailingLmHeadQuantization(CustomTestCase):
    def _model(self, quant_config, prefix="", hidden_states=None):
        config = SimpleNamespace(
            tie_word_embeddings=False, vocab_size=128, hidden_size=128
        )
        parallel = SimpleNamespace(enable_dp_lm_head=False, tp_rank=0, tp_size=1)
        # Keep the real model constructor, head, and quantization dispatch;
        # omit the transformer body and distributed process-group setup.
        with (
            patch.object(
                bailing, "BailingMoELinearModel", return_value=torch.nn.Identity()
            ),
            patch.object(bailing, "LogitsProcessor", return_value=torch.nn.Identity()),
            patch.object(
                bailing, "get_pp_group", return_value=SimpleNamespace(is_last_rank=True)
            ),
            patch.object(bailing, "get_parallel", return_value=parallel),
            patch.object(embedding, "get_parallel", return_value=parallel),
            torch.device("cuda"),
        ):
            model = bailing.BailingMoELinearForCausalLM(
                config=config, quant_config=quant_config, prefix=prefix
            )
        if hidden_states is not None:
            model.model = _HiddenStateModel(hidden_states)
        return model

    def _head(self, quant_config, prefix=""):
        return self._model(quant_config, prefix=prefix).lm_head

    def _processor(self):
        processor = LogitsProcessor.__new__(LogitsProcessor)
        torch.nn.Module.__init__(processor)
        processor.vocab_size = 128
        processor.logit_scale = None
        processor.use_attn_tp_group = False
        processor.use_tp_lm_head_all_to_all = False
        processor.do_tensor_parallel_all_gather = False
        processor.do_tensor_parallel_all_gather_dp_attn = False
        processor.use_fp32_lm_head = True
        processor.rl_on_policy_target = None
        processor.final_logit_softcapping = None
        processor.return_full_logits = False
        return processor

    def _load_int8_head(self, head):
        weights = (
            torch.arange(128 * 128, device="cuda").reshape(128, 128) % 15 - 7
        ).to(torch.int8)
        scales = torch.full((128, 1), 1 / 32, device="cuda")
        head.weight.weight_loader(head.weight, weights)
        head.weight_scale.weight_loader(head.weight_scale, scales)
        head.quant_method.process_weights_after_loading(head)
        return weights, scales

    def test_named_targets_allocate_quantized_head(self):
        for prefix, target in (
            ("", "lm_head"),
            ("language_model", "language_model.lm_head"),
            ("language_model", "lm_head"),
            ("language_model", "re:.*lm_head$"),
        ):
            with self.subTest(prefix=prefix, target=target):
                head = self._head(_quant_config([target]), prefix=prefix)
                self.assertIsInstance(head.quant_method, CompressedTensorsLinearMethod)
                self.assertEqual(head.weight.dtype, torch.int8)
                self.assertEqual(head.weight.device.type, "cuda")
                self.assertEqual(head.weight_scale.shape, (128, 1))

    def test_unquantized_head_conventions(self):
        for quant_config in (
            None,
            _quant_config(["Linear"]),
            _quant_config(["re:.*lm_head$"], ignore=["language_model.lm_head"]),
        ):
            with self.subTest(quant_config=quant_config):
                head = self._head(quant_config, prefix="language_model")
                self.assertIsInstance(head.quant_method, UnquantizedEmbeddingMethod)
                self.assertEqual(head.weight.dtype, torch.float32)

    def test_unsupported_head_quantization_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "Block-quantized lm_head"):
            self._head(_quant_config(["lm_head"], block=True))

    def test_quantized_head_matches_dequantized_reference(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                hidden_states = torch.arange(128, device="cuda", dtype=dtype).repeat(
                    4, 1
                )
                model = self._model(
                    _quant_config(["language_model.lm_head"]),
                    prefix="language_model",
                    hidden_states=hidden_states,
                )
                weights, scales = self._load_int8_head(model.lm_head)
                model.logits_processor = self._processor()
                observed_dtypes = []
                model.logits_processor.register_forward_pre_hook(
                    lambda _module, args: observed_dtypes.append(args[1].dtype)
                )

                output = model(
                    input_ids=torch.zeros(4, device="cuda", dtype=torch.long),
                    positions=torch.zeros(4, device="cuda", dtype=torch.long),
                    forward_batch=LogitsMetadata(forward_mode=ForwardMode.DECODE),
                )

                expected = (hidden_states.float() @ weights.float().T * scales.T).to(
                    dtype
                )
                self.assertEqual(observed_dtypes, [dtype])
                torch.testing.assert_close(
                    output.next_token_logits, expected.float(), rtol=0, atol=0
                )

    def test_unquantized_head_matches_fp32_reference(self):
        hidden_states = torch.arange(128, device="cuda", dtype=torch.bfloat16).repeat(
            4, 1
        )
        model = self._model(None, hidden_states=hidden_states)
        weights = torch.linspace(
            -1, 1, 128 * 128, device="cuda", dtype=torch.float32
        ).reshape(128, 128)
        model.lm_head.weight.data.copy_(weights)
        model.logits_processor = self._processor()
        observed_dtypes = []
        model.logits_processor.register_forward_pre_hook(
            lambda _module, args: observed_dtypes.append(args[1].dtype)
        )

        output = model(
            input_ids=torch.zeros(4, device="cuda", dtype=torch.long),
            positions=torch.zeros(4, device="cuda", dtype=torch.long),
            forward_batch=LogitsMetadata(forward_mode=ForwardMode.DECODE),
        )

        self.assertEqual(observed_dtypes, [torch.float32])
        torch.testing.assert_close(
            output.next_token_logits,
            hidden_states.float() @ weights.T,
            rtol=0,
            atol=0,
        )


class _HiddenStateModel(torch.nn.Module):
    def __init__(self, hidden_states):
        super().__init__()
        self.hidden_states = hidden_states

    def forward(self, **_kwargs):
        return self.hidden_states


if __name__ == "__main__":
    unittest.main()
