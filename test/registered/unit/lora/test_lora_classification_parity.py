"""CPU parity with a real, config-only HF/PEFT sequence classifier.

The reference runs PEFT's complete forward pass. The other path reconstructs
the saved LoRA through SGLang, runs an independent decoder, and applies the
production CPU head. This does not exercise SGLang GPU scheduling or kernels.
"""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    LlamaConfig,
    LlamaForSequenceClassification,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForSequenceClassification,
    Qwen3Config,
    Qwen3ForSequenceClassification,
)

from sglang.srt.lora.classification_export import write_classification_manifest
from sglang.srt.lora.classification_head import prepare_classification_bundle
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_LABELS = {0: "billing", 1: "shipping", 2: "other"}
_MODELS = (
    (LlamaConfig, LlamaForSequenceClassification),
    (Qwen2Config, Qwen2ForSequenceClassification),
    (Qwen3Config, Qwen3ForSequenceClassification),
)


def _tokenizer(padding_side):
    vocabulary = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[UNK]": 2,
        "billing": 3,
        "shipping": 4,
        "other": 5,
        "please": 6,
        "help": 7,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", 1)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        unk_token="[UNK]",
        padding_side=padding_side,
    )


class TestLoRAClassificationParity(CustomTestCase):
    @torch.no_grad()
    def test_saved_peft_adapter_matches_full_classifier_forward(self):
        for config_class, model_class in _MODELS:
            for padding_side in ("left", "right"):
                with self.subTest(
                    model=model_class.__name__, padding_side=padding_side
                ):
                    self._check_forward(config_class, model_class, padding_side)

    def _check_forward(self, config_class, model_class, padding_side):
        config = config_class(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            pad_token_id=0,
            num_labels=3,
            id2label=_LABELS,
            label2id={label: index for index, label in _LABELS.items()},
            problem_type="single_label_classification",
            **({"head_dim": 8} if config_class is Qwen3Config else {}),
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91)
            base = model_class(config).eval()
            independent = copy.deepcopy(base)
            reference = get_peft_model(
                base,
                LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=2,
                    lora_alpha=4,
                    target_modules=["o_proj"],
                    modules_to_save=["score"],
                ),
            ).eval()
            for name, parameter in reference.named_parameters():
                if "lora_" in name:
                    torch.nn.init.normal_(parameter, std=0.2)
                elif "modules_to_save" in name:
                    torch.nn.init.normal_(parameter, std=0.3)

        tokenizer = _tokenizer(padding_side)
        texts = ["billing", "please help shipping other billing"]
        expected_inputs = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=4,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        expected_logits = reference(**expected_inputs).logits

        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory, save_embedding_layers=False)
            write_classification_manifest(
                directory,
                id2label=_LABELS,
                hidden_size=config.hidden_size,
                max_length=4,
                add_special_tokens=True,
            )
            bundle = prepare_classification_bundle(directory, config.hidden_size)
            self.addCleanup(bundle.directory.cleanup)
            saved = load_file(str(Path(bundle.path) / "adapter_model.safetensors"))
            adapter = LoRAAdapter(
                "classifier", LoRAConfig(path=bundle.path), config, None, None
            )
            adapter.initialize_weights_from_tensors(saved)

        prefix = "base_model.model.model.layers.0.self_attn.o_proj"
        weights = adapter.layers[0].weights
        delta = (
            weights[f"{prefix}.lora_B.weight"] @ weights[f"{prefix}.lora_A.weight"]
        ) * adapter.scaling
        self.assertGreater(delta.abs().max().item(), 0)
        independent.model.layers[0].self_attn.o_proj.weight.add_(delta)

        request = GenerateReqInput(text=texts, sampling_params={"max_new_tokens": 0})
        request.normalize_batch_and_arguments()
        bundle.head.prepare_input(request, tokenizer)
        expected_ids = [
            ids[mask.bool()].tolist()
            for ids, mask in zip(
                expected_inputs["input_ids"], expected_inputs["attention_mask"]
            )
        ]
        self.assertEqual(request.input_ids, expected_ids)
        self.assertIsNone(request.text)
        self.assertEqual(tokenizer.padding_side, padding_side)
        actual_inputs = tokenizer.pad(
            {"input_ids": request.input_ids}, padding=True, return_tensors="pt"
        )
        hidden = independent.model(**actual_inputs).last_hidden_state
        mask = actual_inputs["attention_mask"]
        last = (mask * torch.arange(mask.shape[1])).argmax(dim=-1)
        pooled = hidden[torch.arange(len(texts)), last]
        actual_logits = torch.nn.functional.linear(
            pooled.to(bundle.head.weight.dtype), bundle.head.weight, bundle.head.bias
        ).float()
        torch.testing.assert_close(
            actual_logits, expected_logits.float(), rtol=1e-5, atol=1e-6
        )
        responses = bundle.head.classify(
            [{"meta_info": {"hidden_states": row.tolist()}} for row in pooled]
        )
        torch.testing.assert_close(
            torch.tensor([response["probs"] for response in responses]),
            expected_logits.float().softmax(dim=-1),
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertEqual(
            [response["label"] for response in responses],
            [_LABELS[index] for index in expected_logits.argmax(-1).tolist()],
        )
        self.assertEqual([response["index"] for response in responses], [0, 1])


if __name__ == "__main__":
    unittest.main()
