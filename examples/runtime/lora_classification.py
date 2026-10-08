"""Reproduce shared generation/classification serving without downloading models.

Create a tiny random model and real PEFT adapters (not useful trained models)::

    python examples/runtime/lora_classification.py create /tmp/classifiers
    sglang serve --model-path /tmp/classifiers/base --served-model-name tiny-classifier \
        --enable-lora --lora-target-modules o_proj --max-lora-rank 4 \
        --max-loras-per-batch 4 --return-hidden-states-mode last --tp 1 \
        --dtype float16 --attention-backend torch_native --lora-backend triton \
        --disable-cuda-graph --context-length 64 --max-total-tokens 512
    python examples/runtime/lora_classification.py check \
        --base-dir /tmp/classifiers/base --url http://127.0.0.1:30000

The HTTP check compares GPU float16 probabilities against a complete, independent
CPU float32 HF+PEFT forward (absolute and relative tolerance 0.003). It exercises
dynamic heads, raw text/token IDs, truncation, cache isolation, and streaming.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import requests
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    PreTrainedTokenizerFast,
)

from sglang.srt.lora.classification_export import write_classification_manifest

MODEL_NAME = "tiny-classifier"
CLASSIFIERS = ("classifier-2", "classifier-3")
TEXTS = ["billing", "please help shipping other billing"]


def create_fixture(output_dir: str | Path) -> Path:
    """Write one backbone, two classifier adapters, and one generation adapter."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty fixture directory")
    base_dir = output / "base"
    words = [
        "[PAD]",
        "[BOS]",
        "[EOS]",
        "[UNK]",
        "user",
        "assistant",
        "billing",
        "shipping",
        "other",
        "please",
        "help",
        "hello",
    ]
    words += [f"word{i}" for i in range(32 - len(words))]
    backend = Tokenizer(
        WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", 1)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        model_max_length=64,
        chat_template="{{ bos_token }}{% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}",
    )
    config = LlamaConfig(
        vocab_size=len(words),
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(697)
        base = LlamaForCausalLM(config).eval()
        base.save_pretrained(base_dir)
        tokenizer.save_pretrained(base_dir)
        for count in (2, 3):
            labels = {
                i: label
                for i, label in enumerate(
                    ("negative", "positive")
                    if count == 2
                    else ("billing", "shipping", "other")
                )
            }
            classifier_config = copy.deepcopy(config)
            classifier_config.num_labels = count
            classifier_config.id2label = labels
            classifier_config.label2id = {label: i for i, label in labels.items()}
            classifier_config.problem_type = "single_label_classification"
            classifier = LlamaForSequenceClassification(classifier_config).eval()
            classifier.model.load_state_dict(base.model.state_dict())
            classifier.config._name_or_path = str(base_dir)
            adapter = get_peft_model(
                classifier,
                LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=4,
                    lora_alpha=8,
                    target_modules=["o_proj"],
                    modules_to_save=["score"],
                    lora_dropout=0.0,
                ),
            ).eval()
            _initialize_adapter(adapter, 697 + count)
            adapter_dir = output / f"classifier-{count}"
            adapter.save_pretrained(adapter_dir, save_embedding_layers=False)
            write_classification_manifest(
                adapter_dir,
                id2label=labels,
                hidden_size=config.hidden_size,
                max_length=4,
                add_special_tokens=True,
            )
        generation_base = copy.deepcopy(base)
        generation_base.config._name_or_path = str(base_dir)
        generation = get_peft_model(
            generation_base,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=4,
                lora_alpha=8,
                target_modules=["o_proj"],
                lora_dropout=0.0,
            ),
        ).eval()
        _initialize_adapter(generation, 700)
        generation.save_pretrained(output / "generation", save_embedding_layers=False)
    return base_dir


def _initialize_adapter(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.normal_(
                    std=0.08
                )  # Nonzero A and B: adapter drops are observable.
            elif "modules_to_save" in name:
                parameter.normal_(std=0.1)


@torch.inference_mode()
def expected_classification(base_dir, adapter_dir, inputs=TEXTS) -> dict:
    """Run the saved classifier's full HF+PEFT model, without SGLang hidden states."""
    base_dir, adapter_dir = Path(base_dir), Path(adapter_dir)
    manifest = json.loads((adapter_dir / "classification_config.json").read_text())
    labels = {int(index): label for index, label in manifest["id2label"].items()}
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    if isinstance(inputs, str):
        inputs = [inputs]
    if inputs and isinstance(inputs[0], int):
        input_ids = [list(inputs)]
    else:
        input_ids = tokenizer(
            inputs,
            add_special_tokens=manifest["add_special_tokens"],
            truncation=False,
            padding=False,
        )["input_ids"]
    input_ids = [row[: manifest["max_length"]] for row in input_ids]
    encoded = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
    base = LlamaForSequenceClassification.from_pretrained(
        base_dir,
        local_files_only=True,
        num_labels=manifest["num_labels"],
        id2label=labels,
        label2id={label: i for i, label in labels.items()},
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True).eval()
    logits = model(**encoded).logits.float()
    probabilities = logits.softmax(-1)
    return {
        "input_ids": input_ids,
        "logits": logits.tolist(),
        "probs": probabilities.tolist(),
        "labels": [labels[index] for index in probabilities.argmax(-1).tolist()],
        "prompt_tokens": sum(map(len, input_ids)),
    }


def _post(session, url, endpoint, body):
    response = session.post(url.rstrip("/") + endpoint, json=body, timeout=120)
    if not response.ok:
        raise AssertionError(
            f"{endpoint}: HTTP {response.status_code}: {response.text}"
        )
    result = response.json()
    if result.get("success") is False or "error" in result:
        raise AssertionError(f"{endpoint}: {result}")
    return result


def _chat(session, url, model, stream=False):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "please help billing"}],
        "temperature": 0,
        "max_tokens": 4,
        "ignore_eos": True,
        "stream": stream,
    }
    if not stream:
        response = _post(session, url, "/v1/chat/completions", body)
        assert response["usage"]["completion_tokens"] == 4, response
        return response["choices"][0]["message"]["content"] or ""
    chunks, finished, done = [], False, False
    with session.post(
        url.rstrip("/") + "/v1/chat/completions", json=body, stream=True, timeout=120
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"data: "):
                continue
            if line == b"data: [DONE]":
                done = True
                break
            event = json.loads(line[6:])
            assert "error" not in event, event
            for choice in event.get("choices", []):
                chunks.append(choice["delta"].get("content") or "")
                finished |= choice.get("finish_reason") is not None
    assert done and finished, "Incomplete chat event stream"
    return "".join(chunks)


def check_server(base_dir, url, model=MODEL_NAME, tolerance=0.003) -> dict:
    """Exercise actual endpoints and compare both classifier versions with HF."""
    base_dir = Path(base_dir).resolve()
    expected = {
        name: expected_classification(base_dir, base_dir.parent / name)
        for name in CLASSIFIERS
    }
    errors = []
    loaded = set()
    with requests.Session() as session:

        def load(name):
            _post(
                session,
                url,
                "/load_lora_adapter",
                {"lora_name": name, "lora_path": str(base_dir.parent / name)},
            )
            loaded.add(name)

        def unload(name):
            _post(session, url, "/unload_lora_adapter", {"lora_name": name})
            loaded.remove(name)

        def classify(name, inputs, reference):
            response = _post(
                session,
                url,
                "/v1/classify",
                {"model": f"{model}:{name}", "input": inputs},
            )
            actual = torch.tensor([row["probs"] for row in response["data"]])
            wanted = torch.tensor(reference["probs"])
            torch.testing.assert_close(actual, wanted, atol=tolerance, rtol=tolerance)
            assert [row["label"] for row in response["data"]] == reference["labels"], (
                response
            )
            assert [row["index"] for row in response["data"]] == list(
                range(len(wanted))
            ), response
            assert all(
                row["num_classes"] == wanted.shape[1] for row in response["data"]
            ), response
            assert response["usage"]["prompt_tokens"] == reference["prompt_tokens"], (
                response
            )
            assert response["usage"]["completion_tokens"] == 0, response
            errors.append((actual - wanted).abs().max().item())

        try:
            base_before = _chat(session, url, model)
            assert _chat(session, url, model, stream=True) == base_before
            for name in (*CLASSIFIERS, "generation"):
                load(name)
            generation_before = _chat(session, url, f"{model}:generation")
            assert (
                _chat(session, url, f"{model}:generation", stream=True)
                == generation_before
            )
            # Repeat identical inputs across heads to exercise warm-cache isolation.
            for name in (*CLASSIFIERS, *reversed(CLASSIFIERS)):
                classify(name, TEXTS, expected[name])
            for name in CLASSIFIERS:
                reference = expected_classification(
                    base_dir, base_dir.parent / name, [1, 6, 7, 8, 9]
                )
                classify(name, [1, 6, 7, 8, 9], reference)
                unload(name)
                load(name)
                classify(name, TEXTS, expected[name])
            assert _chat(session, url, model) == base_before, (
                "Classifier changed base generation"
            )
            assert _chat(session, url, f"{model}:generation") == generation_before
        finally:
            for name in list(loaded):
                unload(name)
    return {
        "classification_comparisons": len(errors),
        "max_probability_error": max(errors),
        "tolerance": tolerance,
        "base_and_generation_streaming": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("output_dir", type=Path)
    check = commands.add_parser("check")
    check.add_argument("--base-dir", type=Path, required=True)
    check.add_argument("--url", required=True)
    check.add_argument("--model", default=MODEL_NAME)
    check.add_argument("--tolerance", type=float, default=0.003)
    args = parser.parse_args()
    if args.command == "create":
        print(create_fixture(args.output_dir))
    else:
        print(
            json.dumps(
                check_server(args.base_dir, args.url, args.model, args.tolerance),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
