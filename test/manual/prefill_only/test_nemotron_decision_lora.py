import gc
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Ministral3Config, Ministral3ForCausalLM

from sglang.srt.configs.nemotron_decision import load_nemotron_decision_config
from sglang.srt.entrypoints.systemone.nemotron_decide import NemotronDecideEncoder
from sglang.srt.entrypoints.systemone.protocol import SystemOneRequest
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Decision-8B-BF16"
REVISION = "40c36c022455a731f1226594108dac296098102f"
PROJECTIONS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def make_adapter(model, anchor, label, index, path):
    rank = 8
    rng = torch.Generator().manual_seed(920 + index)
    tensors = {}
    for projection in PROJECTIONS:
        area = "self_attn" if projection in PROJECTIONS[:4] else "mlp"
        key = f"encoder.layers.0.{area}.{projection}"
        parameter = model.get_parameter(key.replace("encoder.", "model.") + ".weight")
        out_dim, in_dim = parameter.shape
        tensors[f"base_model.model.{key}.lora_A.weight"] = (
            torch.randn(rank, in_dim, generator=rng) * 0.01
        ).bfloat16()
        tensors[f"base_model.model.{key}.lora_B.weight"] = (
            torch.randn(out_dim, rank, generator=rng) * 0.01
        ).bfloat16()
    aa = torch.zeros(rank, model.config.hidden_size, dtype=torch.bfloat16)
    aa[0] = anchor.bfloat16()
    bb = torch.zeros(model.config.vocab_size, rank, dtype=torch.bfloat16)
    bb[label, 0] = 0.25
    tensors["base_model.model.lm_head.lora_A.weight"] = aa
    tensors["base_model.model.lm_head.lora_B.weight"] = bb
    if index == 1:
        aa = torch.zeros(rank, model.config.vocab_size, dtype=torch.bfloat16)
        aa[0, model.config.mask_token_id] = 1
        bb = (
            torch.randn(model.config.hidden_size, rank, generator=rng) * 0.03
        ).bfloat16()
        tensors["base_model.model.encoder.embed_tokens.lora_embedding_A"] = aa
        tensors["base_model.model.encoder.embed_tokens.lora_embedding_B"] = bb
    path.mkdir()
    save_file(tensors, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(
        json.dumps(
            dict(
                peft_type="LORA",
                task_type="CAUSAL_LM",
                r=rank,
                lora_alpha=rank,
                lora_dropout=0.0,
                bias="none",
                inference_mode=True,
                target_modules=PROJECTIONS
                + ["lm_head"]
                + (["embed_tokens"] if index else []),
            )
        )
    )
    return tensors


def merge_adapter(model, tensors):
    originals = {}
    for key, aa in tensors.items():
        if key.endswith(".lora_A.weight"):
            parameter = key.removeprefix("base_model.model.").replace(
                ".lora_A.weight", ".weight"
            )
            bb = tensors[key.replace(".lora_A.", ".lora_B.")]
            delta = bb.cuda().float() @ aa.cuda().float()
        elif key.endswith(".lora_embedding_A"):
            parameter = key.removeprefix("base_model.model.").replace(
                ".lora_embedding_A", ".weight"
            )
            bb = tensors[key.replace(".lora_embedding_A", ".lora_embedding_B")]
            delta = (bb.cuda().float() @ aa.cuda().float()).T
        else:
            continue
        parameter = parameter.replace("encoder.", "model.")
        weight = model.get_parameter(parameter)
        originals[parameter] = weight.clone()
        weight.add_(delta.to(weight.dtype))
    return originals


def reference_hidden(model, prompt_ids):
    inputs = torch.tensor([prompt_ids], device="cuda")
    length = len(prompt_ids)
    # Override the causal-LM wrapper with full bidirectional attention.
    mask = torch.zeros(1, 1, length, length, device="cuda", dtype=torch.bfloat16)
    return model.model(
        input_ids=inputs, attention_mask=mask, use_cache=False
    ).last_hidden_state[0, -1]


def prepare_cases(model_path, config):
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=REVISION)
    metadata = load_nemotron_decision_config(model_path, REVISION)
    encoder = NemotronDecideEncoder(
        config=config, metadata=metadata, tokenizer=tokenizer, max_prompt_tokens=4096
    )
    examples = [
        (
            {"capital": "Paris"},
            "Which city is listed?",
            {"Paris": "Paris", "Berlin": "Berlin"},
        ),
        (
            {"Stadt": "Berlin"},
            "Welche Stadt ist angegeben?",  # codespell:ignore ist
            {"Berlin": "Berlin", "Paris": "Paris"},
        ),
        (
            {"ville": "Montréal"},
            "Quelle ville est indiquée?",
            {"Montréal": "Montréal", "Lyon": "Lyon"},
        ),
    ]
    cases = []
    for index, (state, instructions, criteria) in enumerate(examples):
        question = dict(type="choice", instructions=instructions, criteria=criteria)
        payload = dict(
            model="decision-test", state=state, questions={"answer": question}
        )
        request = SystemOneRequest(**payload)
        # Compare weights on identical inputs; prompt rendering is covered separately.
        ids, labels, _ = encoder.encode(state, request.questions["answer"], {})
        cases.append(
            dict(
                id=str(index),
                kind="choice",
                payload=payload,
                prompt_ids=ids,
                label_ids=labels,
                api_names=list(criteria),
            )
        )
    return cases, metadata["temperature"].get("choice", 1.0)


@torch.inference_mode()
def make_fixtures(model_path, directory):
    config = Ministral3Config.from_pretrained(model_path, revision=REVISION)
    reference, loading = Ministral3ForCausalLM.from_pretrained(
        model_path,
        revision=REVISION,
        config=config,
        key_mapping={r"^encoder\.": "model.", r"^diffusion_head\.": "lm_head."},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if loading["missing_keys"] or loading["unexpected_keys"]:
        raise AssertionError(f"Reference weight loading failed: {loading}")
    reference = reference.eval().cuda()
    for layer in reference.model.layers:
        layer.self_attn.is_causal = False
    cases, temperature = prepare_cases(model_path, config)
    anchor = reference_hidden(reference, cases[0]["prompt_ids"]).float()
    anchor = (anchor / anchor.norm()).cpu()
    records = {}
    for name in [None, "red", "blue"]:
        originals = {}
        if name is not None:
            index = ["red", "blue"].index(name)
            tensors = make_adapter(
                reference, anchor, cases[0]["label_ids"][index], index, directory / name
            )
            originals = merge_adapter(reference, tensors)
        records[name] = []
        for case in cases:
            hidden = reference_hidden(reference, case["prompt_ids"])
            logits = reference.lm_head(hidden).float()[case["label_ids"]]
            probabilities = (logits / temperature).softmax(-1).cpu().tolist()
            records[name].append(
                {
                    **case,
                    "payload": {
                        **case["payload"],
                        "model": "decision-test" + (":" + name if name else ""),
                    },
                    "probabilities": dict(zip(case["api_names"], probabilities)),
                }
            )
        for key, value in originals.items():
            reference.get_parameter(key).copy_(value)
    return [
        case
        for group in zip(records[None], records["red"], records["blue"])
        for case in group
    ]


class TestNemotronDecisionLoRAServing(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        fixture_path = os.environ.get("NEMOTRON_DECISION_LORA_REFERENCE")
        external_url = os.environ.get("NEMOTRON_DECISION_TEST_URL")
        if bool(fixture_path) != bool(external_url):
            raise ValueError(
                "Set both NEMOTRON_DECISION_LORA_REFERENCE and NEMOTRON_DECISION_TEST_URL for an existing server, or neither to launch one"
            )
        if fixture_path:
            cls.url = external_url
            cls.cases = json.loads(Path(fixture_path).read_text())
            return
        model_path = os.environ.get("NEMOTRON_DECISION_MODEL_PATH", MODEL)
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        cls.cases = make_fixtures(model_path, directory)
        gc.collect()
        torch.cuda.empty_cache()
        cls.url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model_path,
            cls.url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--revision",
                REVISION,
                "--served-model-name",
                "decision-test",
                "--context-length",
                "4096",
                "--max-prefill-tokens",
                "4096",
                "--max-total-tokens",
                "8192",
                "--max-running-requests",
                "4",
                "--mem-fraction-static",
                "0.6",
                "--enable-lora",
                "--max-lora-rank",
                "8",
                "--lora-target-modules",
                *PROJECTIONS,
                "lm_head",
                "embed_tokens",
                "--lora-paths",
                f"red={directory / 'red'}",
                f"blue={directory / 'blue'}",
            ],
        )
        cls.addClassCleanup(kill_process_tree, process.pid)

    def check_case(self, case, route="/v1/systemone"):
        payload = case["payload"]
        if route == "/v1/decisions":
            qid, original = next(iter(payload["questions"].items()))
            question = {"id": qid, "question": original["instructions"]}
            if case["kind"] == "choice":
                question.update(
                    type="choice",
                    options=[
                        {"name": name, "description": value}
                        for name, value in original["criteria"].items()
                    ],
                )
            elif case["kind"] == "score":
                question.update(type="score", levels=original["criteria"])
            else:
                question["type"] = "yes_no"
                for source, target in (("true", "yes"), ("false", "no")):
                    if source in (original.get("criteria") or {}):
                        question[target] = original["criteria"][source]
            payload = {
                "model": payload["model"],
                "input": payload["state"],
                "questions": [question],
            }
        response = requests.post(self.url + route, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        self.assertEqual(set(data["answers"]), set(case["payload"]["questions"]))
        answer = data["answers"][next(iter(case["payload"]["questions"]))]
        actual = (
            {"yes": answer["noul"], "no": 1 - answer["noul"]}
            if case["kind"] == "noul" and route == "/v1/systemone"
            else answer["probabilities"]
        )
        self.assertEqual(set(actual), set(case["probabilities"]))
        for label, expected in case["probabilities"].items():
            self.assertAlmostEqual(
                actual[label],
                expected,
                delta=0.02,
                msg=(case["id"], case["payload"]["model"]),
            )
        input_key, output_key = (
            ("input_tokens", "output_tokens")
            if route == "/v1/systemone"
            else ("prompt_tokens", "completion_tokens")
        )
        self.assertEqual(data["usage"][input_key], len(case["prompt_ids"]))
        self.assertEqual(data["usage"][output_key], 0)
        return actual

    def test_base_and_adapter_scores_stay_isolated_under_concurrency(self):
        # These fixtures use opposing adapter heads, so a base-model fallback fails.
        anchors = self.cases[:3]
        self.assertEqual(len({case["payload"]["model"] for case in anchors}), 3)
        first, second = anchors[1:]
        self.assertGreater(
            max(
                abs(first["probabilities"][k] - second["probabilities"][k])
                for k in first["probabilities"]
            ),
            0.5,
        )
        for route in ("/v1/systemone", "/v1/decisions"):
            for case in self.cases:
                self.check_case(case, route)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda case: self.check_case(case, route), self.cases))

    def test_unknown_adapter_is_rejected_without_affecting_valid_reads(self):
        invalid = {
            **self.cases[0],
            "payload": {
                **self.cases[0]["payload"],
                "model": "decision-test:missing-adapter",
            },
        }
        for route in ("/v1/systemone", "/v1/decisions"):
            with self.assertRaises(requests.HTTPError) as error:
                self.check_case(invalid, route)
            self.assertEqual(error.exception.response.status_code, 400)
            for case in self.cases[:3]:
                self.check_case(case, route)


if __name__ == "__main__":
    unittest.main()
