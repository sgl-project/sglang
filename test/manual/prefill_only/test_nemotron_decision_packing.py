"""Compare packed Decision reads and adapter routing with an explicit masked reference."""

import gc
import math
import os
import runpy
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from test.manual.prefill_only.test_nemotron_decision_lora import (
    MODEL,
    make_adapter,
    merge_adapter,
)

import requests
import torch
from huggingface_hub import snapshot_download
from transformers.utils import cached_file

from sglang.srt.configs.nemotron_decision import load_nemotron_decision_config
from sglang.srt.entrypoints.openai.serving_decisions import label_mass
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

REVISION = os.environ.get("NEMOTRON_DECISION_REVISION")


@torch.inference_mode()
def make_packed_fixtures(model_path, directory):
    model_path = (
        model_path
        if Path(model_path).is_dir()
        else snapshot_download(model_path, revision=REVISION)
    )
    checkpoint = runpy.run_path(cached_file(model_path, "decide_hf.py"))["Decider"](
        model_path
    )
    model, tokenizer = checkpoint.model, checkpoint.tok
    config = model.config
    metadata = load_nemotron_decision_config(model_path, None)
    reference = runpy.run_path(cached_file(model_path, "packed_layout.py"))
    assert config.state_end_token_id is not None
    groups = []
    for state in (
        {"city": "Paris", "note": "Montréal – café"},
        {"city": "Berlin", "note": "Die Stadt liegt in Deutschland."},
    ):
        questions = {
            "city": dict(
                type="choice",
                instructions="Which city is listed?",
                criteria={"Paris": "Paris", "Berlin": "Berlin"},
            ),
            "country": dict(
                type="choice",
                instructions="Which country contains the city?",
                criteria={
                    "France": "France",
                    "Germany": "Germany",
                    "other": "Another country",
                },
            ),
            "in_france": dict(
                type="noul",
                instructions="Is the listed city in France?",
                criteria=None,
            ),
            "france_match": dict(
                type="score",
                instructions="How strongly does the state indicate a French city?",
                criteria=["No match", "Unclear", "Strong match"],
            ),
        }
        prompts, labels = [], []
        for question in questions.values():
            kind = question["type"]
            if kind == "noul":
                entries = [
                    ("false", "No, the condition does not hold."),
                    ("true", "Yes, the condition holds."),
                ]
            elif kind == "score":
                entries = list(
                    zip(
                        map(str, range(len(question["criteria"]))), question["criteria"]
                    )
                )
            else:
                entries = list(question["criteria"].items())
            codes = metadata["labels"][kind][: len(entries)]
            options = [
                dict(code=code, label=name, meaning=description)
                for code, (name, description) in zip(codes, entries)
            ]
            text = reference["render"](
                tokenizer,
                state,
                question["instructions"],
                options,
                config.state_end_token_id,
            )
            ids = tokenizer.encode(text, add_special_tokens=False) + [
                config.mask_token_id
            ]
            prompts.append(ids)
            code_ids = tokenizer.convert_tokens_to_ids(codes)
            labels.append(code_ids[::-1] if kind == "noul" else code_ids)
        shared = prompts[0].index(config.state_end_token_id) + 1
        assert all(
            ids[:shared] == prompts[0][:shared]
            and ids.count(config.state_end_token_id) == 1
            for ids in prompts
        )
        tails = [ids[shared:] for ids in prompts]
        ids = prompts[0][:shared] + [token for tail in tails for token in tail]
        lengths = [shared, *map(len, tails)]
        # Build a dense reference independently of the paged attention implementation.
        segments = torch.cat(
            [
                torch.full((length,), i, device="cuda")
                for i, length in enumerate(lengths)
            ]
        )
        allowed = (segments[None, :] == 0) | (segments[:, None] == segments[None, :])
        mask = torch.zeros(
            1, 1, len(ids), len(ids), dtype=torch.bfloat16, device="cuda"
        )
        mask.masked_fill_(~allowed[None, None], torch.finfo(mask.dtype).min)
        positions = torch.tensor(
            list(range(shared))
            + [
                position
                for tail in tails
                for position in range(shared, shared + len(tail))
            ],
            device="cuda",
        )[None]
        slots = torch.tensor(lengths, device="cuda").cumsum(0)[1:] - 1
        groups.append(
            dict(
                state=state,
                questions=questions,
                labels=labels,
                layout=lengths,
                input_ids=torch.tensor([ids], device="cuda"),
                mask=mask,
                positions=positions,
                slots=slots,
            )
        )
    group = groups[0]
    n = sum(group["layout"][:2])
    groups.append(
        dict(
            group,
            questions={"city": group["questions"]["city"]},
            labels=group["labels"][:1],
            layout=group["layout"][:2],
            input_ids=group["input_ids"][:, :n],
            mask=group["mask"][:, :, :n, :n],
            positions=group["positions"][:, :n],
            slots=group["slots"][:1],
        )
    )
    anchor = (
        model.model(
            input_ids=groups[0]["input_ids"],
            attention_mask=groups[0]["mask"],
            position_ids=groups[0]["positions"],
            use_cache=False,
        )
        .last_hidden_state[0, groups[0]["slots"][0]]
        .float()
    )
    anchor = (anchor / anchor.norm()).cpu()
    cases = []
    for adapter in (None, "red", "blue"):
        originals = {}
        if adapter is not None:
            index = ("red", "blue").index(adapter)
            tensors = make_adapter(
                model, anchor, groups[0]["labels"][0][index], index, directory / adapter
            )
            originals = merge_adapter(model, tensors)
        for group in groups:
            hidden = model.model(
                input_ids=group["input_ids"],
                attention_mask=group["mask"],
                position_ids=group["positions"],
                use_cache=False,
            ).last_hidden_state[0, group["slots"]]
            logits = model.lm_head(hidden).float()
            probabilities = {}
            for row, (name, question), labels in zip(
                logits, group["questions"].items(), group["labels"]
            ):
                kind = question["type"]
                values = (
                    (row[labels] / metadata.get("temperature", {}).get(kind, 1.0))
                    .softmax(-1)
                    .tolist()
                )
                if kind == "noul":
                    names = ["yes", "no"]
                elif kind == "score":
                    names = list(map(str, range(len(question["criteria"]))))
                else:
                    names = list(question["criteria"])
                probabilities[name] = dict(zip(names, values))
            cases.append(
                dict(
                    payload=dict(
                        model="decision-test" + (":" + adapter if adapter else ""),
                        state=group["state"],
                        questions=group["questions"],
                    ),
                    probabilities=probabilities,
                    native_input_ids=group["input_ids"][0].tolist(),
                    layout=group["layout"],
                    slots=group["slots"].tolist(),
                    labels=dict(zip(group["questions"], group["labels"])),
                    adapter_name=adapter,
                    prompt_tokens=group["input_ids"].shape[1],
                )
            )
        for name, original in originals.items():
            model.get_parameter(name).copy_(original)
    return cases


class TestNemotronDecisionPacking(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        model_path = os.environ.get("NEMOTRON_DECISION_MODEL_PATH", MODEL)
        cls.metadata = load_nemotron_decision_config(model_path, REVISION)
        cls.cases = make_packed_fixtures(model_path, directory)
        gc.collect()
        torch.cuda.empty_cache()
        cls.url = DEFAULT_URL_FOR_TEST
        server = popen_launch_server(
            model_path,
            cls.url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(["--revision", REVISION] if REVISION else [])
            + [
                "--json-model-override-args",
                '{"decision_attention_layout":"state_sees_no_questions"}',
                # Fix the GPU batch shape for exact API/native score comparisons.
                "--max-running-requests",
                "1",
                "--served-model-name",
                "decision-test",
                "--lora-paths",
                f"red={directory / 'red'}",
                f"blue={directory / 'blue'}",
            ],
        )
        cls.addClassCleanup(kill_process_tree, server.pid)
        # The first packed request can compile kernels after server startup.
        requests.post(
            cls.url + "/v1/systemone",
            json=cls.cases[0]["payload"],
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        ).raise_for_status()

    def check_case(self, case, route, return_prompt_token_ids=False):
        payload = case["payload"]
        if route == "/v1/decisions":
            questions = []
            for name, question in payload["questions"].items():
                converted = dict(id=name, question=question["instructions"])
                if question["type"] == "choice":
                    converted.update(
                        type="choice",
                        options=[
                            dict(name=label, description=description)
                            for label, description in question["criteria"].items()
                        ],
                    )
                elif question["type"] == "score":
                    converted.update(type="score", levels=question["criteria"])
                else:
                    converted["type"] = "yes_no"
                questions.append(converted)
            payload = dict(
                model=payload["model"],
                input=payload["state"],
                questions=questions,
            )
        if return_prompt_token_ids:
            payload = {**payload, "return_prompt_token_ids": True}
        response = requests.post(self.url + route, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        self.assertEqual(set(data["answers"]), set(case["probabilities"]))
        native = requests.post(
            self.url + "/generate",
            json=dict(
                input_ids=case["native_input_ids"],
                decision_layout=case["layout"],
                token_indices_to_pool=case["slots"],
                lora_path=case["adapter_name"],
                token_ids_logprob=sorted(
                    {token for labels in case["labels"].values() for token in labels}
                ),
                sampling_params={"max_new_tokens": 0},
                return_logprob=True,
                logprob_start_len=0,
            ),
            timeout=120,
        )
        native.raise_for_status()
        rows = native.json()["meta_info"]["input_token_ids_logprobs"]
        native_scores = dict(zip(case["probabilities"], rows))
        for name, expected in case["probabilities"].items():
            answer = data["answers"][name]
            kind = case["payload"]["questions"][name]["type"]
            actual = (
                {"yes": answer["noul"], "no": 1 - answer["noul"]}
                if route == "/v1/systemone" and kind == "noul"
                else answer["probabilities"]
            )
            self.assertEqual(set(actual), set(expected))
            row = {token: value for value, token, *_ in native_scores[name]}
            logprobs = [row[token] for token in case["labels"][name]]
            temperature = self.metadata.get("temperature", {}).get(kind, 1.0)
            weights = [math.exp((v - max(logprobs)) / temperature) for v in logprobs]
            total = math.fsum(weights)
            native_probabilities = dict(zip(expected, [v / total for v in weights]))
            if route == "/v1/systemone" and kind == "noul":
                self.assertEqual(answer["noul"], native_probabilities["yes"])
            else:
                self.assertEqual(actual, native_probabilities)
            self.assertEqual(
                max(actual, key=actual.get), max(expected, key=expected.get)
            )
            mass_key = "x_label_mass" if route == "/v1/systemone" else "label_mass"
            self.assertEqual(answer[mass_key], label_mass(logprobs))
        token_key = "input_tokens" if route == "/v1/systemone" else "prompt_tokens"
        self.assertEqual(data["usage"][token_key], case["prompt_tokens"])
        return data

    def test_packed_masks_and_adapter_readouts_match_reference_under_concurrency(self):
        red = next(
            case for case in self.cases if case["payload"]["model"].endswith(":red")
        )
        blue = next(
            case for case in self.cases if case["payload"]["model"].endswith(":blue")
        )
        self.assertGreater(
            max(
                abs(
                    red["probabilities"]["city"][label]
                    - blue["probabilities"]["city"][label]
                )
                for label in red["probabilities"]["city"]
            ),
            0.2,
        )
        for route in ("/v1/systemone", "/v1/decisions"):
            for case in self.cases:
                self.check_case(case, route)
            with ThreadPoolExecutor(4) as pool:
                list(
                    pool.map(lambda case: self.check_case(case, route), self.cases * 2)
                )
            invalid = {
                **self.cases[0],
                "payload": {
                    **self.cases[0]["payload"],
                    "model": "decision-test:missing-adapter",
                },
            }
            with self.assertRaises(requests.HTTPError) as error:
                self.check_case(invalid, route)
            self.assertEqual(error.exception.response.status_code, 400)
            self.check_case(self.cases[0], route)
            if route == "/v1/decisions":
                with self.assertRaises(requests.HTTPError) as error:
                    self.check_case(self.cases[0], route, return_prompt_token_ids=True)
                self.assertEqual(error.exception.response.status_code, 400)
                self.assertIn("return_prompt_token_ids", error.exception.response.text)
                self.check_case(self.cases[0], route)


if __name__ == "__main__":
    unittest.main()
