"""Compare packed Decision reads and adapter routing with an explicit masked reference."""

import gc
import json
import os
import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from test.manual.prefill_only.test_nemotron_decision_lora import (
    MODEL,
    REVISION,
    make_adapter,
    merge_adapter,
)
from types import SimpleNamespace

import requests
import torch
from transformers import AutoTokenizer, Ministral3Config, Ministral3ForCausalLM

from sglang.srt.configs.nemotron_decision import load_nemotron_decision_config
from sglang.srt.entrypoints.systemone.nemotron_decide import NemotronDecideEncoder
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


@torch.inference_mode()
def make_packed_fixtures(model_path, directory):
    config = Ministral3Config.from_pretrained(model_path, revision=REVISION)
    model = (
        Ministral3ForCausalLM.from_pretrained(
            model_path,
            revision=REVISION,
            config=config,
            key_mapping={r"^encoder\.": "model.", r"^diffusion_head\.": "lm_head."},
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .eval()
        .cuda()
    )
    for layer in model.model.layers:
        layer.self_attn.is_causal = False
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=REVISION)
    metadata = load_nemotron_decision_config(model_path, REVISION)
    encoder = NemotronDecideEncoder(
        config=config,
        metadata=metadata,
        tokenizer=tokenizer,
        max_prompt_tokens=4096,
    )
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
            ids, codes, _ = encoder.encode(state, SimpleNamespace(**question), {})
            prompts.append(ids)
            labels.append(codes)
        first_question = next(iter(questions.values()))
        body = json.dumps(
            dict(
                state=state,
                question=first_question["instructions"],
                options=[
                    dict(code=code, label=name, meaning=description)
                    for code, (name, description) in zip(
                        metadata["labels"]["choice"], first_question["criteria"].items()
                    )
                ],
            ),
            ensure_ascii=False,
        )
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            chat_template=metadata["chat_template"],
            decision_content=body,
            tokenize=False,
            add_generation_prompt=False,
        )
        assert rendered.count(body) == 1
        state_start = re.match(r'\{\s*"state"\s*:\s*', body).end()
        _, state_end = json.JSONDecoder().raw_decode(body, state_start)
        field = re.match(r',\s*"question"\s*:\s*"', body[state_end:])
        boundary = rendered.index(body) + state_end + field.end()
        prefix = tokenizer.encode(rendered[:boundary], add_special_tokens=False)
        assert tokenizer.encode(rendered, add_special_tokens=False) == prompts[0]
        shared = next(
            (
                i
                for i, values in enumerate(zip(prefix, *prompts))
                if len(set(values)) != 1
            ),
            len(prefix),
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
                input_ids=torch.tensor([ids], device="cuda"),
                mask=mask,
                positions=positions,
                slots=slots,
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
            probabilities, masses = {}, {}
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
                masses[name] = row.log_softmax(-1)[labels].exp().sum().item()
            cases.append(
                dict(
                    payload=dict(
                        model="decision-test" + (":" + adapter if adapter else ""),
                        state=group["state"],
                        questions=group["questions"],
                        packing="shared",
                    ),
                    probabilities=probabilities,
                    label_masses=masses,
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
        cls.cases = make_packed_fixtures(model_path, directory)
        gc.collect()
        torch.cuda.empty_cache()
        cls.url = DEFAULT_URL_FOR_TEST
        server = popen_launch_server(
            model_path,
            cls.url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--revision",
                REVISION,
                "--served-model-name",
                "decision-test",
                "--lora-paths",
                f"red={directory / 'red'}",
                f"blue={directory / 'blue'}",
            ],
        )
        cls.addClassCleanup(kill_process_tree, server.pid)

    def check_case(self, case, route):
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
                packing="shared",
                questions=questions,
            )
        response = requests.post(self.url + route, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        self.assertEqual(set(data["answers"]), set(case["probabilities"]))
        for name, expected in case["probabilities"].items():
            answer = data["answers"][name]
            kind = case["payload"]["questions"][name]["type"]
            actual = (
                {"yes": answer["noul"], "no": 1 - answer["noul"]}
                if route == "/v1/systemone" and kind == "noul"
                else answer["probabilities"]
            )
            self.assertEqual(set(actual), set(expected))
            for label, probability in expected.items():
                self.assertAlmostEqual(actual[label], probability, delta=0.02)
            mass_key = "x_label_mass" if route == "/v1/systemone" else "label_mass"
            self.assertAlmostEqual(
                answer[mass_key], case["label_masses"][name], delta=0.02
            )
            if kind == "score":
                expected_score = sum(
                    int(level) * value for level, value in expected.items()
                )
                self.assertAlmostEqual(answer["score"], expected_score, delta=0.04)
        token_key = "input_tokens" if route == "/v1/systemone" else "prompt_tokens"
        self.assertEqual(data["usage"][token_key], case["prompt_tokens"])
        return data

    def test_packed_masks_and_adapter_readouts_match_reference_under_concurrency(self):
        red, blue = self.cases[2], self.cases[4]
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


if __name__ == "__main__":
    unittest.main()
