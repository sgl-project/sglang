"""Independent LM-head references for Qwen3.5 pointwise and setwise scoring."""

import gc
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.qwen35_score_utils import (
    ANCHOR,
    engine_args,
    resolve_model,
    server_args,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=360, stage="extra-a", runner_config="1-gpu-small")


class TestQwen35ScoreAPI(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = resolve_model()
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model, trust_remote_code=False
        )
        cls.labels = []
        for label in (" yes", " no", " maybe"):
            (token_id,) = cls.tokenizer.encode(label, add_special_tokens=False)
            cls.labels.append(token_id)
        assert len(set(cls.labels)) == 3
        cls.anchor_id = cls.tokenizer.convert_tokens_to_ids(ANCHOR)
        assert cls.anchor_id is not None
        assert cls.anchor_id != cls.tokenizer.unk_token_id
        encode = lambda text: cls.tokenizer.encode(text, add_special_tokens=False)
        cls.cases = [
            {
                "query": encode("Judge whether each passage describes an animal: "),
                "items": [
                    encode("A cat sleeps on a warm chair."),
                    encode("A table stands beside the window."),
                    encode("A blue whale swims in the ocean. " * 8),
                ],
            },
            {
                "query": [],
                "items": [encode("cat") * length for length in (63, 64, 65)],
            },
            {
                "query": encode("Judge whether each passage describes an animal: "),
                "items": [
                    encode(
                        f"Candidate A: a cat.{ANCHOR} Candidate B: a table.{ANCHOR}"
                    ),
                    encode(f"Candidate C: a whale.{ANCHOR}"),
                ],
                "setwise": True,
            },
            {
                "query": encode("cat") * 64 + [9999],
                "items": [encode("dog") * 130, [9999, 1000]],
            },
            {
                "query": encode("Judge whether this describes an animal: "),
                "items": [[], encode("cat"), [], []],
            },
            {
                "query": encode("Judge whether this describes an animal: "),
                "items": [[]],
            },
        ]
        # Materialize the oracle before the serving engine reserves GPU memory.
        hf, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            cls.model,
            dtype=torch.float32,
            attn_implementation="eager",
            trust_remote_code=False,
            output_loading_info=True,
        )
        assert not loading.get("missing_keys"), loading
        assert not loading.get("unexpected_keys"), loading
        hf = hf.eval().to("cuda")
        cls.references = []
        with torch.inference_mode():
            for case in cls.cases:
                references = []
                for item in case["items"]:
                    ids = case["query"] + item
                    positions = (
                        [i for i, token in enumerate(ids) if token == cls.anchor_id]
                        if case.get("setwise")
                        else [len(ids) - 1]
                    )
                    logits = (
                        hf(
                            input_ids=torch.tensor([ids], device="cuda"),
                            use_cache=False,
                        )
                        .logits[0, positions]
                        .float()
                    )
                    references.append(
                        torch.log_softmax(logits, dim=-1)[:, cls.labels].cpu().clone()
                    )
                cls.references.append(references)
        del hf
        gc.collect()
        torch.cuda.empty_cache()

    def _payload(self, case, *, normalized, ragged=False, temperature=1.0):
        labels = (
            [
                list(reversed(self.labels)) if i % 2 else self.labels[:2]
                for i in range(len(case["items"]))
            ]
            if ragged
            else self.labels[:2]
        )
        payload = dict(
            query=case["query"],
            items=case["items"],
            label_token_ids=labels,
            apply_softmax=normalized,
            temperature=temperature,
        )
        if case.get("setwise"):
            payload["score_extraction_token"] = ANCHOR
        else:
            payload["return_token_logprobs"] = True
        return payload

    def _assert_reference(self, body, payload, references, setwise=False):
        self.assertEqual(len(body["scores"]), len(references))
        shared = isinstance(payload["label_token_ids"][0], int)
        for i, (actual, logprobs) in enumerate(zip(body["scores"], references)):
            labels = (
                payload["label_token_ids"] if shared else payload["label_token_ids"][i]
            )
            indices = [self.labels.index(label) for label in labels]
            expected_logprobs = logprobs[:, indices].double()
            actual = torch.tensor(actual if setwise else [actual]).double()
            self.assertEqual(actual.shape, expected_logprobs.shape)
            self.assertTrue(torch.isfinite(actual).all())
            if payload["apply_softmax"]:
                expected = torch.softmax(
                    expected_logprobs / payload["temperature"], dim=-1
                )
                torch.testing.assert_close(
                    actual.sum(-1), torch.ones(actual.shape[0]).double()
                )
                # Calibrated against the FP32 HF oracle, including BF16 rounding.
                torch.testing.assert_close(actual, expected, atol=0.06, rtol=0)
            else:
                self.assertTrue((actual > 0).all())
                self.assertTrue((actual <= 1).all())
                # Log-space bounds prevent tiny/all-zero probabilities passing.
                torch.testing.assert_close(
                    actual.log(), expected_logprobs, atol=0.2, rtol=0
                )
            if not setwise:
                returned = torch.tensor(body["token_logprobs"][i]).double()
                torch.testing.assert_close(
                    returned, expected_logprobs[0], atol=0.2, rtol=0
                )
                if payload["apply_softmax"]:
                    torch.testing.assert_close(
                        actual[0],
                        torch.softmax(returned / payload["temperature"], dim=-1),
                        atol=1e-6,
                        rtol=1e-6,
                    )

    def _engine_result(self, engine, payload):
        args = dict(payload)
        if args.pop("score_extraction_token", None) is not None:
            args["score_extraction_token_id"] = self.anchor_id
        result = engine.score(**args)
        return {"scores": result.scores, "token_logprobs": result.token_logprobs}

    def _check_engine(self, enable_mis):
        engine = Engine(model_path=self.model, **engine_args(enable_mis))
        try:
            for index, case in enumerate(self.cases):
                for normalized, temperature, ragged in (
                    (False, 1.0, False),
                    (True, 1.0, False),
                    (True, 2.0, True),
                ):
                    with self.subTest(
                        enable_mis=enable_mis, case=index, normalized=normalized
                    ):
                        payload = self._payload(
                            case,
                            normalized=normalized,
                            ragged=ragged,
                            temperature=temperature,
                        )
                        self._assert_reference(
                            self._engine_result(engine, payload),
                            payload,
                            self.references[index],
                            case.get("setwise", False),
                        )
            self._check_isolation(engine)
            if not enable_mis:
                payload = self._item_first_payload()
                self._assert_reference(
                    self._engine_result(engine, payload),
                    payload,
                    [self.references[0][0]],
                )
        finally:
            engine.shutdown()
            torch.cuda.empty_cache()

    def _check_isolation(self, engine):
        # Existing GDN tests cover pointwise isolation; exercise anchor regrouping here.
        case = self.cases[2]
        payload = self._payload(case, normalized=True)
        target = case["items"][0]
        for items in (
            [target],
            list(reversed(case["items"])),
            [case["items"][-1], target, target],
        ):
            changed = dict(payload, items=items)
            references = [
                self.references[2][case["items"].index(item)] for item in items
            ]
            self._assert_reference(
                self._engine_result(engine, changed), changed, references, setwise=True
            )

    def test_engine_pointwise_and_setwise(self):
        self._check_engine(False)

    def test_engine_mis_and_setwise_mis(self):
        self._check_engine(True)

    def _item_first_payload(self):
        case = self.cases[0]
        payload = self._payload(case, normalized=True)
        payload.update(query=case["items"][0], items=[case["query"]], item_first=True)
        return payload

    def _check_text_inputs(self, post, enable_mis):
        for setwise in (False, True):
            payload = dict(
                query="Rate:",
                items=(
                    [f"cat{ANCHOR} table{ANCHOR}", f"whale{ANCHOR}"]
                    if setwise
                    else ["", "cat", "", "table", ""]
                ),
                label_token_ids=self.labels[:2],
                apply_softmax=True,
            )
            if setwise:
                payload["score_extraction_token"] = ANCHOR
            # SIS tokenizes whole strings; MIS tokenizes query/items separately.
            tokens = dict(
                payload,
                query=self.tokenizer.encode(payload["query"]) if enable_mis else [],
                items=[
                    self.tokenizer.encode(
                        item if enable_mis else payload["query"] + item,
                        add_special_tokens=not enable_mis,
                    )
                    for item in payload["items"]
                ],
            )
            actual_scores, expected_scores = (
                post(payload)["scores"],
                post(tokens)["scores"],
            )
            self.assertEqual(len(actual_scores), len(payload["items"]))
            self.assertEqual(len(actual_scores), len(expected_scores))
            for actual, expected in zip(actual_scores, expected_scores):
                torch.testing.assert_close(
                    torch.tensor(actual), torch.tensor(expected), atol=0.01, rtol=0.02
                )

    def _check_http(self, enable_mis):
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args(enable_mis),
        )
        try:

            def post(payload):
                response = requests.post(
                    DEFAULT_URL_FOR_TEST + "/v1/score", json=payload, timeout=60
                )
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["object"], "scoring")
                self.assertEqual(body["usage"]["completion_tokens"], 0)
                return body

            payloads = []
            for index, case in enumerate(self.cases):
                for normalized in (False, True):
                    payload = self._payload(
                        case,
                        normalized=normalized,
                        ragged=True,
                        temperature=2.0 if normalized else 1.0,
                    )
                    self._assert_reference(
                        post(payload),
                        payload,
                        self.references[index],
                        case.get("setwise", False),
                    )
                payloads.append(self._payload(case, normalized=True))
            with ThreadPoolExecutor(max_workers=6) as pool:
                bodies = list(pool.map(post, payloads * 2))
            for i, body in enumerate(bodies):
                index = i % len(self.cases)
                self._assert_reference(
                    body,
                    payloads[index],
                    self.references[index],
                    self.cases[index].get("setwise", False),
                )
            if not enable_mis:
                payload = self._item_first_payload()
                self._assert_reference(post(payload), payload, [self.references[0][0]])
            self._check_text_inputs(post, enable_mis)
            omitted_query = dict(payloads[1])
            omitted_query.pop("query")
            self._assert_reference(
                post(omitted_query), omitted_query, self.references[1]
            )
            for bad in (
                dict(payloads[2], query=[self.anchor_id]),
                dict(payloads[2], items=[[1000]]),
                dict(payloads[2], item_first=True),
            ):
                response = requests.post(
                    DEFAULT_URL_FOR_TEST + "/v1/score", json=bad, timeout=60
                )
                self.assertIn(response.status_code, (400, 422), response.text)
            self._assert_reference(post(payloads[0]), payloads[0], self.references[0])
        finally:
            terminate_and_kill_process_tree(process)

    def test_http_pointwise_and_setwise(self):
        self._check_http(False)

    def test_http_mis_and_setwise_mis(self):
        self._check_http(True)


if __name__ == "__main__":
    unittest.main()
