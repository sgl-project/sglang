"""CPU HTTP tests of the real classification handler and tokenizer lifecycle.

The ASGI fixture mounts the production handler without starting the server.
Only scheduler transport and its request packing are replaced; no GPU model
is launched, and no production class is extracted or rebuilt from source.
"""

import asyncio
import threading
import unittest
from types import SimpleNamespace

import httpx
import torch
from fastapi import FastAPI, Request
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.entrypoints.openai.protocol import ClassifyRequest
from sglang.srt.entrypoints.openai.serving_classify import OpenAIServingClassify
from sglang.srt.lora.classification_head import ClassificationHead
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CPUSchedulerManager(TokenizerManager):
    """Use production request/lease management around a CPU scheduler double."""

    def __init__(self, server_args):
        # TokenizerManager.__init__ starts sockets and process-management state.
        # Supply that boundary here while inheriting the request implementation.
        self.server_args = server_args
        self.model_path = self.served_model_name = "tiny-base"
        self.is_generation = self.enable_lora = True
        self.enable_priority_scheduling = self.enable_trace = False
        self.disaggregation_mode = DisaggregationMode.NULL
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                id2label={0: "billing", 1: "shipping", 2: "other"}
            )
        )
        self.request_logger = SimpleNamespace(
            log_requests=False,
            log_requests_level=0,
            log_received_request=lambda *args: None,
        )
        backend = Tokenizer(
            WordLevel(
                {"[UNK]": 0, "please": 1, "help": 2, "billing": 3}, unk_token="[UNK]"
            )
        )
        backend.pre_tokenizer = Whitespace()
        self.tokenizer = self.classification_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]"
        )
        self.classification_tokenizer_lock = threading.Lock()
        self.ref = LoRARef(lora_name="classifier", lora_path="unused-local-fixture")
        self.lora_registry = LoRARegistry([self.ref])
        self.lora_ref_cache = {self.ref.lora_name: self.ref}
        self.classification_heads = {
            self.ref.lora_id: ClassificationHead(
                torch.eye(3, 4),
                torch.tensor([0.0, 0.5, -0.5]),
                ("billing", "shipping", "other"),
                3,
                False,
            )
        }
        self.model_update_lock = RWLock()
        self.is_pause_cond = asyncio.Condition()
        self.is_pause = False
        self.rid_to_state = {}
        self.encoder_dispatch_ready = {}
        self._lora_release_tasks = set()
        self.sent = []
        self.response_meta = {"hidden_states": [[0.25, 2.0, 1.0, 9.0]]}

    def auto_create_handle_loop(self):
        pass

    async def _tokenize_one_request(self, obj):
        # ClassificationLease already performs real classifier tokenization.
        # No scheduler-specific tokenized message is needed by this transport.
        if obj.input_ids is None:
            obj.input_ids = self.tokenizer.encode(obj.text)
        return obj

    async def _batch_tokenize_and_process(self, batch_size, obj):
        return [await self._tokenize_one_request(obj[i]) for i in range(batch_size)]

    async def _send_one_request(self, obj):
        self.sent.append(obj)
        self.rid_to_state[obj.rid].dispatched = True

    async def _send_batch_request(self, objects):
        for obj in objects:
            await self._send_one_request(obj)

    async def _wait_one_response(self, obj, request):
        result = {
            "meta_info": {
                **self.response_meta,
                "prompt_tokens": len(obj.input_ids),
            }
        }
        if isinstance(obj, EmbeddingReqInput):
            result["embedding"] = [0.25, 2.5, 0.5]
        # The scheduler's terminal response releases its request before the
        # handler postprocesses the result; the independent CPU lease remains.
        state = self.rid_to_state.pop(obj.rid)
        task = self._release_lora_once(state)
        if task is not None:
            await task
        yield result


class TestLoRAClassificationAPI(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    async def asyncSetUp(self):
        self.enterContext(
            get_context().override_server_args(
                enable_lora=True,
                max_loaded_loras=None,
                tokenizer_worker_num=1,
                disaggregation_mode="null",
                language_only=False,
                enable_tokenizer_batch_encode=False,
                enable_dp_attention=False,
            )
        )
        self.manager = _CPUSchedulerManager(SimpleNamespace())
        self.handler = OpenAIServingClassify(self.manager, None)
        app = FastAPI()

        @app.post("/v1/classify")
        async def classify(request: ClassifyRequest, raw_request: Request):
            return await self.handler.handle_request(request, raw_request)

        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)

    async def assert_no_leases(self):
        tasks = list(self.manager._lora_release_tasks)
        if tasks:
            await asyncio.gather(*tasks)
        self.assertEqual(
            self.manager.lora_registry._counters[self.manager.ref.lora_id].value(), 0
        )
        self.assertFalse(self.manager.rid_to_state)

    async def test_repeated_text_batch_and_token_ids_use_zero_token_generation(self):
        for input_, count, tokens in (
            ("please help billing please", 1, 3),
            (["please", "help billing please help"], 2, 4),
            ([1, 2, 3, 4], 1, 3),
        ):
            with self.subTest(input=input_):
                response = await self.client.post(
                    "/v1/classify",
                    json={"model": "tiny-base:classifier", "input": input_},
                )
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["model"], "tiny-base:classifier")
                self.assertEqual(len(body["data"]), count)
                self.assertEqual(body["usage"]["prompt_tokens"], tokens)
                self.assertEqual(body["usage"]["completion_tokens"], 0)
                self.assertEqual(body["usage"]["total_tokens"], tokens)
                self.assertNotIn("hidden_states", response.text)
                for index, item in enumerate(body["data"]):
                    self.assertEqual(item["index"], index)
                    self.assertEqual(item["label"], "shipping")
                    self.assertEqual(item["num_classes"], 3)
                    torch.testing.assert_close(
                        torch.tensor(item["probs"]),
                        torch.tensor([0.25, 2.5, 0.5]).softmax(-1),
                    )
                for sent in self.manager.sent:
                    self.assertIsInstance(sent, GenerateReqInput)
                    self.assertEqual(sent.sampling_params["max_new_tokens"], 0)
                    self.assertEqual(sent.lora_id, self.manager.ref.lora_id)
                    self.assertFalse(sent.stream)
                await self.assert_no_leases()

    async def test_invalid_or_unknown_adapter_requests_do_not_reach_scheduler(self):
        for model, input_ in (
            ("tiny-base", "please"),
            ("wrong-base:classifier", "please"),
            ("tiny-base:unknown", "please"),
            ("tiny-base:classifier", " "),
            ("tiny-base:classifier", []),
        ):
            with self.subTest(model=model, input=input_):
                response = await self.client.post(
                    "/v1/classify", json={"model": model, "input": input_}
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertFalse(self.manager.sent)
                await self.assert_no_leases()

    async def test_loaded_generation_adapter_without_head_has_no_fallback(self):
        self.manager.classification_heads.clear()
        response = await self.client.post(
            "/v1/classify", json={"model": "tiny-base:classifier", "input": [1, 2]}
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("classification head", response.text)
        self.assertFalse(self.manager.sent)
        await self.assert_no_leases()

    async def test_invalid_hidden_state_returns_error_and_releases_leases(self):
        for meta in (
            {},
            {"hidden_states": [1, 2]},
            {"finish_reason": {"type": "abort"}},
        ):
            with self.subTest(meta=meta):
                self.manager.response_meta = meta
                response = await self.client.post(
                    "/v1/classify",
                    json={"model": "tiny-base:classifier", "input": [1, 2]},
                )
                self.assertEqual(response.status_code, 400, response.text)
                await self.assert_no_leases()

    async def test_protocol_rejects_non_classification_input(self):
        response = await self.client.post(
            "/v1/classify",
            json={"model": "tiny-base:classifier", "input": {"text": "please"}},
        )
        self.assertEqual(response.status_code, 422)
        self.assertFalse(self.manager.sent)

    async def test_native_embedding_classifier_preserves_its_response(self):
        self.manager.is_generation = False
        self.handler = OpenAIServingClassify(self.manager, None)
        response = await self.client.post(
            "/v1/classify", json={"model": "tiny-base", "input": [1, 2]}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsInstance(self.manager.sent[-1], EmbeddingReqInput)
        self.assertEqual(response.json()["data"][0]["label"], "shipping")
        await self.assert_no_leases()


if __name__ == "__main__":
    unittest.main()
