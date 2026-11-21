import asyncio
import json
import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.entrypoints.engine_score_mixin import EngineScoreMixin
from sglang.srt.entrypoints.openai.protocol import ScoringRequest
from sglang.srt.entrypoints.openai.serving_score import OpenAIServingScore
from sglang.srt.layers.attention.mis_swa_guard import merge_scheduler_swa_windows
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager_score_mixin import TokenizerManagerScoreMixin
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
from sglang.srt.runtime_context import publish, restore_context, snapshot_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class ScoringManager(TokenizerManagerScoreMixin):
    """Replace only model execution; keep request construction and score extraction real."""

    def __init__(self, enable_mis=False, generation=True):
        self.server_args = ServerArgs(model_path="dummy", enable_mis=enable_mis)
        publish(self.server_args, role="test")
        self.is_generation = generation
        self.tokenizer = SimpleNamespace(vocab_size=8)
        self.logits = torch.tensor([-1000.0, -999.0, -997.0, -996.0, 0, 1, 2, 3])
        self.requests = []

    async def generate_request(self, request, raw_request):
        self.requests.append(request)
        request.normalize_batch_and_arguments()
        results = []
        for index, ids in enumerate(request.input_ids):
            meta = {"prompt_tokens": len(ids)}
            if self.is_generation:
                assert request.sampling_params[index]["max_new_tokens"] == 0
                labels = request.token_ids_logprob[index]
                values = torch.log_softmax(self.logits, dim=0)
                logprobs = [(values[token].item(), token, None) for token in labels]
                if self.server_args.enable_mis:
                    count = len(request.multi_item_delimiter_indices[index])
                    meta["input_token_ids_logprobs"] = [logprobs] * count
                else:
                    meta["output_token_ids_logprobs"] = [logprobs]
                results.append({"meta_info": meta})
            else:
                embedding = [1.0, 3.0]
                if self.server_args.enable_mis:
                    count = len(request.multi_item_delimiter_indices[index])
                    embedding = [embedding] * count
                results.append({"meta_info": meta, "embedding": embedding})
        yield results


class TestTokenScoring(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(restore_context, snapshot_context())

    async def test_http_to_manager_ragged_candidates(self):
        for enable_mis in (False, True):
            with self.subTest(enable_mis=enable_mis):
                manager = ScoringManager(enable_mis=enable_mis)
                request = ScoringRequest(
                    query=[4],
                    items=[[5], [6, 7]],
                    label_token_ids=[[3, 1], [0, 2, 1]],
                    apply_softmax=True,
                    temperature=2.0,
                    return_token_logprobs=True,
                )
                handler = OpenAIServingScore(manager)
                response = await handler._handle_non_streaming_request(
                    request, request, None
                )
                body = json.loads(response.body)
                for labels, scores, logprobs in zip(
                    request.label_token_ids, body["scores"], body["token_logprobs"]
                ):
                    expected = torch.softmax(manager.logits[labels] / 2, dim=0)
                    torch.testing.assert_close(torch.tensor(scores), expected)
                    torch.testing.assert_close(
                        torch.tensor(logprobs),
                        torch.log_softmax(manager.logits, dim=0)[labels],
                    )
                self.assertEqual(body["usage"]["completion_tokens"], 0)

    async def test_rid_is_threaded_from_http_to_sub_requests(self):
        for generation in (True, False):
            with self.subTest(generation=generation):
                manager = ScoringManager(generation=generation)
                request = ScoringRequest(
                    query=[4], items=[[5], [6]], label_token_ids=[3, 1], rid="corr"
                )
                await OpenAIServingScore(manager)._handle_non_streaming_request(
                    request, request, None
                )
                self.assertEqual(manager.requests[-1].rid, ["corr_0", "corr_1"])

    async def test_default_scores_are_vocabulary_probabilities(self):
        manager = ScoringManager()
        result = await manager.score_request(
            query=[], items=[[4], [5]], label_token_ids=[7, 4]
        )
        expected = torch.softmax(manager.logits, dim=0)[[7, 4]]
        for scores in result.scores:
            torch.testing.assert_close(torch.tensor(scores), expected)
        self.assertIsNone(result.token_logprobs)

    async def test_mis_text_items_do_not_add_special_tokens(self):
        vocab = {"[UNK]": 0, "[BOS]": 1, "Rate": 2, ":": 3, "Option": 4, "A": 5, "B": 6}
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A", special_tokens=[("[BOS]", vocab["[BOS]"])]
        )
        manager = ScoringManager(enable_mis=True)
        manager.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, bos_token="[BOS]", unk_token="[UNK]"
        )
        items = [" Option A", " Option B"]
        for query in ("Rate:", ""):
            with self.subTest(query=query):
                await manager.score_request(
                    query=query, items=items, label_token_ids=[5, 6]
                )
                request = manager.requests[-1]
                packed = request.input_ids[0]
                delimiters = request.multi_item_delimiter_indices[0]
                prefix = packed[: delimiters[0]]
                for item, start, end in zip(items, delimiters, delimiters[1:]):
                    self.assertEqual(
                        prefix + packed[start + 1 : end],
                        manager.tokenizer.encode(query + item),
                    )

    async def test_async_engine_and_full_prompts(self):
        manager = ScoringManager()
        engine = EngineScoreMixin()
        engine.tokenizer_manager = manager
        kwargs = dict(
            label_token_ids=[[1, 3], [2]],
            apply_softmax=True,
            temperature=0.5,
            return_token_logprobs=True,
        )
        actual = await engine.async_score(query=[], items=[[4], [5]], **kwargs)
        expected = await manager.score_prompts([[4], [5]], **kwargs)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.scores[1], [1.0])

    async def test_classification_temperature(self):
        for enable_mis in (False, True):
            manager = ScoringManager(enable_mis=enable_mis, generation=False)
            result = await manager.score_request(
                query=[4], items=[[5], [6]], apply_softmax=True, temperature=2.0
            )
            for scores in result.scores:
                torch.testing.assert_close(
                    torch.tensor(scores), torch.softmax(torch.tensor([0.5, 1.5]), 0)
                )
            with self.assertRaisesRegex(ValueError, "only supported for CausalLM"):
                await manager.score_request(
                    query=[], items=[[4]], return_token_logprobs=True
                )

    async def test_invalid_candidates_and_temperature(self):
        manager = ScoringManager()
        for labels in ([], [[]], [[1]], [[1], []], [1, 1], [-1], [8], [True], "bad"):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                await manager.score_request(
                    query=[], items=[[4], [5]], label_token_ids=labels
                )
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                await manager.score_request(
                    query=[],
                    items=[[4]],
                    label_token_ids=[1],
                    temperature=temperature,
                    apply_softmax=True,
                )
        with self.assertRaisesRegex(ValueError, "requires apply_softmax"):
            await manager.score_request(
                query=[], items=[[4]], label_token_ids=[1], temperature=2.0
            )
        self.assertEqual(manager.requests, [])

    async def test_http_out_of_vocabulary_error(self):
        for enable_mis in (False, True):
            for token_id in (8, 999999):
                for labels in ([7, token_id], [[7], [1, token_id]]):
                    with self.subTest(enable_mis=enable_mis, labels=labels):
                        manager = ScoringManager(enable_mis=enable_mis)
                        request = ScoringRequest(
                            query=[],
                            items=[[4], [5]],
                            label_token_ids=labels,
                            apply_softmax=True,
                        )
                        response = await OpenAIServingScore(
                            manager
                        )._handle_non_streaming_request(request, request, None)
                        self.assertEqual(response.status_code, 400)
                        body = json.loads(response.body)
                        self.assertEqual(body["type"], "BadRequestError")
                        self.assertEqual(
                            body["message"],
                            f"Token ID {token_id} is out of vocabulary (vocab size: 8)",
                        )
                        self.assertEqual(manager.requests, [])

    async def test_small_temperature_is_finite(self):
        for enable_mis in (False, True):
            for generation in (False, True):
                manager = ScoringManager(enable_mis=enable_mis, generation=generation)
                result = await manager.score_request(
                    query=[4],
                    items=[[5]],
                    label_token_ids=[1, 3],
                    apply_softmax=True,
                    temperature=1e-300,
                )
                self.assertEqual(result.scores, [[0.0, 1.0]])

    async def test_empty_batch(self):
        manager = ScoringManager()
        result = await manager.score_request(
            query=[], items=[], label_token_ids=[], return_token_logprobs=True
        )
        self.assertEqual(result.scores, [])
        self.assertEqual(result.token_logprobs, [])
        self.assertEqual(manager.requests, [])


class TestSyncTokenScoring(unittest.TestCase):
    def test_engine_score(self):
        self.addCleanup(restore_context, snapshot_context())
        engine = EngineScoreMixin()
        engine.tokenizer_manager = ScoringManager()
        engine.loop = asyncio.new_event_loop()
        try:
            result = engine.score(
                query=[],
                items=[[4]],
                label_token_ids=[[1, 3]],
                apply_softmax=True,
                temperature=2.0,
                return_token_logprobs=True,
            )
            torch.testing.assert_close(
                torch.tensor(result.scores[0]),
                torch.softmax(engine.tokenizer_manager.logits[[1, 3]] / 2, 0),
            )
        finally:
            engine.loop.run_until_complete(engine.loop.shutdown_asyncgens())
            engine.loop.close()


class TestSWAMultiItemScoringIngressGuard(CustomTestCase):
    def setUp(self):
        self.addCleanup(restore_context, snapshot_context())

    def _make_intake_scheduler(
        self, *, min_positive_swa_window: int | None
    ) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_session_radix_cache = False
        scheduler.model_config = SimpleNamespace(hf_eos_token_id={1}, vocab_size=128)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.tokenizer = None
        scheduler.dllm_config = None
        scheduler.validate_dllm_request = lambda req: None
        scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
        scheduler._record_kv_event_lora_name = MagicMock()
        scheduler.spec_algorithm = SimpleNamespace(
            is_dflash_family=lambda: False,
            is_uno=lambda: False,
            is_none=lambda: True,
        )
        scheduler.init_req_max_new_tokens = MagicMock()
        scheduler.max_req_input_len = 2048
        scheduler.invariant_checker = MagicMock()
        scheduler.grammar_manager = SimpleNamespace(
            process_req_with_grammar=lambda req: False
        )
        scheduler._add_request_to_queue = MagicMock()
        scheduler.output_streamer = MagicMock()
        scheduler.tree_cache = MagicMock(cache_controller=None)
        scheduler.beam_coordinator = MagicMock()
        scheduler.hisparse_coordinator = None
        scheduler.uocr_runtime = None
        scheduler.send_abort_output = MagicMock()
        scheduler.min_positive_swa_window = min_positive_swa_window
        scheduler.init_request_retirer()
        return scheduler

    def _make_tokenized_intake_req(
        self, *, input_ids: list[int], delimiters: list[int]
    ) -> SimpleNamespace:
        return SimpleNamespace(
            rid="swa-intake-req",
            input_text="",
            input_ids=array("q", input_ids),
            sampling_params=SamplingParams(max_new_tokens=0),
            mm_inputs=None,
            session_params=None,
            session_id=None,
            input_embeds=None,
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            return_sampling_mask=False,
            sampling_logprobs_mode="top_k",
            return_flat_raw_top_logprobs=False,
            stream=False,
            lora_id=None,
            lora_name=None,
            positional_embed_overrides=None,
            token_type_ids=None,
            custom_logit_processor=None,
            require_reasoning=False,
            return_hidden_states=False,
            return_routed_experts=False,
            routed_experts_start_len=0,
            return_indexer_topk=False,
            bootstrap_host=None,
            bootstrap_port=1,
            bootstrap_room=None,
            routed_dp_rank=None,
            disagg_prefill_dp_rank=None,
            kv_hints=None,
            priority=None,
            routing_key=None,
            extra_key=None,
            cache_salt=None,
            http_worker_ipc=None,
            time_stats=None,
            token_indices_to_pool=None,
            pp_prefetch_ticketed=False,
            dimensions=None,
            encoding_format=None,
            return_pooled_hidden_states=False,
            decision_layout=None,
            multi_item_delimiter_indices=delimiters,
        )

    def test_ingress_guard_boundary_and_same_instance_recovery(self):
        manager = ScoringManager(enable_mis=True)
        manager.min_positive_swa_window = 31

        combined, delims = manager._build_multi_item_token_sequence(
            [1] * 31, [[2]], 9999
        )
        self.assertEqual(len(combined), 34)
        self.assertEqual(delims, [31, 33])

        with self.assertRaises(ValueError):
            manager._build_multi_item_token_sequence([1] * 31, [[2, 3]], 9999)
        with self.assertRaises(ValueError):
            manager._build_multi_item_token_sequence([1] * 32, [[2]], 9999)

        handler = OpenAIServingScore(manager)
        bad_req = ScoringRequest(
            query=[4] * 32, items=[[5]], label_token_ids=[1, 3], apply_softmax=False
        )
        bad_resp = asyncio.run(
            handler._handle_non_streaming_request(bad_req, bad_req, None)
        )
        self.assertEqual(bad_resp.status_code, 400)
        self.assertEqual(json.loads(bad_resp.body)["type"], "BadRequestError")
        self.assertEqual(manager.requests, [])

        ok_req = ScoringRequest(
            query=[4] * 31, items=[[5]], label_token_ids=[1, 3], apply_softmax=False
        )
        ok_resp = asyncio.run(
            handler._handle_non_streaming_request(ok_req, ok_req, None)
        )
        self.assertEqual(ok_resp.status_code, 200)
        self.assertEqual(len(manager.requests), 1)

        engine = EngineScoreMixin()
        engine.tokenizer_manager = manager
        engine.loop = asyncio.new_event_loop()
        try:
            with self.assertRaises(ValueError):
                engine.score(
                    query=[4] * 31,
                    items=[[5, 6]],
                    label_token_ids=[1, 3],
                    apply_softmax=False,
                )
            self.assertEqual(len(manager.requests), 1)

            recovered = engine.score(
                query=[4] * 31,
                items=[[5]],
                label_token_ids=[1, 3],
                apply_softmax=False,
            )
            self.assertEqual(len(recovered.scores), 1)
            self.assertEqual(len(manager.requests), 2)
        finally:
            engine.loop.run_until_complete(engine.loop.shutdown_asyncgens())
            engine.loop.close()

        non_mis_manager = ScoringManager(enable_mis=False)
        non_mis_manager.min_positive_swa_window = 31
        non_mis_res = asyncio.run(
            non_mis_manager.score_request(
                query=[4] * 64,
                items=[[5] * 8, [6] * 8],
                label_token_ids=[1, 3],
                apply_softmax=False,
            )
        )
        self.assertEqual(len(non_mis_res.scores), 2)

    def test_scheduler_retires_overflow_without_queuing_and_respects_enable_mis(self):
        for method_name in ("handle_generate_request", "handle_embedding_request"):
            with self.subTest(entrypoint=method_name, case="overflow_rejected"):
                publish(ServerArgs(model_path="dummy", enable_mis=True), role="test")
                scheduler = self._make_intake_scheduler(min_positive_swa_window=31)
                bad_req = self._make_tokenized_intake_req(
                    input_ids=list(range(35)), delimiters=[32, 34]
                )
                getattr(scheduler, method_name)(bad_req)
                scheduler._add_request_to_queue.assert_not_called()
                scheduler.output_streamer.stream_output.assert_called_once()
                streamed_reqs = scheduler.output_streamer.stream_output.call_args.args[
                    0
                ]
                self.assertEqual(len(streamed_reqs), 1)
                retired_req = streamed_reqs[0]
                self.assertTrue(retired_req.finished())
                self.assertEqual(
                    retired_req.finished_reason.status_code, HTTPStatus.BAD_REQUEST
                )
                self.assertIn("window_left", retired_req.finished_reason.message)
                scheduler.tree_cache.finish.assert_called_once()
                self.assertEqual(
                    scheduler.tree_cache.finish.call_args.args[1],
                    CacheRequestOutcome.ABORT,
                )

            with self.subTest(entrypoint=method_name, case="valid_boundary_queued"):
                publish(ServerArgs(model_path="dummy", enable_mis=True), role="test")
                scheduler = self._make_intake_scheduler(min_positive_swa_window=31)
                ok_req = self._make_tokenized_intake_req(
                    input_ids=list(range(34)), delimiters=[31, 33]
                )
                getattr(scheduler, method_name)(ok_req)
                scheduler._add_request_to_queue.assert_called_once()
                scheduler.output_streamer.stream_output.assert_not_called()

            with self.subTest(entrypoint=method_name, case="enable_mis_false_ignored"):
                publish(ServerArgs(model_path="dummy", enable_mis=False), role="test")
                scheduler = self._make_intake_scheduler(min_positive_swa_window=31)
                ignored_req = self._make_tokenized_intake_req(
                    input_ids=list(range(35)), delimiters=[32, 34]
                )
                getattr(scheduler, method_name)(ignored_req)
                scheduler._add_request_to_queue.assert_called_once()
                scheduler.output_streamer.stream_output.assert_not_called()

        infos = [
            {"min_positive_swa_window": None},
            {"min_positive_swa_window": 32},
            {"min_positive_swa_window": 31},
        ]
        self.assertEqual(merge_scheduler_swa_windows(infos=infos), 31)
        self.assertTrue(all(info["min_positive_swa_window"] == 31 for info in infos))


if __name__ == "__main__":
    unittest.main()
