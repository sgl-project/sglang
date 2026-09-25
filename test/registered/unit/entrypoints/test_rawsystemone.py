"""CPU contract/dispatcher tests; no model weights or GPU imports required.

Run: python test/registered/unit/entrypoints/test_rawsystemone.py
"""

import argparse
import ast
import asyncio
import importlib.util
import json
import math
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

# Import the standalone endpoint package without sglang's GPU-heavy __init__.
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/sglang/srt/entrypoints"))
from rawsystemone.protocol import (
    RawSystemOneError,
    RawSystemOneRequest,
    RawSystemOneResponse,
)
from rawsystemone.scoring import (
    aggregate,
    native_rows,
    partition_candidates,
    plan_tokens,
    softmax,
)
from rawsystemone.service import RawSystemOneConfig, RawSystemOneService

spec = importlib.util.spec_from_file_location(
    "rawsystemone_ci", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ci
spec.loader.exec_module(ci)
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def request(prefix="ab", suffixes=None, **kwargs):
    return RawSystemOneRequest(
        prefix=prefix, suffixes=suffixes or ["c", "d", "e"], **kwargs
    )


def score_token(ids, position):
    # Depends on preceding context, not just the target ID; repeated IDs expose
    # position/alignment bugs. Includes exact zeros for denominator coverage.
    return -float(sum(ids[: position + 1]) % 13) / 8


class FakeManager:
    num_reserved_tokens = 0
    context_len = 2048
    max_req_input_len = 2048
    served_model_name = "fake-causal-model"
    model_update_epoch = 0
    enable_trace = False

    def __init__(self):
        self.calls = []
        self.active = set()
        self.aborted = set()
        self.closed = set()
        self.texts = []
        self.hook = None
        self.after_yield = None
        self.encoder = lambda text: [ord(c) for c in text]
        self.option_encoder = lambda text: [ord(c) for c in text]
        self.tokenizer = SimpleNamespace(encode=self.encode_option)

    def encode_option(self, text, *, add_special_tokens=True):
        tokens = self.option_encoder(text)
        return [1] + tokens if add_special_tokens else tokens

    async def _tokenize_texts(self, texts):
        self.texts.extend(texts)
        return [self.encoder(text) for text in texts], None

    def abort_request(self, rid):
        self.aborted.add(rid)
        self.active.discard(rid)

    async def generate_request(self, obj, raw_request):
        self.calls.append(obj)
        self.active.update(obj.rid)
        try:
            if self.hook:
                await self.hook(obj)
            result = []
            for rid, ids in zip(obj.rid, obj.input_ids):
                start = obj.logprob_start_len
                rows = [(None, ids[start], None)] + [
                    (score_token(ids, t), ids[t], None)
                    for t in range(start + 1, len(ids))
                ]
                result.append(
                    {
                        "meta_info": {
                            "id": rid,
                            "input_token_logprobs": rows,
                            "completion_tokens": 0,
                            "cached_tokens": start,
                            "finish_reason": {"type": "length", "length": 0},
                        }
                    }
                )
            # Force collection by ID, rather than relying on completion order.
            yield result[::-1]
            if self.after_yield:
                await self.after_yield(obj)
        finally:
            self.closed.update(obj.rid)
            self.active.difference_update(obj.rid)


def service(manager=None, **config):
    manager = manager or FakeManager()
    result = RawSystemOneService(manager, RawSystemOneConfig(**config))
    result._make_request = lambda **kwargs: SimpleNamespace(**kwargs)
    return result


async def eventually(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2)


def source_method(path, class_name, name, globals=None):
    """Execute the actual dependency-free method, without importing CUDA modules."""
    tree = ast.parse((ROOT / path).read_text())
    cls = (
        next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        if class_name
        else tree
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    namespace = globals or {}
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


class ContractTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("msgspec"), "Install msgspec for config/CLI checks"
    )
    def test_declared_cli_flags_and_runtime_factory(self):
        def load_module(name, relative):
            spec = importlib.util.spec_from_file_location(name, ROOT / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        arg_utils = load_module(
            "rawsystemone_arg_utils", "python/sglang/srt/arg_groups/arg_utils.py"
        )
        fake_common = ModuleType("sglang.srt.utils.common")
        fake_common.json_list_type = json.loads
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.arg_groups.arg_utils": arg_utils,
                "sglang.srt.utils.common": fake_common,
            },
        ):
            serving_module = load_module(
                "rawsystemone_serving_fields",
                "python/sglang/srt/arg_groups/fields/serving.py",
            )
            parser = argparse.ArgumentParser()
            arg_utils.add_cli_args_from_dataclass(parser, serving_module.Serving)
            args = parser.parse_args(
                [
                    "--rawsystemone-max-options",
                    "96",
                    "--rawsystemone-max-inflight-batches-per-request",
                    "4",
                ]
            )
        self.assertEqual(args.rawsystemone_max_options, 96)
        self.assertEqual(args.rawsystemone_max_inflight_batches_per_request, 4)
        serving = serving_module.Serving()
        fake_runtime = ModuleType("sglang.srt.runtime_context")
        fake_runtime.get_serving = lambda: serving
        fake_runtime.get_disagg = lambda: SimpleNamespace(disaggregation_mode="null")
        fake_runtime.get_parallel = lambda: SimpleNamespace(dp_size=1, pp_size=1)
        fake_runtime.get_exec = lambda: SimpleNamespace(
            dllm=SimpleNamespace(dllm_algorithm=None),
            features=SimpleNamespace(enable_mis=False),
        )
        fake_runtime.get_spec = lambda: SimpleNamespace(speculative_algorithm=None)
        fake_runtime.get_memory = lambda: SimpleNamespace(disable_radix_cache=False)
        manager = FakeManager()
        manager.is_generation = True
        manager.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            is_multimodal=False,
            is_encoder_decoder=False,
        )
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": fake_runtime}):
            scorer = RawSystemOneService.from_runtime(manager)
            self.assertEqual(scorer.config, RawSystemOneConfig())
            self.assertIsNone(scorer.unsupported_reason)
            serving.tokenizer_worker_num = 2
            self.assertIsNotNone(
                RawSystemOneService.from_runtime(manager).unsupported_reason
            )

    def test_strict_schema(self):
        for data in [
            {},
            {"prefix": None, "suffixes": ["x"]},
            {"prefix": 7, "suffixes": ["x"]},
            {"prefix": "", "suffixes": []},
            {"prefix": "", "suffixes": [1]},
            {"prefix": "", "suffixes": [None]},
            {"prefix": "", "suffixes": "x"},
            {"prefix": "", "suffixes": ("x",)},
            {"prefix": "", "suffixes": ["x"], "temperature": 1},
            {"prefix": "", "suffixes": ["x"], "return_token_logprobs": 1},
        ]:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                RawSystemOneRequest.model_validate(data)
        self.assertEqual(request("", [""]).suffixes, [""])

    def test_planning_and_duplicates(self):
        plan = plan_tokens([[1, 2, 3], [1, 2], [1, 2, 3], [1, 2, 4]])
        self.assertEqual(plan.common_length, 2)
        self.assertEqual(plan.original_to_unique, (1, 0, 1, 2))
        self.assertEqual(plan_tokens([[1, 2], [3, 4]]).common_length, 0)
        self.assertEqual(len(plan_tokens([[1, 2], [1, 2]]).sequences), 1)

    def test_partition_count_tokens_singletons(self):
        ids = ((1,) * 2, (1,) * 4, (1,) * 7, (1,) * 2, (1,) * 2)
        self.assertEqual(
            partition_candidates(list(range(5)), ids, 2, 6, 8), [[0, 1], [2], [3, 4]]
        )
        with self.assertRaises(RawSystemOneError):
            partition_candidates([2], ids, 2, 6, 6)

    def test_full_aggregation_and_zero(self):
        ids = (1, 2, 2, 4)
        records = native_rows(
            ids, 0, [(None, 1, None), (0.0, 2, None), (-4.0, 2, None), (-2.0, 4, None)]
        )
        self.assertEqual(aggregate(ids, records), (-6.0, 3))
        self.assertEqual(aggregate(ids[:-1], records[:-1]), (-4.0, 2))
        self.assertEqual(aggregate(ids, records, start=3), (-2.0, 1))

    def test_stable_softmax(self):
        # Large magnitudes must neither overflow nor produce 0/0. Equal
        # scores share weight, and a common shift leaves the result unchanged.
        expected = [1 / (1 + math.exp(-1)), 1 / (1 + math.exp(1))]
        for values in ([10000, 9999], [-10000, -10001]):
            for actual, target in zip(softmax(values), expected):
                self.assertAlmostEqual(actual, target)
        self.assertEqual(softmax([-1e300, -1e300]), [0.5, 0.5])
        self.assertEqual(softmax([-1e300]), [1.0])
        self.assertEqual(softmax([0, -1e300]), [1.0, 0.0])

    def test_invalid_native_rows(self):
        rows = [(None, 1, None), (-1.0, 2, None)]
        for bad in [
            [],
            rows[:1],
            [(None, 2, None), rows[1]],
            [rows[0], (None, 2, None)],
            [rows[0], (math.nan, 2, None)],
            [rows[0], (math.inf, 2, None)],
            [rows[0], (-math.inf, 2, None)],
            [rows[0], (True, 2, None)],
            [(0.0, 1, None), rows[1]],
        ]:
            with self.subTest(bad=bad), self.assertRaises(RawSystemOneError):
                native_rows((1, 2), 0, bad)

    def test_native_scheduler_offset_and_chunked_accumulation(self):
        method = source_method(
            "python/sglang/srt/managers/scheduler_components/logprob_result_processor.py",
            "SchedulerLogprobResultProcessor",
            "_process_input_token_logprobs",
        )
        processor = SimpleNamespace(
            _is_multi_item_scoring=lambda _: False,
            model_config=SimpleNamespace(vocab_size=100),
        )
        ids = [7, 7, 7, 8, 7, 99]  # includes the last valid vocabulary ID
        for start in (0, 1, 2, 4):
            req = SimpleNamespace(
                origin_input_ids=ids, logprob_start_len=start, logprob=SimpleNamespace()
            )
            # Model predictor outputs, accumulated across two prefill chunks.
            predictions = [score_token(ids, t) for t in range(start + 1, len(ids))] + [
                -99.0
            ]
            accumulated = predictions[:2] + predictions[2:]
            method(processor, req, accumulated)
            rows = list(
                zip(
                    req.logprob.input_token_logprobs_val,
                    req.logprob.input_token_logprobs_idx,
                    [None] * (len(ids) - start),
                )
            )
            records = native_rows(tuple(ids), start, rows)
            self.assertIsNone(records[0].logprob)
            for record in records[1:]:
                self.assertEqual(record.logprob, score_token(ids, record.position))
        cache_limit = source_method(
            "python/sglang/srt/managers/schedule_batch.py",
            "Req",
            "_compute_max_prefix_len",
        )
        req = SimpleNamespace(
            dllm_config=None, return_logprob=True, logprob_start_len=3
        )
        self.assertEqual(cache_limit(req, 6), 3)

    def test_actual_chunked_native_assembly(self):
        path = "python/sglang/srt/managers/scheduler_components/logprob_result_processor.py"
        cls = "SchedulerLogprobResultProcessor"
        processor = SimpleNamespace(
            _is_multi_item_scoring=lambda _: False,
            model_config=SimpleNamespace(vocab_size=100),
        )
        for name in (
            "_process_input_token_logprobs",
            "_process_input_top_logprobs",
            "_process_input_token_ids_logprobs",
            "_calculate_relevant_tokens_len",
        ):
            fn = source_method(path, cls, name)
            setattr(processor, name, lambda *args, fn=fn: fn(processor, *args))
        add_chunk = source_method(path, cls, "add_input_logprob_return_values")
        ids = (7, 7, 8, 7, 99)
        req = SimpleNamespace(
            origin_input_ids=ids,
            logprob_start_len=1,
            return_logprob=True,
            return_flat_raw_top_logprobs=False,
            input_token_logprobs=None,
            temp_input_top_logprobs_val=None,
            temp_input_top_logprobs_idx=None,
            temp_input_token_ids_logprobs_val=None,
            temp_input_token_ids_logprobs_idx=None,
            logprob=SimpleNamespace(
                input_token_logprobs_val=None,
                input_token_logprobs_idx=None,
                input_top_logprobs_val=None,
                input_top_logprobs_idx=None,
                top_logprobs_num=0,
                token_ids_logprob=None,
            ),
        )
        add_chunk(
            processor,
            0,
            req,
            SimpleNamespace(input_token_logprobs=(-0.5, -0.75)),
            0,
            2,
            False,
        )
        self.assertIsNone(req.logprob.input_token_logprobs_val)
        add_chunk(
            processor,
            0,
            req,
            SimpleNamespace(input_token_logprobs=(-1.0, -99.0)),
            0,
            2,
            True,
        )
        rows = list(
            zip(
                req.logprob.input_token_logprobs_val,
                req.logprob.input_token_logprobs_idx,
                [None] * 4,
            )
        )
        mapped = native_rows(ids, 1, rows)
        self.assertEqual([r.position for r in mapped], [1, 2, 3, 4])
        self.assertEqual([r.logprob for r in mapped], [None, -0.5, -0.75, -1.0])

    def test_no_logs_disables_native_input_and_result_logging(self):
        path = "python/sglang/srt/utils/request_logger.py"
        received = source_method(path, "RequestLogger", "log_received_request")
        finished = source_method(path, "RequestLogger", "log_finished_request")
        logger = SimpleNamespace(log_requests=True)  # accessing anything else fails
        obj = SimpleNamespace(no_logs=True)
        received(logger, obj)
        finished(logger, obj, {})

    def test_weight_epoch_covers_all_native_update_transports(self):
        path = "python/sglang/srt/managers/model_update_epoch.py"
        tree = ast.parse((ROOT / path).read_text())
        names = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        for family in [
            "UpdateWeightFromDiskReq",
            "UpdateWeightsFromTensorReq",
            "UpdateWeightsFromDistributedReq",
            "UpdateWeightsFromIPCReq",
            "BeginWeightUpdateReq",
            "EndWeightUpdateReq",
        ]:
            for suffix in ["Input", "Output"]:
                self.assertIn(family + suffix, names)
        message_type = type("WeightUpdate", (), {})
        sends = []
        fn = source_method(
            "python/sglang/srt/managers/tokenizer_manager.py",
            "TokenizerManager",
            "_dispatch_to_scheduler",
            {
                "WEIGHT_UPDATE_MESSAGES": (message_type,),
                "sock_send": lambda sock, obj: sends.append(obj),
            },
        )
        manager = SimpleNamespace(
            model_update_epoch=0, tokenizer_ipc_name=None, send_to_scheduler=None
        )
        fn(manager, message_type())
        fn(manager, object())
        fn(manager, message_type())
        self.assertEqual(manager.model_update_epoch, 2)

    def test_invalid_config(self):
        for kwargs in [
            {"max_options": 0},
            {"timeout_seconds": math.inf},
            {"max_inflight_batches": -1},
        ]:
            with self.assertRaises(ValueError):
                RawSystemOneConfig(**kwargs)

    @unittest.skipUnless(
        importlib.util.find_spec("msgspec"),
        "Install msgspec for native sampling-default checks",
    )
    def test_native_request_neutralizes_every_sampling_default(self):
        native_spec = importlib.util.spec_from_file_location(
            "rawsystemone_native_sampling",
            ROOT / "python/sglang/srt/sampling/sampling_params.py",
        )
        native = importlib.util.module_from_spec(native_spec)
        sys.modules[native_spec.name] = native
        native_spec.loader.exec_module(native)
        fake_io = ModuleType("sglang.srt.managers.io_struct")
        fake_io.GenerateReqInput = SimpleNamespace
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.managers.io_struct": fake_io,
                "sglang.srt.sampling.sampling_params": native,
            },
        ):
            obj = RawSystemOneService._make_request(
                input_ids=[[1, 2]], rid=["child"], logprob_start_len=0
            )
        self.assertEqual(obj.sampling_params["max_new_tokens"], 0)
        self.assertEqual(obj.sampling_params["temperature"], 1.0)
        self.assertEqual(obj.sampling_params["top_p"], 1.0)
        self.assertEqual(obj.sampling_params["min_p"], 0.0)
        self.assertEqual(obj.sampling_params["repetition_penalty"], 1.0)
        self.assertEqual(obj.sampling_params["n"], 1)
        for field in [
            "json_schema",
            "regex",
            "ebnf",
            "structural_tag",
            "custom_params",
            "logit_bias",
            "beam_width",
        ]:
            self.assertIsNone(obj.sampling_params[field])
        self.assertEqual(
            set(obj.sampling_params), set(native.SamplingParams.__struct_fields__)
        )
        self.assertTrue(obj.no_logs)
        self.assertTrue(obj.return_logprob)
        self.assertFalse(obj.stream)


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_native_manager_batch_submits_before_awaiting_results(self):
        manager_path = "python/sglang/srt/managers/tokenizer_manager.py"
        handle = source_method(
            manager_path,
            "TokenizerManager",
            "_handle_batch_request",
            {
                "asyncio": asyncio,
                "nullcontext": nullcontext,
                "get_bool_env_var": lambda _: False,
            },
        )
        collect = source_method(
            manager_path,
            "TokenizerManager",
            "_collect_batch_responses",
            {"asyncio": asyncio},
        )
        gates = [asyncio.Event(), asyncio.Event()]
        started = []
        children = [
            SimpleNamespace(rid=str(i), return_prompt_token_ids=False) for i in range(2)
        ]

        class Batch:
            rid = ["0", "1"]
            batch_size = 2
            parallel_sample_num = 1
            stream = False

            def __getitem__(self, i):
                return children[i]

        async def tokenize(obj):
            return obj

        async def send(obj):
            started.append(obj.rid)

        async def wait(obj, request):
            await gates[int(obj.rid)].wait()
            yield {"meta_info": {"id": obj.rid}}

        manager = SimpleNamespace(
            _should_use_batch_tokenization=lambda *_: False,
            _tokenize_one_request=tokenize,
            _send_one_request=send,
            _wait_one_response=wait,
            rid_to_state={obj.rid: SimpleNamespace() for obj in children},
        )
        manager._collect_batch_responses = lambda generators: collect(
            manager, generators
        )
        generator = handle(manager, Batch())
        task = asyncio.create_task(generator.__anext__())
        await eventually(lambda: len(started) == 2)
        gates[1].set()
        self.assertFalse(task.done())
        gates[0].set()
        self.assertEqual(len(await task), 2)
        await generator.aclose()

    async def test_http_route_validation_and_token_null(self):
        import httpx
        from fastapi import FastAPI, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse

        namespace = dict(
            RawSystemOneRequest=RawSystemOneRequest,
            Request=Request,
            RawSystemOneError=RawSystemOneError,
            ORJSONResponse=JSONResponse,
        )
        handler = source_method(
            "python/sglang/srt/entrypoints/http_server.py",
            None,
            "rawsystemone_request",
            namespace,
        )
        tree = ast.parse(
            (ROOT / "python/sglang/srt/entrypoints/http_server.py").read_text()
        )
        # There are two functions named validation_exception_handler. Select
        # the RequestValidationError handler that contains this route's branch.
        validation = next(
            n
            for n in tree.body
            if isinstance(n, ast.AsyncFunctionDef)
            and n.name == "validation_exception_handler"
            and "Invalid rawsystemone" in ast.unparse(n)
        )
        validation.decorator_list = []
        namespace["RequestValidationError"] = RequestValidationError
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[validation], type_ignores=[])
                ),
                "validation",
                "exec",
            ),
            namespace,
        )
        app = FastAPI()
        app.state.rawsystemone = service()
        app.add_api_route(
            "/v1/rawsystemone",
            handler,
            methods=["POST"],
            response_model=RawSystemOneResponse,
        )
        app.add_exception_handler(
            RequestValidationError, namespace["validation_exception_handler"]
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            result = await client.post(
                "/v1/rawsystemone", json=dict(prefix="ab", suffixes=["c"])
            )
            self.assertEqual(result.status_code, 200, result.text)
            self.assertNotIn("token_logprobs", result.json()["data"][0])
            self.assertNotIn("input_token_count", result.json()["data"][0])
            self.assertNotIn("scored_token_count", result.json()["data"][0])
            self.assertEqual(result.json()["data"][0]["option_token_count"], 1)
            self.assertEqual(
                result.json()["scoring"],
                "softmax_mean_suffix_logprob",
            )
            self.assertEqual(result.json()["tokenization"], "prefix_suffix_tokens_v1")
            self.assertEqual(result.json()["data"][0]["score"], 1.0)
            result = await client.post(
                "/v1/rawsystemone",
                json=dict(prefix="ab", suffixes=["c"], return_token_logprobs=True),
            )
            self.assertIsNone(result.json()["data"][0]["token_logprobs"][0]["logprob"])
            candidate = result.json()["data"][0]
            self.assertNotIn("input_token_count", candidate)
            self.assertNotIn("scored_token_count", candidate)
            self.assertEqual(len(candidate["token_logprobs"]), 3)
            self.assertEqual(candidate["option_token_count"], 1)
            for bad in [
                dict(prefix=1, suffixes=["x"]),
                dict(prefix="secret", suffixes=["x"], context="secret"),
            ]:
                result = await client.post("/v1/rawsystemone", json=bad)
                self.assertEqual(result.status_code, 400)
                self.assertEqual(result.json()["type"], "invalid_request")
                self.assertNotIn("secret", result.text)

    async def test_fixed_prefix_tokens_and_exact_suffix_whitespace(self):
        manager = FakeManager()
        # Whole-text encoding would merge "ab" into token 99. The endpoint
        # instead keeps the prompt's tokens fixed and appends each option.
        manager.encoder = lambda s: [1, 99] if s == "ab" else [1] + [ord(c) for c in s]
        scorer = service(manager)
        suffixes = ["b", " b", "\nλ", "  "]
        out = await scorer.score(request("a", suffixes, return_token_logprobs=True))
        self.assertEqual(manager.texts, ["a"])
        self.assertEqual(manager.calls[0].input_ids, [[1, ord("a")]])
        for suffix, candidate in zip(suffixes, out.data):
            self.assertEqual(
                [record.token_id for record in candidate.token_logprobs],
                [1, ord("a")] + [ord(c) for c in suffix],
            )
        self.assertEqual(out.data[0].option_token_count, 1)

    async def test_reference_parity_duplicates_shared_option_text_and_order(self):
        scorer = service(max_candidates_per_batch=1)
        req = request("a", ["bc", "b", "bc", "bdd", "be"], return_token_logprobs=True)
        diagnostics = {}
        out = await scorer.score(req, _diagnostics=diagnostics)
        reference = await scorer.score(req, _reference=True)
        self.assertEqual(out.data, reference.data)
        self.assertEqual(out.usage.input_tokens, 15)
        self.assertEqual(out.usage.scored_tokens, 10)
        # Shared option text stays part of the suffix score, not the prefix.
        self.assertEqual(diagnostics["shared_tokens"], 1)
        self.assertEqual(
            [b["phase"] for b in diagnostics["batches"]].count("shared_prefix"), 0
        )
        for candidate in out.data:
            self.assertIsNone(candidate.token_logprobs[0].logprob)
            self.assertTrue(
                all(r.logprob is not None for r in candidate.token_logprobs[1:])
            )

    async def test_one_unique_candidate_has_no_warmup(self):
        for suffixes in [["c"], ["c", "c", "c"]]:
            scorer = service()
            result = await scorer.score(request(suffixes=suffixes))
            self.assertEqual(len(scorer.manager.calls), 1)
            self.assertEqual(len(scorer.manager.calls[0].input_ids), 1)
            self.assertEqual(result.best_index, 0)
            self.assertEqual(
                [c.score for c in result.data], [1 / len(suffixes)] * len(suffixes)
            )

    async def test_empty_prefix_when_scoreable(self):
        scorer = service()
        scorer.manager.encoder = lambda _: [1]  # Native BOS supplies the predictor.
        await scorer.score(request("", ["ab"]))

    async def test_suffix_mean_subtracts_prefix_and_normalizes_options(self):
        scorer = service(max_candidates_per_batch=1)
        # Prefix logprob is -3/8. Suffix targets have known conditional
        # logprobs; adding prefix logprob before length normalization is wrong.
        scorer.manager.encoder = lambda _: [1, 2]
        scorer.manager.option_encoder = lambda text: {"x": [3], "long": [4, 5, 6]}[text]
        req = request("ab", ["x", "long"], return_token_logprobs=True)
        out = await scorer.score(req)
        self.assertEqual([c.option_token_count for c in out.data], [1, 3])
        self.assertEqual([c.logprob_sum for c in out.data], [-0.75, -3.0])
        weights = [math.exp(-0.75), math.exp(-1.0)]
        for candidate, weight in zip(out.data, weights):
            self.assertAlmostEqual(candidate.score, weight / sum(weights))
            full = math.fsum(r.logprob for r in candidate.token_logprobs[1:])
            self.assertEqual(candidate.logprob_sum, full - (-3 / 8))
        self.assertAlmostEqual(sum(c.score for c in out.data), 1.0)
        self.assertEqual(out.best_index, 0)
        self.assertEqual(out.usage.input_tokens, 8)
        self.assertEqual(out.usage.scored_tokens, 6)
        self.assertEqual(len(scorer.manager.calls), 3)
        self.assertEqual(scorer.manager.calls[0].input_ids, [[1, 2]])
        reference = await scorer.score(req, _reference=True)
        self.assertEqual(out.data, reference.data)

    async def test_token_equivalent_options_dedup_before_softmax(self):
        scorer = service()
        scorer.manager.encoder = lambda _: [1, 2]
        scorer.manager.option_encoder = lambda _: [3, 4]
        req = request("a", ["b", "bb", "b"])
        out = await scorer.score(req)
        self.assertEqual([c.option_token_count for c in out.data], [2, 2, 2])
        self.assertEqual([c.logprob_sum for c in out.data], [-2.0] * 3)
        self.assertEqual([c.score for c in out.data], [1 / 3] * 3)
        self.assertEqual(out.best_index, 0)
        self.assertEqual(len(scorer.manager.calls), 1)

    async def test_zero_token_options_fail_before_prefill(self):
        for suffix in ["", "ignored"]:
            scorer = service()
            scorer.manager.option_encoder = lambda _: []
            with self.assertRaises(RawSystemOneError) as caught:
                await scorer.score(request("ab", [suffix]))
            self.assertEqual(caught.exception.code, "no_option_tokens")
            self.assertEqual(caught.exception.status, 400)
            self.assertEqual(scorer.manager.calls, [])
            self.assertEqual(scorer.pending_requests, 0)

    async def test_full_validation_precedes_any_gpu_work(self):
        scenarios = [
            (dict(max_options=2), request()),
            (dict(max_request_tokens=8), request()),
            (dict(max_inflight_tokens=3), request("ab", ["c", "de"])),
            ({}, request("", ["ab", ""])),
        ]
        for cfg, req in scenarios:
            scorer = service(**cfg)
            with self.assertRaises(RawSystemOneError):
                await scorer.score(req)
            self.assertEqual(scorer.manager.calls, [])
            self.assertEqual(scorer.pending_requests, 0)
        for limit_field in ["context_len", "max_req_input_len"]:
            scorer = service()
            setattr(scorer.manager, limit_field, 3)
            with self.assertRaises(RawSystemOneError) as caught:
                await scorer.score(request())
            self.assertEqual(caught.exception.code, "context_length_exceeded")
            self.assertEqual(scorer.manager.calls, [])

    async def test_unavailable_and_unsupported(self):
        scorer = service()
        scorer.unsupported_reason = "Unsupported test mode"
        with self.assertRaises(RawSystemOneError) as caught:
            await scorer.score(request())
        self.assertEqual(caught.exception.code, "unsupported_mode")
        scorer.unsupported_reason = None
        scorer.manager.tokenizer = None
        with self.assertRaises(RawSystemOneError) as caught:
            await scorer.score(request())
        self.assertEqual(caught.exception.code, "tokenizer_unavailable")

    async def test_zero_token_prefix_fails_before_prefill(self):
        scorer = service()
        with self.assertRaises(RawSystemOneError) as caught:
            await scorer.score(request("", ["ab"]))
        self.assertEqual(caught.exception.code, "no_prefix_tokens")
        self.assertEqual(scorer.manager.calls, [])

    async def test_cache_disabled_and_short_prefix_use_batched_fallback(self):
        for disabled in [True, False]:
            scorer = service()
            scorer.cache_enabled = not disabled
            diagnostics = {}
            await scorer.score(
                request("ab" if disabled else "a", ["cd", "ef"]),
                _diagnostics=diagnostics,
            )
            self.assertEqual(len(scorer.manager.calls), 1)
            self.assertEqual(len(scorer.manager.calls[0].input_ids), 2)
            self.assertEqual(scorer.manager.calls[0].logprob_start_len, 0)
            self.assertEqual(diagnostics["mode"], "full_sequence")

    async def test_concurrent_and_work_conserving_dispatch(self):
        scorer = service(max_candidates_per_batch=1, max_inflight_batches_per_request=2)
        gates = {ord(c): asyncio.Event() for c in "cde"}
        prefix_result_ready = asyncio.Event()
        finish_prefix = asyncio.Event()
        started = set()

        async def after_yield(obj):
            if obj.logprob_start_len == 0:
                prefix_result_ready.set()
                await finish_prefix.wait()

        async def hook(obj):
            if obj.logprob_start_len:
                prefix_call = scorer.manager.calls[0]
                self.assertTrue(set(prefix_call.rid) <= scorer.manager.closed)
                self.assertEqual(obj.cache_salt, prefix_call.cache_salt)
                self.assertEqual(obj.input_ids[0][:2], prefix_call.input_ids[0])
                tail = obj.input_ids[0][-1]
                started.add(tail)
                await gates[tail].wait()

        scorer.manager.hook = hook
        scorer.manager.after_yield = after_yield
        diagnostics = {}
        task = asyncio.create_task(scorer.score(request(), _diagnostics=diagnostics))
        await asyncio.wait_for(prefix_result_ready.wait(), timeout=2)
        self.assertEqual(len(scorer.manager.calls), 1)
        self.assertEqual(scorer.manager.calls[0].input_ids, [[ord("a"), ord("b")]])
        self.assertEqual(started, set())
        finish_prefix.set()
        await eventually(lambda: len(started) == 2)
        self.assertEqual(started, {ord("c"), ord("d")})
        gates[ord("d")].set()
        await eventually(lambda: ord("e") in started)
        self.assertFalse(gates[ord("c")].is_set())
        gates[ord("c")].set()
        gates[ord("e")].set()
        await task
        self.assertEqual(diagnostics["max_inflight"], 2)
        self.assertEqual(len(scorer.manager.calls), 4)
        self.assertEqual(scorer.admission.used, [0, 0, 0])

    async def test_global_admission_bounds_multiple_parents(self):
        scorer = service(
            max_candidates_per_batch=1,
            max_inflight_batches=2,
            max_inflight_candidates=2,
            max_inflight_tokens=6,
        )
        scorer.cache_enabled = False
        gate = asyncio.Event()
        scorer.manager.hook = lambda _: gate.wait()
        tasks = [asyncio.create_task(scorer.score(request())) for _ in range(3)]
        await eventually(lambda: len(scorer.manager.calls) == 2)
        self.assertEqual(scorer.admission.used, [2, 2, 6])
        gate.set()
        await asyncio.gather(*tasks)
        self.assertEqual(len(scorer.manager.calls), 9)
        self.assertEqual(scorer.admission.used, [0, 0, 0])

    async def test_cancel_queued_and_running_batches(self):
        scorer = service(max_candidates_per_batch=1, max_inflight_batches=1)
        scorer.cache_enabled = False
        gate = asyncio.Event()
        scorer.manager.hook = lambda _: gate.wait()
        task = asyncio.create_task(scorer.score(request()))
        await eventually(
            lambda: (
                len(scorer.manager.calls) == 1 and len(scorer.admission.waiters) == 1
            )
        )
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(scorer.manager.calls), 1)
        self.assertEqual(scorer.manager.aborted, set(scorer.manager.calls[0].rid))
        self.assertEqual(scorer.admission.used, [0, 0, 0])
        self.assertFalse(scorer.admission.waiters)
        self.assertFalse(scorer.manager.active)
        self.assertEqual(scorer.pending_requests, 0)

    async def test_cancel_while_consuming_native_generator(self):
        scorer = service()
        yielded = asyncio.Event()

        async def after_yield(obj):
            yielded.set()
            await asyncio.Event().wait()

        scorer.manager.after_yield = after_yield
        task = asyncio.create_task(scorer.score(request()))
        await yielded.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        ids = set(scorer.manager.calls[0].rid)
        self.assertEqual(scorer.manager.closed, ids)
        self.assertEqual(scorer.manager.aborted, ids)
        self.assertEqual(scorer.admission.used, [0, 0, 0])
        self.assertEqual(scorer.pending_requests, 0)

    async def test_cancel_parent_before_any_admission(self):
        scorer = service(max_inflight_batches=1, max_inflight_batches_per_request=1)
        scorer.cache_enabled = False
        gate = asyncio.Event()
        scorer.manager.hook = lambda _: gate.wait()
        first = asyncio.create_task(scorer.score(request()))
        await eventually(lambda: bool(scorer.manager.calls))
        queued = asyncio.create_task(scorer.score(request()))
        await eventually(lambda: len(scorer.admission.waiters) == 1)
        queued.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await queued
        self.assertFalse(scorer.admission.waiters)
        self.assertEqual(scorer.pending_requests, 1)
        gate.set()
        await first
        self.assertEqual(len(scorer.manager.calls), 1)
        self.assertEqual(scorer.admission.used, [0, 0, 0])

    async def test_native_overload_is_typed_and_releases_admission(self):
        from fastapi import HTTPException

        scorer = service()

        async def overload(obj):
            raise HTTPException(status_code=503, detail="Private native request detail")

        scorer.manager.hook = overload
        with self.assertRaises(RawSystemOneError) as caught:
            await scorer.score(request())
        self.assertEqual(caught.exception.code, "overloaded")
        self.assertEqual(caught.exception.status, 503)
        self.assertNotIn("Private", str(caught.exception))
        self.assertEqual(scorer.admission.used, [0, 0, 0])

    async def test_failure_aborts_sibling_and_stops_queue(self):
        scorer = service(max_candidates_per_batch=1)
        started = set()

        async def hook(obj):
            if obj.logprob_start_len:
                tail = obj.input_ids[0][-1]
                started.add(tail)
                await eventually(lambda: len(started) == 2)
                if tail == ord("c"):
                    raise RuntimeError("private prompt must not escape")
                await asyncio.Event().wait()

        scorer.manager.hook = hook
        with self.assertRaises(RawSystemOneError) as caught:
            await scorer.score(request())
        self.assertEqual(str(caught.exception), "Native scoring failed.")
        self.assertEqual(started, {ord("c"), ord("d")})
        self.assertEqual(len(scorer.manager.aborted), 2)
        self.assertFalse(scorer.manager.active)
        self.assertEqual(scorer.admission.used, [0, 0, 0])

    async def test_model_change_between_prefix_and_branches(self):
        scorer = service()

        async def hook(obj):
            if obj.logprob_start_len:
                scorer.manager.model_update_epoch += 1

        scorer.manager.hook = hook
        with self.assertRaises(RawSystemOneError) as caught:
            await scorer.score(request())
        self.assertEqual(caught.exception.code, "model_changed")
        self.assertFalse(scorer.manager.active)
        self.assertEqual(scorer.admission.used, [0, 0, 0])

    async def test_timeout_and_parent_overload(self):
        scorer = service(timeout_seconds=0.05, max_pending_requests=1)
        scorer.manager.hook = lambda _: asyncio.Event().wait()
        task = asyncio.create_task(scorer.score(request()))
        await eventually(lambda: bool(scorer.manager.calls))
        with self.assertRaises(RawSystemOneError) as caught:
            await scorer.score(request())
        self.assertEqual(caught.exception.status, 429)
        with self.assertRaises(RawSystemOneError) as caught:
            await task
        self.assertEqual(caught.exception.code, "timeout")
        self.assertEqual(scorer.pending_requests, 0)
        self.assertEqual(scorer.admission.used, [0, 0, 0])

    async def test_disconnect_and_salt_propagation(self):
        scorer = service()
        disconnected = asyncio.Event()

        async def is_disconnected():
            return disconnected.is_set()

        raw = SimpleNamespace(
            headers={},
            state=SimpleNamespace(cache_salt="private-tenant"),
            is_disconnected=is_disconnected,
        )
        scorer.manager.hook = lambda _: asyncio.Event().wait()
        task = asyncio.create_task(scorer.score(request(), raw))
        await eventually(lambda: bool(scorer.manager.calls))
        self.assertEqual(scorer.manager.calls[0].cache_salt, "private-tenant")
        disconnected.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(scorer.admission.used, [0, 0, 0])

    async def test_permutation_and_candidate_set_independence(self):
        scorer = service(max_candidates_per_batch=2)
        a = await scorer.score(request("ab", ["c", "d", "eee"]))
        b = await scorer.score(request("ab", ["eee", "d", "c"]))
        alone = await scorer.score(request("ab", ["d"]))
        self.assertEqual([x.score for x in a.data], [x.score for x in reversed(b.data)])
        self.assertEqual(a.data[1].logprob_sum, alone.data[0].logprob_sum)
        self.assertEqual(a.data[1].option_token_count, alone.data[0].option_token_count)
        self.assertEqual(alone.data[0].score, 1.0)
        self.assertLess(a.data[1].score, alone.data[0].score)
        self.assertAlmostEqual(sum(x.score for x in a.data), 1.0)


if __name__ == "__main__":
    unittest.main()
