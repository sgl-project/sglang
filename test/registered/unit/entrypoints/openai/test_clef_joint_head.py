"""CPU checks for Clef serving; no model weights or GPU needed."""

import asyncio
import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.entrypoints.openai.protocol import DecisionRequest
from sglang.srt.entrypoints.openai.serving_clef import (
    decision_record,
    handle_clef_request,
)
from sglang.srt.entrypoints.openai.serving_decisions import OpenAIServingDecisions
from sglang.srt.entrypoints.systemone.protocol import SystemOneRequest
from sglang.srt.entrypoints.systemone.serving import SystemOneServing
from sglang.srt.environ import envs
from sglang.srt.layers.clef import validate_clef_settings, validate_record
from sglang.srt.layers.clef_reference import encode_record
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(character) for character in text])


def request():
    return DecisionRequest(
        model="clef",
        input={"state": "Unicode ✓"},
        questions=[
            {
                "id": "bool",
                "type": "yes_no",
                "question": "Allowed?",
                "yes": "Yes!",
                "no": "No!",
            },
            {
                "id": "choice",
                "type": "choice",
                "question": "Which?",
                "options": [
                    {"name": "z", "description": "last"},
                    {"name": "a", "description": "first"},
                ],
            },
            {
                "id": "score",
                "type": "score",
                "question": "How much?",
                "levels": ["low", "high"],
            },
        ],
    )


class TestClefJointHead(CustomTestCase):
    def test_joint_encoding_and_wire_round_trip(self):
        record = decision_record(request())
        encoded = encode_record(Tokenizer(), record, max_length=10000)
        assert len(encoded.questions) == 3
        assert encoded.questions[0].option_ids == ("true", "false")
        assert encoded.questions[1].option_ids == ("a", "z")
        assert encoded.questions[2].option_ids == ("0", "1")
        assert (
            validate_record(json.dumps(asdict(encoded)), list(encoded.input_ids))
            == encoded
        )
        for question in encoded.questions:
            assert (
                "".join(map(chr, encoded.input_ids[slice(*question.question_span)]))
                == record["questions"][question.question_id]["instructions"]
            )
        malformed = asdict(encoded)
        malformed["questions"][0]["question_span"] = (0, len(encoded.input_ids) + 1)
        with self.assertRaisesRegex(ValueError, "span"):
            validate_record(malformed, list(encoded.input_ids))

    def test_hidden_cache_preserves_chunks_across_slot_reuse(self):
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
        from sglang.srt.mem_cache.clef import ClefHybridLinearKVPool

        with envs.SGLANG_NATIVE_MOVE_KV_CACHE.override(True):
            pool = ClefHybridLinearKVPool(
                size=16,
                dtype=torch.bfloat16,
                page_size=1,
                head_num=1,
                head_dim=4,
                full_attention_layer_ids=[0],
                device="cpu",
                mamba_pool=None,
                clef_hidden_size=4,
            )

        allocator = TokenToKVPoolAllocator(
            16, torch.bfloat16, "cpu", pool, need_sort=False
        )
        self.assertIsNotNone(allocator.alloc(16))

        def write(slots, values):
            pool.store_clef_hidden(
                torch.tensor(slots), torch.tensor(values, dtype=torch.bfloat16)
            )

        write([1, 5, 3], [[10] * 4, [20] * 4, [11] * 4])
        write([7, 2], [[21] * 4, [12] * 4])
        torch.testing.assert_close(
            pool.gather_clef_hidden(torch.tensor([1, 3, 2])),
            torch.tensor([[10] * 4, [11] * 4, [12] * 4], dtype=torch.bfloat16),
        )
        allocator.free(torch.tensor([5, 7]))
        reused = allocator.alloc(2)
        self.assertIsNotNone(reused)
        write(reused.tolist(), [[30] * 4, [31] * 4])
        torch.testing.assert_close(
            pool.gather_clef_hidden(torch.tensor([1, 5, 7])),
            torch.tensor([[10] * 4, [30] * 4, [31] * 4], dtype=torch.bfloat16),
        )
        self.assertEqual(
            pool.clef_hidden_states.numel() * pool.clef_hidden_states.element_size(),
            17 * 4 * 2,
        )

    def test_ragged_results_follow_request_attempts_after_host_copy(self):
        from sglang.srt.layers.clef import ClefDeviceOutput, ClefResult
        from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH, Req
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
        from sglang.srt.sampling.sampling_params import SamplingParams

        records = [
            encode_record(
                Tokenizer(),
                {
                    "state": {},
                    "questions": questions,
                },
                max_length=10000,
            )
            for questions in (
                {
                    "a": {
                        "type": "choice",
                        "instructions": "Pick",
                        "criteria": {"x": "X", "y": "Y", "z": "Z"},
                    }
                },
                {
                    "b": {"type": "noul", "instructions": "Accept"},
                    "c": {
                        "type": "score",
                        "instructions": "Rate",
                        "criteria": ["low", "high"],
                    },
                },
            )
        ]
        reqs = [
            Req(str(i), "", list(record.input_ids), SamplingParams(max_new_tokens=0))
            for i, record in enumerate(records)
        ]
        for req in reqs:
            req.finished_reason = FINISH_LENGTH(0)
        output = ClefDeviceOutput(
            torch.tensor([0.1, 0.2, 0.7, 0.8, 0.2, 0.3, 0.7]),
            tuple(
                ClefResult(i, req.cache_request_handle, record)
                for i, (req, record) in enumerate(zip(reqs, records))
            ),
        )
        host = output.copy_to_host(torch.clone)
        output.probabilities.zero_()
        batch = SimpleNamespace(reqs=reqs)
        host.consume(batch, [None, None])
        self.assertEqual(set(reqs[0].customized_info["clef_probabilities"][0]), {"a"})
        self.assertAlmostEqual(
            reqs[0].customized_info["clef_probabilities"][0]["a"]["z"], 0.7, places=6
        )
        self.assertEqual(
            set(reqs[1].customized_info["clef_probabilities"][0]), {"b", "c"}
        )
        self.assertAlmostEqual(
            reqs[1].customized_info["clef_probabilities"][0]["b"]["true"], 0.8, places=6
        )
        for reason in ("retry", "aborted", "unfinished"):
            with self.subTest(reason=reason):
                req = reqs[0]
                req.customized_info = None
                req.cache_request_handle = output.results[0].request_handle
                req.finished_reason = FINISH_LENGTH(0)
                if reason == "retry":
                    req.cache_request_handle = CacheRequestHandle(req.rid, 1)
                elif reason == "aborted":
                    req.finished_reason = FINISH_ABORT()
                else:
                    req.finished_reason = None
                reqs[1].customized_info = None
                host.consume(batch, [None, None])
                self.assertIsNone(req.customized_info)
                live = reqs[1].customized_info["clef_probabilities"][0]
                self.assertAlmostEqual(live["b"]["true"], 0.8, places=6)
                self.assertAlmostEqual(live["c"]["0"], 0.3, places=6)

    def test_default_offload_parameters_do_not_enable_offloading(self):
        from sglang.srt.arg_groups.model_override_base import resolving_view
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="dummy", dtype="bfloat16", disable_prefill_cuda_graph=True
        )
        validate_clef_settings(resolving_view(args), torch.bfloat16, None)
        for field in ("cpu_offload_gb", "offload_group_size"):
            with self.subTest(field=field):
                setattr(args, field, 1)
                with self.assertRaisesRegex(ValueError, "offload"):
                    validate_clef_settings(resolving_view(args), torch.bfloat16, None)
                setattr(args, field, 0)

    def test_restricted_settings_fail_closed(self):
        args = SimpleNamespace(
            tp_size=1,
            pp_size=1,
            dp_size=1,
            max_running_requests=1,
            disable_radix_cache=True,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            enable_lora=False,
            allow_auto_truncate=False,
            chunked_prefill_size=-1,
            speculative_algorithm=None,
        )
        validate_clef_settings(args, torch.bfloat16, None)
        args.enable_lora = None
        validate_clef_settings(args, torch.bfloat16, None)
        args.enable_lora = True
        with self.assertRaisesRegex(ValueError, "LoRA"):
            validate_clef_settings(args, torch.bfloat16, None)
        args.enable_lora = None
        args.lora_paths = ["adapter"]
        with self.assertRaisesRegex(ValueError, "LoRA"):
            validate_clef_settings(args, torch.bfloat16, None)
        args.lora_paths = None
        args.disable_radix_cache = False
        args.max_running_requests = 8
        args.disable_cuda_graph = False
        args.disable_overlap_schedule = False
        args.chunked_prefill_size = 512
        validate_clef_settings(args, torch.bfloat16, None)

    def test_api_uses_one_prefill_and_preserves_probabilities(self):
        calls = []

        async def generate(internal, raw):
            calls.append(internal)
            yield {
                "meta_info": {
                    "clef_probabilities": [
                        {
                            "bool": {"true": 0.7, "false": 0.3},
                            "choice": {"a": 0.2, "z": 0.8},
                            "score": {"0": 0.6, "1": 0.4},
                        }
                    ],
                }
            }

        manager = SimpleNamespace(
            tokenizer=Tokenizer(),
            context_len=10000,
            num_reserved_tokens=0,
            generate_request=generate,
            served_model_name="clef",
        )
        serving = SimpleNamespace(
            tokenizer_manager=manager,
            route="/v1/decisions",
            _validate_server=lambda model: None,
        )
        response = asyncio.run(handle_clef_request(serving, request(), None))
        data = json.loads(response.body)
        assert len(calls) == 1
        assert calls[0].sampling_params["max_new_tokens"] == 0
        assert isinstance(calls[0].sampling_params["custom_params"]["clef_record"], str)
        assert data["answers"]["bool"]["probabilities"] == {"yes": 0.7, "no": 0.3}
        assert data["answers"]["choice"]["choice"] == "z"
        assert data["answers"]["score"]["score"] == 0.4
        assert data["usage"]["completion_tokens"] == 0
        assert "label_mass" not in data["answers"]["bool"]

    def test_empty_state_and_large_choices_use_native_encoding_on_both_routes(self):
        for count in (2, 3, 77):
            native = {
                "state": {},
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Classify this request",
                        "criteria": {
                            f"option_{i}": f"Intent {i}" for i in range(count)
                        },
                    }
                },
            }
            expected = encode_record(Tokenizer(), native, max_length=2**63 - 1)
            probabilities = {
                name: 1 / count for name in expected.questions[0].option_ids
            }
            for serving_class in (OpenAIServingDecisions, SystemOneServing):
                with self.subTest(count=count, route=serving_class.route):
                    calls = []

                    async def generate(internal, raw):
                        calls.append(internal)
                        yield {
                            "meta_info": {
                                "clef_probabilities": [{"q": probabilities}],
                            }
                        }

                    serving = object.__new__(serving_class)
                    serving.tokenizer_manager = SimpleNamespace(
                        tokenizer=Tokenizer(),
                        context_len=10000,
                        num_reserved_tokens=0,
                        generate_request=generate,
                        served_model_name="clef",
                        model_config=SimpleNamespace(clef_config={}),
                    )
                    serving._validate_server = lambda model: None
                    if serving_class is OpenAIServingDecisions:
                        wire = DecisionRequest.model_validate(
                            {
                                "model": "clef",
                                "input": native["state"],
                                "questions": [
                                    {
                                        "id": "q",
                                        "type": "choice",
                                        "question": native["questions"]["q"][
                                            "instructions"
                                        ],
                                        "options": [
                                            {"name": name, "description": description}
                                            for name, description in native[
                                                "questions"
                                            ]["q"]["criteria"].items()
                                        ],
                                    }
                                ],
                            }
                        )
                    else:
                        wire = SystemOneRequest.model_validate(
                            {"model": "clef", **native}
                        )
                    response = asyncio.run(serving.handle_request(wire, None))
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(
                        len(json.loads(response.body)["answers"]["q"]["probabilities"]),
                        count,
                    )
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0].input_ids, list(expected.input_ids))
                    self.assertEqual(
                        validate_record(
                            calls[0].sampling_params["custom_params"]["clef_record"],
                            calls[0].input_ids,
                        ),
                        expected,
                    )

    def test_aborted_requests_preserve_worker_errors_on_both_routes(self):
        from fastapi import HTTPException

        for serving_class in (OpenAIServingDecisions, SystemOneServing):
            for status in (None, 400, 503, 500, "missing_head"):
                with self.subTest(route=serving_class.route, status=status):

                    async def generate(internal, raw):
                        if status == "missing_head":
                            yield {"meta_info": {"finish_reason": {"type": "length"}}}
                            return
                        if status == 400:
                            raise ValueError("worker refused request")
                        if status is not None:
                            raise HTTPException(
                                status_code=status, detail="worker interrupted request"
                            )
                        yield {
                            "meta_info": {
                                "finish_reason": {"type": "abort", "message": "Aborted"}
                            }
                        }

                    serving = object.__new__(serving_class)
                    serving.tokenizer_manager = SimpleNamespace(
                        tokenizer=Tokenizer(),
                        context_len=10000,
                        num_reserved_tokens=0,
                        generate_request=generate,
                        served_model_name="clef",
                        model_config=SimpleNamespace(clef_config={}),
                    )
                    serving._validate_server = lambda model: None
                    wire = (
                        request()
                        if serving_class is OpenAIServingDecisions
                        else SystemOneRequest.model_validate(
                            {"model": "clef", **decision_record(request())}
                        )
                    )
                    if status == "missing_head":
                        with self.assertRaisesRegex(RuntimeError, "did not execute"):
                            asyncio.run(serving.handle_request(wire, None))
                        continue
                    response = asyncio.run(serving.handle_request(wire, None))
                    body = json.loads(response.body)
                    self.assertEqual(response.status_code, status or 503)
                    self.assertEqual(body["code"], status or 503)
                    self.assertEqual(body["object"], "error")
                    self.assertNotIn("answers", body)
                    if status is None:
                        self.assertEqual(body["type"], "RequestAborted")
                        self.assertEqual(body["message"], "Aborted")
                    else:
                        self.assertIn("worker", body["message"])

    def test_metadata_validator_rejects_unsupported_request_features(self):
        from sglang.srt.layers.clef import validate_clef_request

        encoded = encode_record(
            Tokenizer(), decision_record(request()), max_length=10000
        )
        params = {
            "max_new_tokens": 0,
            "custom_params": {"clef_record": json.dumps(asdict(encoded))},
        }
        config = SimpleNamespace(clef_config={}, vocab_size=100000)
        wire = SimpleNamespace(
            input_ids=list(encoded.input_ids),
            sampling_params=params,
            return_logprob=None,
        )
        validate_clef_request(wire, config)
        wire.sampling_params = [params]
        with self.assertRaisesRegex(ValueError, "batched"):
            validate_clef_request(wire, config)
        wire.sampling_params = params | {"n": 2}
        with self.assertRaisesRegex(ValueError, "one text-only"):
            validate_clef_request(wire, config)
        wire.sampling_params = params
        wire.audio_data = "audio"
        with self.assertRaisesRegex(ValueError, "one text-only"):
            validate_clef_request(wire, config)

    def test_serialized_probabilities_sum_to_one(self):
        from sglang.srt.layers.clef import normalized_probabilities

        probabilities = normalized_probabilities(
            torch.tensor([0.1, 0.2, 0.3], dtype=torch.bfloat16)
            .float()
            .softmax(-1)
            .tolist()
        )
        assert abs(sum(probabilities) - 1) < 1e-12
        assert (
            max(
                abs(a - b)
                for a, b in zip(
                    probabilities,
                    torch.tensor([0.1, 0.2, 0.3], dtype=torch.bfloat16)
                    .float()
                    .softmax(-1)
                    .tolist(),
                )
            )
            < 1e-7
        )

    def _check_headless_model(self, lm_head):
        import sglang.srt.layers.clef as integration
        import sglang.srt.runtime_context as context

        with (
            patch.object(
                context,
                "get_model",
                return_value=SimpleNamespace(model_path="ordinary-qwen", revision=None),
            ),
            patch.object(integration, "load_clef_config", return_value=None) as config,
        ):
            self.assertIsNone(integration.load_clef_head(lm_head, None))
            config.return_value = {"hidden_size": 8}
            with self.assertRaisesRegex(ValueError, "requires a decoder LM head"):
                integration.load_clef_head(lm_head, None)

    def test_non_clef_encoder_only_remains_supported(self):
        self._check_headless_model(None)

    def test_non_clef_pipeline_shard_remains_supported(self):
        self._check_headless_model(torch.nn.Identity())

    def test_clef_rejects_cpu_weights_without_restricting_other_models(self):
        import sglang.srt.layers.clef as integration
        import sglang.srt.runtime_context as context

        lm_head = torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16, device="cpu")
        with (
            patch.object(
                context,
                "get_model",
                return_value=SimpleNamespace(model_path="checkpoint", revision=None),
            ),
            patch.object(integration, "load_clef_config", return_value=None) as config,
        ):
            self.assertIsNone(integration.load_clef_head(lm_head, None))
            config.return_value = {"hidden_size": 8}
            with self.assertRaisesRegex(ValueError, "requires CUDA-resident"):
                integration.load_clef_head(lm_head, None)

    def test_kv_cache_storage_remains_bfloat16(self):
        from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype

        args = SimpleNamespace(
            tp_size=1,
            pp_size=1,
            dp_size=1,
            max_running_requests=1,
            disable_radix_cache=True,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            enable_lora=None,
            allow_auto_truncate=False,
            chunked_prefill_size=-1,
            speculative_algorithm=None,
        )
        for value in ("auto", "bf16", "bfloat16"):
            with self.subTest(accepted=value):
                args.kv_cache_dtype = value
                validate_clef_settings(args, torch.bfloat16, None)
                _, actual_dtype = configure_kv_cache_dtype(
                    server_args_kv_cache_dtype=value,
                    model=SimpleNamespace(quant_config=None),
                    model_dtype=torch.bfloat16,
                    is_draft_worker=False,
                    is_dflash=False,
                    speculative_draft_attention_backend="",
                )
                self.assertEqual(actual_dtype, torch.bfloat16)
        for value in (
            "fp8_e4m3",
            "fp8_e5m2",
            "mxfp8",
            "nvfp4",
            "fp4_mx_block16",
            "float16",
            "float32",
            None,
        ):
            with self.subTest(rejected=value):
                args.kv_cache_dtype = value
                with self.assertRaisesRegex(ValueError, "kv_cache_dtype"):
                    validate_clef_settings(args, torch.bfloat16, None)

    def test_rust_server_is_refused(self):
        args = SimpleNamespace(
            tp_size=1,
            pp_size=1,
            dp_size=1,
            max_running_requests=1,
            disable_radix_cache=True,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            allow_auto_truncate=False,
        )
        with envs.SGLANG_RUST_SERVER.override(False):
            validate_clef_settings(args, torch.bfloat16, None)
        with envs.SGLANG_RUST_SERVER.override(True):
            with self.assertRaisesRegex(ValueError, "SGLANG_RUST_SERVER=0"):
                validate_clef_settings(args, torch.bfloat16, None)

    def _check_unsupported_execution_mode(self, feature, value):
        args = SimpleNamespace(
            tp_size=1,
            pp_size=1,
            dp_size=1,
            max_running_requests=1,
            disable_radix_cache=True,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            enable_lora=None,
            allow_auto_truncate=False,
            chunked_prefill_size=-1,
            speculative_algorithm=None,
        )
        setattr(args, feature, value)
        with self.assertRaisesRegex(ValueError, feature):
            validate_clef_settings(args, torch.bfloat16, None)

    def test_torch_compile_is_refused(self):
        self._check_unsupported_execution_mode("enable_torch_compile", True)

    def test_unmirrored_cache_modes_are_refused(self):
        for feature in (
            "enable_unified_memory",
            "enable_hierarchical_cache",
            "enable_lmcache",
            "enable_flexkv",
        ):
            with self.subTest(feature=feature):
                self._check_unsupported_execution_mode(feature, True)

    def test_disaggregation_is_refused(self):
        self._check_unsupported_execution_mode("disaggregation_mode", "prefill")


if __name__ == "__main__":
    unittest.main()
