"""Parity of the three `/generate` vocabularies: the hand-written Python
request structs (`io_struct.GenerateReqInput`, `SamplingParams`), the
generated Python types (`sglang.api.v1.api_types`), and the generated Rust
types (`rust/sglang-api-types`).

The cross-language half runs through `rust/sglang-api-types/testdata/
json_parity.json`: both generators must turn each fixture input into the same
canonical bytes (`tests/json_parity.rs` checks the Rust side). The Python half
pins the field sets against the hand-written structs and proves a schema
request still normalizes through the dataclass the Python server runs.
"""

import dataclasses
import json
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from sglang.api.v1 import api_types
from sglang.api.v1.api_types import (
    GenerateMetaInfo,
    GenerateRequest,
    GenerateResponse,
    SamplingParams,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.sampling.sampling_params import SamplingParams as PySamplingParams

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[5]
FIXTURES = REPO_ROOT / "rust" / "sglang-api-types" / "testdata" / "json_parity.json"

# `GenerateReqInput` fields with no schema counterpart. Each is either
# Engine-only (tensors, callbacks), set by the server after admission, or a
# feature the HTTP contract has not adopted. A new dataclass field must be
# added to the proto or listed here, so a client-facing field cannot slip
# into the Python server without the Rust server and the contract seeing it.
PYTHON_ONLY_REQUEST_FIELDS = frozenset(
    {
        "background",
        "cache_salt",
        "conversation_id",
        "custom_labels",
        "custom_logit_processor",
        "data_parallel_rank",
        "encoder_urls",
        "external_trace_header",
        "extra_key",
        "http_worker_ipc",
        "image_max_dynamic_patch",
        "images_config",
        "input_embeds",
        "kv_hints",
        "log_metrics",
        "lora_id",
        "lora_path",
        "max_dynamic_patch",
        "max_thinking_tokens",
        "min_dynamic_patch",
        "mm_content_hashes",
        "modalities",
        "multi_item_delimiter_indices",
        "need_wait_for_mm_inputs",
        "no_logs",
        "num_items_assigned",
        "positional_embed_overrides",
        "priority",
        "received_time",
        "require_reasoning",
        "return_bytes",
        "return_entropy",
        "return_flat_raw_output_top_logprobs",
        "return_flat_raw_top_logprobs",
        "return_flat_raw_top_logprobs_b64",
        "return_indexer_topk",
        "return_prompt_token_ids",
        "return_routed_experts",
        "return_sampling_mask",
        "routed_experts_start_len",
        "routing_key",
        "sampling_logprobs_mode",
        "session_id",
        "session_params",
        "token_indices_to_pool",
        "use_audio_in_video",
        "video_config",
        "video_max_dynamic_patch",
    }
)

# `SamplingParams` members that `normalize()` derives; never on the wire.
DERIVED_SAMPLING_FIELDS = frozenset(
    {
        "stop_strs",
        "stop_regex_strs",
        "stop_str_max_len",
        "stop_regex_max_len",
        "is_normalized",
        "ebnf_full_assistant",
    }
)


def compact(value) -> str:
    """serde_json::to_string parity: no whitespace."""
    return json.dumps(value, separators=(",", ":"))


def load_fixtures():
    with open(FIXTURES) as f:
        return json.load(f)["cases"]


class TestCrossLanguageFixtures(CustomTestCase):
    def test_fixtures_decode_to_their_canonical_bytes(self):
        """The Python emitter agrees with the Rust emitter on every fixture:
        same keys, same order, same presence and null rules, and the
        canonical form is a fixed point."""
        cases = load_fixtures()
        self.assertTrue(cases)
        for case in cases:
            with self.subTest(case=f"{case['type']}: {case['name']}"):
                cls = getattr(api_types, case["type"])
                decoded = cls.from_json_value(case["input"])
                self.assertEqual(compact(decoded.to_json_value()), case["canonical"])
                again = cls.from_json_value(json.loads(case["canonical"]))
                self.assertEqual(compact(again.to_json_value()), case["canonical"])

    def test_every_fixture_type_is_a_generated_struct(self):
        """A fixture naming a type the Python module lacks would be skipped
        silently by a lookup-by-name; make it a failure."""
        for case in load_fixtures():
            self.assertTrue(hasattr(api_types, case["type"]), case["type"])


class TestRequestFieldParity(CustomTestCase):
    def test_schema_request_fields_are_a_subset_of_the_dataclass(self):
        """Every schema field is a `GenerateReqInput` field, so a schema body
        constructs the dataclass as-is, and the dataclass-only fields are
        exactly the pinned list."""
        schema = set(GenerateRequest.__struct_fields__)
        dataclass_fields = {f.name for f in dataclasses.fields(GenerateReqInput)}
        self.assertEqual(schema - dataclass_fields, set())
        self.assertEqual(dataclass_fields - schema, PYTHON_ONLY_REQUEST_FIELDS)

    def test_schema_sampling_fields_match_the_python_api(self):
        """The schema's SamplingParams is the Python struct minus the fields
        `normalize()` derives, in the same order."""
        python_api = [
            name
            for name in PySamplingParams.__struct_fields__
            if name not in DERIVED_SAMPLING_FIELDS
        ]
        self.assertEqual(set(python_api), set(SamplingParams.__struct_fields__))
        self.assertEqual(
            set(PySamplingParams.__struct_fields__) - set(python_api),
            DERIVED_SAMPLING_FIELDS,
        )

    def test_request_fixtures_normalize_through_the_dataclass(self):
        """A schema-decoded body, re-encoded, is a valid `GenerateReqInput`
        and survives `normalize_batch_and_arguments` with its prompt, batch
        shape, stream flag, and sampling values intact."""
        for case in load_fixtures():
            if case["type"] != "GenerateRequest":
                continue
            with self.subTest(case=case["name"]):
                req = GenerateRequest.from_json_value(case["input"])
                wire = req.to_json_value()
                dataclass_req = GenerateReqInput(**wire)
                dataclass_req.normalize_batch_and_arguments()
                self.assertEqual(dataclass_req.stream, wire["stream"])
                if "text" in wire:
                    self.assertEqual(dataclass_req.text, wire["text"])
                    batch = isinstance(wire["text"], list)
                else:
                    self.assertEqual(dataclass_req.input_ids, wire["input_ids"])
                    batch = isinstance(wire["input_ids"][0], list)
                self.assertEqual(dataclass_req.is_single, not batch)
                if "sampling_params" in wire:
                    sent = wire["sampling_params"]
                    got = dataclass_req.sampling_params
                    if isinstance(sent, list):
                        self.assertEqual(len(got), len(sent))
                        for sent_item, got_item in zip(sent, got):
                            self.assertEqual({**got_item, **sent_item}, got_item)
                    else:
                        self.assertEqual({**got, **sent}, got)
                if "rid" in wire:
                    self.assertEqual(dataclass_req.rid, wire["rid"])


class TestResponseParity(CustomTestCase):
    def test_python_server_meta_info_decodes_to_the_schema_subset(self):
        """The Python server's `meta_info` carries keys the schema does not
        model (weight_version, num_retractions, spec stats, ...). The
        generated types ignore them and re-emit the schema subset, so a
        Rust-server client and a Python-server client see one shape."""
        python_frame = {
            "text": "ok",
            "meta_info": {
                "id": "rid",
                "finish_reason": {"type": "stop", "matched": 2},
                "prompt_tokens": 3,
                "weight_version": "default",
                "num_retractions": 0,
                "completion_tokens": 2,
                "cached_tokens": 1,
                "spec_accept_rate": 0.5,
                "e2e_latency": 0.25,
            },
        }
        frame = GenerateResponse.from_json_value(python_frame)
        back = frame.to_json_value()
        self.assertEqual(
            list(back["meta_info"]),
            [
                "id",
                "prompt_tokens",
                "completion_tokens",
                "finish_reason",
                "e2e_latency",
            ],
        )
        self.assertEqual(
            back["meta_info"]["finish_reason"], {"type": "stop", "matched": 2}
        )
        self.assertNotIn("output_ids", back)

    def test_response_key_order_is_the_proto_field_order(self):
        """The Rust HTTP server's memoized cumulative frame writes these keys
        by hand in this order (`Accumulated::frame_json`); the generated
        serializers must keep matching it."""
        self.assertEqual(
            list(GenerateResponse.__struct_fields__),
            ["text", "meta_info", "output_ids", "index"],
        )
        self.assertEqual(
            list(GenerateMetaInfo.__struct_fields__)[:4],
            ["id", "prompt_tokens", "completion_tokens", "finish_reason"],
        )
        self.assertEqual(list(GenerateMetaInfo.__struct_fields__)[-1], "e2e_latency")


if __name__ == "__main__":
    unittest.main()
