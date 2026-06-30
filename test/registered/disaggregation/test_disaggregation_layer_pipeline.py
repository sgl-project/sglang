"""Phase 1 e2e test for the layer-pipeline disaggregated KV transfer feature.

Starts prefill+decode with `--enable-disagg-layer-pipeline` and asserts gsm8k
accuracy + logprob structure; KV-byte corruption from layer-group transfer
collapses the gsm8k score and diverges logprob shapes.

Issue tracker: docs/developer_guide/disagg_layer_pipeline_issues.md #4
Plan: docs/developer_guide/disagg_layer_pipeline_plan.md (Phase 1 verification)
"""

import json
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

# Reuse the baseline disagg suite/budget; bump est_time for layer-group bookkeeping.
register_cuda_ci(est_time=420, suite="stage-b-test-2-gpu-large")


_LAYER_PIPELINE_ARGS = [
    "--enable-disagg-layer-pipeline",
    "--disagg-layer-group-size",
    "4",
    # Lower the threshold so short gsm8k prompts (~200 tokens) hit the layer-group
    # path; the production default of 2048 would silently fall back to single-chunk.
    "--disagg-layer-pipeline-min-prefill-len",
    "0",
]


class TestDisaggregationLayerPipeline(PDDisaggregationServerBase):
    """Layer-pipeline ON, default BF16 KV. Should match the BF16 baseline."""

    extra_prefill_args = list(_LAYER_PIPELINE_ARGS)
    extra_decode_args = list(_LAYER_PIPELINE_ARGS)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.launch_all()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics (layer-pipeline ON): {metrics}")
        # Mirror the threshold from test_disaggregation_basic; KV corruption would collapse the score.
        self.assertGreater(metrics["score"], 0.62)

    def test_logprob_shape_unchanged(self):
        """Logprob shape must be identical regardless of how KV was sliced on the wire."""
        prompt = "The capital of France is "
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0},
                "return_logprob": True,
                "return_input_logprob": True,
                "logprob_start_len": 0,
            },
        )

        j = response.json()
        completion_tokens = j["meta_info"]["completion_tokens"]
        input_logprobs = j["meta_info"]["input_token_logprobs"]
        output_logprobs = j["meta_info"]["output_token_logprobs"]

        self.assertEqual(
            len(output_logprobs),
            completion_tokens,
            "Layer-pipeline run dropped or duplicated output logprobs",
        )
        self.assertGreater(
            len(input_logprobs), 0, "Layer-pipeline run produced no input logprobs"
        )

    def test_structured_output(self):
        """Constrained decoding must still emit valid JSON when KV streamed in groups."""
        json_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[\\w]+$"},
                    "population": {"type": "integer"},
                },
                "required": ["name", "population"],
            }
        )
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": (
                    "Here is the information of the capital of France in the "
                    "JSON format.\n"
                ),
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 64,
                    "json_schema": json_schema,
                },
            },
        )
        output = response.json()["text"]
        json.loads(output)  # raises on invalid JSON


class TestDisaggregationLayerPipelineGroupSizeOne(PDDisaggregationServerBase):
    """Stress the slicing path with layer_group_size=1 — every layer is its own transfer chunk,
    catching off-by-one errors hidden at the default group_size=4."""

    extra_prefill_args = [
        "--enable-disagg-layer-pipeline",
        "--disagg-layer-group-size",
        "1",
        "--disagg-layer-pipeline-min-prefill-len",
        "0",
    ]
    extra_decode_args = list(extra_prefill_args)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.launch_all()

    def test_gsm8k_smaller_sample(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=256,
            num_examples=50,
            num_threads=64,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics (layer-pipeline ON, group_size=1): {metrics}")
        # Smaller sample → looser threshold; mainly checks no crash + non-trivial accuracy.
        self.assertGreater(metrics["score"], 0.55)


class TestDisaggregationLayerPipelineFallback(PDDisaggregationServerBase):
    """Bootstrap mismatch: prefill has layer pipeline, decode does not. The
    connection must succeed via per-connection fallback and gsm8k must hold."""

    extra_prefill_args = list(_LAYER_PIPELINE_ARGS)
    extra_decode_args = []  # decode runs in legacy single-chunk mode

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.launch_all()

    def test_gsm8k_with_mismatched_pipeline_flag(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=256,
            num_examples=50,
            num_threads=64,
        )
        metrics = run_eval(args)
        print(
            f"Evaluation metrics (mismatched pipeline flag, fallback path): {metrics}"
        )
        self.assertGreater(metrics["score"], 0.55)
