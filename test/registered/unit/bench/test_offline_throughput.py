"""Regression tests for offline benchmark input token accounting.

The random sampler shortens each sampled prompt by the number of special
tokens the server-side tokenizer will add back, so ``DatasetRow.prompt_len``
is a nominal sampling length, not the length of the final submitted text.
The benchmark must report the tokens the backend actually counted in
``meta_info.prompt_tokens`` of each generation result.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from sglang.benchmark import offline_throughput as offline_throughput_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_LATENCY = 2.0

_VOCAB = {
    "[UNK]": 0,
    "[BOS]": 1,
    "alpha": 2,
    "beta": 3,
    "gamma": 4,
    "delta": 5,
    "epsilon": 6,
}


def _make_tokenizer(with_bos):
    raw = Tokenizer(models.WordLevel(_VOCAB, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    if with_bos:
        raw.post_processor = processors.TemplateProcessing(
            single="[BOS] $A", special_tokens=[("[BOS]", 1)]
        )
    return PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", bos_token="[BOS]"
    )


class _RecordingBackend:
    """Boundary stand-in for Engine/Runtime: it records the final prompt
    texts, independently re-encodes them with the real tokenizer to produce
    ``meta_info.prompt_tokens``, and reports the requested completion count.
    It never touches the dataset rows' ``prompt_len``."""

    def __init__(self, tokenizer, backend_name):
        self._tokenizer = tokenizer
        self._backend_name = backend_name
        self.prompts = []
        self.outputs = []
        self.stopped = False

    def generate(self, prompt, sampling_params, **kwargs):
        self.prompts = list(prompt)
        # Mirror the documented upstream fan-out: with parallel sampling
        # (n > 1) each request yields n consecutive outputs, all reporting
        # that request's prompt_tokens.
        self.outputs = [
            {
                "text": "alpha beta",
                "meta_info": {
                    "prompt_tokens": len(self._tokenizer.encode(p)),
                    "completion_tokens": params["max_new_tokens"],
                },
            }
            for p, params in zip(prompt, sampling_params)
            for _ in range(params.get("n", 1))
        ]
        if self._backend_name == "runtime":
            # Runtime.generate returns the raw JSON string of the response.
            return json.dumps(self.outputs)
        return self.outputs

    def get_server_info(self):
        info = {"internal_states": [{"last_gen_throughput": 17.0}]}
        if self._backend_name == "runtime":

            async def _get_info():
                return info

            return _get_info()
        return info

    def shutdown(self):
        self.stopped = True


class TestOfflineThroughputInputAccounting(CustomTestCase):
    def _run_throughput_test(
        self,
        tmpdir,
        backend_name,
        dataset_name,
        tokenizer,
        input_len,
        output_len=2,
        num_prompts=3,
        extra_request_body=None,
        random_range_ratio=1.0,
        backend=None,
    ):
        dataset_path = os.path.join(tmpdir, f"{dataset_name}-dataset.json")
        with open(dataset_path, "w") as f:
            json.dump(
                [
                    {
                        "conversations": [
                            {"value": "alpha beta gamma delta epsilon"},
                            {"value": "alpha beta"},
                        ]
                    }
                ]
                * num_prompts,
                f,
            )
        result_path = os.path.join(tmpdir, f"{dataset_name}-result.jsonl")
        bench = offline_throughput_mod.BenchArgs(
            backend=backend_name,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            num_prompts=num_prompts,
            random_input_len=input_len,
            random_output_len=output_len,
            random_range_ratio=random_range_ratio,
            skip_warmup=True,
            result_filename=result_path,
            extra_request_body=extra_request_body,
        )
        if backend is None:
            backend = _RecordingBackend(tokenizer, backend_name)
        # Engine/Runtime construction and the macOS launch-time config
        # resolution are boundaries of this contract test; the statistics,
        # dataset sampling and JSONL writer run for real.
        cfg = SimpleNamespace(
            use_ray=False,
            tokenizer_path=tmpdir,
            model_path=tmpdir,
            _raw_input={},
            resolve_once=lambda: None,
        )
        stdout = io.StringIO()
        with (
            patch.object(offline_throughput_mod, "resolving_view", return_value=cfg),
            patch.object(offline_throughput_mod, "Engine", return_value=backend),
            patch.object(offline_throughput_mod, "Runtime", return_value=backend),
            patch.object(
                offline_throughput_mod, "get_tokenizer", return_value=tokenizer
            ),
            patch.object(offline_throughput_mod, "set_ulimit"),
            patch.object(
                offline_throughput_mod.time,
                "perf_counter",
                side_effect=[10.0, 10.0 + _LATENCY],
            ),
            contextlib.redirect_stdout(stdout),
        ):
            result = offline_throughput_mod.throughput_test(object(), bench)
        return result, backend, stdout.getvalue(), result_path

    def _assert_accounting(self, result, backend, stdout, result_path):
        expected_input = sum(len(backend._tokenizer.encode(p)) for p in backend.prompts)
        expected_output = sum(
            o["meta_info"]["completion_tokens"] for o in backend.outputs
        )
        self.assertEqual(result["total_input_tokens"], expected_input)
        self.assertEqual(result["total_output_tokens"], expected_output)
        self.assertAlmostEqual(result["input_throughput"], expected_input / _LATENCY)
        self.assertAlmostEqual(
            result["total_throughput"],
            (expected_input + expected_output) / _LATENCY,
        )
        self.assertAlmostEqual(result["output_throughput"], expected_output / _LATENCY)
        self.assertAlmostEqual(
            result["request_throughput"], len(backend.prompts) / _LATENCY
        )
        self.assertEqual(result["last_gen_throughput"], 17.0)
        # The same numbers reach the result panel and the JSONL record.
        self.assertIn(
            f"{'Total input tokens:':<40} {result['total_input_tokens']:<10}", stdout
        )
        with open(result_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(records, [result])
        self.assertTrue(backend.stopped)
        self.assertEqual(len(backend.prompts), 3)

    def test_aborted_response_invalidates_trial_and_shuts_down_runtime(self):
        tokenizer = _make_tokenizer(with_bos=True)
        for aborted_index in (0, 1):
            with (
                self.subTest(aborted_index=aborted_index),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                backend = _RecordingBackend(tokenizer, "runtime")
                generate = backend.generate

                def interrupted_generate(**kwargs):
                    outputs = json.loads(generate(**kwargs))
                    outputs[aborted_index]["meta_info"] = {
                        "completion_tokens": 0,
                        "finish_reason": {
                            "type": "abort",
                            "message": "Abort in waiting queue",
                        },
                    }
                    return json.dumps(outputs)

                with patch.object(
                    backend, "generate", side_effect=interrupted_generate
                ):
                    with self.assertRaisesRegex(RuntimeError, "aborted"):
                        self._run_throughput_test(
                            tmpdir,
                            "runtime",
                            "random",
                            tokenizer,
                            input_len=5,
                            extra_request_body='{"n": 2}',
                            backend=backend,
                        )
                self.assertTrue(backend.stopped)
                self.assertFalse(
                    os.path.exists(os.path.join(tmpdir, "random-result.jsonl"))
                )

    def test_random_with_bos_engine_backend(self):
        tokenizer = _make_tokenizer(with_bos=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            result, backend, stdout, result_path = self._run_throughput_test(
                tmpdir, "engine", "random", tokenizer, input_len=5
            )
            self._assert_accounting(result, backend, stdout, result_path)
            # Nominal sampling length is 5 - 1 (BOS), but the submitted text
            # re-encodes to 5 tokens.
            self.assertEqual(result["total_input_tokens"], 15)

    def test_random_with_bos_runtime_backend(self):
        tokenizer = _make_tokenizer(with_bos=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            result, backend, stdout, result_path = self._run_throughput_test(
                tmpdir, "runtime", "random", tokenizer, input_len=5
            )
            self._assert_accounting(result, backend, stdout, result_path)
            self.assertEqual(result["total_input_tokens"], 15)

    def test_random_with_bos_short_input(self):
        # A target length of 1 becomes a 2-token prompt once the BOS token is
        # added back, so the CLI target length cannot be used as the count.
        tokenizer = _make_tokenizer(with_bos=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            result, backend, stdout, result_path = self._run_throughput_test(
                tmpdir, "engine", "random", tokenizer, input_len=1
            )
            self._assert_accounting(result, backend, stdout, result_path)
            self.assertEqual(result["total_input_tokens"], 6)

    def test_random_without_bos_control(self):
        tokenizer = _make_tokenizer(with_bos=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            result, backend, stdout, result_path = self._run_throughput_test(
                tmpdir, "engine", "random", tokenizer, input_len=5
            )
            self._assert_accounting(result, backend, stdout, result_path)
            self.assertEqual(result["total_input_tokens"], 15)

    def test_random_without_bos_parallel_sampling_n2(self):
        # n > 1 fans each request out into two output sequences, but the
        # prompt is still submitted (and counted) once per original request.
        tokenizer = _make_tokenizer(with_bos=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            result, backend, stdout, result_path = self._run_throughput_test(
                tmpdir,
                "engine",
                "random",
                tokenizer,
                input_len=5,
                extra_request_body='{"n": 2}',
            )
            self._assert_accounting(result, backend, stdout, result_path)
            self.assertEqual(len(backend.outputs), 6)
            self.assertEqual(result["successful_requests"], 3)
            self.assertEqual(result["total_output_tokens"], 12)
            self.assertEqual(result["total_input_tokens"], 15)

    def test_random_bos_parallel_sampling_n2_heterogeneous_runtime(self):
        # Heterogeneous prompt lengths, a BOS-adding tokenizer, the runtime
        # backend and parallel sampling combined: every output sequence
        # counts toward the output total, but each original prompt only once.
        tokenizer = _make_tokenizer(with_bos=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            result, backend, stdout, result_path = self._run_throughput_test(
                tmpdir,
                "runtime",
                "random",
                tokenizer,
                input_len=6,
                extra_request_body='{"n": 2}',
                random_range_ratio=0.5,
            )
            self._assert_accounting(result, backend, stdout, result_path)
            self.assertEqual(len(backend.outputs), 6)
            self.assertEqual(result["successful_requests"], 3)

    def test_sharegpt_with_bos_control(self):
        # ShareGPT rows keep their true encoded length, so this path already
        # agreed with the backend count before the fix and must not change.
        tokenizer = _make_tokenizer(with_bos=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            result, backend, stdout, result_path = self._run_throughput_test(
                tmpdir, "engine", "sharegpt", tokenizer, input_len=5
            )
            self._assert_accounting(result, backend, stdout, result_path)


if __name__ == "__main__":
    unittest.main()
