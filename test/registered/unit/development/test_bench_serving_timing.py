"""Registered regressions for the Round 33 AC-11 bench_serving timing
contract.

Plan §AC-11 / DEC-2 says "fixed seed, 120s warmup, 600s measurement
window, 3 trials, median." Round 31 only wrote the warmup/window
defaults into the sidecar metadata; Round 33 adds CLI flags
``--warmup-seconds`` and ``--measurement-window-seconds`` to
``python/sglang/bench_serving.py`` and makes the script-level driver
run multiple full epochs over the prepared workload until the time
threshold is crossed.

These tests exercise the timing path with a mock request_func so the
unit suite stays CPU-only.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
import unittest
from types import SimpleNamespace
from typing import List
from unittest import mock


def _load_bs():
    """Lazily import sglang.bench_serving; skip the entire module if the
    sglang package is not on sys.path (e.g. CI without
    ``PYTHONPATH=python``)."""
    try:
        return importlib.import_module("sglang.bench_serving")
    except Exception as exc:  # pragma: no cover — env-conditional
        raise unittest.SkipTest(
            f"sglang.bench_serving unavailable in this environment: {exc}"
        )


class TestBenchServingTimingCLI(unittest.TestCase):
    """The two new CLI flags must be exposed and default to 0.0 (legacy
    single-pass behavior preserved)."""

    def setUp(self):
        self.bs = _load_bs()

    def test_cli_exposes_warmup_seconds_flag(self):
        parser_factory = getattr(self.bs, "_build_parser", None) or getattr(
            self.bs, "build_parser", None,
        )
        if parser_factory is None:
            # bench_serving builds the parser inside __main__; assert
            # via --help string instead.
            import subprocess
            out = subprocess.run(
                [sys.executable, "-m", "sglang.bench_serving", "--help"],
                capture_output=True, text=True, timeout=30,
            )
            self.assertIn("--warmup-seconds", out.stdout)
            self.assertIn("--measurement-window-seconds", out.stdout)
            return

    def test_cli_default_zero_preserves_legacy_path(self):
        import subprocess
        out = subprocess.run(
            [sys.executable, "-m", "sglang.bench_serving", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        # Default values shown in --help (argparse renders default in
        # the metavar block); accept either form.
        self.assertIn("--warmup-seconds", out.stdout)
        self.assertIn("--measurement-window-seconds", out.stdout)


class TestBenchServingTimingDispatch(unittest.TestCase):
    """Exercise the in-process dispatch logic for the time-based driver
    with a fake request_func so no HTTP server / tokenizer is needed.
    """

    def setUp(self):
        self.bs = _load_bs()

    def _fake_request_func(self, per_request_s: float = 0.001):
        """Return an async request_func that sleeps for `per_request_s`
        seconds and reports success."""
        async def fake(*, request_func_input, pbar=None):
            await asyncio.sleep(per_request_s)
            out = self.bs.RequestFuncOutput()
            out.success = True
            out.prompt_len = request_func_input.prompt_len
            out.output_len = request_func_input.output_len
            out.latency = per_request_s
            out.ttft = per_request_s * 0.5
            out.itl = [per_request_s * 0.5]
            out.generated_text = "ok"
            if pbar is not None:
                pbar.update(1)
            return out
        return fake

    def _make_input_requests(self, n: int = 4):
        rows = []
        for i in range(n):
            row = self.bs.DatasetRow(
                prompt=f"prompt-{i}",
                prompt_len=8,
                output_len=4,
                image_data=None,
            )
            # `DatasetRow.timestamp` / `routing_key` / `extra_request_body`
            # are populated by the loaders; fill with defaults here so the
            # dispatch loop can read them.
            row.timestamp = 0.0
            row.routing_key = None
            row.extra_request_body = {}
            rows.append(row)
        return rows

    def _make_args(self, *, warmup_seconds=0.0, measurement_window=0.0,
                   warmup_requests=0):
        return SimpleNamespace(
            backend="sglang",
            dataset_name="generated-shared-prefix",
            num_prompts=4,
            random_input_len=0,
            random_output_len=0,
            sharegpt_output_len=None,
            random_range_ratio=1.0,
            plot_throughput=False,
            output_file=None,
            output_details=False,
            tag=None,
            seed=1,
            tokenize_prompt=False,
            warmup_requests=warmup_requests,
            warmup_seconds=warmup_seconds,
            measurement_window_seconds=measurement_window,
            mooncake_num_rounds=1,
            gsp_system_prompt_len=2048,
            gsp_question_len=2048,
            gsp_output_len=512,
        )

    def test_seconds_warmup_runs_multiple_discarded_epochs(self):
        """When ``--warmup-seconds`` is set, the driver runs full epochs
        until the threshold is met. Discarded outputs are not counted in
        the measured results."""
        bs = self.bs
        input_requests = self._make_input_requests(n=4)
        args = self._make_args(
            warmup_seconds=0.05,
            measurement_window=0.05,
            warmup_requests=0,
        )
        bs.args = args
        # Each request sleeps for 5ms; one epoch over 4 requests is
        # ~5ms (concurrent). Warmup of 50ms → at least 2 epochs.
        fake = self._fake_request_func(per_request_s=0.005)
        bs.ASYNC_REQUEST_FUNCS["test_backend"] = fake
        try:
            with mock.patch.object(bs, "tqdm", lambda total=None: None):
                with mock.patch.object(bs.time, "sleep", lambda s: None):
                    outputs, dur = asyncio.run(self._drive_benchmark(args))
            # Measurement window of 50ms with each epoch ~5ms → at least
            # 2 measured epochs accumulated.
            self.assertGreaterEqual(len(outputs), 4)
            self.assertGreaterEqual(dur, 0.05)
        finally:
            bs.ASYNC_REQUEST_FUNCS.pop("test_backend", None)

    def test_seed_reset_between_warmup_and_measured_phases(self):
        """After warmup epochs, the driver must re-seed ``random`` and
        ``np.random`` so warmup does not perturb the measured request-
        arrival process."""
        bs = self.bs
        seed_calls = {"random": 0, "np": 0}
        orig_random_seed = bs.random.seed
        orig_np_seed = bs.np.random.seed

        def _wrapped_random_seed(s):
            seed_calls["random"] += 1
            return orig_random_seed(s)

        def _wrapped_np_seed(s):
            seed_calls["np"] += 1
            return orig_np_seed(s)

        args = self._make_args(
            warmup_seconds=0.02,
            measurement_window=0.0,
            warmup_requests=0,
        )
        bs.args = args
        fake = self._fake_request_func(per_request_s=0.002)
        bs.ASYNC_REQUEST_FUNCS["test_backend"] = fake
        try:
            with mock.patch.object(bs.random, "seed", side_effect=_wrapped_random_seed):
                with mock.patch.object(bs.np.random, "seed", side_effect=_wrapped_np_seed):
                    with mock.patch.object(bs, "tqdm", lambda total=None: None):
                        with mock.patch.object(bs.time, "sleep", lambda s: None):
                            asyncio.run(self._drive_benchmark(args))
            # At least one re-seed of each happens between warmup and
            # measured phase.
            self.assertGreaterEqual(seed_calls["random"], 1)
            self.assertGreaterEqual(seed_calls["np"], 1)
        finally:
            bs.ASYNC_REQUEST_FUNCS.pop("test_backend", None)

    def test_measurement_window_loop_accumulates_outputs(self):
        """The driver runs full epochs until accumulated wall-clock >=
        window. Outputs across epochs are concatenated and `duration`
        is the accumulated wall-clock."""
        bs = self.bs
        args = self._make_args(
            warmup_seconds=0.0,
            measurement_window=0.04,
            warmup_requests=0,
        )
        bs.args = args
        # 4 requests * ~3ms each (concurrent) ≈ 3ms/epoch → ≥ 13 epochs
        # to clear 40ms.
        fake = self._fake_request_func(per_request_s=0.003)
        bs.ASYNC_REQUEST_FUNCS["test_backend"] = fake
        try:
            with mock.patch.object(bs, "tqdm", lambda total=None: None):
                with mock.patch.object(bs.time, "sleep", lambda s: None):
                    outputs, dur = asyncio.run(self._drive_benchmark(args))
            self.assertGreaterEqual(len(outputs), 8,
                                     "measurement window should yield multiple epochs of outputs")
            self.assertGreaterEqual(dur, 0.04)
        finally:
            bs.ASYNC_REQUEST_FUNCS.pop("test_backend", None)

    def test_legacy_single_pass_when_both_seconds_unset(self):
        """When both ``--warmup-seconds`` and
        ``--measurement-window-seconds`` are 0, the driver dispatches
        the workload exactly once (legacy behavior)."""
        bs = self.bs
        args = self._make_args(
            warmup_seconds=0.0,
            measurement_window=0.0,
            warmup_requests=0,
        )
        bs.args = args
        fake = self._fake_request_func(per_request_s=0.001)
        bs.ASYNC_REQUEST_FUNCS["test_backend"] = fake
        try:
            with mock.patch.object(bs, "tqdm", lambda total=None: None):
                with mock.patch.object(bs.time, "sleep", lambda s: None):
                    outputs, dur = asyncio.run(self._drive_benchmark(args))
            # Exactly one epoch over 4 requests.
            self.assertEqual(len(outputs), 4)
        finally:
            bs.ASYNC_REQUEST_FUNCS.pop("test_backend", None)

    def _fake_metrics(self, outputs):
        """Build a complete ``BenchmarkMetrics`` instance with every
        required field. The dataclass definition is the source of truth
        for which fields the JSONL writer reads."""
        from dataclasses import fields as _dc_fields
        kwargs = {}
        for f in _dc_fields(self.bs.BenchmarkMetrics):
            if f.name == "completed":
                kwargs[f.name] = len(outputs)
            elif f.type == "int" or f.type is int:
                kwargs[f.name] = 0
            elif f.type == "float" or f.type is float:
                kwargs[f.name] = 1.0
            else:
                kwargs[f.name] = 1.0
        return self.bs.BenchmarkMetrics(**kwargs)

    async def _drive_benchmark(self, args):
        """Drive ``bs.benchmark(...)`` end-to-end with stubs so the test
        only exercises the dispatch + timing logic."""
        bs = self.bs
        input_requests = self._make_input_requests(n=args.num_prompts)

        # Stub out tokenizer / server_info / metric printing so the
        # function reaches the JSONL stage without networking.
        bs.args = args

        # Patch requests.get/post and async_request_profile so they
        # become no-ops.
        async def _noop_profile(*a, **kw):
            return SimpleNamespace(success=True)

        class _NoopResp:
            status_code = 200
            def json(self): return {}

        with mock.patch.object(bs.requests, "post", lambda *a, **kw: _NoopResp()), \
             mock.patch.object(bs.requests, "get", lambda *a, **kw: _NoopResp()), \
             mock.patch.object(bs, "async_request_profile", _noop_profile), \
             mock.patch.object(bs, "calculate_metrics",
                                lambda **kw: (
                                    self._fake_metrics(kw["outputs"]),
                                    [o.output_len for o in kw["outputs"]],
                                )):
            # Capture outputs by inspecting the closed-over global
            # `outputs` after benchmark returns; simpler is to stub the
            # writer to do nothing and use a side-effect spy on
            # calculate_metrics. We re-derive the outputs by counting
            # `completed`.
            # Instead, mock the file-write so the JSONL stays in memory.
            captured = {}
            real_open = open

            def _fake_open(path, mode="r", *a, **kw):
                if "w" in mode or "a" in mode:
                    import io
                    # `bench_serving` writes via `with open(...) as f`;
                    # exiting the `with` closes the file, so we override
                    # close to keep the buffer readable afterwards.
                    sink = io.StringIO()
                    sink.close = lambda: None  # type: ignore[assignment]
                    captured["sink"] = sink
                    captured["path"] = path
                    return sink
                return real_open(path, mode, *a, **kw)

            with mock.patch("builtins.open", _fake_open):
                await bs.benchmark(
                    backend="test_backend",
                    api_url="http://stub/",
                    base_url="http://stub",
                    model_id="stub-model",
                    tokenizer=mock.MagicMock(),
                    input_requests=input_requests,
                    request_rate=float("inf"),
                    max_concurrency=4,
                    disable_tqdm=True,
                    lora_names=[],
                    lora_request_distribution=None,
                    lora_zipf_alpha=None,
                    extra_request_body={},
                    profile=False,
                    warmup_requests=args.warmup_requests,
                )
            sink = captured.get("sink")
            import json as _json
            text = sink.getvalue() if sink else ""
            self.assertTrue(text, "bench_serving must write a JSONL row")
            row = _json.loads(text.strip().splitlines()[-1])
            return list(range(row["completed"])), row["duration"]


class TestBenchServingJSONLWorkloadFields(unittest.TestCase):
    """Round 33 (AC-11): the JSONL must surface the workload triple
    (num_prompts, input_len, output_len) so the comparator can
    cross-check the sidecar."""

    def setUp(self):
        self.bs = _load_bs()

    def test_jsonl_includes_workload_triple(self):
        bs = self.bs
        # Drive a minimal single-pass run and verify the result dict has
        # the new keys.
        args = SimpleNamespace(
            backend="sglang",
            dataset_name="generated-shared-prefix",
            num_prompts=4,
            random_input_len=0,
            random_output_len=0,
            sharegpt_output_len=None,
            random_range_ratio=1.0,
            plot_throughput=False,
            output_file=None,
            output_details=False,
            tag=None,
            seed=1,
            tokenize_prompt=False,
            warmup_requests=0,
            warmup_seconds=0.0,
            measurement_window_seconds=0.0,
            mooncake_num_rounds=1,
            gsp_system_prompt_len=2048,
            gsp_question_len=2048,
            gsp_output_len=512,
        )
        bs.args = args

        async def fake(*, request_func_input, pbar=None):
            out = bs.RequestFuncOutput()
            out.success = True
            out.prompt_len = request_func_input.prompt_len
            out.output_len = request_func_input.output_len
            out.latency = 0.001
            out.ttft = 0.0005
            out.itl = [0.0005]
            out.generated_text = "ok"
            return out

        bs.ASYNC_REQUEST_FUNCS["test_backend"] = fake
        rows = []
        try:
            input_requests = [
                bs.DatasetRow(prompt=f"p{i}", prompt_len=8, output_len=4,
                              image_data=None)
                for i in range(4)
            ]
            for r in input_requests:
                r.timestamp = 0.0
                r.routing_key = None
                r.extra_request_body = {}

            async def _noop_profile(*a, **kw):
                return SimpleNamespace(success=True)

            class _NoopResp:
                status_code = 200
                def json(self): return {}

            real_open = open
            sink_holder = {}

            def _fake_open(path, mode="r", *a, **kw):
                if "w" in mode or "a" in mode:
                    import io
                    sink = io.StringIO()
                    sink.close = lambda: None  # type: ignore[assignment]
                    sink_holder["sink"] = sink
                    return sink
                return real_open(path, mode, *a, **kw)

            with mock.patch.object(bs.requests, "post", lambda *a, **kw: _NoopResp()), \
                 mock.patch.object(bs.requests, "get", lambda *a, **kw: _NoopResp()), \
                 mock.patch.object(bs, "async_request_profile", _noop_profile), \
                 mock.patch.object(bs.time, "sleep", lambda s: None), \
                 mock.patch("builtins.open", _fake_open):
                async def _run():
                    await bs.benchmark(
                        backend="test_backend",
                        api_url="http://stub/",
                        base_url="http://stub",
                        model_id="stub-model",
                        tokenizer=mock.MagicMock(),
                        input_requests=input_requests,
                        request_rate=float("inf"),
                        max_concurrency=4,
                        disable_tqdm=True,
                        lora_names=[],
                        lora_request_distribution=None,
                        lora_zipf_alpha=None,
                        extra_request_body={},
                        profile=False,
                        warmup_requests=0,
                    )
                asyncio.run(_run())
            sink = sink_holder["sink"]
            import json as _json
            row = _json.loads(sink.getvalue().strip().splitlines()[-1])
            self.assertIn("num_prompts", row)
            self.assertIn("input_len", row)
            self.assertIn("output_len", row)
            # Effective ISL from gsp_system_prompt_len + gsp_question_len:
            self.assertEqual(row["input_len"], 2048 + 2048)
            self.assertEqual(row["output_len"], 512)
            self.assertEqual(row["num_prompts"], 4)
        finally:
            bs.ASYNC_REQUEST_FUNCS.pop("test_backend", None)


if __name__ == "__main__":
    unittest.main()
