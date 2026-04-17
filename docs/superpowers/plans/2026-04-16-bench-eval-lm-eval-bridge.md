# bench_eval: lm-eval × bench_serving Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `sglang.bench_eval` — a tool that runs any generative lm-evaluation-harness task through `sglang.bench_serving`, producing a single report with both accuracy (from lm-eval) and serving performance (TTFT, ITL, output throughput, per-request output length) for the exact same workload, with first-class support for chain-of-thought / thinking-mode chat templates.

**Architecture:** Subclass `lm_eval.api.model.LM` and route its `generate_until(requests)` through `sglang.bench_serving.benchmark(...)`. Let `lm_eval.simple_evaluate(...)` drive prompt construction (fewshot, chat template, `enable_thinking`), filter application (e.g. gsm8k strict-match vs flexible-extract), aggregation, and stderr — we only own the generation step. The custom LM stashes the bench_serving perf metrics on `self.last_perf`; a thin report module merges those with `simple_evaluate`'s result dict.

**Tech Stack:** Python 3.10+, `lm-eval[api]>=0.4.9.2` (already declared in `python/pyproject.toml:143`), `sglang.bench_serving` internals, `aiohttp` (for the mock integration test), `transformers` tokenizers.

---

## Design Decisions

Up-front so a reviewer can push back before tasks start:

1. **Who builds prompts:** `simple_evaluate`, not us. It handles fewshot context, chat template application, `fewshot_as_multiturn`, description/system prompt, stop strings, and all filter pipelines. Our LM's `generate_until` receives ready-to-send prompt strings via `Instance.args[0]`.
2. **Who scores:** `simple_evaluate`, not us. `results["results"][task_name]` has aggregated metrics with filter-tagged keys (`exact_match,strict-match`, `exact_match,flexible-extract`) and stderr.
3. **What we own:** one class `BenchServingLM(LM)` that implements `generate_until` by building `DatasetRow`s from the incoming Instances and invoking `sglang.bench_serving.benchmark(...)`. Plus a small report module.
4. **Backend default:** `sglang-oai` (/v1/completions) — mirrors lm-eval's `local-completions` path (raw-string prompts, plain completion response).
5. **Thinking mode:** we override `LM.apply_chat_template` so `simple_evaluate(apply_chat_template=True)` injects `enable_thinking=True` whenever the tool is launched with `--enable-thinking`. No reliance on `simple_evaluate` passing a `chat_template_kwargs` argument through.
6. **`<think>…</think>` parsing:** handled by the server (`--reasoning-parser`) OR by lm-eval's filters (most CoT tasks regex the final answer). We don't need to strip anything in our LM.
7. **Task support:** every generative (`output_type=="generate_until"`) task lm-eval ships — gsm8k, mmlu_flan_cot_zeroshot, gpqa_diamond_cot_zeroshot, mmlu_pro, math, aime, humaneval, ... Loglikelihood tasks raise `NotImplementedError` from `loglikelihood` / `loglikelihood_rolling` with a clear message.
8. **Report:** append-mode JSONL — one line per run, containing `task`, `accuracy` (lm-eval's per-metric dict), `perf` (bench_serving subset), and the run's config.
9. **Prereq install:** lm-eval is not yet in the `sglang` conda env. Task 1 installs it.
10. **Tests:** live in `test/registered/bench_fn/` (matches `test_benchmark_datasets_api.py`), register as CPU CI. Integration test uses an in-process aiohttp mock — no GPU required.

## File Structure

```
python/sglang/bench_eval.py                                 # NEW — CLI + orchestration
python/sglang/benchmark/eval_harness/__init__.py            # NEW — public re-exports
python/sglang/benchmark/eval_harness/bench_serving_lm.py    # NEW — BenchServingLM(LM)
python/sglang/benchmark/eval_harness/report.py              # NEW — merge perf + lm-eval results
test/registered/bench_fn/test_bench_eval.py                 # NEW — unit + integration tests
```

No existing files are modified. `python/sglang/bench_serving.py` stays unchanged — we import `benchmark` and `set_global_args` from it.

## Test Conventions

- `unittest.TestCase` (matches `test/registered/bench_fn/test_benchmark_datasets_api.py:32`). Pytest picks it up via `test/pytest.ini` (asyncio_mode=auto).
- Register CPU CI: `from sglang.test.ci.ci_register import register_cpu_ci; register_cpu_ci(est_time=10, suite="stage-a-test-cpu")` at module top.
- Run: `conda activate sglang && CUDA_VISIBLE_DEVICES=4 pytest test/registered/bench_fn/test_bench_eval.py -v`

## Commit Style

Prefix with `bench-eval:` (mirrors `heter-moe:` prefix in recent history).

---

### Task 1: Prereqs + Scaffold

**Files:**
- Create: `python/sglang/benchmark/eval_harness/__init__.py`
- Create: `test/registered/bench_fn/test_bench_eval.py`

- [x] **Step 1: Install lm-eval into the sglang env** _(already done — `lm-eval==0.4.11` verified)_

Prereq verified via:
```bash
/home/huanchen/.conda/envs/sglang/bin/python -c "import lm_eval; print(lm_eval.__version__)"
# → 0.4.11
```

- [ ] **Step 2: Write the failing smoke test**

Create `test/registered/bench_fn/test_bench_eval.py`:
```python
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class TestBenchEvalImports(unittest.TestCase):
    def test_subpackage_importable(self):
        from sglang.benchmark import eval_harness  # noqa: F401

    def test_public_symbols(self):
        from sglang.benchmark.eval_harness import (
            BenchServingLM,
            merge_report,
            write_report,
        )
        self.assertTrue(callable(BenchServingLM))
        self.assertTrue(callable(merge_report))
        self.assertTrue(callable(write_report))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run test to verify it fails**

```bash
conda activate sglang && CUDA_VISIBLE_DEVICES=4 pytest test/registered/bench_fn/test_bench_eval.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'sglang.benchmark.eval_harness'`.

- [ ] **Step 4: Create scaffold with stub symbols**

Create `python/sglang/benchmark/eval_harness/__init__.py`:
```python
"""Bridge: run lm-evaluation-harness tasks through sglang.bench_serving."""

from __future__ import annotations


def _not_implemented(*_args, **_kwargs):
    raise NotImplementedError("eval_harness helper not yet implemented")


BenchServingLM = _not_implemented
merge_report = _not_implemented
write_report = _not_implemented

__all__ = ["BenchServingLM", "merge_report", "write_report"]
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest test/registered/bench_fn/test_bench_eval.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/benchmark/eval_harness/__init__.py \
        test/registered/bench_fn/test_bench_eval.py
git commit -m "bench-eval: scaffold eval_harness subpackage"
```

---

### Task 2: BenchServingLM — core class

Single class: `generate_until` forwards requests to `bench_serving.benchmark`, `apply_chat_template` optionally injects `enable_thinking`, loglikelihood methods raise.

**Files:**
- Create: `python/sglang/benchmark/eval_harness/bench_serving_lm.py`
- Modify: `python/sglang/benchmark/eval_harness/__init__.py` — export `BenchServingLM`
- Modify: `test/registered/bench_fn/test_bench_eval.py` — add LM tests

- [ ] **Step 1: Write the failing tests**

Append to `test/registered/bench_fn/test_bench_eval.py`:
```python
from types import SimpleNamespace
from unittest.mock import patch

from test.registered.bench_fn.test_benchmark_datasets_api import (
    create_lightweight_tokenizer,
)


def _fake_request(prompt, until=None, max_gen_toks=64, **kw):
    """Minimal object matching lm_eval.api.instance.Instance.args[0..1]."""
    gen_kwargs = {"until": until or [], "max_gen_toks": max_gen_toks, **kw}
    return SimpleNamespace(args=(prompt, gen_kwargs))


class TestBenchServingLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = create_lightweight_tokenizer()

    def _make_lm(self, **overrides):
        from sglang.benchmark.eval_harness import BenchServingLM

        defaults = dict(
            base_url="http://mock:0",
            backend="sglang-oai",
            model_id="mock-model",
            tokenizer=self.tokenizer,
            request_rate=float("inf"),
            max_concurrency=4,
            enable_thinking=False,
        )
        defaults.update(overrides)
        return BenchServingLM(**defaults)

    def test_rejects_loglikelihood(self):
        lm = self._make_lm()
        with self.assertRaises(NotImplementedError):
            lm.loglikelihood([_fake_request("x")])
        with self.assertRaises(NotImplementedError):
            lm.loglikelihood_rolling([_fake_request("x")])

    def test_generate_until_calls_bench_serving(self):
        lm = self._make_lm()
        requests = [
            _fake_request("Q1?", until=["\n"], max_gen_toks=32),
            _fake_request("Q2?", until=["\n"], max_gen_toks=32),
        ]

        async def fake_benchmark(**kwargs):
            # Assert we passed proper DatasetRows with our prompts.
            rows = kwargs["input_requests"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].prompt, "Q1?")
            self.assertEqual(rows[0].output_len, 32)
            self.assertEqual(kwargs["request_rate"], float("inf"))
            self.assertEqual(kwargs["max_concurrency"], 4)
            return {
                "generated_texts": ["A1", "A2"],
                "mean_ttft_ms": 1.0,
                "mean_itl_ms": 0.5,
                "output_throughput": 10.0,
                "completed": 2,
            }

        with patch("sglang.bench_serving.benchmark", new=fake_benchmark), \
             patch("sglang.bench_serving.set_global_args"):
            outputs = lm.generate_until(requests)

        self.assertEqual(outputs, ["A1", "A2"])
        self.assertEqual(lm.last_perf["completed"], 2)
        self.assertAlmostEqual(lm.last_perf["mean_ttft_ms"], 1.0)

    def test_generate_until_forwards_stop_strings(self):
        lm = self._make_lm()
        requests = [_fake_request("Q?", until=["\n", "Question:"], max_gen_toks=16)]

        captured = {}

        async def fake_benchmark(**kwargs):
            captured["rows"] = kwargs["input_requests"]
            captured["extra"] = kwargs["extra_request_body"]
            return {"generated_texts": ["."], "completed": 1}

        with patch("sglang.bench_serving.benchmark", new=fake_benchmark), \
             patch("sglang.bench_serving.set_global_args"):
            lm.generate_until(requests)

        # Per-request stop strings go on DatasetRow.extra_request_body.
        self.assertEqual(
            captured["rows"][0].extra_request_body.get("stop"),
            ["\n", "Question:"],
        )

    def test_apply_chat_template_plain(self):
        lm = self._make_lm()
        msg = [{"role": "user", "content": "hi"}]
        out = lm.apply_chat_template(msg)
        self.assertIn("user:", out)  # lightweight tokenizer template
        self.assertIn("hi", out)

    def test_apply_chat_template_with_thinking(self):
        """When enable_thinking=True, apply_chat_template must forward it."""
        lm = self._make_lm(enable_thinking=True)
        calls = []
        real_apply = lm.tokenizer.apply_chat_template

        def spy(chat, **kwargs):
            calls.append(kwargs)
            # Strip the thinking kwarg so the lightweight tokenizer doesn't choke.
            kwargs.pop("enable_thinking", None)
            return real_apply(chat, **kwargs)

        lm.tokenizer.apply_chat_template = spy
        lm.apply_chat_template([{"role": "user", "content": "hi"}])
        self.assertTrue(calls, "apply_chat_template was not called on tokenizer")
        self.assertTrue(calls[0].get("enable_thinking"))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test/registered/bench_fn/test_bench_eval.py::TestBenchServingLM -v
```
Expected: 5 FAILs (`NotImplementedError` from the stub).

- [ ] **Step 3: Implement `BenchServingLM`**

Create `python/sglang/benchmark/eval_harness/bench_serving_lm.py`:
```python
"""BenchServingLM — an lm_eval.api.model.LM that routes generation through
sglang.bench_serving.

Usage:
    lm = BenchServingLM(base_url="http://host:port", backend="sglang-oai",
                        model_id="Qwen/...", tokenizer=tok, ...)
    results = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"], ...)
    perf    = lm.last_perf   # populated after the run

simple_evaluate handles prompt construction (fewshot + chat template +
enable_thinking via our apply_chat_template override), filter application,
and aggregation. We only implement generate_until; loglikelihood raises.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace
from typing import Any, Dict, List, Optional

from lm_eval.api.model import LM

from sglang.benchmark.datasets.common import DatasetRow


class BenchServingLM(LM):
    def __init__(
        self,
        *,
        base_url: str,
        backend: str,
        model_id: str,
        tokenizer,
        request_rate: float = float("inf"),
        max_concurrency: Optional[int] = None,
        enable_thinking: bool = False,
        warmup_requests: int = 0,
        flush_cache: bool = False,
        extra_request_body: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.backend = backend
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.request_rate = request_rate
        self.max_concurrency = max_concurrency
        self.enable_thinking = enable_thinking
        self.warmup_requests = warmup_requests
        self.flush_cache = flush_cache
        self._global_extra = dict(extra_request_body or {})
        self.last_perf: Optional[Dict[str, Any]] = None

        # lm_eval inspects these attributes.
        self._rank = 0
        self._world_size = 1
        self.batch_size_per_gpu = max_concurrency or 1

    # ---- chat template ---------------------------------------------------

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:
        kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
        if self.enable_thinking:
            kwargs["enable_thinking"] = True
        prompt = self.tokenizer.apply_chat_template(chat_history, **kwargs)
        bos = getattr(self.tokenizer, "bos_token", None)
        if bos and prompt.startswith(bos):
            prompt = prompt[len(bos):]  # bench_serving backend re-adds it.
        return prompt

    @property
    def tokenizer_name(self) -> str:
        return getattr(self.tokenizer, "name_or_path", "bench_serving_lm")

    # ---- loglikelihood: unsupported -------------------------------------

    def loglikelihood(self, requests):
        raise NotImplementedError(
            "BenchServingLM supports only generative tasks. Use a CoT / "
            "generative variant (e.g. mmlu_flan_cot_zeroshot instead of mmlu)."
        )

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "BenchServingLM supports only generative tasks."
        )

    # ---- generate_until: the whole point --------------------------------

    def generate_until(self, requests) -> List[str]:
        from sglang import bench_serving

        rows: List[DatasetRow] = []
        for req in requests:
            prompt, gen_kwargs = req.args[0], req.args[1]
            stop = gen_kwargs.get("until") or []
            max_gen_toks = gen_kwargs.get("max_gen_toks") or 2048
            temperature = gen_kwargs.get("temperature")

            per_req_extra: Dict[str, Any] = {}
            if stop:
                per_req_extra["stop"] = list(stop)
            if temperature is not None:
                per_req_extra["temperature"] = temperature

            prompt_ids = self.tokenizer.encode(prompt)
            rows.append(DatasetRow(
                prompt=prompt,
                prompt_len=len(prompt_ids),
                output_len=max_gen_toks,
                extra_request_body=per_req_extra,
            ))

        api_url = self.base_url + (
            "/v1/completions" if self.backend == "sglang-oai" else "/generate"
        )

        # bench_serving reads module-level args in a few places (warmup
        # branching, flush_cache, dataset_name guards). Populate it.
        args = Namespace(
            dataset_name="bench_eval",
            backend=self.backend,
            tag=None,
            sharegpt_output_len=None,
            random_input_len=0,
            random_output_len=0,
            random_range_ratio=1.0,
            output_file=None,
            output_details=False,
            num_prompts=len(rows),
        )
        bench_serving.set_global_args(args)

        perf = asyncio.run(bench_serving.benchmark(
            backend=self.backend,
            api_url=api_url,
            base_url=self.base_url,
            model_id=self.model_id,
            tokenizer=self.tokenizer,
            input_requests=rows,
            request_rate=self.request_rate,
            max_concurrency=self.max_concurrency,
            disable_tqdm=False,
            lora_names=[],
            lora_request_distribution=None,
            lora_zipf_alpha=None,
            extra_request_body=dict(self._global_extra),
            profile=False,
            flush_cache=self.flush_cache,
            warmup_requests=self.warmup_requests,
        ))
        self.last_perf = perf
        return list(perf["generated_texts"])
```

- [ ] **Step 4: Wire into `__init__.py`**

Replace `BenchServingLM = _not_implemented` with:
```python
from sglang.benchmark.eval_harness.bench_serving_lm import BenchServingLM
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest test/registered/bench_fn/test_bench_eval.py::TestBenchServingLM -v
```
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/benchmark/eval_harness/bench_serving_lm.py \
        python/sglang/benchmark/eval_harness/__init__.py \
        test/registered/bench_fn/test_bench_eval.py
git commit -m "bench-eval: route lm_eval generate_until through bench_serving"
```

---

### Task 3: Report Merger

Small module that combines the `simple_evaluate` result dict with `lm.last_perf` into one record.

**Files:**
- Create: `python/sglang/benchmark/eval_harness/report.py`
- Modify: `python/sglang/benchmark/eval_harness/__init__.py`
- Modify: `test/registered/bench_fn/test_bench_eval.py`

- [ ] **Step 1: Write the failing test**

Append to `test/registered/bench_fn/test_bench_eval.py`:
```python
import json
import tempfile
from pathlib import Path


class TestReport(unittest.TestCase):
    def test_merge_report_shape(self):
        from sglang.benchmark.eval_harness import merge_report

        lm_eval_results = {
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": 0.62,
                    "exact_match_stderr,strict-match": 0.013,
                    "exact_match,flexible-extract": 0.68,
                    "exact_match_stderr,flexible-extract": 0.012,
                    "alias": "gsm8k",
                }
            },
            "n-samples": {"gsm8k": {"original": 1319, "effective": 1319}},
            "config": {"num_fewshot": 5},
        }
        perf = {
            "mean_ttft_ms": 12.0, "mean_itl_ms": 4.5,
            "output_throughput": 1000.0, "request_throughput": 2.0,
            "completed": 1319, "duration": 600.0,
            "total_output_tokens": 2_500_000,
            "mean_e2e_latency_ms": 500.0,
        }
        merged = merge_report(
            task_name="gsm8k", lm_eval_results=lm_eval_results, perf=perf,
            run_config={"backend": "sglang-oai", "request_rate": 4.0},
        )
        self.assertEqual(merged["task"], "gsm8k")
        self.assertIn("exact_match,strict-match", merged["accuracy"])
        self.assertAlmostEqual(merged["accuracy"]["exact_match,strict-match"], 0.62)
        self.assertAlmostEqual(merged["perf"]["mean_ttft_ms"], 12.0)
        self.assertEqual(merged["n_samples"]["effective"], 1319)
        self.assertEqual(merged["run"]["backend"], "sglang-oai")

    def test_write_report_appends(self):
        from sglang.benchmark.eval_harness import merge_report, write_report

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "out.jsonl"
            r1 = merge_report(task_name="t", lm_eval_results={
                "results": {"t": {"acc": 0.5}}, "n-samples": {"t": {"effective": 10}},
                "config": {"num_fewshot": 0},
            }, perf={"mean_ttft_ms": 1.0}, run_config={})
            r2 = merge_report(task_name="t", lm_eval_results={
                "results": {"t": {"acc": 0.6}}, "n-samples": {"t": {"effective": 10}},
                "config": {"num_fewshot": 0},
            }, perf={"mean_ttft_ms": 2.0}, run_config={})
            write_report(str(path), r1)
            write_report(str(path), r2)

            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertAlmostEqual(json.loads(lines[0])["accuracy"]["acc"], 0.5)
            self.assertAlmostEqual(json.loads(lines[1])["accuracy"]["acc"], 0.6)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/registered/bench_fn/test_bench_eval.py::TestReport -v
```
Expected: 2 FAILs.

- [ ] **Step 3: Implement `report.py`**

Create `python/sglang/benchmark/eval_harness/report.py`:
```python
"""Combine simple_evaluate's result dict with bench_serving perf metrics.

The output is one JSON record per run, appended to a JSONL file so multiple
runs (e.g. a sweep across request_rate) accumulate naturally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


_PERF_FIELDS = (
    "duration", "completed",
    "total_input_tokens", "total_output_tokens",
    "request_throughput", "input_throughput", "output_throughput", "total_throughput",
    "mean_e2e_latency_ms", "median_e2e_latency_ms", "p99_e2e_latency_ms",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_itl_ms", "median_itl_ms", "p95_itl_ms", "p99_itl_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "concurrency", "max_output_tokens_per_s",
)


def merge_report(
    *,
    task_name: str,
    lm_eval_results: Dict[str, Any],
    perf: Dict[str, Any],
    run_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    task_results = dict(lm_eval_results["results"].get(task_name, {}))
    # `alias` is lm-eval bookkeeping, not a metric.
    task_results.pop("alias", None)

    n_samples = lm_eval_results.get("n-samples", {}).get(task_name, {})

    return {
        "task": task_name,
        "accuracy": task_results,
        "n_samples": n_samples,
        "lm_eval_config": lm_eval_results.get("config", {}),
        "perf": {k: perf[k] for k in _PERF_FIELDS if k in perf},
        "run": dict(run_config or {}),
    }


def write_report(path: str, report: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")
```

- [ ] **Step 4: Wire into `__init__.py`**

Replace `merge_report = _not_implemented` / `write_report = _not_implemented` with:
```python
from sglang.benchmark.eval_harness.report import merge_report, write_report
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest test/registered/bench_fn/test_bench_eval.py::TestReport -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/benchmark/eval_harness/report.py \
        python/sglang/benchmark/eval_harness/__init__.py \
        test/registered/bench_fn/test_bench_eval.py
git commit -m "bench-eval: merge simple_evaluate results with bench_serving perf"
```

---

### Task 4: Orchestration — `run_bench_eval`

One function that loads the tokenizer, constructs `BenchServingLM`, calls `simple_evaluate`, merges, optionally writes.

**Files:**
- Create: `python/sglang/bench_eval.py`
- Modify: `test/registered/bench_fn/test_bench_eval.py`

- [ ] **Step 1: Write the failing integration test (aiohttp mock)**

Append to `test/registered/bench_fn/test_bench_eval.py`:
```python
class TestRunBenchEvalEndToEnd(unittest.TestCase):
    """
    Stand up an in-process aiohttp server that mimics SGLang's
    /v1/completions. Call run_bench_eval(task='gsm8k', limit=3, ...). Assert
    both accuracy and perf fields are populated.

    The mock returns '#### 42' for every prompt. gsm8k's flexible-extract
    filter pulls '42' out; the test expects a deterministic accuracy.
    """

    def test_end_to_end_with_mock_server(self):
        import asyncio
        import json
        import tempfile
        from pathlib import Path

        from aiohttp import web

        async def completions(request):
            resp = {
                "id": "cmpl", "object": "text_completion",
                "choices": [{"text": " reasoning... #### 42",
                             "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12},
            }
            return web.Response(text=json.dumps(resp), content_type="application/json")

        async def models(request):
            return web.json_response({"data": [{"id": "mock-model"}]})

        async def flush(request):
            return web.Response(text="ok")

        async def _run():
            app = web.Application()
            app.router.add_post("/v1/completions", completions)
            app.router.add_get("/v1/models", models)
            app.router.add_get("/flush_cache", flush)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            try:
                from sglang.bench_eval import run_bench_eval

                with tempfile.TemporaryDirectory() as d:
                    out_file = Path(d) / "out.jsonl"
                    report = run_bench_eval(
                        task="gsm8k",
                        base_url=f"http://127.0.0.1:{port}",
                        backend="sglang-oai",
                        model="mock-model",
                        tokenizer_path="hf-internal-testing/llama-tokenizer",
                        num_fewshot=0,
                        limit=3,
                        max_gen_toks=32,
                        request_rate=float("inf"),
                        max_concurrency=2,
                        apply_chat_template=False,
                        enable_thinking=False,
                        output_file=str(out_file),
                        include_per_doc=False,
                    )
                    return report, out_file.read_text()
            finally:
                await runner.cleanup()

        report, jsonl = asyncio.run(_run())
        self.assertEqual(report["task"], "gsm8k")
        # gsm8k flexible-extract should pull "42" and compare to the doc target.
        # We can't assume the target is 42, so just check the field is present.
        self.assertTrue(any("exact_match" in k for k in report["accuracy"]))
        self.assertIn("mean_ttft_ms", report["perf"])
        self.assertEqual(report["n_samples"].get("effective"), 3)
        # Report also landed on disk.
        self.assertEqual(len(jsonl.strip().splitlines()), 1)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/registered/bench_fn/test_bench_eval.py::TestRunBenchEvalEndToEnd -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'sglang.bench_eval'`.

- [ ] **Step 3: Implement `bench_eval.py` core**

Create `python/sglang/bench_eval.py`:
```python
"""bench_eval — run lm-evaluation-harness tasks through sglang.bench_serving.

Produces one unified report with accuracy (from lm-eval, including filter-
tagged metrics like exact_match,strict-match) and serving performance
(TTFT, ITL, throughput, output-tokens-per-sec) for the same workload.

Entry points:
    run_bench_eval(...)   — programmatic; returns the merged report dict.
    main()                — CLI (added in Task 5).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def run_bench_eval(
    *,
    task: str,
    base_url: str,
    backend: str,
    model: str,
    tokenizer_path: str,
    num_fewshot: int,
    limit: Optional[int],
    max_gen_toks: int,
    request_rate: float,
    max_concurrency: Optional[int],
    apply_chat_template: bool,
    enable_thinking: bool,
    output_file: Optional[str],
    include_per_doc: bool,
    fewshot_as_multiturn: bool = False,
    flush_cache: bool = False,
    extra_request_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import lm_eval
    from transformers import AutoTokenizer

    from sglang.benchmark.eval_harness import (
        BenchServingLM, merge_report, write_report,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    lm = BenchServingLM(
        base_url=base_url,
        backend=backend,
        model_id=model,
        tokenizer=tokenizer,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        enable_thinking=enable_thinking,
        flush_cache=flush_cache,
        extra_request_body=extra_request_body,
    )

    gen_kwargs = f"max_gen_toks={max_gen_toks}"

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=num_fewshot,
        limit=limit,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        gen_kwargs=gen_kwargs,
        batch_size=max_concurrency or "auto",
    )

    if lm.last_perf is None:
        raise RuntimeError(
            "simple_evaluate returned without invoking generate_until. "
            "Check that the task is generative (output_type=='generate_until')."
        )

    report = merge_report(
        task_name=task,
        lm_eval_results=results,
        perf=lm.last_perf,
        run_config={
            "backend": backend,
            "model": model,
            "request_rate": request_rate,
            "max_concurrency": max_concurrency,
            "max_gen_toks": max_gen_toks,
            "num_fewshot": num_fewshot,
            "apply_chat_template": apply_chat_template,
            "enable_thinking": enable_thinking,
            "limit": limit,
        },
    )

    if include_per_doc:
        # Raw lm-eval samples are in results["samples"][task].
        report["per_doc"] = results.get("samples", {}).get(task, [])

    if output_file:
        write_report(output_file, report)

    return report
```

- [ ] **Step 4: Run the integration test**

```bash
pytest test/registered/bench_fn/test_bench_eval.py::TestRunBenchEvalEndToEnd -v
```
Expected: 1 passed. First run downloads gsm8k (`limit=3` so small) and a Llama tokenizer (~small); may take 10–30s.

If the test hangs: check that `simple_evaluate` is batching all 3 requests into a single `generate_until` call (lm-eval default for generate_until). If it makes one call per doc, `asyncio.run` inside `generate_until` will still work — just slower.

If it fails because `bench_serving.benchmark` writes a stray JSONL to cwd, change the test to run inside the tempdir (`os.chdir(d)` inside `TemporaryDirectory`).

- [ ] **Step 5: Commit**

```bash
git add python/sglang/bench_eval.py \
        test/registered/bench_fn/test_bench_eval.py
git commit -m "bench-eval: orchestrate simple_evaluate + BenchServingLM end-to-end"
```

---

### Task 5: CLI — `python -m sglang.bench_eval`

**Files:**
- Modify: `python/sglang/bench_eval.py` — add `build_parser()`, `main()`, `__main__` guard

- [ ] **Step 1: Write the failing CLI test**

Append to `test/registered/bench_fn/test_bench_eval.py`:
```python
class TestBenchEvalCLI(unittest.TestCase):
    def test_parser_accepts_all_flags(self):
        from sglang.bench_eval import build_parser

        args = build_parser().parse_args([
            "--task", "gsm8k",
            "--base-url", "http://localhost:30000",
            "--backend", "sglang-oai",
            "--model", "mock",
            "--tokenizer", "mock",
            "--num-fewshot", "5",
            "--limit", "10",
            "--max-gen-toks", "512",
            "--request-rate", "8.0",
            "--max-concurrency", "16",
            "--apply-chat-template",
            "--enable-thinking",
            "--fewshot-as-multiturn",
            "--output-file", "out.jsonl",
            "--include-per-doc",
        ])
        self.assertEqual(args.task, "gsm8k")
        self.assertEqual(args.num_fewshot, 5)
        self.assertEqual(args.max_gen_toks, 512)
        self.assertTrue(args.apply_chat_template)
        self.assertTrue(args.enable_thinking)
        self.assertTrue(args.fewshot_as_multiturn)
        self.assertTrue(args.include_per_doc)
        self.assertAlmostEqual(args.request_rate, 8.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/registered/bench_fn/test_bench_eval.py::TestBenchEvalCLI -v
```
Expected: FAIL — `ImportError: cannot import name 'build_parser'`.

- [ ] **Step 3: Add CLI to `python/sglang/bench_eval.py`**

Append to `python/sglang/bench_eval.py`:
```python
import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sglang.bench_eval",
        description=("Run an lm-eval task through sglang.bench_serving. "
                     "Reports both accuracy (lm-eval) and serving performance "
                     "(TTFT, ITL, throughput) for the same workload."),
    )
    p.add_argument("--task", required=True,
                   help="lm-eval task name (must be generative). Examples: "
                        "gsm8k, mmlu_flan_cot_zeroshot, gpqa_diamond_cot_zeroshot, "
                        "mmlu_pro.")
    p.add_argument("--base-url", required=True,
                   help="SGLang server base URL, e.g. http://127.0.0.1:30000.")
    p.add_argument("--backend", default="sglang-oai",
                   choices=["sglang", "sglang-oai"])
    p.add_argument("--model", required=True, help="Model id (sent as model field).")
    p.add_argument("--tokenizer", required=True,
                   help="HF tokenizer path or repo id.")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of eval docs (None = full task).")
    p.add_argument("--max-gen-toks", type=int, default=2048)
    p.add_argument("--request-rate", type=float, default=float("inf"),
                   help="Requests per second. 'inf' = unlimited (default).")
    p.add_argument("--max-concurrency", type=int, default=None)
    p.add_argument("--apply-chat-template", action="store_true")
    p.add_argument("--enable-thinking", action="store_true",
                   help="Adds enable_thinking=True to apply_chat_template.")
    p.add_argument("--fewshot-as-multiturn", action="store_true")
    p.add_argument("--output-file", default=None,
                   help="Append-mode JSONL path for the merged report.")
    p.add_argument("--include-per-doc", action="store_true")
    p.add_argument("--flush-cache", action="store_true",
                   help="Flush KV cache before the run (CI parity).")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.enable_thinking and not args.apply_chat_template:
        parser.error("--enable-thinking requires --apply-chat-template")

    report = run_bench_eval(
        task=args.task,
        base_url=args.base_url,
        backend=args.backend,
        model=args.model,
        tokenizer_path=args.tokenizer,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        max_gen_toks=args.max_gen_toks,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
        apply_chat_template=args.apply_chat_template,
        enable_thinking=args.enable_thinking,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        output_file=args.output_file,
        include_per_doc=args.include_per_doc,
        flush_cache=args.flush_cache,
    )

    print("=" * 60)
    print(f"Task: {report['task']}")
    print(f"N samples: {report['n_samples']}")
    print("Accuracy:")
    for k, v in report["accuracy"].items():
        print(f"  {k}: {v}")
    print("Performance:")
    for k, v in report["perf"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
pytest test/registered/bench_fn/test_bench_eval.py -v
```
Expected: all tests passing.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/bench_eval.py \
        test/registered/bench_fn/test_bench_eval.py
git commit -m "bench-eval: CLI entry point (python -m sglang.bench_eval)"
```

---

### Task 6: Manual smoke test on Qwen3-1.7B

Validates end-to-end behavior with a real model. Per user preference: `sglang` conda env, GPU 4.

- [ ] **Step 1: Launch an SGLang server in one terminal**

```bash
conda activate sglang
CUDA_VISIBLE_DEVICES=4 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-1.7B \
    --port 30000 \
    --reasoning-parser qwen3
```
Wait for `The server is fired up and ready to roll!`.

- [ ] **Step 2: Baseline run — gsm8k, 5-shot, no thinking**

```bash
conda activate sglang
python -m sglang.bench_eval \
    --task gsm8k --limit 50 --num-fewshot 5 \
    --base-url http://127.0.0.1:30000 \
    --backend sglang-oai \
    --model Qwen/Qwen3-1.7B \
    --tokenizer Qwen/Qwen3-1.7B \
    --max-gen-toks 512 \
    --max-concurrency 8 \
    --output-file /tmp/bench_eval_smoke.jsonl
```
Expected: prints both `exact_match,strict-match` / `exact_match,flexible-extract` accuracy and `mean_ttft_ms` / `mean_itl_ms` / `output_throughput`. Accuracy should be >20% (Qwen3-1.7B 5-shot on gsm8k, limit=50).

- [ ] **Step 3: Thinking-mode run — same task, chat template + enable_thinking**

```bash
python -m sglang.bench_eval \
    --task gsm8k --limit 50 --num-fewshot 0 \
    --base-url http://127.0.0.1:30000 \
    --backend sglang-oai \
    --model Qwen/Qwen3-1.7B \
    --tokenizer Qwen/Qwen3-1.7B \
    --max-gen-toks 8192 \
    --max-concurrency 4 \
    --apply-chat-template --enable-thinking \
    --output-file /tmp/bench_eval_smoke.jsonl
```
Expected:
- Higher output_throughput stability vs. run 2 (longer per-request decode).
- Larger `total_output_tokens` (think tokens dominate).
- Accuracy should be >= baseline; often notably higher.
- Because the server was launched with `--reasoning-parser qwen3`, `<think>…</think>` is stripped before the completion text reaches us — so lm-eval's gsm8k flexible-extract sees a clean answer.

- [ ] **Step 4: Confirm the JSONL captured both runs**

```bash
wc -l /tmp/bench_eval_smoke.jsonl   # expect 2
python -c "
import json
for line in open('/tmp/bench_eval_smoke.jsonl'):
    r = json.loads(line)
    print(r['run']['enable_thinking'], r['accuracy'], r['perf']['mean_ttft_ms'])
"
```

- [ ] **Step 5: Marker commit recording validation**

```bash
git commit --allow-empty -m "bench-eval: smoke test on Qwen3-1.7B gsm8k — acc + perf both populated"
```

---

## Self-Review Checklist (engineer runs after Task 5)

1. **Spec coverage** — each requirement maps to a task:
   - Request-rate control → Task 2 (`request_rate` param) + Task 5 (`--request-rate`). ✓
   - TTFT / ITL / throughput reported → Task 3 (`_PERF_FIELDS`) + Task 4 returns them in `perf`. ✓
   - Output generation length → captured via `total_output_tokens` and per-request `output_len`. ✓
   - Thinking / CoT chat template → Task 2 (`apply_chat_template` override forwards `enable_thinking`) + Task 5 (`--enable-thinking`). ✓
   - lm-eval API (phase 1 prompt construction + phase 3 scoring) → Task 4 uses `simple_evaluate` for both. ✓
   - Log results for later eval → Task 3 `write_report` appends JSONL; `--include-per-doc` keeps raw samples. ✓

2. **Placeholder scan** — no `TBD`, no "handle appropriate errors", no "similar to above". Each step has concrete code or commands.

3. **Type consistency** — `BenchServingLM.last_perf` matches `_PERF_FIELDS` subset consumed by `merge_report`. `generate_until` returns `list[str]` (lm-eval's expected return). `Instance.args` destructured as `(prompt, gen_kwargs)` in Task 2 — matches lm-eval 0.4.x convention.

4. **Edge cases** — loglikelihood task rejected (Task 2 test), apply_chat_template with missing chat_template falls through to the tokenizer's default (warns automatically), thinking without chat template rejected (Task 5 `main()` guard).

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-16-bench-eval-lm-eval-bridge.md`.**

## Execution Handoff

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
