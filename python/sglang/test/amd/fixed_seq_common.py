"""Shared harness for the MI35x fixed-sequence benchmark (random ISL/OSL sweeps).

Each recipe is one model as its owner serves it, in its own file
``test/registered/amd/perf/mi35x/test_<model>_fixed_seq_mi35x.py`` with its own
suite and its own leg in ``nightly-benchmark-amd.yml``. A recipe is split into
server cells: one server configuration (flags and env) and the points measured
on it. A run measures one cell, FIXED_SEQ_CELL (``tp2-c16``, ``1k1k-tp4-dp4``,
...):

1. start the server with the cell's flags and env;
2. gate on GSM8K (sgl-eval, GSM8K_EXAMPLES examples, thinking off, greedy):
   below the recipe's threshold the cell measures nothing, so a broken build
   never reports numbers;
3. run ``sglang.benchmark.serving`` on the random dataset for each point,
   ``prompts_per_conc * conc`` prompts after ``warmups_per_conc * conc`` warmup
   requests, at ``--max-concurrency conc``;
4. write one record (``fixed_seq_report`` renders it) to CI_ARTIFACT_DIR.

Reports rather than gates on performance: a cell fails only when the server,
GSM8K or a benchmark point fails.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import msgspec

from sglang.srt.utils import find_local_repo_dir, kill_process_tree
from sglang.test.amd.fixed_seq_report import render_markdown
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server

CELL = os.environ.get("FIXED_SEQ_CELL", "")
SERVER_LAUNCH_TIMEOUT = 5400
# The longest point, 70K/300 at the highest concurrency, prefills tens of
# millions of tokens.
POINT_TIMEOUT = 5400
# A point with more failed requests than this is marked failed; the report
# flags any point that lost a request.
MAX_FAILED_FRACTION = 0.05
GSM8K_EXAMPLES = 200
# Thinking off keeps the answer inside the smallest cell's context (DSV4 1K/1K:
# 2304 tokens) and the gate to a few minutes.
GSM8K_MAX_TOKENS = 1024


class Workload(msgspec.Struct, frozen=True):
    isl: int
    osl: int
    prompts_per_conc: int
    warmups_per_conc: int
    range_ratio: float


# The owners' client settings: the DSV4 1K/8K sweep's for 1K/1K and 8K/1K, the
# DSV4 70K sweep's for 70K/300.
WORKLOAD_1K1K = Workload(
    isl=1024, osl=1024, prompts_per_conc=10, warmups_per_conc=2, range_ratio=0.8
)
WORKLOAD_8K1K = Workload(
    isl=8192, osl=1024, prompts_per_conc=10, warmups_per_conc=2, range_ratio=0.8
)
WORKLOAD_70K300 = Workload(
    isl=70000, osl=300, prompts_per_conc=4, warmups_per_conc=1, range_ratio=1.0
)


class Cell(msgspec.Struct, frozen=True):
    tp: int
    other_args: list[str]
    env: dict[str, str] = {}
    # (workload, concurrency), in the order they run.
    points: list[tuple[Workload, int]] = []


def powers_of_two(low: int, high: int) -> list:
    return [1 << n for n in range(20) if low <= 1 << n <= high]


def _sglang_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _gsm8k(
    base_url: str, model: str, chat_template_kwargs: dict, threshold: float
) -> dict:
    metrics = run_sgl_eval(
        SimpleNamespace(
            eval_name="gsm8k",
            base_url=base_url,
            model=model,
            num_examples=GSM8K_EXAMPLES,
            num_threads=64,
            max_tokens=GSM8K_MAX_TOKENS,
            temperature=0.0,
            chat_template_kwargs=chat_template_kwargs,
        )
    )
    return {
        "accuracy": metrics["score"],
        "num_examples": GSM8K_EXAMPLES,
        "threshold": threshold,
        "passed": metrics["score"] >= threshold,
    }


# What a record keeps of a bench_serving result (it also carries server_info).
_RESULT_KEYS = (
    "duration",
    "completed",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "concurrency",
    "accept_length",
    *(
        f"{stat}_{metric}_ms"
        for metric in ("ttft", "tpot", "itl", "e2e_latency")
        for stat in ("mean", "median", "p90", "p99")
    ),
)


def _bench_point(
    base_url: str, model: str, tokenizer: str, workload: Workload, conc: int
) -> dict:
    point = {
        "isl": workload.isl,
        "osl": workload.osl,
        "conc": conc,
        "num_prompts": workload.prompts_per_conc * conc,
        "warmups": workload.warmups_per_conc * conc,
        "range_ratio": workload.range_ratio,
        "result": None,
        "error": None,
    }
    with tempfile.TemporaryDirectory(prefix="fixed-seq-") as tmp:
        output_file = Path(tmp) / "result.jsonl"
        command = _client_command(
            base_url=base_url,
            model=model,
            tokenizer=tokenizer,
            workload=workload,
            conc=conc,
            output_file=output_file,
        )
        _run_client(point=point, command=command, output_file=output_file)
    return point


def _client_command(
    base_url: str,
    model: str,
    tokenizer: str,
    workload: Workload,
    conc: int,
    output_file: Path,
) -> list:
    return [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        # The OpenAI completions endpoint, as the owners' clients use.
        "--backend",
        "sglang-oai",
        "--base-url",
        base_url,
        "--model",
        model,
        "--tokenizer",
        tokenizer,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(workload.isl),
        "--random-output-len",
        str(workload.osl),
        "--random-range-ratio",
        str(workload.range_ratio),
        "--num-prompts",
        str(workload.prompts_per_conc * conc),
        "--max-concurrency",
        str(conc),
        "--request-rate",
        "inf",
        "--warmup-requests",
        str(workload.warmups_per_conc * conc),
        "--output-file",
        str(output_file),
        "--disable-tqdm",
    ]


def _run_client(point: dict, command: list, output_file: Path) -> None:
    """Run the client and fill ``point``'s result, or its error."""
    print(
        f"[fixed-seq] {point['isl']}/{point['osl']} conc={point['conc']} "
        f"prompts={point['num_prompts']} warmups={point['warmups']}",
        flush=True,
    )
    try:
        subprocess.run(command, check=True, timeout=POINT_TIMEOUT)
    except subprocess.TimeoutExpired:
        point["error"] = f"timed out after {POINT_TIMEOUT} s"
        return
    except subprocess.CalledProcessError as e:
        point["error"] = f"client exited {e.returncode}"
        return
    lines = output_file.read_text().splitlines() if output_file.is_file() else []
    if not lines:
        point["error"] = "no result written"
        return
    result = json.loads(lines[-1])
    point["result"] = {key: result.get(key) for key in _RESULT_KEYS}
    # The client exits 0 when some requests fail and reports over the rest.
    failed = point["num_prompts"] - (result.get("completed") or 0)
    if failed > MAX_FAILED_FRACTION * point["num_prompts"]:
        point["error"] = (
            f"{result.get('completed')}/{point['num_prompts']} requests completed"
        )


def run_cell(
    test_case: unittest.TestCase,
    *,
    recipe: str,
    title: str,
    model: str,
    cells: dict,
    gsm8k_threshold: float,
    gsm8k_chat_template_kwargs: dict,
) -> None:
    """Serve cell FIXED_SEQ_CELL of ``cells``, gate it on GSM8K, measure its
    points and write its record; ``title`` heads the report.
    ``gsm8k_chat_template_kwargs`` turns the model's thinking off."""
    test_case.assertIn(CELL, cells, f"FIXED_SEQ_CELL must be one of {list(cells)}")
    cell = cells[CELL]
    record = {
        "recipe": recipe,
        "title": title,
        "cell": CELL,
        "model": model,
        "model_revision": None,
        "gpus": cell.tp,
        "server_args": cell.other_args,
        "server_env": cell.env,
        "sglang_commit": _sglang_commit(),
        "image": os.environ.get("CI_IMAGE", ""),
        "error": None,
        "gsm8k": None,
        "points": [],
    }
    base_url = DEFAULT_URL_FOR_TEST
    try:
        server = popen_launch_server(
            model,
            base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=cell.other_args,
            env={**os.environ, **cell.env},
        )
    except BaseException as e:
        record["error"] = f"server did not start: {type(e).__name__}"
        _write_artifacts(record)
        raise
    try:
        _measure(
            record=record,
            cell=cell,
            base_url=base_url,
            gsm8k_threshold=gsm8k_threshold,
            gsm8k_chat_template_kwargs=gsm8k_chat_template_kwargs,
        )
    except BaseException as e:
        record["error"] = f"{type(e).__name__}: {e}"[:300]
        raise
    finally:
        # Before the kill, which raises when the server outlives its timeout.
        _write_artifacts(record)
        kill_process_tree(server.pid)

    test_case.assertTrue(
        record["gsm8k"]["passed"],
        f"GSM8K {record['gsm8k']['accuracy']:.3f} < {gsm8k_threshold}",
    )
    failed = [p for p in record["points"] if p["error"]]
    test_case.assertFalse(
        failed,
        "Failed points: "
        + ", ".join(f"{p['isl']}/{p['osl']}@{p['conc']}: {p['error']}" for p in failed),
    )


def _measure(
    record: dict,
    cell: Cell,
    base_url: str,
    gsm8k_threshold: float,
    gsm8k_chat_template_kwargs: dict,
) -> None:
    """GSM8K, then the points while it passes; the record is rewritten after each
    step so a cell killed by its time limit keeps what it measured."""
    model = record["model"]
    local_dir = find_local_repo_dir(model)
    record["model_revision"] = Path(local_dir).name if local_dir else None
    record["gsm8k"] = _gsm8k(
        base_url=base_url,
        model=model,
        chat_template_kwargs=gsm8k_chat_template_kwargs,
        threshold=gsm8k_threshold,
    )
    print(f"[fixed-seq] GSM8K {record['gsm8k']}", flush=True)
    _write_artifacts(record)
    if not record["gsm8k"]["passed"]:
        return
    for workload, conc in cell.points:
        record["points"].append(
            _bench_point(
                base_url=base_url,
                model=model,
                tokenizer=local_dir or model,
                workload=workload,
                conc=conc,
            )
        )
        _write_artifacts(record)


def _write_artifacts(record: dict) -> None:
    artifact_dir = Path(os.environ.get("CI_ARTIFACT_DIR", "fixed_seq_artifacts"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fixedseq-{record['recipe']}-{record['cell']}"
    (artifact_dir / f"{stem}.json").write_text(json.dumps(record, indent=1))
    (artifact_dir / f"{stem}.md").write_text(render_markdown([record]))
