"""MI35x InferenceX AgentX curve for DeepSeek-V4.1-Flash, one point per run.

Runs the point AGENTX_POINT (``tp2-c1``, ``tp2-dpa-c64``, ...) of InferenceX's
``dsv41flash-fp4-mi355x-sglang-agentic-dspark`` sweep against this checkout.
The point's server flags, env, golden acceptance length and the AgentX client
all come from InferenceX at INFERENCEX_COMMIT, bound by its own srt-slurm
planner, so a result is comparable with InferenceX's for the same image. The
only departures are CI ones: no power telemetry, server and router addresses.

Reports rather than gates: fails only when the server, router or client fails,
including InferenceX's result validation.

Registry: nightly-perf-8-gpu-mi35x-dsv41-flash-agentx suite
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import requests
import yaml

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    find_available_port,
    popen_launch_server,
)

# Register for AMD CI - one AgentX point: model load, up to an hour of warmup,
# then InferenceX's one-hour measurement window.
register_amd_ci(
    est_time=10800, suite="nightly-perf-8-gpu-mi35x-dsv41-flash-agentx", nightly=True
)

INFERENCEX_REPO = "https://github.com/SemiAnalysisAI/InferenceX.git"
INFERENCEX_COMMIT = "4538f1bc9784426ae92da31b7e8daae4ba637b84"
CONFIG_KEY = "dsv41flash-fp4-mi355x-sglang-agentic-dspark"
# What InferenceX's benchmark workflow sets for every AgentX point.
THINKING_MODE = "thinking_on"

POINT = os.environ.get("AGENTX_POINT", "")
SERVER_LAUNCH_TIMEOUT = 5400
ROUTER_READY_TIMEOUT = 600

SUMMARY_HEADER = (
    "| point | requests | total tok/s/GPU | output tok/s/GPU | p50 interactivity "
    "(tok/s/user) | p50 TTFT (s) | p50 ITL (ms) | GPU cache hit |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
)


def _checkout_inferencex() -> Path:
    # A temp dir, not /sgl-data: the clone takes seconds and goes with the container.
    root = Path(tempfile.mkdtemp(prefix="inferencex-"))
    git = ["git", "-C", str(root)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run(
        [*git, "fetch", "-q", "--depth", "1", INFERENCEX_REPO, INFERENCEX_COMMIT],
        check=True,
    )
    subprocess.run([*git, "checkout", "-q", "FETCH_HEAD"], check=True)
    return root / "inferencex-e2e"


def _matrix_entry(master: dict, point: str) -> tuple:
    """``(entry, tp, dp_attention, conc)`` for ``point`` (``tp<N>[-dpa]-c<CONC>``)."""
    match = re.fullmatch(r"tp(\d+)(-dpa)?-c(\d+)", point)
    if not match:
        raise ValueError(
            f"AGENTX_POINT must look like tp2-c1 or tp2-dpa-c64: {point!r}"
        )
    tp, dp_attention, conc = int(match[1]), bool(match[2]), int(match[3])
    entries = [
        entry
        for scenario in master["scenarios"]["agentic-coding"]
        for entry in scenario["search-space"]
        if entry["tp"] == tp
        and bool(entry.get("dp-attn")) == dp_attention
        and conc in entry["conc-list"]
    ]
    if len(entries) != 1:
        raise ValueError(
            f"{point} is not one point of {CONFIG_KEY} at InferenceX {INFERENCEX_COMMIT}"
        )
    return entries[0], tp, dp_attention, conc


def _bind_point(e2e: Path, point: str) -> tuple:
    """``(variant, recipe, environment)`` for ``point``, bound by InferenceX's planner."""
    sys.path.insert(0, str(e2e))
    from infx.matrix.validation import DEFAULT_AGENTIC_DURATION_SECONDS, srt_recipe_path
    from infx.srt_slurm.single_node import select_recipe
    from infx.srt_slurm.synthetic_acceptance import build_overrides
    from infx.srt_slurm.variants import expand_variants

    master = yaml.safe_load((e2e / "configs" / "amd-master.yaml").read_text())
    master = master[CONFIG_KEY]
    entry, tp, dp_attention, conc = _matrix_entry(master=master, point=point)
    # The inputs InferenceX's benchmark template gives a single-node AgentX point.
    environment = {
        "IMAGE": master["image"],
        "MODEL": master["model"],
        "MODEL_PREFIX": master["model-prefix"],
        "FRAMEWORK": master["framework"],
        "PRECISION": master["precision"],
        "TP": str(tp),
        "EP_SIZE": str(entry["ep"]),
        "PP_SIZE": "1",
        "DCP_SIZE": "1",
        "PCP_SIZE": "1",
        "DP_ATTENTION": "true" if dp_attention else "false",
        "GPU_COUNT": str(tp),
        "CONC": str(conc),
        "SPEC_DECODING": entry["spec-decoding"],
        "KV_OFFLOADING": entry["kv-offloading"],
        "IS_AGENTIC": "1",
        "SCENARIO_TYPE": "agentic-coding",
        "IS_MULTINODE": "false",
        "DISAGG": "false",
        "RUN_EVAL": "false",
        "EVAL_ONLY": "false",
        "THINKING_MODE": THINKING_MODE,
        "DURATION": str(DEFAULT_AGENTIC_DURATION_SECONDS),
        "AIPERF_EXPERIMENTAL_FAST": "0",
    }
    path = e2e / srt_recipe_path(False, master["srt-recipe-dir"], entry["srt-recipe"])
    variant, recipe = select_recipe(
        str(path), environment, root=e2e, expand=expand_variants
    )
    _apply_srtctl_overrides(
        recipe=recipe,
        overrides=build_overrides(recipe, environment["FRAMEWORK"], environment),
    )
    return variant, recipe, environment


def _apply_srtctl_overrides(recipe: dict, overrides: list) -> None:
    """Apply srtctl ``--set path=<json>`` / ``--unset path`` pairs to ``recipe``."""
    pairs = iter(overrides)
    for op in pairs:
        key, _, value = next(pairs).partition("=")
        *parents, leaf = key.split(".")
        node = recipe
        for part in parents:
            node = node.setdefault(part, {})
        if op == "--set":
            node[leaf] = json.loads(value)
        else:
            node.pop(leaf, None)


def _engine_cli_args(config: dict) -> list:
    # srt-slurm's rendering of engine args: sorted, a true bool is a bare flag.
    args = []
    for key, value in sorted(config.items()):
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, list):
            args += [flag, *map(str, value)]
        elif value is not None:
            args += [flag, str(value)]
    return args


def _router_cli_args(config: dict) -> list:
    # srt-slurm's rendering of router args: recipe order, a list repeats its flag.
    args = []
    for key, value in config.items():
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            args.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            for item in value:
                args += [flag, str(item)]
        else:
            args += [flag, str(value)]
    return args


def _server_launch(recipe: dict) -> tuple:
    """``(model, served_model_name, other_args, env)`` of the recipe's sglang worker."""
    role = recipe["roles"]["agg"]
    args = dict(role["args"])
    model = recipe["model"]["path"].removeprefix("hf:")
    served_model_name = args.pop("served-model-name", model)
    other_args = ["--served-model-name", served_model_name]
    if "enable-metrics" not in args:
        other_args.append("--enable-metrics")
    other_args += _engine_cli_args(args)
    env = os.environ.copy()
    for scope in (recipe.get("environment") or {}, role.get("env") or {}):
        env.update((key, str(value)) for key, value in scope.items())
    return model, served_model_name, other_args, env


def _launch_router(worker_url: str, frontend_args: dict) -> tuple:
    """Start sglang-router in front of ``worker_url`` as srt-slurm does; ``(process, url)``."""
    port = find_available_port(int(worker_url.rsplit(":", 1)[1]))
    command = [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--worker-urls",
        worker_url,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--prometheus-port",
        str(find_available_port(port)),
        "--prometheus-host",
        "127.0.0.1",
        *_router_cli_args(frontend_args),
    ]
    print(f"router command={' '.join(command)}", flush=True)
    process = subprocess.Popen(command)
    url = f"http://127.0.0.1:{port}"
    # Ready once its worker registry lists healthy workers, as srt-slurm waits.
    deadline = time.monotonic() + ROUTER_READY_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"sglang-router exited with code {process.returncode}")
        try:
            workers = requests.get(url + "/workers", timeout=5).json()["workers"]
            if workers and all(worker.get("is_healthy") for worker in workers):
                return process, url
        except (requests.RequestException, ValueError, KeyError, TypeError):
            pass
        time.sleep(5)
    kill_process_tree(process.pid)
    raise TimeoutError(f"sglang-router not ready after {ROUTER_READY_TIMEOUT}s")


def _runtime_settings(e2e: Path) -> dict:
    """The exports of InferenceX's ``benchmarks/runtime_settings.sh``."""
    pattern = re.compile(r"export ([A-Za-z_][A-Za-z0-9_]*)='([^']*)'")
    text = (e2e / "benchmarks" / "runtime_settings.sh").read_text()
    return dict(m.groups() for m in map(pattern.fullmatch, text.splitlines()) if m)


def _client_env(
    e2e: Path,
    recipe: dict,
    environment: dict,
    served_model_name: str,
    server_url: str,
    metrics_url: str,
    result_dir: Path,
    result_filename: str,
) -> dict:
    return {
        **os.environ,
        **_runtime_settings(e2e),
        **environment,
        **{key: str(value) for key, value in recipe["benchmark"]["env"].items()},
        # No GPU power telemetry in sglang CI.
        "ENABLE_AGENTX_POWER": "0",
        "REQUIRE_POWER": "0",
        # This CI's image, not the one InferenceX's matrix row names.
        "IMAGE": "",
        "RUNNER_TYPE": "mi355x-sglang-ci",
        "SERVED_MODEL_NAME": served_model_name,
        "AIPERF_SERVER_URL": server_url,
        "AIPERF_SERVER_METRICS_URLS": metrics_url,
        "RESULT_DIR": str(result_dir),
        "RESULT_FILENAME": result_filename,
        "AGENTIC_OUTPUT_DIR": str(result_dir),
        # As benchmarks/srt_agentic.sh runs the client.
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(e2e), os.environ.get("PYTHONPATH")])
        ),
    }


def _sglang_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _summary_row(point: str, aggregate: dict) -> str:
    metrics = aggregate.get("request_metrics", {})
    latency = metrics.get("latency", {})
    per_gpu = metrics.get("throughput", {}).get("per_gpu", {})
    cache = aggregate.get("server_metrics", {}).get("cache", {})

    def cell(value, fmt, scale=1):
        return format(value * scale, fmt) if isinstance(value, (int, float)) else "n/a"

    return (
        f"| {point} | {aggregate.get('num_requests_successful', 'n/a')} "
        f"| {cell(per_gpu.get('total_tput_tps'), ',.0f')} "
        f"| {cell(per_gpu.get('output_tput_tps'), ',.1f')} "
        f"| {cell(latency.get('intvty', {}).get('p50'), '.1f')} "
        f"| {cell(latency.get('ttft', {}).get('p50'), '.2f')} "
        f"| {cell(latency.get('itl', {}).get('p50'), '.2f', scale=1000)} "
        f"| {cell(cache.get('gpu_cache_hit_rate'), '.1%')} |\n"
    )


def _write_artifacts(
    artifact_dir: Path, variant: str, aggregate: dict, client_exit_code: int
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "config_key": CONFIG_KEY,
        "point": POINT,
        "variant": variant,
        "inferencex_commit": INFERENCEX_COMMIT,
        "sglang_commit": _sglang_commit(),
        "image": os.environ.get("CI_IMAGE", ""),
        "client_exit_code": client_exit_code,
        "aggregate": aggregate,
    }
    (artifact_dir / f"agentx-{POINT}.json").write_text(json.dumps(record))
    (artifact_dir / f"agentx-{POINT}.md").write_text(
        f"### DeepSeek-V4.1-Flash AgentX on MI35x: {POINT}\n\n"
        + SUMMARY_HEADER
        + _summary_row(point=POINT, aggregate=aggregate)
    )


# Not CustomTestCase: its CI retry would rerun a failed two-hour point and report
# the second attempt's numbers as a pass.
@unittest.skipUnless(POINT, "AGENTX_POINT is not set")
class TestDeepSeekV41FlashAgentXMI35x(unittest.TestCase):
    def test_agentx_point(self):
        e2e = _checkout_inferencex()
        variant, recipe, environment = _bind_point(e2e=e2e, point=POINT)
        print(
            f"AgentX {POINT}: {variant} at InferenceX {INFERENCEX_COMMIT}", flush=True
        )
        frontend = recipe.get("frontend") or {}
        if frontend.get("type") not in ("sglang", "sglang-router"):
            raise ValueError(f"Unsupported frontend {frontend.get('type')!r}")

        model, served_model_name, other_args, server_env = _server_launch(recipe)
        result_dir = Path(tempfile.mkdtemp(prefix=f"agentx-{POINT}-"))
        result_filename = f"agentx_{CONFIG_KEY}_{POINT}"
        worker_url = DEFAULT_URL_FOR_TEST
        server = popen_launch_server(
            model,
            worker_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=server_env,
        )
        router = None
        try:
            server_url = worker_url
            if frontend["type"] == "sglang-router":
                router, server_url = _launch_router(
                    worker_url=worker_url, frontend_args=frontend.get("args") or {}
                )
            client_env = _client_env(
                e2e=e2e,
                recipe=recipe,
                environment=environment,
                served_model_name=served_model_name,
                server_url=server_url,
                metrics_url=f"{worker_url}/metrics",
                result_dir=result_dir,
                result_filename=result_filename,
            )
            client = subprocess.run(
                [sys.executable, "-m", "infx.bench", "agentic"], env=client_env
            )
        finally:
            if router is not None:
                kill_process_tree(router.pid)
            kill_process_tree(server.pid)

        aggregate_path = result_dir / f"{result_filename}.json"
        if aggregate_path.is_file():
            _write_artifacts(
                artifact_dir=Path(
                    os.environ.get("CI_ARTIFACT_DIR", "agentx_artifacts")
                ),
                variant=variant,
                aggregate=json.loads(aggregate_path.read_text()),
                client_exit_code=client.returncode,
            )
        self.assertEqual(client.returncode, 0, "InferenceX AgentX client failed")
        self.assertTrue(aggregate_path.is_file(), f"{aggregate_path} was not written")


if __name__ == "__main__":
    unittest.main()
