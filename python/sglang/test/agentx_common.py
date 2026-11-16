"""Shared harness for the MI35x InferenceX AgentX curve tests.

Each curve is one row of InferenceX's ``amd-master.yaml`` and lives in its own
file, ``test/registered/amd/perf/mi35x/test_<model>_agentx_mi35x.py``, with its
own suite and its own job in ``nightly-benchmark-amd.yml``, so a curve can be
added or dropped without touching the others.

A run measures one point of the row, AGENTX_POINT (``tp2-c1``, ``tp2-dpa-c64``,
``tp4-dcp4-c16``, ...). The point's server flags, env, KV-offload budget, golden
acceptance length and the AgentX client all come from InferenceX at
INFERENCEX_COMMIT, bound by its own srt-slurm planner. The departures are CI
ones: no power telemetry, the server and router addresses, and the image.

The image is the day's sglang CI image (``image`` in the record), not the one
the row names (``inferencex_image``). A point is comparable with the one
InferenceX publishes only when the two are the same ROCm and sglang build;
InferenceX runs some rows on another ROCm (the Qwen3.5 and GLM-5.2 MXFP4 rows on
rocm720, while the nightly runs rocm10).

Reports rather than gates: a point fails only when the server, router or client
fails, including InferenceX's result validation.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from sglang.srt.utils import kill_process_tree
from sglang.test.server_fixtures.disaggregation_fixture import find_sgl_router_binary
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    find_available_port,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

INFERENCEX_REPO = "https://github.com/SemiAnalysisAI/InferenceX.git"
INFERENCEX_COMMIT = "4538f1bc9784426ae92da31b7e8daae4ba637b84"
# What InferenceX's benchmark workflow sets for every AgentX point.
THINKING_MODE = "thinking_on"

POINT = os.environ.get("AGENTX_POINT", "")
SERVER_LAUNCH_TIMEOUT = 5400
ROUTER_READY_TIMEOUT = 600
# sgl-router's cache_aware learns prefixes only from the worker's KV events. It
# finds them through the worker's /server_info: DP rank r publishes on 5557 + r,
# and --load-publish-endpoint auto puts the load reports on the ports after.
KV_EVENTS_CONFIG = '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}'

SUMMARY_HEADER = (
    "| point | requests | total tok/s/GPU | output tok/s/GPU | p50 interactivity "
    "(tok/s/user) | p50 TTFT (s) | p50 ITL (ms) | GPU cache hit |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
)


def _checkout_inferencex(commit: str) -> Path:
    # A temp dir, not /sgl-data: the clone takes seconds and goes with the container.
    root = Path(tempfile.mkdtemp(prefix="inferencex-"))
    git = ["git", "-C", str(root)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run(
        [*git, "fetch", "-q", "--depth", "1", INFERENCEX_REPO, commit], check=True
    )
    subprocess.run([*git, "checkout", "-q", "FETCH_HEAD"], check=True)
    return root / "inferencex-e2e"


def _matrix_entry(master: dict, point: str, label: str) -> tuple:
    """``(scenario, entry, tp, dp_attention, dcp, conc)`` for ``point``
    (``tp<N>[-dpa][-dcp<M>]-c<CONC>``)."""
    match = re.fullmatch(r"tp(\d+)(-dpa)?(?:-dcp(\d+))?-c(\d+)", point)
    if not match:
        raise ValueError(
            f"AGENTX_POINT must look like tp2-c1, tp2-dpa-c64 or tp4-dcp4-c16: {point!r}"
        )
    tp, dp_attention = int(match[1]), bool(match[2])
    dcp, conc = int(match[3] or 1), int(match[4])
    entries = [
        (scenario, entry)
        for scenario in master["scenarios"]["agentic-coding"]
        for entry in scenario["search-space"]
        if entry["tp"] == tp
        and bool(entry.get("dp-attn")) == dp_attention
        and entry.get("dcp-size", 1) == dcp
        and conc in entry["conc-list"]
    ]
    if len(entries) != 1:
        raise ValueError(f"{point} is not one point of {label}")
    return *entries[0], tp, dp_attention, dcp, conc


def _bind_point(e2e: Path, config_key: str, commit: str, point: str) -> tuple:
    """``(variant, recipe, environment)`` for ``point``, bound by InferenceX's planner."""
    sys.path.insert(0, str(e2e))
    from infx.matrix.generate import agentic_dram_offload_gb
    from infx.matrix.validation import DEFAULT_AGENTIC_DURATION_SECONDS, srt_recipe_path
    from infx.srt_slurm.single_node import select_recipe
    from infx.srt_slurm.synthetic_acceptance import build_overrides
    from infx.srt_slurm.variants import expand_variants

    master = yaml.safe_load((e2e / "configs" / "amd-master.yaml").read_text())
    master = master[config_key]
    scenario, entry, tp, dp_attention, dcp, conc = _matrix_entry(
        master=master, point=point, label=f"{config_key} at InferenceX {commit}"
    )
    # The host-DRAM budget InferenceX's matrix gives a KV-offload point; 0 otherwise.
    total_cpu_dram_gb, _ = agentic_dram_offload_gb(
        scenario,
        entry,
        master["runner"],
        yaml.safe_load((e2e / "configs" / "runners.yaml").read_text()),
    )
    offload_backend = entry.get("kv-offload-backend")
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
        "DCP_SIZE": str(dcp),
        "PCP_SIZE": "1",
        "DP_ATTENTION": "true" if dp_attention else "false",
        "GPU_COUNT": str(tp),
        "CONC": str(conc),
        "SPEC_DECODING": entry["spec-decoding"],
        "KV_OFFLOADING": entry["kv-offloading"],
        "KV_OFFLOAD_BACKEND": offload_backend["name"] if offload_backend else "",
        "KV_OFFLOAD_BACKEND_METADATA": (
            json.dumps(offload_backend) if offload_backend else ""
        ),
        "TOTAL_CPU_DRAM_GB": str(total_cpu_dram_gb),
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


# Mirrors toSglRouter in docs/src/snippets/_agentx.jsx: the dropped set and the
# policy, request-timeout-secs and retry-max-retries branches must match. Unlike
# the page, an arg in neither set raises here.
# Args whose behavior is sgl-router's default, or that it has no counterpart for.
_ROUTER_ARGS_DROPPED = {
    "pd-disaggregation",
    "request-id-headers",
    "connect-timeout-secs",
    "disable-health-check",
    "disable-retries",
    "balance-abs-threshold",
    "retry-backoff-multiplier",
}
_ROUTER_ARGS_PASSED = {
    "dp-aware",
    "disable-circuit-breaker",
    "retry-initial-backoff-ms",
    "retry-max-backoff-ms",
}


def _router_policy(frontend_args: dict) -> str | None:
    policy = frontend_args.get("policy")
    return "cache_aware" if policy == "consistent_hashing" else policy


def _router_cli_args(config: dict) -> list:
    args = []
    for key, value in config.items():
        key = key.replace("_", "-")
        if key == "policy":
            args += ["--policy", _router_policy(config)]
        elif key == "retry-max-retries":
            args += ["--retry-max-attempts", str(int(value) + 1)]
        elif key in _ROUTER_ARGS_PASSED:
            if value is True:
                args.append(f"--{key}")
            elif value is not False:
                args += [f"--{key}", str(value)]
        elif key == "request-timeout-secs":
            # The recipe bounds the whole request; sgl-router bounds the time to response
            # headers, the silence between chunks and the request lifetime separately.
            for flag in (
                "--request-timeout-secs",
                "--stream-idle-timeout-secs",
                "--stale-request-timeout-secs",
            ):
                args += [flag, str(value)]
        elif key not in _ROUTER_ARGS_DROPPED:
            raise ValueError(f"No sgl-router flag for the frontend arg {key!r}")
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
    frontend_args = (recipe.get("frontend") or {}).get("args") or {}
    if (
        _router_policy(frontend_args) == "cache_aware"
        and "kv-events-config" not in args
    ):
        other_args += [
            "--kv-events-config",
            KV_EVENTS_CONFIG,
            "--load-publish-endpoint",
            "auto",
        ]
    env = os.environ.copy()
    for scope in (recipe.get("environment") or {}, role.get("env") or {}):
        env.update((key, str(value)) for key, value in scope.items())
    return model, served_model_name, other_args, env


def _launch_router(
    model: str, served_model_name: str, worker_url: str, frontend_args: dict, env: dict
) -> tuple:
    """Start sgl-router in front of ``worker_url``; ``(process, url)``."""
    port = find_available_port(int(worker_url.rsplit(":", 1)[1]))
    # cache_aware matches prefixes on the worker's tokenizer; other policies
    # need no request tokens and leave tokenizing to the worker.
    if _router_policy(frontend_args) == "cache_aware":
        tokenizer = ["--tokenizer-path", model]
    else:
        tokenizer = ["--no-tokenizer"]
    command = [
        find_sgl_router_binary(),
        "--model-id",
        served_model_name,
        *tokenizer,
        "--worker-urls",
        worker_url,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *_router_cli_args(frontend_args),
    ]
    print(f"router command={' '.join(command)}", flush=True)
    # The worker's env: a router that renders chats reads the same render defaults
    # (SGLANG_DEFAULT_THINKING, SGLANG_DSV4_REASONING_EFFORT, SGLANG_DSV41_REASONING_EFFORT).
    process = subprocess.Popen(command, env=env)
    url = f"http://127.0.0.1:{port}"
    try:
        wait_for_http_ready(
            url=f"{url}/readiness", timeout=ROUTER_READY_TIMEOUT, process=process
        )
    except BaseException:
        kill_process_tree(process.pid)
        raise
    return process, url


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


def run_point(
    test_case: unittest.TestCase,
    *,
    config_key: str,
    title: str,
    inferencex_commit: str = INFERENCEX_COMMIT,
) -> None:
    """Measure AGENTX_POINT of the InferenceX row ``config_key`` (an
    ``amd-master.yaml`` key) and write its record; ``title`` heads the job summary.
    A curve passes its own ``inferencex_commit`` only when its row moves on alone."""
    e2e = _checkout_inferencex(inferencex_commit)
    variant, recipe, environment = _bind_point(
        e2e=e2e, config_key=config_key, commit=inferencex_commit, point=POINT
    )
    print(
        f"AgentX {config_key} {POINT}: {variant} at InferenceX {inferencex_commit}; "
        f"image {os.environ.get('CI_IMAGE', '?')}, "
        f"InferenceX ran this row on {environment['IMAGE']}",
        flush=True,
    )
    result_dir = Path(tempfile.mkdtemp(prefix=f"agentx-{POINT}-"))
    result_filename = f"agentx_{config_key}_{POINT}"
    client_exit_code = _serve_and_run_client(
        e2e=e2e,
        recipe=recipe,
        environment=environment,
        result_dir=result_dir,
        result_filename=result_filename,
    )

    aggregate_path = result_dir / f"{result_filename}.json"
    if aggregate_path.is_file():
        _write_artifacts(
            artifact_dir=Path(os.environ.get("CI_ARTIFACT_DIR", "agentx_artifacts")),
            title=title,
            record={
                "config_key": config_key,
                "point": POINT,
                "variant": variant,
                "inferencex_commit": inferencex_commit,
                "sglang_commit": _sglang_commit(),
                "image": os.environ.get("CI_IMAGE", ""),
                "inferencex_image": environment["IMAGE"],
                "client_exit_code": client_exit_code,
                "aggregate": json.loads(aggregate_path.read_text()),
            },
        )
    test_case.assertEqual(client_exit_code, 0, "InferenceX AgentX client failed")
    test_case.assertTrue(aggregate_path.is_file(), f"{aggregate_path} was not written")


def _serve_and_run_client(
    e2e: Path, recipe: dict, environment: dict, result_dir: Path, result_filename: str
) -> int:
    """Serve the point as srt-slurm does, run InferenceX's AgentX client against
    it, and return the client's exit code."""
    frontend = recipe.get("frontend") or {}
    if frontend.get("type") not in ("sglang", "sglang-router"):
        raise ValueError(f"Unsupported frontend {frontend.get('type')!r}")

    model, served_model_name, other_args, server_env = _server_launch(recipe)
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
                model=model,
                served_model_name=served_model_name,
                worker_url=worker_url,
                frontend_args=frontend.get("args") or {},
                env=server_env,
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
    return client.returncode


def _write_artifacts(artifact_dir: Path, title: str, record: dict) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    point = record["point"]
    (artifact_dir / f"agentx-{point}.json").write_text(json.dumps(record))
    (artifact_dir / f"agentx-{point}.md").write_text(
        f"### {title} AgentX on MI35x: {point}\n\n"
        + SUMMARY_HEADER
        + _summary_row(point=point, aggregate=record["aggregate"])
    )
