"""JSONL records of benchmark cases, one per case result.

Readers parse them as plain JSON keyed by the field names here; bump
`SCHEMA_VERSION` when a field is renamed, removed or changes meaning.
"""

import importlib.metadata
import os
import re
import socket
import subprocess
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

import msgspec
import pynvml
import torch

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.jit.utils.compile.toolchain import device_compiler_path
from sglang.srt.environ import envs

from .case import BenchInfo, Case, CaseResult, Tolerance

SCHEMA_VERSION = 1

_GITHUB_EVENT_TO_RUN_KIND = {
    "schedule": "nightly",
    "push": "main",
    "pull_request": "pr",
    "workflow_dispatch": "manual",
}


class Metrics(msgspec.Struct, frozen=True, kw_only=True):
    p50_us: float
    avg_us: float
    n: int
    samples_us: List[float]


class GPUState(msgspec.Struct, frozen=True, kw_only=True):
    sm_clock_mhz: int
    temperature_c: int
    throttle_reasons: int


class EnvInfo(msgspec.Struct, frozen=True, kw_only=True):
    gpu_name: str
    sm_count: int
    compute_capability: str
    driver: str
    cuda_runtime: str
    torch: str
    triton: str
    nvcc: str
    commit: str
    hostname: str


class RunInfo(msgspec.Struct, frozen=True, kw_only=True):
    kind: str
    id: Optional[str]  # None outside GitHub Actions
    process: str


# No omit_defaults: every record carries every field, None ones included.
class BenchRecord(msgspec.Struct, frozen=True, kw_only=True):
    schema_version: int = SCHEMA_VERSION
    bench_id: str
    params: Dict[str, Any]
    line: str
    world_size: int
    case_id: str
    status: Literal["ok", "skip"]
    skip_reason: Optional[str]
    metrics: Optional[Metrics]
    gbps: Optional[float]
    tflops: Optional[float]
    gate: bool
    gate_reason: str
    tolerance: Tolerance
    gpu_before: GPUState
    gpu_after: GPUState
    env: EnvInfo
    run: RunInfo
    timestamp: float


class RecordReporter:
    """Appends one `BenchRecord` per case result to a per-process JSONL file."""

    def __init__(self, *, output_dir: str) -> None:
        self._output_dir = output_dir

    def begin(self, *, info: BenchInfo) -> None:
        self._info = info

    def before_case(self, case: Case) -> None:
        self._gpu_before = sample_gpu_state()

    def on_result(self, result: CaseResult) -> None:
        rec = make_record(info=self._info, result=result, gpu_before=self._gpu_before)
        append_record(output_dir=self._output_dir, record=rec)

    def finish(self) -> None:
        pass


def make_record(
    *, info: BenchInfo, result: CaseResult, gpu_before: GPUState
) -> BenchRecord:
    case, bench = result.case, result.result
    return BenchRecord(
        bench_id=case.bench_id,
        params=case.params,
        line=case.line,
        world_size=case.world_size,
        case_id=case.case_id,
        status="skip" if bench is None else "ok",
        skip_reason=result.skip_reason,
        metrics=None if bench is None else make_metrics(list(bench.samples)),
        gbps=None if bench is None else bench.gbps(),
        tflops=None if bench is None else bench.tflops(),
        gate=case.line in info.gated_lines,
        gate_reason=info.gate_reason,
        tolerance=info.tolerance,
        gpu_before=gpu_before,
        gpu_after=sample_gpu_state(),
        env=collect_env_info(
            device_id=torch.cuda.current_device(), source_file=info.source_file
        ),
        run=collect_run_info(),
        timestamp=time.time(),
    )


def make_metrics(samples_s: List[float]) -> Metrics:
    assert samples_s, "a recorded BenchResult needs its per-iteration samples"
    samples_us = sorted(s * 1e6 for s in samples_s)
    n = len(samples_us)
    return Metrics(
        p50_us=samples_us[n // 2],
        avg_us=sum(samples_us) / n,
        n=n,
        samples_us=samples_us,
    )


@cache_once
def _nvml_handle(device_id: int) -> Any:
    pynvml.nvmlInit()
    uuid_str = str(torch.cuda.get_device_properties(device_id).uuid)
    return pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid_str}")


def sample_gpu_state() -> GPUState:
    handle = _nvml_handle(torch.cuda.current_device())
    return GPUState(
        sm_clock_mhz=pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
        temperature_c=pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        ),
        throttle_reasons=pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
    )


def _run_text(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, text=True, timeout=30)


def _nvcc_version() -> str:
    output = _run_text([device_compiler_path(), "--version"])
    match = re.search(r"V(\d+\.\d+\.\d+)", output)
    assert match is not None, f"no version in nvcc --version output:\n{output}"
    return match.group(1)


def _git_commit(source_file: str) -> str:
    """The commit of the checkout the benchmark source lives in: the code measured."""
    source_dir = os.path.dirname(os.path.abspath(source_file))
    return _run_text(["git", "-C", source_dir, "rev-parse", "HEAD"]).strip()


@cache_once
def collect_env_info(*, device_id: int, source_file: str) -> EnvInfo:
    props = torch.cuda.get_device_properties(device_id)
    _nvml_handle(device_id)  # nvmlInit before the system query below
    assert torch.version.cuda is not None, "benchmark records need a CUDA build"
    return EnvInfo(
        gpu_name=props.name,
        sm_count=props.multi_processor_count,
        compute_capability=f"{props.major}.{props.minor}",
        driver=pynvml.nvmlSystemGetDriverVersion(),
        cuda_runtime=torch.version.cuda,
        torch=str(torch.__version__),
        triton=importlib.metadata.version("triton"),
        nvcc=_nvcc_version(),
        commit=_git_commit(source_file),
        hostname=socket.gethostname(),
    )


@cache_once
def collect_run_info() -> RunInfo:
    in_actions = "GITHUB_RUN_ID" in os.environ
    kind = envs.SGLANG_JIT_BENCHMARK_RUN_KIND.get() or (
        _GITHUB_EVENT_TO_RUN_KIND[os.environ["GITHUB_EVENT_NAME"]]
        if in_actions
        else "local"
    )
    run_id = (
        f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
        if in_actions
        else None
    )
    return RunInfo(kind=kind, id=run_id, process=_process_token())


@cache_once
def _process_token() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


@cache_once
def _sink_path(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"bench-{_process_token()}.jsonl")


def append_record(*, output_dir: str, record: BenchRecord) -> None:
    with open(_sink_path(output_dir), "ab") as f:
        f.write(msgspec.json.encode(record) + b"\n")
