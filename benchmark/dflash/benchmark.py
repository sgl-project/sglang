"""Benchmark a Kimi-K3 target-only baseline against DFlash on TP8 B300."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlretrieve

CACHE_DIR = Path(
    os.environ.get(
        "SGLANG_DFLASH_BENCHMARK_CACHE",
        Path.home() / ".cache" / "sglang" / "dflash-benchmark",
    )
).expanduser()
SUPPORTED_WORKLOADS = ("gsm8k", "math500", "humaneval", "mbpp", "mt-bench")
DEFAULT_WORKLOADS = "gsm8k"
DEFAULT_TARGET_MODEL = "moonshotai/Kimi-K3"
DEFAULT_TARGET_MODEL_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
DEFAULT_DFLASH_DRAFT_MODEL = "/tmp/dflash/draft-epoch-10"
DEFAULT_DFLASH_DRAFT_ATTENTION_BACKEND = "trtllm_mha"
DEFAULT_LINEAR_ATTN_PREFILL_BACKEND = "ptx_kda"
DEFAULT_LINEAR_ATTN_DECODE_BACKEND = "triton"
DEFAULT_LINEAR_ATTN_VERIFY_BACKEND = "nv_cutedsl"
LINEAR_ATTN_PREFILL_BACKEND_CHOICES = (
    "triton",
    "cutedsl",
    "flashkda",
    "nvidia_kda",
    "ptx_kda",
)
LINEAR_ATTN_DECODE_BACKEND_CHOICES = (
    "triton",
    "cutedsl",
    "flashinfer",
)
LINEAR_ATTN_VERIFY_BACKEND_CHOICES = (
    "triton",
    "nv_cutedsl",
    "flashinfer",
)
DEFAULT_TRTLLM_GEN_MOE_CUBIN_POOL = os.environ.get(
    "SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL",
    "/home/modal/trtllm_gen_moe_cubin_pool_20260617_v0613rc1",
)
DEFAULT_DFLASH_BLOCK_SIZES = "default"
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_TIMEOUT_S = 3600
SERVER_SHUTDOWN_DRAIN_TIMEOUT_S = 30.0
SERVER_SHUTDOWN_TIMEOUT_S = 120.0
GSM8K_TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
MT_BENCH_QUESTION_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)


@dataclass(frozen=True)
class SharedServerConfig:
    tp_size: int = 8
    attention_backend: str = "auto"
    mem_fraction_static: Optional[float] = 0.85
    max_running_requests: int = 128
    cuda_graph_max_bs_decode: int = 128
    context_length: int = 131072
    moe_runner_backend: str = "flashinfer_mxfp4"
    model_loader_num_threads: int = 64
    load_format: str = "auto"
    random_seed: int = DEFAULT_RANDOM_SEED
    linear_attn_prefill_backend: str = DEFAULT_LINEAR_ATTN_PREFILL_BACKEND
    linear_attn_decode_backend: str = DEFAULT_LINEAR_ATTN_DECODE_BACKEND
    linear_attn_verify_backend: str = DEFAULT_LINEAR_ATTN_VERIFY_BACKEND

    def to_args(self) -> list[str]:
        args = [
            "--tp",
            str(self.tp_size),
            "--trust-remote-code",
            "--load-format",
            self.load_format,
            "--model-loader-extra-config",
            json.dumps(
                {
                    "enable_multithread_load": True,
                    "num_threads": self.model_loader_num_threads,
                },
                separators=(",", ":"),
            ),
            "--moe-runner-backend",
            self.moe_runner_backend,
            "--context-length",
            str(self.context_length),
            "--random-seed",
            str(self.random_seed),
            "--max-running-requests",
            str(self.max_running_requests),
            "--cuda-graph-max-bs-decode",
            str(self.cuda_graph_max_bs_decode),
            "--skip-server-warmup",
            "--enable-metrics",
        ]
        if self.mem_fraction_static is not None:
            args.extend(["--mem-fraction-static", str(self.mem_fraction_static)])
        for flag, backend in (
            ("--linear-attn-prefill-backend", self.linear_attn_prefill_backend),
            ("--linear-attn-decode-backend", self.linear_attn_decode_backend),
            ("--linear-attn-verify-backend", self.linear_attn_verify_backend),
        ):
            args.extend([flag, backend])
        return args

    def summary_label(self) -> str:
        return (
            f"tp:{self.tp_size},mem_fraction:{self.mem_fraction_static},"
            f"attention_backend:{self.attention_backend},"
            f"max_running_requests:{self.max_running_requests},"
            f"cuda_graph_max_bs_decode:{self.cuda_graph_max_bs_decode},"
            f"context_length:{self.context_length},"
            f"moe_runner_backend:{self.moe_runner_backend},"
            f"model_loader_num_threads:{self.model_loader_num_threads},"
            f"load_format:{self.load_format},"
            f"random_seed:{self.random_seed},"
            f"linear_attn_prefill_backend:{self.linear_attn_prefill_backend},"
            f"linear_attn_decode_backend:{self.linear_attn_decode_backend},"
            f"linear_attn_verify_backend:{self.linear_attn_verify_backend}"
        )


BASE_SHARED_SERVER_CONFIG = SharedServerConfig()


@dataclass(frozen=True)
class DFlashConfig:
    draft_model: str
    block_size: Optional[int] = None
    draft_attention_backend: str = DEFAULT_DFLASH_DRAFT_ATTENTION_BACKEND
    # Replace per-token SSM snapshots with ReplaySSM raw-input rings.
    enable_replayssm: bool = True

    @property
    def mode_key(self) -> str:
        suffix = "" if self.enable_replayssm else "_noring"
        if self.block_size is None:
            return f"dflash{suffix}"
        return f"dflash_b{self.block_size}{suffix}"

    @property
    def display_name(self) -> str:
        suffix = "" if self.enable_replayssm else " (no ReplaySSM)"
        if self.block_size is None:
            return f"DFLASH{suffix}"
        return f"DFLASH block={self.block_size}{suffix}"

    @property
    def expect_spec(self) -> bool:
        return True

    def _replayssm_cache_len(self) -> int:
        # Server validation requires a power-of-two ring at least 2x the block size.
        block = 16 if self.block_size is None else int(self.block_size)
        return max(32, 1 << (2 * block - 1).bit_length())

    def to_args(self) -> list[str]:
        args = [
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            self.draft_model,
            "--speculative-draft-attention-backend",
            self.draft_attention_backend,
        ]
        if self.block_size is not None:
            args.extend(["--speculative-dflash-block-size", str(int(self.block_size))])
        if self.enable_replayssm:
            args.extend(
                [
                    "--enable-gdn-replayssm-spec",
                    "--linear-replayssm-cache-len",
                    str(self._replayssm_cache_len()),
                ]
            )
        return args


@dataclass(frozen=True)
class BaselineConfig:
    @property
    def mode_key(self) -> str:
        return "baseline"

    @property
    def display_name(self) -> str:
        return "Baseline"

    @property
    def expect_spec(self) -> bool:
        return False

    def to_args(self) -> list[str]:
        return []


@dataclass(frozen=True)
class ServerDeployment:
    shared_config: SharedServerConfig
    mode_config: BaselineConfig | DFlashConfig

    @property
    def mode_key(self) -> str:
        return self.mode_config.mode_key

    @property
    def display_name(self) -> str:
        return self.mode_config.display_name

    @property
    def expect_spec(self) -> bool:
        return self.mode_config.expect_spec

    @property
    def server_args(self) -> list[str]:
        return [*self.shared_config.to_args(), *self.mode_config.to_args()]

    @property
    def dflash_block_size(self) -> Optional[int]:
        if isinstance(self.mode_config, DFlashConfig):
            return self.mode_config.block_size
        return None


@dataclass(frozen=True)
class DeploymentSweep:
    include_baseline: bool
    dflash_draft_model: str
    dflash_block_sizes: tuple[Optional[int], ...]
    dflash_draft_attention_backend: str = DEFAULT_DFLASH_DRAFT_ATTENTION_BACKEND
    dflash_enable_replayssm: bool = True

    @property
    def mode_keys(self) -> list[str]:
        return [
            DFlashConfig(
                "",
                block_size=block_size,
                enable_replayssm=self.dflash_enable_replayssm,
            ).mode_key
            for block_size in self.dflash_block_sizes
        ]


@dataclass(frozen=True)
class SamplingConfig:
    enable_thinking: bool
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int


@dataclass(frozen=True)
class BenchmarkMethodologyConfig:
    num_samples: Optional[int]
    min_generation_turns_per_config: int
    min_warmup_generation_turns: int
    runs_per_config: int
    timeout_s: int = DEFAULT_TIMEOUT_S
    server_shutdown_drain_timeout_s: float = SERVER_SHUTDOWN_DRAIN_TIMEOUT_S
    server_shutdown_timeout_s: float = SERVER_SHUTDOWN_TIMEOUT_S


@dataclass(frozen=True)
class SweepConfig:
    target_model: str
    target_model_revision: Optional[str]
    trtllm_gen_moe_cubin_pool: Optional[str]
    load_format: str
    linear_attn_prefill_backend: str
    linear_attn_decode_backend: str
    linear_attn_verify_backend: str
    workloads: tuple[str, ...]
    concurrencies: tuple[int, ...]
    sampling: SamplingConfig
    methodology: BenchmarkMethodologyConfig
    deployment_sweep: DeploymentSweep
    csv_output: Optional[str]
    random_seed: int = DEFAULT_RANDOM_SEED


@dataclass(frozen=True)
class RunKey:
    workload: str
    backend: str
    tp: int
    concurrency: int
    mode: str

    def metric_key(self) -> tuple[str, int, int, str]:
        return (self.backend, self.tp, self.concurrency, self.mode)


@dataclass(frozen=True)
class BenchmarkPlan:
    measured_samples: list[list[str]]
    warmup_samples: list[list[str]]

    @property
    def measured_sample_count(self) -> int:
        return len(self.measured_samples)

    @property
    def measured_generation_turn_count(self) -> int:
        return _generation_turn_count(self.measured_samples)

    @property
    def warmup_generation_turn_count(self) -> int:
        return _generation_turn_count(self.warmup_samples)


@dataclass(frozen=True)
class BenchmarkJob:
    target_model: str
    target_model_revision: Optional[str]
    trtllm_gen_moe_cubin_pool: Optional[str]
    workload: str
    deployment: ServerDeployment
    concurrency: int
    run_index: int
    sampling: SamplingConfig
    methodology: BenchmarkMethodologyConfig

    @property
    def key(self) -> RunKey:
        shared_config = self.deployment.shared_config
        return RunKey(
            workload=self.workload,
            backend=shared_config.attention_backend,
            tp=shared_config.tp_size,
            concurrency=self.concurrency,
            mode=self.deployment.mode_key,
        )

    @property
    def label(self) -> str:
        key = self.key
        return (
            f"workload={key.workload} backend={key.backend} tp={key.tp} "
            f"conc={key.concurrency} ({self.deployment.display_name})"
        )

    @property
    def run_label(self) -> str:
        return f"run={self.run_index + 1}/{self.methodology.runs_per_config}"


def _parse_int_csv(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _parse_optional_int_csv(value: str) -> list[Optional[int]]:
    values: list[Optional[int]] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in ("default", "none"):
            values.append(None)
        else:
            values.append(int(item))
    return values or [None]


def _parse_str_csv(value: str) -> list[str]:
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def _duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _shared_server_config_to_payload(config: SharedServerConfig) -> dict[str, Any]:
    return {
        "tp_size": config.tp_size,
        "attention_backend": config.attention_backend,
        "mem_fraction_static": config.mem_fraction_static,
        "max_running_requests": config.max_running_requests,
        "cuda_graph_max_bs_decode": config.cuda_graph_max_bs_decode,
        "context_length": config.context_length,
        "moe_runner_backend": config.moe_runner_backend,
        "model_loader_num_threads": config.model_loader_num_threads,
        "load_format": config.load_format,
        "random_seed": config.random_seed,
        "linear_attn_prefill_backend": config.linear_attn_prefill_backend,
        "linear_attn_decode_backend": config.linear_attn_decode_backend,
        "linear_attn_verify_backend": config.linear_attn_verify_backend,
    }


def _shared_server_config_from_payload(payload: dict[str, Any]) -> SharedServerConfig:
    return SharedServerConfig(
        tp_size=int(payload["tp_size"]),
        attention_backend=str(payload.get("attention_backend", "auto")),
        mem_fraction_static=payload.get("mem_fraction_static"),
        max_running_requests=int(payload.get("max_running_requests", 128)),
        cuda_graph_max_bs_decode=int(payload.get("cuda_graph_max_bs_decode", 128)),
        context_length=int(payload.get("context_length", 131072)),
        moe_runner_backend=str(payload.get("moe_runner_backend", "flashinfer_mxfp4")),
        model_loader_num_threads=int(payload.get("model_loader_num_threads", 64)),
        load_format=str(payload.get("load_format", "auto")),
        random_seed=int(payload.get("random_seed", DEFAULT_RANDOM_SEED)),
        linear_attn_prefill_backend=str(
            payload.get("linear_attn_prefill_backend")
            or DEFAULT_LINEAR_ATTN_PREFILL_BACKEND
        ),
        linear_attn_decode_backend=str(
            payload.get("linear_attn_decode_backend")
            or DEFAULT_LINEAR_ATTN_DECODE_BACKEND
        ),
        linear_attn_verify_backend=str(
            payload.get("linear_attn_verify_backend")
            or DEFAULT_LINEAR_ATTN_VERIFY_BACKEND
        ),
    )


def _mode_config_to_payload(
    config: BaselineConfig | DFlashConfig,
) -> dict[str, Any]:
    if isinstance(config, BaselineConfig):
        return {"kind": "baseline"}
    if isinstance(config, DFlashConfig):
        return {
            "kind": "dflash",
            "draft_model": config.draft_model,
            "block_size": config.block_size,
            "draft_attention_backend": config.draft_attention_backend,
            "enable_replayssm": config.enable_replayssm,
        }
    raise TypeError(f"Unsupported mode config type: {type(config).__name__}")


def _mode_config_from_payload(
    payload: dict[str, Any],
) -> BaselineConfig | DFlashConfig:
    kind = payload["kind"]
    if kind == "baseline":
        return BaselineConfig()
    if kind == "dflash":
        return DFlashConfig(
            draft_model=str(payload["draft_model"]),
            block_size=payload.get("block_size"),
            draft_attention_backend=str(
                payload.get("draft_attention_backend", "trtllm_mha")
            ),
            enable_replayssm=bool(payload.get("enable_replayssm", True)),
        )
    raise ValueError(f"Unsupported mode config kind: {kind}")


def _deployment_to_payload(deployment: ServerDeployment) -> dict[str, Any]:
    return {
        "shared_config": _shared_server_config_to_payload(deployment.shared_config),
        "mode_config": _mode_config_to_payload(deployment.mode_config),
    }


def _deployment_from_payload(payload: dict[str, Any]) -> ServerDeployment:
    return ServerDeployment(
        shared_config=_shared_server_config_from_payload(payload["shared_config"]),
        mode_config=_mode_config_from_payload(payload["mode_config"]),
    )


def _sampling_config_to_payload(config: SamplingConfig) -> dict[str, Any]:
    return {
        "enable_thinking": config.enable_thinking,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
    }


def _sampling_config_from_payload(payload: dict[str, Any]) -> SamplingConfig:
    return SamplingConfig(
        enable_thinking=bool(payload["enable_thinking"]),
        max_new_tokens=int(payload["max_new_tokens"]),
        temperature=float(payload["temperature"]),
        top_p=float(payload["top_p"]),
        top_k=int(payload["top_k"]),
    )


def _methodology_to_payload(config: BenchmarkMethodologyConfig) -> dict[str, Any]:
    return {
        "num_samples": config.num_samples,
        "min_generation_turns_per_config": config.min_generation_turns_per_config,
        "min_warmup_generation_turns": config.min_warmup_generation_turns,
        "runs_per_config": config.runs_per_config,
        "timeout_s": config.timeout_s,
        "server_shutdown_drain_timeout_s": config.server_shutdown_drain_timeout_s,
        "server_shutdown_timeout_s": config.server_shutdown_timeout_s,
    }


def _methodology_from_payload(
    payload: dict[str, Any],
) -> BenchmarkMethodologyConfig:
    return BenchmarkMethodologyConfig(
        num_samples=payload.get("num_samples"),
        min_generation_turns_per_config=int(payload["min_generation_turns_per_config"]),
        min_warmup_generation_turns=int(payload["min_warmup_generation_turns"]),
        runs_per_config=int(payload["runs_per_config"]),
        timeout_s=int(payload["timeout_s"]),
        server_shutdown_drain_timeout_s=float(
            payload["server_shutdown_drain_timeout_s"]
        ),
        server_shutdown_timeout_s=float(payload["server_shutdown_timeout_s"]),
    )


def benchmark_job_to_payload(job: BenchmarkJob) -> dict[str, Any]:
    return {
        "target_model": job.target_model,
        "target_model_revision": job.target_model_revision,
        "trtllm_gen_moe_cubin_pool": job.trtllm_gen_moe_cubin_pool,
        "workload": job.workload,
        "deployment": _deployment_to_payload(job.deployment),
        "concurrency": job.concurrency,
        "run_index": job.run_index,
        "sampling": _sampling_config_to_payload(job.sampling),
        "methodology": _methodology_to_payload(job.methodology),
    }


def benchmark_job_from_payload(payload: dict[str, Any]) -> BenchmarkJob:
    return BenchmarkJob(
        target_model=str(payload["target_model"]),
        target_model_revision=payload.get("target_model_revision"),
        trtllm_gen_moe_cubin_pool=payload.get("trtllm_gen_moe_cubin_pool"),
        workload=str(payload["workload"]),
        deployment=_deployment_from_payload(payload["deployment"]),
        concurrency=int(payload["concurrency"]),
        run_index=int(payload["run_index"]),
        sampling=_sampling_config_from_payload(payload["sampling"]),
        methodology=_methodology_from_payload(payload["methodology"]),
    )


def _parse_workload_selection(value: str) -> list[str]:
    values = _parse_str_csv(value)
    if values == ["all"]:
        return list(SUPPORTED_WORKLOADS)
    unknown = sorted(set(values) - set(SUPPORTED_WORKLOADS))
    if unknown:
        raise ValueError(
            f"Unknown workloads: {','.join(unknown)}. Supported: "
            f"{','.join(SUPPORTED_WORKLOADS)} or all."
        )
    if not values:
        raise ValueError("--workloads must include at least one workload.")
    duplicates = _duplicate_values(values)
    if duplicates:
        raise ValueError(f"Duplicate workloads: {','.join(duplicates)}.")
    return values


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _download_to_cache(url: str, filename: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / filename
    if out_path.exists():
        return out_path

    tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
    print(f"[download] {url}")
    urlretrieve(url, tmp_path)
    os.replace(tmp_path, out_path)
    return out_path


def _resolve_model_reference(model: str) -> str:
    local_path = Path(model).expanduser()
    if local_path.exists():
        return str(local_path.resolve())
    return model


def _effective_model_revision(model: str, revision: Optional[str]) -> Optional[str]:
    return None if Path(model).expanduser().exists() else revision


def _load_gsm8k_user_prompts() -> list[str]:
    path = _download_to_cache(GSM8K_TEST_URL, "gsm8k_test.jsonl")
    if not path.is_file():
        raise RuntimeError(f"GSM8K data file does not exist: {path}")

    prompts: list[str] = []
    for row in _read_jsonl(path):
        prompts.append(
            row["question"]
            + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
    return prompts


def _load_math500_user_prompts() -> list[str]:
    rows = _load_hf_dataset_rows("HuggingFaceH4/MATH-500", split="test")

    prompts: list[str] = []
    for row in rows:
        prompts.append(
            row["problem"]
            + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
    return prompts


def _load_hf_dataset_rows(*load_args, **load_kwargs) -> list[dict]:
    from datasets import load_dataset

    return list(load_dataset(*load_args, **load_kwargs))


def _load_humaneval_user_prompts() -> list[str]:
    rows = _load_hf_dataset_rows("openai/openai_humaneval", split="test")

    return [row["prompt"] for row in rows]


def _load_mbpp_user_prompts() -> list[str]:
    rows = _load_hf_dataset_rows(
        "google-research-datasets/mbpp", "sanitized", split="test"
    )

    return [row["prompt"] for row in rows]


def _load_mt_bench_user_turns() -> list[list[str]]:
    path = _download_to_cache(MT_BENCH_QUESTION_URL, "mt_bench_question.jsonl")
    if not path.is_file():
        raise RuntimeError(f"MT-bench data file does not exist: {path}")
    rows = _read_jsonl(path)

    prompts: list[list[str]] = []
    for row in rows:
        turns = row.get("turns", row.get("prompt"))
        if not isinstance(turns, list):
            raise RuntimeError(
                "MT-bench rows must contain a list-valued `turns` or `prompt` field."
            )
        turns = [str(turn) for turn in turns[:2]]
        if len(turns) != 2:
            raise RuntimeError(
                f"MT-bench rows must contain exactly two turns; got {len(turns)}."
            )
        prompts.append(turns)
    return prompts


def _load_user_turns(workload: str) -> list[list[str]]:
    if workload == "gsm8k":
        return [[prompt] for prompt in _load_gsm8k_user_prompts()]
    if workload == "math500":
        return [[prompt] for prompt in _load_math500_user_prompts()]
    if workload == "humaneval":
        return [[prompt] for prompt in _load_humaneval_user_prompts()]
    if workload == "mbpp":
        return [[prompt] for prompt in _load_mbpp_user_prompts()]
    if workload == "mt-bench":
        return _load_mt_bench_user_turns()
    raise ValueError(f"Unknown workload: {workload}")


def _request_flush_cache(base_url: str, timeout_s: float) -> None:
    import requests

    requests.get(
        base_url + "/flush_cache",
        params={"timeout": float(timeout_s)},
        timeout=max(float(timeout_s) + 5.0, 10.0),
    ).raise_for_status()


def _flush_cache(
    base_url: str, timeout_s: float = SERVER_SHUTDOWN_DRAIN_TIMEOUT_S
) -> None:
    try:
        _request_flush_cache(base_url, timeout_s)
    except Exception as exc:
        raise RuntimeError(
            "Failed to flush cache before the next benchmark phase; "
            "SGLang still had pending requests after waiting for drain."
        ) from exc


def _flush_cache_best_effort(base_url: str, timeout_s: float) -> None:
    try:
        _request_flush_cache(base_url, timeout_s)
    except Exception as exc:
        print(f"[shutdown] /flush_cache failed before server shutdown: {exc}")


def _shutdown_server(
    proc, base_url: str, *, drain_timeout_s: float, kill_timeout_s: float
) -> None:
    from sglang.srt.utils import kill_process_tree

    if proc.poll() is not None:
        return

    _flush_cache_best_effort(base_url, drain_timeout_s)

    if proc.poll() is not None:
        return

    print(f"[shutdown] sending SIGTERM to server pid={proc.pid}")
    proc.terminate()
    try:
        proc.wait(timeout=float(kill_timeout_s))
        return
    except Exception:
        print(
            f"[shutdown] server pid={proc.pid} did not exit within "
            f"{kill_timeout_s}s; falling back to kill_process_tree."
        )

    kill_process_tree(proc.pid, wait_timeout=30)


def _send_generate(
    base_url: str,
    prompt: str | list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_s: int,
) -> dict:
    import requests

    sampling_params: dict = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
        "skip_special_tokens": False,
    }
    request: dict[str, Any] = {
        "sampling_params": sampling_params,
        "stream": False,
    }
    if isinstance(prompt, str):
        request["text"] = prompt
    else:
        request["input_ids"] = [int(token_id) for token_id in prompt]
    resp = requests.post(
        base_url + "/generate",
        json=request,
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    out = resp.json()
    if isinstance(out, list):
        raise RuntimeError(
            "Expected an object response for single /generate, but got "
            f"type={type(out).__name__}."
        )
    return out


@dataclass(frozen=True)
class BenchMetrics:
    sample_count: int
    generation_turn_count: int
    latency_s: float
    output_tokens: int
    output_toks_per_s: float
    # Tokens generated after prefill (completion_tokens - 1 per generation turn).
    decode_output_tokens: int
    # Union of measured scheduler decode intervals, so concurrent requests do
    # not double-count wall time.
    decode_active_s: float
    decode_output_toks_per_s: Optional[float]
    peak_request_decode_output_toks_per_s: Optional[float]
    # Aggregate/server definition: sum(completion_tokens) / sum(spec_verify_ct).
    spec_accept_length: Optional[float]
    # Equal-weight mean of completion_tokens / spec_verify_ct per request.
    spec_accept_length_mean_per_request: Optional[float]
    spec_accept_length_request_count: int
    spec_verify_ct_sum: int


@dataclass(frozen=True)
class JobResult:
    key: RunKey
    deployment: ServerDeployment
    source_sample_count: int
    source_generation_turn_count: int
    warmup_generation_turn_count: int
    run_index: int
    metrics: BenchMetrics


@dataclass(frozen=True)
class JobFailure:
    key: RunKey
    deployment: ServerDeployment
    run_index: int
    error_type: str
    error_message: str


@dataclass(frozen=True)
class ConfigResult:
    key: RunKey
    deployment: ServerDeployment
    source_sample_count: Optional[int]
    source_generation_turn_count: Optional[int]
    warmup_generation_turn_count: Optional[int]
    metrics: Optional[BenchMetrics]
    repeat_metrics: tuple[BenchMetrics, ...]
    successful_run_indices: tuple[int, ...]
    failures: tuple[JobFailure, ...]

    @property
    def run_count(self) -> int:
        return self.successful_run_count + self.failed_run_count

    @property
    def successful_run_count(self) -> int:
        return len(self.repeat_metrics)

    @property
    def failed_run_count(self) -> int:
        return len(self.failures)

    @property
    def status(self) -> str:
        if self.failed_run_count == 0:
            return "ok"
        if self.successful_run_count == 0:
            return "failed"
        return "partial_failed"


def _run_key_to_payload(key: RunKey) -> dict[str, Any]:
    return {
        "workload": key.workload,
        "backend": key.backend,
        "tp": key.tp,
        "concurrency": key.concurrency,
        "mode": key.mode,
    }


def _run_key_from_payload(payload: dict[str, Any]) -> RunKey:
    return RunKey(
        workload=str(payload["workload"]),
        backend=str(payload["backend"]),
        tp=int(payload["tp"]),
        concurrency=int(payload["concurrency"]),
        mode=str(payload["mode"]),
    )


def _bench_metrics_to_payload(metrics: BenchMetrics) -> dict[str, Any]:
    return {
        "sample_count": metrics.sample_count,
        "generation_turn_count": metrics.generation_turn_count,
        "latency_s": metrics.latency_s,
        "output_tokens": metrics.output_tokens,
        "output_toks_per_s": metrics.output_toks_per_s,
        "decode_output_tokens": metrics.decode_output_tokens,
        "decode_active_s": metrics.decode_active_s,
        "decode_output_toks_per_s": metrics.decode_output_toks_per_s,
        "peak_request_decode_output_toks_per_s": (
            metrics.peak_request_decode_output_toks_per_s
        ),
        "spec_accept_length": metrics.spec_accept_length,
        "spec_accept_length_mean_per_request": (
            metrics.spec_accept_length_mean_per_request
        ),
        "spec_accept_length_request_count": metrics.spec_accept_length_request_count,
        "spec_verify_ct_sum": metrics.spec_verify_ct_sum,
    }


def _bench_metrics_from_payload(payload: dict[str, Any]) -> BenchMetrics:
    return BenchMetrics(
        sample_count=int(payload["sample_count"]),
        generation_turn_count=int(payload["generation_turn_count"]),
        latency_s=float(payload["latency_s"]),
        output_tokens=int(payload["output_tokens"]),
        output_toks_per_s=float(payload["output_toks_per_s"]),
        decode_output_tokens=int(payload["decode_output_tokens"]),
        decode_active_s=float(payload["decode_active_s"]),
        decode_output_toks_per_s=(
            None
            if payload["decode_output_toks_per_s"] is None
            else float(payload["decode_output_toks_per_s"])
        ),
        peak_request_decode_output_toks_per_s=(
            None
            if payload.get("peak_request_decode_output_toks_per_s") is None
            else float(payload["peak_request_decode_output_toks_per_s"])
        ),
        spec_accept_length=payload.get("spec_accept_length"),
        spec_accept_length_mean_per_request=payload.get(
            "spec_accept_length_mean_per_request"
        ),
        spec_accept_length_request_count=int(
            payload.get("spec_accept_length_request_count", 0)
        ),
        spec_verify_ct_sum=int(payload["spec_verify_ct_sum"]),
    )


def job_outcome_to_payload(result: JobResult | JobFailure) -> dict[str, Any]:
    if isinstance(result, JobFailure):
        return {
            "kind": "failure",
            "key": _run_key_to_payload(result.key),
            "deployment": _deployment_to_payload(result.deployment),
            "run_index": result.run_index,
            "error_type": result.error_type,
            "error_message": result.error_message,
        }
    return {
        "kind": "result",
        "key": _run_key_to_payload(result.key),
        "deployment": _deployment_to_payload(result.deployment),
        "source_sample_count": result.source_sample_count,
        "source_generation_turn_count": result.source_generation_turn_count,
        "warmup_generation_turn_count": result.warmup_generation_turn_count,
        "run_index": result.run_index,
        "metrics": _bench_metrics_to_payload(result.metrics),
    }


def job_outcome_from_payload(payload: dict[str, Any]) -> JobResult | JobFailure:
    kind = payload["kind"]
    if kind == "failure":
        return JobFailure(
            key=_run_key_from_payload(payload["key"]),
            deployment=_deployment_from_payload(payload["deployment"]),
            run_index=int(payload["run_index"]),
            error_type=str(payload["error_type"]),
            error_message=str(payload["error_message"]),
        )
    if kind == "result":
        return JobResult(
            key=_run_key_from_payload(payload["key"]),
            deployment=_deployment_from_payload(payload["deployment"]),
            source_sample_count=int(payload["source_sample_count"]),
            source_generation_turn_count=int(payload["source_generation_turn_count"]),
            warmup_generation_turn_count=int(payload["warmup_generation_turn_count"]),
            run_index=int(payload["run_index"]),
            metrics=_bench_metrics_from_payload(payload["metrics"]),
        )
    raise ValueError(f"Unsupported job outcome kind: {kind}")


@dataclass(frozen=True)
class SampleMetrics:
    generation_turn_count: int
    output_tokens: int
    decode_output_tokens: int
    decode_intervals: tuple[tuple[float, float], ...]
    peak_request_decode_output_toks_per_s: Optional[float]
    spec_verify_ct_sum: int
    spec_accept_lengths_per_request: tuple[float, ...]


def _extract_generated_text(out: dict) -> str:
    text = out.get("text")
    if isinstance(text, str):
        return text
    if isinstance(text, list) and len(text) == 1 and isinstance(text[0], str):
        return text[0]
    raise RuntimeError(
        "Expected /generate response to include generated text in `text`; "
        f"got keys={sorted(out.keys())}."
    )


def _extract_generate_stats(
    out: dict,
) -> tuple[int, int, int, Optional[tuple[float, float]]]:
    meta = out.get("meta_info", {}) or {}
    if "completion_tokens" not in meta:
        raise RuntimeError(
            "/generate response is missing `meta_info.completion_tokens`."
        )
    output_tokens = int(meta["completion_tokens"])
    spec_verify_ct = int(meta.get("spec_verify_ct", 0))
    decode_output_tokens = max(output_tokens - 1, 0)
    if decode_output_tokens == 0:
        return output_tokens, spec_verify_ct, decode_output_tokens, None

    timestamp_fields = ("prefill_finished_time", "last_decode_finish_time")
    missing_fields = [field for field in timestamp_fields if field not in meta]
    if missing_fields:
        raise RuntimeError(
            "/generate response is missing decode timing metadata: "
            + ", ".join(f"`meta_info.{field}`" for field in missing_fields)
            + "."
        )
    prefill_finished_time = float(meta["prefill_finished_time"])
    last_decode_finish_time = float(meta["last_decode_finish_time"])
    if not (
        math.isfinite(prefill_finished_time) and math.isfinite(last_decode_finish_time)
    ):
        raise RuntimeError(
            "Decode timing metadata must be finite, got "
            f"prefill_finished_time={prefill_finished_time}, "
            f"last_decode_finish_time={last_decode_finish_time}."
        )
    if last_decode_finish_time <= prefill_finished_time:
        raise RuntimeError(
            "Decode timing metadata is not ordered, got "
            f"prefill_finished_time={prefill_finished_time}, "
            f"last_decode_finish_time={last_decode_finish_time}."
        )
    return (
        output_tokens,
        spec_verify_ct,
        decode_output_tokens,
        (prefill_finished_time, last_decode_finish_time),
    )


def _interval_union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0

    sorted_intervals = sorted(intervals)
    current_start, current_end = sorted_intervals[0]
    union_duration = 0.0
    for start, end in sorted_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        union_duration += current_end - current_start
        current_start, current_end = start, end
    union_duration += current_end - current_start
    return float(union_duration)


def _run_sample(
    base_url: str,
    *,
    turns: list[str],
    tokenizer,
    sampling: SamplingConfig,
    timeout_s: int,
) -> SampleMetrics:
    from sglang.srt.parser.reasoning_parser import KimiK3Detector

    messages: list[dict[str, Any]] = []
    total_tokens = 0
    decode_output_tokens = 0
    decode_intervals: list[tuple[float, float]] = []
    peak_request_decode_output_toks_per_s = None
    spec_verify_ct_sum = 0
    accept_lengths_per_request: list[float] = []

    for turn_idx, user_content in enumerate(turns):
        messages.append({"role": "user", "content": user_content})
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            thinking=sampling.enable_thinking,
        )
        out = _send_generate(
            base_url=base_url,
            prompt=prompt_ids,
            max_new_tokens=sampling.max_new_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            timeout_s=timeout_s,
        )
        (
            output_tokens,
            spec_verify_ct,
            turn_decode_output_tokens,
            decode_interval,
        ) = _extract_generate_stats(out)
        total_tokens += output_tokens
        decode_output_tokens += turn_decode_output_tokens
        if decode_interval is not None:
            decode_intervals.append(decode_interval)
            decode_duration = decode_interval[1] - decode_interval[0]
            request_decode_toks_per_s = (
                float(turn_decode_output_tokens) / decode_duration
            )
            peak_request_decode_output_toks_per_s = max(
                peak_request_decode_output_toks_per_s or 0.0,
                request_decode_toks_per_s,
            )
        spec_verify_ct_sum += spec_verify_ct
        if spec_verify_ct > 0:
            accept_lengths_per_request.append(
                float(output_tokens) / float(spec_verify_ct)
            )

        if turn_idx + 1 < len(turns):
            parsed = KimiK3Detector(
                force_reasoning=sampling.enable_thinking
            ).detect_and_parse(_extract_generated_text(out))
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": parsed.normal_text,
            }
            if parsed.reasoning_text:
                assistant_message["reasoning_content"] = parsed.reasoning_text
            messages.append(assistant_message)

    return SampleMetrics(
        generation_turn_count=len(turns),
        output_tokens=int(total_tokens),
        decode_output_tokens=int(decode_output_tokens),
        decode_intervals=tuple(decode_intervals),
        peak_request_decode_output_toks_per_s=peak_request_decode_output_toks_per_s,
        spec_verify_ct_sum=int(spec_verify_ct_sum),
        spec_accept_lengths_per_request=tuple(accept_lengths_per_request),
    )


def _run_unmeasured_requests(
    base_url: str,
    *,
    samples: list[list[str]],
    tokenizer,
    sampling: SamplingConfig,
    concurrency: int,
    timeout_s: int,
) -> None:
    if not samples:
        return

    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        futures = [
            pool.submit(
                _run_sample,
                base_url=base_url,
                turns=turns,
                tokenizer=tokenizer,
                sampling=sampling,
                timeout_s=timeout_s,
            )
            for turns in samples
        ]
        for fut in as_completed(futures):
            fut.result()


def _take_samples(
    samples: list[list[str]], *, start: int, count: int
) -> list[list[str]]:
    if count <= 0:
        return []
    if not samples:
        raise RuntimeError("Cannot take benchmark samples from an empty workload.")
    return [samples[(start + i) % len(samples)] for i in range(count)]


def _generation_turn_count(samples: list[list[str]]) -> int:
    return sum(len(turns) for turns in samples)


def _take_samples_for_min_generation_turns(
    samples: list[list[str]], *, start: int, min_generation_turns: int
) -> list[list[str]]:
    if min_generation_turns <= 0:
        return []
    if not samples:
        raise RuntimeError("Cannot take benchmark samples from an empty workload.")

    out: list[list[str]] = []
    generation_turns = 0
    idx = 0
    while generation_turns < int(min_generation_turns):
        sample = samples[(start + idx) % len(samples)]
        out.append(sample)
        generation_turns += len(sample)
        idx += 1
    return out


def _build_measured_samples(
    samples: list[list[str]], *, num_samples: Optional[int], min_generation_turns: int
) -> list[list[str]]:
    if not samples:
        raise RuntimeError("Cannot build measured samples from an empty workload.")
    if num_samples is not None:
        if num_samples <= 0:
            raise RuntimeError(f"--num-samples must be > 0, got {num_samples}.")
        return _take_samples(samples, start=0, count=int(num_samples))
    if min_generation_turns < 0:
        raise RuntimeError(
            "--min-generation-turns-per-config must be >= 0, "
            f"got {min_generation_turns}."
        )
    source_generation_turns = _generation_turn_count(samples)
    if source_generation_turns <= 0:
        raise RuntimeError("Cannot build measured samples with zero generation turns.")
    repeats = max(1, math.ceil(int(min_generation_turns) / source_generation_turns))
    return samples * repeats


def _build_measured_samples_for_concurrency(
    samples: list[list[str]],
    *,
    methodology: BenchmarkMethodologyConfig,
    concurrency: int,
) -> list[list[str]]:
    # Concurrency 1 is the stable accept-length pass; use one full workload by
    # default instead of cache-favorable repeated copies.
    if methodology.num_samples is None and int(concurrency) == 1:
        return samples
    return _build_measured_samples(
        samples,
        num_samples=methodology.num_samples,
        min_generation_turns=int(methodology.min_generation_turns_per_config),
    )


def _build_benchmark_plan(
    samples: list[list[str]],
    *,
    concurrency: int,
    methodology: BenchmarkMethodologyConfig,
) -> BenchmarkPlan:
    measured_samples = _build_measured_samples_for_concurrency(
        samples,
        methodology=methodology,
        concurrency=int(concurrency),
    )
    warmup_min_generation_turns = max(
        int(methodology.min_warmup_generation_turns), 2 * int(concurrency)
    )
    warmup_samples = _take_samples_for_min_generation_turns(
        measured_samples,
        start=0,
        min_generation_turns=warmup_min_generation_turns,
    )
    return BenchmarkPlan(
        measured_samples=measured_samples,
        warmup_samples=warmup_samples,
    )


def _run_requests(
    base_url: str,
    *,
    samples: list[list[str]],
    tokenizer,
    sampling: SamplingConfig,
    concurrency: int,
    timeout_s: int,
    expect_spec: bool,
) -> BenchMetrics:
    start = time.perf_counter()
    total_tokens = 0
    decode_output_tokens = 0
    decode_intervals: list[tuple[float, float]] = []
    peak_request_decode_output_toks_per_s = None
    spec_verify_ct_sum = 0
    accept_lengths_per_request: list[float] = []
    generation_turn_count = 0

    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        measured_futures = [
            pool.submit(
                _run_sample,
                base_url=base_url,
                turns=turns,
                tokenizer=tokenizer,
                sampling=sampling,
                timeout_s=timeout_s,
            )
            for turns in samples
        ]
        for fut in as_completed(measured_futures):
            sample_metrics = fut.result()
            total_tokens += sample_metrics.output_tokens
            decode_output_tokens += sample_metrics.decode_output_tokens
            decode_intervals.extend(sample_metrics.decode_intervals)
            if sample_metrics.peak_request_decode_output_toks_per_s is not None:
                peak_request_decode_output_toks_per_s = max(
                    peak_request_decode_output_toks_per_s or 0.0,
                    sample_metrics.peak_request_decode_output_toks_per_s,
                )
            spec_verify_ct_sum += sample_metrics.spec_verify_ct_sum
            accept_lengths_per_request.extend(
                sample_metrics.spec_accept_lengths_per_request
            )
            generation_turn_count += sample_metrics.generation_turn_count

        latency = time.perf_counter() - start

    toks_per_s = total_tokens / max(latency, 1e-6)
    decode_active_s = _interval_union_duration(decode_intervals)
    decode_output_toks_per_s = (
        float(decode_output_tokens) / decode_active_s
        if decode_output_tokens > 0 and decode_active_s > 0.0
        else None
    )

    if expect_spec and spec_verify_ct_sum <= 0:
        raise RuntimeError(
            "Speculative decoding sanity check failed: did not observe any "
            "`spec_verify_ct` in responses (speculative decoding may not have been enabled)."
        )

    spec_accept_length = (
        float(total_tokens) / float(spec_verify_ct_sum)
        if spec_verify_ct_sum > 0
        else None
    )
    spec_accept_length_mean_per_request = (
        float(statistics.mean(accept_lengths_per_request))
        if accept_lengths_per_request
        else None
    )

    return BenchMetrics(
        sample_count=len(samples),
        generation_turn_count=int(generation_turn_count),
        latency_s=float(latency),
        output_tokens=int(total_tokens),
        output_toks_per_s=float(toks_per_s),
        decode_output_tokens=int(decode_output_tokens),
        decode_active_s=float(decode_active_s),
        decode_output_toks_per_s=decode_output_toks_per_s,
        peak_request_decode_output_toks_per_s=peak_request_decode_output_toks_per_s,
        spec_accept_length=spec_accept_length,
        spec_accept_length_mean_per_request=spec_accept_length_mean_per_request,
        spec_accept_length_request_count=len(accept_lengths_per_request),
        spec_verify_ct_sum=int(spec_verify_ct_sum),
    )


def _format_table(
    *,
    tp_sizes: list[int],
    concurrencies: list[int],
    values: dict[tuple[int, int], Optional[float]],
    float_fmt: str,
) -> str:
    header = ["tp\\conc"] + [str(c) for c in concurrencies]
    rows: list[list[str]] = [header]
    for tp in tp_sizes:
        row = [str(tp)]
        for c in concurrencies:
            v = values.get((tp, c), None)
            row.append("N/A" if v is None else format(v, float_fmt))
        rows.append(row)

    col_widths = [
        max(len(row[col_idx]) for row in rows) for col_idx in range(len(rows[0]))
    ]

    lines: list[str] = []
    lines.append("  ".join(cell.rjust(col_widths[i]) for i, cell in enumerate(rows[0])))
    lines.append("  ".join("-" * w for w in col_widths))
    for row in rows[1:]:
        lines.append("  ".join(cell.rjust(col_widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def _build_shared_server_configs(
    *,
    device_sm: int,
    visible_gpus: int,
    max_concurrency: int,
    load_format: str = "auto",
    random_seed: int = DEFAULT_RANDOM_SEED,
    linear_attn_prefill_backend: str = DEFAULT_LINEAR_ATTN_PREFILL_BACKEND,
    linear_attn_decode_backend: str = DEFAULT_LINEAR_ATTN_DECODE_BACKEND,
    linear_attn_verify_backend: str = DEFAULT_LINEAR_ATTN_VERIFY_BACKEND,
) -> list[SharedServerConfig]:
    if device_sm != 103:
        raise RuntimeError(
            f"This benchmark profile targets B300 (SM103); got SM{device_sm}."
        )
    configs = [
        replace(
            BASE_SHARED_SERVER_CONFIG,
            load_format=load_format,
            random_seed=int(random_seed),
            linear_attn_prefill_backend=linear_attn_prefill_backend,
            linear_attn_decode_backend=linear_attn_decode_backend,
            linear_attn_verify_backend=linear_attn_verify_backend,
            max_running_requests=max(
                BASE_SHARED_SERVER_CONFIG.max_running_requests,
                int(max_concurrency),
            ),
        )
    ]
    runnable_configs = [
        config for config in configs if 1 <= config.tp_size <= visible_gpus
    ]
    if not runnable_configs:
        raise RuntimeError(
            f"No shared server configs are runnable with visible_gpus={visible_gpus}. "
            "Set CUDA_VISIBLE_DEVICES accordingly."
        )
    return runnable_configs


def _build_deployments(
    shared_config: SharedServerConfig, sweep: DeploymentSweep
) -> list[ServerDeployment]:
    deployments: list[ServerDeployment] = []
    if sweep.include_baseline:
        deployments.append(
            ServerDeployment(
                shared_config=shared_config,
                mode_config=BaselineConfig(),
            )
        )

    for block_size in sweep.dflash_block_sizes:
        deployments.append(
            ServerDeployment(
                shared_config=shared_config,
                mode_config=DFlashConfig(
                    draft_model=sweep.dflash_draft_model,
                    block_size=block_size,
                    draft_attention_backend=sweep.dflash_draft_attention_backend,
                    enable_replayssm=sweep.dflash_enable_replayssm,
                ),
            )
        )
    return deployments


def _build_benchmark_jobs(
    config: SweepConfig, shared_configs: list[SharedServerConfig]
) -> list[BenchmarkJob]:
    jobs: list[BenchmarkJob] = []
    for shared_config in shared_configs:
        deployments = _build_deployments(shared_config, config.deployment_sweep)
        for deployment in deployments:
            for workload in config.workloads:
                for concurrency in config.concurrencies:
                    run_deployment = replace(
                        deployment,
                        shared_config=replace(
                            shared_config,
                            max_running_requests=int(concurrency),
                            cuda_graph_max_bs_decode=int(concurrency),
                        ),
                    )
                    for run_index in range(config.methodology.runs_per_config):
                        jobs.append(
                            BenchmarkJob(
                                target_model=config.target_model,
                                target_model_revision=config.target_model_revision,
                                trtllm_gen_moe_cubin_pool=(
                                    config.trtllm_gen_moe_cubin_pool
                                ),
                                workload=workload,
                                deployment=run_deployment,
                                concurrency=concurrency,
                                run_index=run_index,
                                sampling=config.sampling,
                                methodology=config.methodology,
                            )
                        )
    return jobs


def _mode_display_name(mode: str) -> str:
    if mode.startswith("dflash_b"):
        return f"DFLASH block={mode.removeprefix('dflash_b')}"
    return {
        "baseline": "Baseline",
        "dflash": "DFLASH",
    }.get(mode, mode)


def _collect_metric(
    *,
    results: dict[tuple[str, int, int, str], BenchMetrics],
    backend: str,
    tp_sizes: list[int],
    concurrencies: list[int],
    mode: str,
    field: str,
) -> dict[tuple[int, int], Optional[float]]:
    out: dict[tuple[int, int], Optional[float]] = {}
    for tp in tp_sizes:
        for conc in concurrencies:
            metrics = results.get((backend, tp, conc, mode), None)
            out[(tp, conc)] = None if metrics is None else getattr(metrics, field)
    return out


def _compute_speedup(
    baseline: dict[tuple[int, int], Optional[float]],
    speculative: dict[tuple[int, int], Optional[float]],
) -> dict[tuple[int, int], Optional[float]]:
    return {
        key: None if (b is None or d is None or b <= 0) else (d / b)
        for key, b in baseline.items()
        for d in [speculative.get(key, None)]
    }


def _metric_map_from_config_results(
    config_results: list[ConfigResult],
) -> dict[tuple[str, int, int, str], BenchMetrics]:
    return {
        result.key.metric_key(): result.metrics
        for result in config_results
        if result.metrics is not None
    }


def _print_kv_lines(items: list[tuple[str, object]]) -> None:
    for key, value in items:
        print(f"{key}={value}")


def _print_failure_summary(config_results: list[ConfigResult]) -> None:
    failed_results = [
        result for result in config_results if result.failed_run_count > 0
    ]
    if not failed_results:
        return

    print("\n=== Failed/Partial Runs ===")
    for result in failed_results:
        key = result.key
        print(
            f"workload={key.workload} backend={key.backend} tp={key.tp} "
            f"mode={key.mode} conc={key.concurrency} status={result.status} "
            f"successful_runs={result.successful_run_count} "
            f"failed_runs={result.failed_run_count} "
            f"successful_run_numbers={_format_successful_run_numbers(result)} "
            f"failed_run_numbers={_format_failed_run_numbers(result.failures)} "
            f"errors={_format_failure_messages(result.failures)}"
        )


def _server_env_for_job(job: BenchmarkJob) -> dict[str, str]:
    env = {
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
        "SGLANG_CUDA_COREDUMP_BEFORE_CRASH": "0",
    }
    if isinstance(job.deployment.mode_config, DFlashConfig):
        env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"
    if job.trtllm_gen_moe_cubin_pool is not None:
        env["SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL"] = job.trtllm_gen_moe_cubin_pool
    return env


def _run_benchmark_job(job: BenchmarkJob) -> JobResult:
    from transformers import AutoTokenizer

    from sglang.test.test_utils import (
        DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH as SGLANG_DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    )
    from sglang.test.test_utils import (
        find_available_port,
        popen_launch_server,
    )

    key = job.key
    print(f"\n=== {job.label} {job.run_label} ===")
    if job.trtllm_gen_moe_cubin_pool is not None:
        cubin_pool = Path(job.trtllm_gen_moe_cubin_pool).expanduser()
        if not cubin_pool.is_dir():
            raise RuntimeError(
                "TRTLLM-Gen cubin pool does not exist or is not a directory: "
                f"{cubin_pool}"
            )
    samples = _load_user_turns(job.workload)
    if not samples:
        raise RuntimeError(f"Workload '{job.workload}' did not produce any prompts.")

    source_sample_count = len(samples)
    source_generation_turn_count = _generation_turn_count(samples)
    plan = _build_benchmark_plan(
        samples,
        concurrency=job.concurrency,
        methodology=job.methodology,
    )
    if plan.measured_sample_count > source_sample_count:
        print(
            "[config] measured sample count exceeds workload size; "
            "repeating whole workload copies with radix cache enabled."
        )

    base_url = f"http://127.0.0.1:{find_available_port(20000)}"
    model_path = _resolve_model_reference(job.target_model)
    model_revision = _effective_model_revision(
        job.target_model, job.target_model_revision
    )
    print(f"model_reference={model_path}")
    print(f"model_revision={model_revision}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=model_revision,
        trust_remote_code=True,
    )
    server_start_timeout_s = int(
        max(SGLANG_DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, job.methodology.timeout_s)
    )
    server_env = _server_env_for_job(job)
    print(
        "server_env="
        + ",".join(f"{key}:{value}" for key, value in sorted(server_env.items()))
    )
    server_args = list(job.deployment.server_args)
    if model_revision is not None:
        server_args.extend(["--revision", model_revision])
    proc = popen_launch_server(
        model_path,
        base_url,
        timeout=server_start_timeout_s,
        other_args=server_args,
        env=server_env,
    )
    try:
        _send_generate(
            base_url,
            "Hello",
            max_new_tokens=8,
            temperature=job.sampling.temperature,
            top_p=job.sampling.top_p,
            top_k=job.sampling.top_k,
            timeout_s=min(job.methodology.timeout_s, 300),
        )

        _flush_cache(base_url)
        print(
            f"[warmup {job.run_label}] run {len(plan.warmup_samples)} samples / "
            f"{plan.warmup_generation_turn_count} generation turns after "
            "/flush_cache; excluded from metrics."
        )
        _run_unmeasured_requests(
            base_url,
            samples=plan.warmup_samples,
            tokenizer=tokenizer,
            sampling=job.sampling,
            concurrency=job.concurrency,
            timeout_s=job.methodology.timeout_s,
        )
        _flush_cache(base_url)
        print(
            f"[warmup {job.run_label}] flushed cache after warmup; "
            "starting measured workload."
        )
        metrics = _run_requests(
            base_url,
            samples=plan.measured_samples,
            tokenizer=tokenizer,
            sampling=job.sampling,
            concurrency=job.concurrency,
            timeout_s=job.methodology.timeout_s,
            expect_spec=job.deployment.expect_spec,
        )
        decode_toks_per_s = (
            "N/A"
            if metrics.decode_output_toks_per_s is None
            else f"{metrics.decode_output_toks_per_s:,.2f}"
        )
        peak_decode_toks_per_s = (
            "N/A"
            if metrics.peak_request_decode_output_toks_per_s is None
            else f"{metrics.peak_request_decode_output_toks_per_s:,.2f}"
        )
        line = (
            f"[{job.label} {job.run_label}] samples={plan.measured_sample_count:<4} "
            f"turns={plan.measured_generation_turn_count:<4} "
            f"toks/s={metrics.output_toks_per_s:,.2f} "
            f"decode_toks/s={decode_toks_per_s} "
            f"peak_request_decode_toks/s={peak_decode_toks_per_s} "
            f"latency={metrics.latency_s:.1f}s "
            f"warmup_turns={plan.warmup_generation_turn_count}"
        )
        if job.deployment.expect_spec:
            accept_len_aggregate = (
                "N/A"
                if metrics.spec_accept_length is None
                else f"{metrics.spec_accept_length:.3f}"
            )
            accept_len_mean = (
                "N/A"
                if metrics.spec_accept_length_mean_per_request is None
                else f"{metrics.spec_accept_length_mean_per_request:.3f}"
            )
            line += (
                f" accept_len_aggregate={accept_len_aggregate} "
                f"accept_len_mean_per_request={accept_len_mean} "
                f"accept_len_request_count={metrics.spec_accept_length_request_count} "
                f"spec_verify_ct_sum={metrics.spec_verify_ct_sum}"
            )
        print(line)
        return JobResult(
            key=key,
            deployment=job.deployment,
            source_sample_count=source_sample_count,
            source_generation_turn_count=source_generation_turn_count,
            warmup_generation_turn_count=plan.warmup_generation_turn_count,
            run_index=job.run_index,
            metrics=metrics,
        )
    finally:
        _shutdown_server(
            proc,
            base_url,
            drain_timeout_s=job.methodology.server_shutdown_drain_timeout_s,
            kill_timeout_s=job.methodology.server_shutdown_timeout_s,
        )


def _run_benchmark_job_gracefully(job: BenchmarkJob) -> JobResult | JobFailure:
    try:
        return _run_benchmark_job(job)
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = _one_line(str(exc) or repr(exc))
        print(f"[failed {job.label} {job.run_label}] " f"{error_type}: {error_message}")
        return JobFailure(
            key=job.key,
            deployment=job.deployment,
            run_index=job.run_index,
            error_type=error_type,
            error_message=error_message,
        )


def _print_summary(
    *,
    config: SweepConfig,
    workload: str,
    config_results: list[ConfigResult],
    shared_configs: list[SharedServerConfig],
    attention_backends: list[str],
    tp_sizes: list[int],
    concurrencies: list[int],
    device_sm: int,
    mode_keys: list[str],
    source_sample_count: Optional[int],
    source_generation_turn_count: Optional[int],
    results: dict[tuple[str, int, int, str], BenchMetrics],
) -> None:
    print("\n=== Kimi-K3 DFlash Benchmark Summary ===")
    _print_kv_lines(
        [
            ("workload", workload),
            ("source_sample_count", source_sample_count),
            ("source_generation_turn_count", source_generation_turn_count),
            ("target_model", config.target_model),
            ("target_model_revision", config.target_model_revision),
            ("load_format", config.load_format),
            ("random_seed", config.random_seed),
            (
                "linear_attn_prefill_backend",
                config.linear_attn_prefill_backend,
            ),
            (
                "linear_attn_decode_backend",
                config.linear_attn_decode_backend,
            ),
            (
                "linear_attn_verify_backend",
                config.linear_attn_verify_backend,
            ),
            (
                "dflash_draft_model",
                config.deployment_sweep.dflash_draft_model,
            ),
            (
                "speculative_draft_attention_backend",
                config.deployment_sweep.dflash_draft_attention_backend,
            ),
            (
                "trtllm_gen_moe_cubin_pool",
                config.trtllm_gen_moe_cubin_pool,
            ),
            ("max_new_tokens", config.sampling.max_new_tokens),
            ("enable_thinking", bool(config.sampling.enable_thinking)),
            ("timeout_s", config.methodology.timeout_s),
            (
                "server_shutdown_drain_timeout_s",
                config.methodology.server_shutdown_drain_timeout_s,
            ),
            (
                "server_shutdown_timeout_s",
                config.methodology.server_shutdown_timeout_s,
            ),
            (
                "shared_server_configs",
                ";".join(
                    server_config.summary_label() for server_config in shared_configs
                ),
            ),
            (
                "sampling",
                f"temperature:{config.sampling.temperature}, "
                f"top_p:{config.sampling.top_p}, top_k:{config.sampling.top_k}",
            ),
            ("attention_backends", ",".join(attention_backends)),
            (
                "dflash_block_sizes",
                ",".join(
                    "default" if x is None else str(x)
                    for x in config.deployment_sweep.dflash_block_sizes
                ),
            ),
            (
                "dflash_enable_replayssm",
                config.deployment_sweep.dflash_enable_replayssm,
            ),
            ("tp_sizes", ",".join(str(x) for x in tp_sizes)),
            ("concurrencies", ",".join(str(x) for x in concurrencies)),
            ("num_samples", config.methodology.num_samples),
            ("runs_per_config", config.methodology.runs_per_config),
            (
                "decode_output_tps_definition",
                "sum(max(completion_tokens - 1, 0)) / "
                "duration(union([prefill_finished_time, "
                "last_decode_finish_time]))",
            ),
            (
                "peak_request_decode_output_tps_definition",
                "max((completion_tokens - 1) / "
                "(last_decode_finish_time - prefill_finished_time) "
                "for completion_tokens > 1)",
            ),
            (
                "min_generation_turns_per_config",
                config.methodology.min_generation_turns_per_config,
            ),
            (
                "min_warmup_generation_turns",
                config.methodology.min_warmup_generation_turns,
            ),
            ("disable_radix_cache", False),
            ("device_sm", device_sm),
            ("skip_baseline", not config.deployment_sweep.include_baseline),
        ]
    )
    _print_failure_summary(config_results)

    for backend in attention_backends:
        print(f"\n=== Backend: {backend} ===")
        baseline_output_tps = _collect_metric(
            results=results,
            backend=backend,
            tp_sizes=tp_sizes,
            concurrencies=concurrencies,
            mode="baseline",
            field="output_toks_per_s",
        )
        baseline_decode_output_tps = _collect_metric(
            results=results,
            backend=backend,
            tp_sizes=tp_sizes,
            concurrencies=concurrencies,
            mode="baseline",
            field="decode_output_toks_per_s",
        )
        baseline_peak_request_decode_output_tps = _collect_metric(
            results=results,
            backend=backend,
            tp_sizes=tp_sizes,
            concurrencies=concurrencies,
            mode="baseline",
            field="peak_request_decode_output_toks_per_s",
        )
        sections: list[tuple[str, dict[tuple[int, int], Optional[float]], str]] = [
            ("Baseline output tok/s", baseline_output_tps, ",.2f"),
            (
                "Baseline decode-only output tok/s",
                baseline_decode_output_tps,
                ",.2f",
            ),
            (
                "Baseline peak request decode-only output tok/s",
                baseline_peak_request_decode_output_tps,
                ",.2f",
            ),
        ]

        for spec_mode in mode_keys:
            display_name = _mode_display_name(spec_mode)
            spec_output_tps = _collect_metric(
                results=results,
                backend=backend,
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                mode=spec_mode,
                field="output_toks_per_s",
            )
            spec_decode_output_tps = _collect_metric(
                results=results,
                backend=backend,
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                mode=spec_mode,
                field="decode_output_toks_per_s",
            )
            spec_peak_request_decode_output_tps = _collect_metric(
                results=results,
                backend=backend,
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                mode=spec_mode,
                field="peak_request_decode_output_toks_per_s",
            )
            spec_accept_length = _collect_metric(
                results=results,
                backend=backend,
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                mode=spec_mode,
                field="spec_accept_length",
            )
            spec_accept_length_mean_per_request = _collect_metric(
                results=results,
                backend=backend,
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                mode=spec_mode,
                field="spec_accept_length_mean_per_request",
            )
            sections.extend(
                [
                    (f"{display_name} output tok/s", spec_output_tps, ",.2f"),
                    (
                        f"{display_name} decode-only output tok/s",
                        spec_decode_output_tps,
                        ",.2f",
                    ),
                    (
                        f"{display_name} peak request decode-only output tok/s",
                        spec_peak_request_decode_output_tps,
                        ",.2f",
                    ),
                    (
                        f"Speedup ({display_name} / baseline)",
                        _compute_speedup(baseline_output_tps, spec_output_tps),
                        ".3f",
                    ),
                    (
                        f"Decode-only speedup ({display_name} / baseline)",
                        _compute_speedup(
                            baseline_decode_output_tps,
                            spec_decode_output_tps,
                        ),
                        ".3f",
                    ),
                    (
                        f"{display_name} aggregate acceptance length",
                        spec_accept_length,
                        ".3f",
                    ),
                    (
                        f"{display_name} mean acceptance length per request",
                        spec_accept_length_mean_per_request,
                        ".3f",
                    ),
                ]
            )

        for title, values, fmt in sections:
            print(f"\n{title}")
            print(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values=values,
                    float_fmt=fmt,
                )
            )


CSV_FIELDS = [
    "workload",
    "backend",
    "linear_attn_prefill_backend",
    "linear_attn_decode_backend",
    "linear_attn_verify_backend",
    "tp",
    "mode",
    "dflash_block_size",
    "concurrency",
    "source_sample_count",
    "source_generation_turn_count",
    "runs_per_config",
    "successful_runs",
    "failed_runs",
    "status",
    "successful_run_numbers",
    "failed_run_numbers",
    "failure_messages",
    "measured_sample_count",
    "measured_generation_turn_count",
    "output_toks_per_s",
    "output_toks_per_s_std",
    "decode_output_toks_per_s",
    "decode_output_toks_per_s_std",
    "peak_request_decode_output_toks_per_s",
    "latency_s",
    "latency_s_std",
    "decode_active_s",
    "decode_active_s_std",
    "output_tokens",
    "decode_output_tokens",
    "speedup_vs_baseline",
    "decode_speedup_vs_baseline",
    "accept_length_aggregate_from_conc1",
    "accept_length_mean_from_conc1",
    "accept_length_aggregate_this_conc",
    "accept_length_aggregate_this_conc_std",
    "accept_length_mean_this_conc",
    "accept_length_mean_this_conc_std",
    "accept_length_request_count",
    "spec_verify_ct_sum",
]


def _one_line(value: str) -> str:
    return " ".join(str(value).split())


def _fmt_optional_int(value: Optional[int]) -> str:
    if value is None:
        return ""
    return str(value)


def _fmt_csv_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _format_failed_run_numbers(failures: tuple[JobFailure, ...]) -> str:
    return ",".join(str(failure.run_index + 1) for failure in failures)


def _format_successful_run_numbers(result: ConfigResult) -> str:
    return ",".join(str(run_index + 1) for run_index in result.successful_run_indices)


def _format_failure_messages(failures: tuple[JobFailure, ...]) -> str:
    return " | ".join(
        (
            f"run={failure.run_index + 1} "
            f"{failure.error_type}: {failure.error_message}"
        )
        for failure in failures
    )


def _mean_optional(values: list[Optional[float]]) -> Optional[float]:
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    return float(statistics.mean(present_values))


def _stdev_optional(values: list[Optional[float]]) -> Optional[float]:
    present_values = [value for value in values if value is not None]
    if len(present_values) < 2:
        return None
    return float(statistics.stdev(present_values))


def _metric_stdev(metrics: tuple[BenchMetrics, ...], field: str) -> Optional[float]:
    return _stdev_optional([getattr(metric, field) for metric in metrics])


def _aggregate_bench_metrics(metrics: list[BenchMetrics]) -> BenchMetrics:
    if not metrics:
        raise RuntimeError("Cannot aggregate an empty metrics list.")
    first = metrics[0]
    total_output_tokens = sum(metric.output_tokens for metric in metrics)
    total_spec_verify_ct = sum(metric.spec_verify_ct_sum for metric in metrics)
    total_accept_length_requests = sum(
        metric.spec_accept_length_request_count for metric in metrics
    )
    total_accept_length_per_request = sum(
        metric.spec_accept_length_mean_per_request
        * metric.spec_accept_length_request_count
        for metric in metrics
        if metric.spec_accept_length_mean_per_request is not None
    )
    return BenchMetrics(
        sample_count=first.sample_count,
        generation_turn_count=first.generation_turn_count,
        latency_s=float(statistics.mean(metric.latency_s for metric in metrics)),
        output_tokens=int(
            round(statistics.mean(metric.output_tokens for metric in metrics))
        ),
        output_toks_per_s=float(
            statistics.mean(metric.output_toks_per_s for metric in metrics)
        ),
        decode_output_tokens=int(
            round(statistics.mean(metric.decode_output_tokens for metric in metrics))
        ),
        decode_active_s=float(
            statistics.mean(metric.decode_active_s for metric in metrics)
        ),
        decode_output_toks_per_s=_mean_optional(
            [metric.decode_output_toks_per_s for metric in metrics]
        ),
        peak_request_decode_output_toks_per_s=max(
            (
                metric.peak_request_decode_output_toks_per_s
                for metric in metrics
                if metric.peak_request_decode_output_toks_per_s is not None
            ),
            default=None,
        ),
        spec_accept_length=(
            float(total_output_tokens) / float(total_spec_verify_ct)
            if total_spec_verify_ct > 0
            else None
        ),
        spec_accept_length_mean_per_request=(
            float(total_accept_length_per_request) / float(total_accept_length_requests)
            if total_accept_length_requests > 0
            else None
        ),
        spec_accept_length_request_count=int(
            round(
                statistics.mean(
                    metric.spec_accept_length_request_count for metric in metrics
                )
            )
        ),
        spec_verify_ct_sum=int(
            round(statistics.mean(metric.spec_verify_ct_sum for metric in metrics))
        ),
    )


def _aggregate_job_results(
    job_results: list[JobResult | JobFailure],
) -> list[ConfigResult]:
    grouped_results: dict[RunKey, list[JobResult | JobFailure]] = {}
    ordered_keys: list[RunKey] = []
    for result in job_results:
        if result.key not in grouped_results:
            grouped_results[result.key] = []
            ordered_keys.append(result.key)
        grouped_results[result.key].append(result)

    config_results: list[ConfigResult] = []
    for key in ordered_keys:
        results = sorted(grouped_results[key], key=lambda result: result.run_index)
        successful_results = [
            result for result in results if isinstance(result, JobResult)
        ]
        failures = tuple(result for result in results if isinstance(result, JobFailure))
        first = results[0]
        first_success = successful_results[0] if successful_results else None
        repeat_metrics = tuple(result.metrics for result in successful_results)
        successful_run_indices = tuple(
            result.run_index for result in successful_results
        )
        metrics = (
            _aggregate_bench_metrics(list(repeat_metrics)) if repeat_metrics else None
        )
        config_results.append(
            ConfigResult(
                key=key,
                deployment=first.deployment,
                source_sample_count=(
                    None if first_success is None else first_success.source_sample_count
                ),
                source_generation_turn_count=(
                    None
                    if first_success is None
                    else first_success.source_generation_turn_count
                ),
                warmup_generation_turn_count=(
                    None
                    if first_success is None
                    else first_success.warmup_generation_turn_count
                ),
                metrics=metrics,
                repeat_metrics=repeat_metrics,
                successful_run_indices=successful_run_indices,
                failures=failures,
            )
        )
    return config_results


def _build_csv_rows(
    *,
    config_results: list[ConfigResult],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    results_by_key = {result.key: result for result in config_results}

    for result in config_results:
        key = result.key
        metrics = result.metrics
        baseline_result = results_by_key.get(
            RunKey(
                workload=key.workload,
                backend=key.backend,
                tp=key.tp,
                concurrency=key.concurrency,
                mode="baseline",
            )
        )
        speedup = None
        decode_speedup = None
        if (
            metrics is not None
            and key.mode != "baseline"
            and baseline_result is not None
            and baseline_result.metrics is not None
            and baseline_result.metrics.output_toks_per_s > 0
        ):
            speedup = (
                metrics.output_toks_per_s / baseline_result.metrics.output_toks_per_s
            )
        if (
            metrics is not None
            and metrics.decode_output_toks_per_s is not None
            and key.mode != "baseline"
            and baseline_result is not None
            and baseline_result.metrics is not None
            and baseline_result.metrics.decode_output_toks_per_s is not None
            and baseline_result.metrics.decode_output_toks_per_s > 0
        ):
            decode_speedup = (
                metrics.decode_output_toks_per_s
                / baseline_result.metrics.decode_output_toks_per_s
            )

        accept_source_result = results_by_key.get(
            RunKey(
                workload=key.workload,
                backend=key.backend,
                tp=key.tp,
                concurrency=1,
                mode=key.mode,
            )
        )
        accept_length_aggregate_from_conc1 = (
            None
            if accept_source_result is None or accept_source_result.metrics is None
            else accept_source_result.metrics.spec_accept_length
        )
        accept_length_mean_from_conc1 = (
            None
            if accept_source_result is None or accept_source_result.metrics is None
            else accept_source_result.metrics.spec_accept_length_mean_per_request
        )

        rows.append(
            {
                "workload": key.workload,
                "backend": key.backend,
                "linear_attn_prefill_backend": result.deployment.shared_config.linear_attn_prefill_backend,
                "linear_attn_decode_backend": result.deployment.shared_config.linear_attn_decode_backend,
                "linear_attn_verify_backend": result.deployment.shared_config.linear_attn_verify_backend,
                "tp": key.tp,
                "mode": key.mode,
                "dflash_block_size": result.deployment.dflash_block_size or "",
                "concurrency": key.concurrency,
                "source_sample_count": _fmt_optional_int(result.source_sample_count),
                "source_generation_turn_count": _fmt_optional_int(
                    result.source_generation_turn_count
                ),
                "runs_per_config": result.run_count,
                "successful_runs": result.successful_run_count,
                "failed_runs": result.failed_run_count,
                "status": result.status,
                "successful_run_numbers": _format_successful_run_numbers(result),
                "failed_run_numbers": _format_failed_run_numbers(result.failures),
                "failure_messages": _format_failure_messages(result.failures),
                "measured_sample_count": (
                    "" if metrics is None else metrics.sample_count
                ),
                "measured_generation_turn_count": (
                    "" if metrics is None else metrics.generation_turn_count
                ),
                "output_toks_per_s": _fmt_csv_value(
                    None if metrics is None else metrics.output_toks_per_s
                ),
                "output_toks_per_s_std": _fmt_csv_value(
                    _metric_stdev(result.repeat_metrics, "output_toks_per_s")
                ),
                "decode_output_toks_per_s": _fmt_csv_value(
                    None if metrics is None else metrics.decode_output_toks_per_s
                ),
                "decode_output_toks_per_s_std": _fmt_csv_value(
                    _metric_stdev(
                        result.repeat_metrics,
                        "decode_output_toks_per_s",
                    )
                ),
                "peak_request_decode_output_toks_per_s": _fmt_csv_value(
                    None
                    if metrics is None
                    else metrics.peak_request_decode_output_toks_per_s
                ),
                "latency_s": _fmt_csv_value(
                    None if metrics is None else metrics.latency_s
                ),
                "latency_s_std": _fmt_csv_value(
                    _metric_stdev(result.repeat_metrics, "latency_s")
                ),
                "decode_active_s": _fmt_csv_value(
                    None if metrics is None else metrics.decode_active_s
                ),
                "decode_active_s_std": _fmt_csv_value(
                    _metric_stdev(result.repeat_metrics, "decode_active_s")
                ),
                "output_tokens": "" if metrics is None else metrics.output_tokens,
                "decode_output_tokens": (
                    "" if metrics is None else metrics.decode_output_tokens
                ),
                "speedup_vs_baseline": _fmt_csv_value(speedup),
                "decode_speedup_vs_baseline": _fmt_csv_value(decode_speedup),
                "accept_length_aggregate_from_conc1": _fmt_csv_value(
                    accept_length_aggregate_from_conc1
                ),
                "accept_length_mean_from_conc1": _fmt_csv_value(
                    accept_length_mean_from_conc1
                ),
                "accept_length_aggregate_this_conc": _fmt_csv_value(
                    None if metrics is None else metrics.spec_accept_length
                ),
                "accept_length_aggregate_this_conc_std": _fmt_csv_value(
                    _metric_stdev(result.repeat_metrics, "spec_accept_length")
                ),
                "accept_length_mean_this_conc": _fmt_csv_value(
                    None
                    if metrics is None
                    else metrics.spec_accept_length_mean_per_request
                ),
                "accept_length_mean_this_conc_std": _fmt_csv_value(
                    _metric_stdev(
                        result.repeat_metrics,
                        "spec_accept_length_mean_per_request",
                    )
                ),
                "accept_length_request_count": (
                    "" if metrics is None else metrics.spec_accept_length_request_count
                ),
                "spec_verify_ct_sum": (
                    "" if metrics is None else metrics.spec_verify_ct_sum
                ),
            }
        )
    return rows


def _print_csv_summary(rows: list[dict[str, object]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    print("\n=== CSV Summary ===")
    print(buffer.getvalue(), end="", flush=True)


def _write_csv_summary(path: str, rows: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] wrote {len(rows)} rows to {out_path}", flush=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Kimi-K3 target-only decoding against DFlash on one "
            "TP8 B300 node."
        )
    )
    parser.add_argument(
        "--workloads",
        default=DEFAULT_WORKLOADS,
        help="Comma-separated workloads to run, or `all`.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional path to write the final CSV summary.",
    )
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument(
        "--load-format",
        default="auto",
        help=(
            "SGLang model load format, for example `auto`, `safetensors`, or "
            "`fastsafetensors` (default: auto)."
        ),
    )
    parser.add_argument(
        "--linear-attn-prefill-backend",
        type=str.lower,
        choices=LINEAR_ATTN_PREFILL_BACKEND_CHOICES,
        default=DEFAULT_LINEAR_ATTN_PREFILL_BACKEND,
        help=(
            "Target linear-attention prefill backend "
            f"(benchmark default: {DEFAULT_LINEAR_ATTN_PREFILL_BACKEND})."
        ),
    )
    parser.add_argument(
        "--linear-attn-decode-backend",
        type=str.lower,
        choices=LINEAR_ATTN_DECODE_BACKEND_CHOICES,
        default=DEFAULT_LINEAR_ATTN_DECODE_BACKEND,
        help=(
            "Target linear-attention fallback decode backend "
            f"(benchmark default: {DEFAULT_LINEAR_ATTN_DECODE_BACKEND})."
        ),
    )
    parser.add_argument(
        "--linear-attn-verify-backend",
        type=str.lower,
        choices=LINEAR_ATTN_VERIFY_BACKEND_CHOICES,
        default=DEFAULT_LINEAR_ATTN_VERIFY_BACKEND,
        help=(
            "Target linear-attention speculative-verify backend "
            f"(benchmark default: {DEFAULT_LINEAR_ATTN_VERIFY_BACKEND})."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="SGLang server random seed (default: 42).",
    )
    parser.add_argument(
        "--target-model-revision",
        default=DEFAULT_TARGET_MODEL_REVISION,
        help=(
            "Optional target-model revision. Pass an empty string to use the "
            "model repository default; ignored for a local model path."
        ),
    )
    parser.add_argument(
        "--dflash-draft-model",
        default=DEFAULT_DFLASH_DRAFT_MODEL,
        help="DFlash draft checkpoint path.",
    )
    parser.add_argument(
        "--speculative-draft-attention-backend",
        default=DEFAULT_DFLASH_DRAFT_ATTENTION_BACKEND,
        help=(
            "Attention backend for the DFlash draft worker "
            f"(default: {DEFAULT_DFLASH_DRAFT_ATTENTION_BACKEND})."
        ),
    )
    parser.add_argument(
        "--trtllm-gen-moe-cubin-pool",
        default=DEFAULT_TRTLLM_GEN_MOE_CUBIN_POOL,
        help=(
            "TRTLLM-Gen cubin-pool overlay used by the flashinfer_mxfp4 "
            "target MoE backend. Pass an empty string to inherit no override."
        ),
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help=(
            "Skip the target-only baseline; run only DFlash and report N/A "
            "for baseline speedup."
        ),
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=True,
        help="Pass thinking=True to the Kimi-K3 chat template (default).",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Pass thinking=False to the Kimi-K3 chat template.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--concurrencies", default="1,32")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help=(
            "Exact number of measured samples per config. Repeats the selected "
            "workload if this exceeds the workload size. Default: unset."
        ),
    )
    parser.add_argument(
        "--runs-per-config",
        type=int,
        default=1,
        help=(
            "Number of repeated measured runs per benchmark configuration. "
            "The final reported metrics are averaged across these runs."
        ),
    )
    parser.add_argument(
        "--min-generation-turns-per-config",
        type=int,
        default=1024,
        help=(
            "When --num-samples is unset and concurrency > 1, repeat whole workload "
            "copies until each config measures at least this many generation turns. "
            "Use 0 for one full workload copy."
        ),
    )
    parser.add_argument(
        "--min-warmup-generation-turns",
        type=int,
        default=8,
        help=(
            "Minimum generation turns to run after /flush_cache before measured "
            "timing. Effective warmup is max(this value, 2 * concurrency)."
        ),
    )
    parser.add_argument(
        "--disable-replayssm",
        action="store_true",
        help=(
            "Disable the KDA ReplaySSM spec-verify ring (on by default: the "
            "DFlash servers get --enable-gdn-replayssm-spec with a "
            "block-size-derived --linear-replayssm-cache-len, 32 for blocks "
            "up to 16)."
        ),
    )
    parser.add_argument(
        "--dflash-block-sizes",
        default=DEFAULT_DFLASH_BLOCK_SIZES,
        help=(
            "Comma-separated DFlash block-size sweep. Use `default` to omit "
            "--speculative-dflash-block-size and let the server choose."
        ),
    )
    args = parser.parse_args(argv)
    args.load_format = str(args.load_format).strip().lower()
    if not args.load_format:
        parser.error("--load-format must not be empty")
    try:
        workloads = _parse_workload_selection(args.workloads)
    except ValueError as exc:
        parser.error(str(exc))

    if not args.dflash_draft_model:
        parser.error("--dflash-draft-model is required")
    args.speculative_draft_attention_backend = (
        str(args.speculative_draft_attention_backend).strip().lower()
    )
    if not args.speculative_draft_attention_backend:
        parser.error("--speculative-draft-attention-backend must not be empty")
    try:
        dflash_block_sizes = _parse_optional_int_csv(str(args.dflash_block_sizes))
    except ValueError as exc:
        parser.error(f"--dflash-block-sizes must be integers/default: {exc}")
    if any(x is not None and x <= 0 for x in dflash_block_sizes):
        parser.error(
            "--dflash-block-sizes values must be > 0, " f"got {args.dflash_block_sizes}"
        )
    mode_keys = DeploymentSweep(
        include_baseline=not args.skip_baseline,
        dflash_draft_model=args.dflash_draft_model,
        dflash_block_sizes=tuple(dflash_block_sizes),
        dflash_draft_attention_backend=args.speculative_draft_attention_backend,
        dflash_enable_replayssm=not args.disable_replayssm,
    ).mode_keys
    duplicate_mode_keys = _duplicate_values(mode_keys)
    if duplicate_mode_keys:
        parser.error(
            "Duplicate deployment modes from sweep flags: "
            + ",".join(duplicate_mode_keys)
        )
    args.workloads = workloads
    args.dflash_block_sizes = dflash_block_sizes
    return args


def build_sweep_config_from_args(args: argparse.Namespace) -> SweepConfig:
    sampling = SamplingConfig(
        enable_thinking=bool(args.enable_thinking),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
    )
    methodology = BenchmarkMethodologyConfig(
        num_samples=args.num_samples,
        min_generation_turns_per_config=int(args.min_generation_turns_per_config),
        min_warmup_generation_turns=int(args.min_warmup_generation_turns),
        runs_per_config=int(args.runs_per_config),
    )
    deployment_sweep = DeploymentSweep(
        include_baseline=not args.skip_baseline,
        dflash_draft_model=args.dflash_draft_model,
        dflash_block_sizes=tuple(args.dflash_block_sizes),
        dflash_draft_attention_backend=args.speculative_draft_attention_backend,
        dflash_enable_replayssm=not args.disable_replayssm,
    )

    if sampling.temperature < 0.0:
        raise RuntimeError(f"--temperature must be >= 0, got {sampling.temperature}.")
    if sampling.max_new_tokens <= 0:
        raise RuntimeError(
            f"--max-new-tokens must be > 0, got {sampling.max_new_tokens}."
        )
    if not (0.0 < sampling.top_p <= 1.0):
        raise RuntimeError(f"--top-p must be in (0, 1], got {sampling.top_p}.")
    if sampling.top_k == 0 or sampling.top_k < -1:
        raise RuntimeError(
            f"--top-k must be -1 (all vocab) or >= 1, got {sampling.top_k}."
        )
    if methodology.num_samples is not None and methodology.num_samples <= 0:
        raise RuntimeError(f"--num-samples must be > 0, got {methodology.num_samples}.")
    if methodology.runs_per_config <= 0:
        raise RuntimeError(
            f"--runs-per-config must be > 0, got {methodology.runs_per_config}."
        )
    if (
        methodology.min_generation_turns_per_config < 0
        or methodology.min_warmup_generation_turns < 0
    ):
        raise RuntimeError(
            "--min-generation-turns-per-config and "
            "--min-warmup-generation-turns must be >= 0."
        )

    try:
        concurrencies = _parse_int_csv(args.concurrencies)
    except ValueError as exc:
        raise RuntimeError("--concurrencies must be comma-separated integers.") from exc
    if not concurrencies:
        raise RuntimeError("No concurrencies specified.")
    if any(c < 1 for c in concurrencies):
        raise RuntimeError(f"--concurrencies values must be >= 1, got {concurrencies}.")
    duplicate_concurrencies = _duplicate_values([str(c) for c in concurrencies])
    if duplicate_concurrencies:
        raise RuntimeError(
            "Duplicate concurrencies: " + ",".join(duplicate_concurrencies)
        )

    return SweepConfig(
        target_model=args.target_model,
        target_model_revision=(args.target_model_revision or None),
        trtllm_gen_moe_cubin_pool=(args.trtllm_gen_moe_cubin_pool or None),
        load_format=args.load_format,
        linear_attn_prefill_backend=args.linear_attn_prefill_backend,
        linear_attn_decode_backend=args.linear_attn_decode_backend,
        linear_attn_verify_backend=args.linear_attn_verify_backend,
        random_seed=int(args.random_seed),
        workloads=tuple(args.workloads),
        concurrencies=tuple(concurrencies),
        sampling=sampling,
        methodology=methodology,
        deployment_sweep=deployment_sweep,
        csv_output=args.csv_output,
    )


def _get_current_cuda_runtime() -> tuple[int, int]:
    import torch

    from sglang.srt.utils import get_device_sm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep.")
    return int(torch.cuda.device_count()), int(get_device_sm())


def build_shared_configs_for_runtime(
    sweep_config: SweepConfig,
) -> tuple[list[SharedServerConfig], int]:
    visible_gpus, device_sm = _get_current_cuda_runtime()
    shared_configs = _build_shared_server_configs(
        device_sm=device_sm,
        visible_gpus=visible_gpus,
        max_concurrency=max(sweep_config.concurrencies),
        load_format=sweep_config.load_format,
        random_seed=sweep_config.random_seed,
        linear_attn_prefill_backend=sweep_config.linear_attn_prefill_backend,
        linear_attn_decode_backend=sweep_config.linear_attn_decode_backend,
        linear_attn_verify_backend=sweep_config.linear_attn_verify_backend,
    )
    return shared_configs, device_sm


def build_shared_configs_for_modal(
    sweep_config: SweepConfig,
    *,
    device_sm: int,
    visible_gpus: int,
) -> list[SharedServerConfig]:
    return _build_shared_server_configs(
        device_sm=int(device_sm),
        visible_gpus=int(visible_gpus),
        max_concurrency=max(sweep_config.concurrencies),
        load_format=sweep_config.load_format,
        random_seed=sweep_config.random_seed,
        linear_attn_prefill_backend=sweep_config.linear_attn_prefill_backend,
        linear_attn_decode_backend=sweep_config.linear_attn_decode_backend,
        linear_attn_verify_backend=sweep_config.linear_attn_verify_backend,
    )


def build_benchmark_jobs(
    sweep_config: SweepConfig, shared_configs: list[SharedServerConfig]
) -> list[BenchmarkJob]:
    return _build_benchmark_jobs(sweep_config, shared_configs)


def run_benchmark_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    job = benchmark_job_from_payload(payload)
    return job_outcome_to_payload(_run_benchmark_job_gracefully(job))


def aggregate_job_outcomes(
    outcomes: list[JobResult | JobFailure],
) -> list[ConfigResult]:
    return _aggregate_job_results(outcomes)


def render_results(
    *,
    sweep_config: SweepConfig,
    shared_configs: list[SharedServerConfig],
    device_sm: int,
    config_results: list[ConfigResult],
) -> list[dict[str, object]]:
    mode_keys = sweep_config.deployment_sweep.mode_keys
    for workload in sweep_config.workloads:
        workload_results = [
            result for result in config_results if result.key.workload == workload
        ]
        if not workload_results:
            continue
        workload_shared_configs: list[SharedServerConfig] = []
        seen_shared_configs: set[SharedServerConfig] = set()
        for result in workload_results:
            shared_config = result.deployment.shared_config
            if shared_config not in seen_shared_configs:
                seen_shared_configs.add(shared_config)
                workload_shared_configs.append(shared_config)
        attention_backends = sorted(
            {config.attention_backend for config in workload_shared_configs}
        )
        tp_sizes = sorted({config.tp_size for config in workload_shared_configs})
        source_sample_count = next(
            (
                result.source_sample_count
                for result in workload_results
                if result.source_sample_count is not None
            ),
            None,
        )
        source_generation_turn_count = next(
            (
                result.source_generation_turn_count
                for result in workload_results
                if result.source_generation_turn_count is not None
            ),
            None,
        )
        print(f"\n\n##### Workload Summary: {workload} #####")
        _print_summary(
            config=sweep_config,
            workload=workload,
            config_results=workload_results,
            shared_configs=workload_shared_configs,
            attention_backends=attention_backends,
            tp_sizes=tp_sizes,
            concurrencies=list(sweep_config.concurrencies),
            device_sm=device_sm,
            mode_keys=mode_keys,
            source_sample_count=source_sample_count,
            source_generation_turn_count=source_generation_turn_count,
            results=_metric_map_from_config_results(workload_results),
        )

    csv_rows = _build_csv_rows(config_results=config_results)
    _print_csv_summary(csv_rows)
    if sweep_config.csv_output is not None:
        _write_csv_summary(sweep_config.csv_output, csv_rows)
    return csv_rows


def run_local_sweep(sweep_config: SweepConfig) -> list[ConfigResult]:
    shared_configs, device_sm = build_shared_configs_for_runtime(sweep_config)
    jobs = build_benchmark_jobs(sweep_config, shared_configs)
    job_results = [_run_benchmark_job_gracefully(job) for job in jobs]
    config_results = aggregate_job_outcomes(job_results)
    render_results(
        sweep_config=sweep_config,
        shared_configs=shared_configs,
        device_sm=device_sm,
        config_results=config_results,
    )
    return config_results


def main() -> None:
    args = parse_args()
    sweep_config = build_sweep_config_from_args(args)
    run_local_sweep(sweep_config)


if __name__ == "__main__":
    main()
