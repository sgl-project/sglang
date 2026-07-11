# /// script
# requires-python = ">=3.10"
# dependencies = ["msgspec", "requests", "plotly", "kaleido", "numpy"]
# ///
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import random
import statistics
import threading
import time
from pathlib import Path
from typing import Optional

import msgspec
import numpy as np
import requests

logger = logging.getLogger(__name__)


def _load_module_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    from sglang.benchmark.one_batch_server import (
        DEFAULT_TIMEOUT,
        should_skip_due_to_max_running_requests,
        should_skip_due_to_token_capacity,
    )
    from sglang.benchmark.utils import get_tokenizer
    from sglang.srt.speculative.dspark_components.dspark_sps import (
        SpsAdditiveCostTable,
        load_sps_table_from_path,
        profile_sps_table,
    )
except ImportError as exc:
    logger.warning(
        "Full sglang runtime unavailable (%s); using a torch-free fallback import. "
        "The 'fit' subcommand works; 'run' requires the full sglang install.",
        exc,
    )
    _benchmark_dir = Path(__file__).resolve().parent
    _table_module = _load_module_by_path(
        "dspark_sps",
        _benchmark_dir.parent / "srt/speculative/dspark_components/dspark_sps.py",
    )
    SpsAdditiveCostTable = _table_module.SpsAdditiveCostTable
    load_sps_table_from_path = _table_module.load_sps_table_from_path
    profile_sps_table = _table_module.profile_sps_table
    DEFAULT_TIMEOUT = 60
    get_tokenizer = None
    should_skip_due_to_max_running_requests = None
    should_skip_due_to_token_capacity = None

DEFAULT_OUT = "dspark_sps.json"
DEFAULT_MAX_BATCH_SIZE = 256
DEFAULT_INPUT_LEN = 16
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MIN_STEADY_STEPS = 32
DEFAULT_MIN_STEADY_SECONDS = 10.0
DEFAULT_ROUND_TIMEOUT_SECONDS = 300.0
ROUND_WARMUP_STEPS = 8
ROUND_STEP_SLACK = 64
STEP_TIME_FLOOR_SECONDS = 0.02
WARMUP_ROUND_STEADY_STEPS = 16
POLL_INTERVAL_SECONDS = 2.0
LOAD_JOIN_TIMEOUT_SECONDS = 60.0
MATCH_FRACTION_WARN = 0.9
MATCH_FRACTION_ERROR = 0.5
PROFILE_SEED = 42
REQUIRED_SIMULATE_ACC_LEN = 1.0
RANDOM_TOKEN_LOW = 1000
RANDOM_TOKEN_HIGH_MARGIN = 1000

STATIC_CONDITIONING_CAVEAT = (
    "Profiled with SGLANG_RAGGED_VERIFY_MODE=static: a verify step of B tokens "
    "comes from B/(gamma+1) requests, the fewest possible for that B. A "
    "compact-mode step with the same B usually spans more requests and reads "
    "more KV history, so the table slightly over-estimates steps_per_sec and "
    "the scheduler may admit slightly more than optimal. Much smaller bias "
    "than the retired non-spec decode proxy, and in the opposite direction."
)
CONVERSION_FORMULA = (
    "batch_tokens = num_running_reqs_per_rank * verify_num_draft_tokens; "
    "steps_per_sec = 1 / median(server-side step_time over aligned steady steps)"
)


class SpsRow(msgspec.Struct, frozen=True):
    forward_ct: int
    num_running_reqs: int
    num_verify_tokens: int
    step_time: float


class RecordSource(msgspec.Struct, frozen=True):
    name: str
    payload_key: str
    enable_hint: str
    step_time_ms: bool


SPS_RECORD_SOURCE = RecordSource(
    name="sps",
    payload_key="dspark_sps_record",
    enable_hint="SGLANG_DSPARK_ENABLE_SPS_RECORD=1 (and SGLANG_RAGGED_VERIFY_MODE=static)",
    step_time_ms=False,
)
INFO_RECORD_SOURCE = RecordSource(
    name="info",
    payload_key="dspark_info_record",
    enable_hint=(
        "SGLANG_DSPARK_DEBUG_DUMP=core,step_cpu_time "
        "(and SGLANG_RAGGED_VERIFY_MODE=static)"
    ),
    step_time_ms=True,
)
RECORD_SOURCES = {
    SPS_RECORD_SOURCE.name: SPS_RECORD_SOURCE,
    INFO_RECORD_SOURCE.name: INFO_RECORD_SOURCE,
}


class ServerContext(msgspec.Struct, frozen=True):
    base_url: str
    tokenizer_path: str
    tp_size: int
    dp_size: int
    verify_num_draft_tokens: int
    simulate_acc_len: float
    cuda_graph_max_bs: Optional[int]
    skip_max_running_requests_threshold: float
    skip_token_capacity_threshold: float
    record_source: RecordSource


class RoundSettings(msgspec.Struct, frozen=True):
    input_len: int
    temperature: float
    min_steady_steps: int
    min_steady_seconds: float
    round_timeout_seconds: float
    ramp_token_slack: int = 0


class LoadInfo(msgspec.Struct, frozen=True):
    num_requests: int
    max_new_tokens: int
    wall_seconds: float
    reached_target: bool


class RoundOutcome(msgspec.Struct, frozen=True):
    batch_size: int
    batch_size_per_rank: int
    batch_tokens: int
    steps_per_sec: float
    num_steady_steps: int
    match_fraction: float
    per_rank_median_step_time: list[float]
    rank_rows: list[list[SpsRow]]
    load_info: LoadInfo
    frac: Optional[float] = None


def out_paths(*, out: str) -> dict[str, Path]:
    out_path = Path(out).expanduser()
    return {
        "table": out_path,
        "records": out_path.with_name(out_path.stem + ".records.jsonl"),
        "rounds": out_path.with_name(out_path.stem + ".rounds.jsonl"),
        "manifest": out_path.with_name(out_path.name + ".manifest.json"),
        "plot": out_path.with_name(out_path.stem + ".plot.png"),
    }


def run_profile(
    *,
    base_url: str,
    batch_sizes: list[int],
    settings: RoundSettings,
    out: str,
    repeats: int,
    local_tokenizer_path: Optional[str],
    recorder_source: str,
    fracs: Optional[list[float]],
) -> None:
    if get_tokenizer is None:
        raise RuntimeError(
            "'run' needs the full sglang runtime (torch, tokenizers, ...), but this "
            "process loaded the torch-free fallback. Run 'run' where sglang is "
            "installed; 'fit' works in either environment."
        )
    if not base_url:
        raise ValueError(
            "dspark_sps_profiler connects to an already-running DSpark server "
            "(SGLANG_RAGGED_VERIFY_MODE=static, SGLANG_DSPARK_ENABLE_SPS_RECORD=1); "
            "pass --base-url <url> (it never launches a server)."
        )

    offdiag = fracs is not None
    if offdiag:
        for frac in fracs:
            if not 0.0 < frac <= 1.0:
                raise ValueError(
                    f"--fracs values must be in (0, 1], got {frac}. The off-diagonal "
                    "budget pin runs frac * full verify, so frac <= 1.0 keeps M below "
                    "the full uniform tier and inside the captured cuda graphs."
                )

    paths = out_paths(out=out)
    paths["table"].parent.mkdir(parents=True, exist_ok=True)
    for path in (paths["records"], paths["rounds"]):
        if path.exists():
            path.unlink()

    context = fetch_server_context(
        base_url=base_url,
        local_tokenizer_path=local_tokenizer_path,
        record_source=RECORD_SOURCES[recorder_source],
        allowed_modes=("compact", "cap-accept") if offdiag else ("static",),
    )
    vocab_size = len(get_tokenizer(context.tokenizer_path))
    batch_sizes = sorted(set(batch_sizes))
    validate_sweep_against_server(context=context, batch_sizes=batch_sizes)
    rng = random.Random(PROFILE_SEED)

    frac_sweep: list[Optional[float]] = sorted(fracs) if offdiag else [None]

    run_warmup_round(
        context=context,
        vocab_size=vocab_size,
        batch_sizes=batch_sizes,
        settings=settings,
        rng=rng,
        frac=frac_sweep[-1],
    )

    rounds: list[RoundOutcome] = []
    for repeat in range(max(1, repeats)):
        for batch_size_per_rank in batch_sizes:
            for frac in frac_sweep:
                outcome = run_one_round(
                    context=context,
                    vocab_size=vocab_size,
                    batch_size_per_rank=batch_size_per_rank,
                    settings=settings,
                    rng=rng,
                    frac=frac,
                )
                if outcome is None:
                    continue
                logger.info(
                    "Round bs=%s (per-rank %s, frac=%s, batch_tokens=%s) "
                    "repeat=%s/%s: steps_per_sec=%.3f over %s steady steps "
                    "(match_fraction=%.2f, wall=%.1fs, per-rank median "
                    "step_time=%s)",
                    outcome.batch_size,
                    outcome.batch_size_per_rank,
                    outcome.frac,
                    outcome.batch_tokens,
                    repeat + 1,
                    max(1, repeats),
                    outcome.steps_per_sec,
                    outcome.num_steady_steps,
                    outcome.match_fraction,
                    outcome.load_info.wall_seconds,
                    ["%.4f" % value for value in outcome.per_rank_median_step_time],
                )
                append_round_files(
                    records_path=paths["records"],
                    rounds_path=paths["rounds"],
                    outcome=outcome,
                    repeat=repeat,
                )
                rounds.append(outcome)

    if not rounds:
        raise RuntimeError(
            "No usable rounds (all were skipped by capacity guards or failed); "
            "check the batch-size sweep against the server's "
            "max_running_requests / KV capacity."
        )

    write_manifest(
        manifest_path=paths["manifest"],
        records_path=paths["records"],
        rounds_path=paths["rounds"],
        context=context,
        batch_sizes=batch_sizes,
        settings=settings,
        repeats=repeats,
        rounds=rounds,
        fracs=fracs,
    )
    logger.info(
        "Collected %s rounds; wrote %s, %s and %s",
        len(rounds),
        paths["rounds"].name,
        paths["records"].name,
        paths["manifest"].name,
    )


def fit_profile(
    *,
    out: str,
    max_batch_tokens: Optional[int],
    self_check: bool,
    plot: bool,
) -> None:
    paths = out_paths(out=out)
    if not paths["rounds"].exists():
        raise FileNotFoundError(
            f"No rounds file at {paths['rounds']}; run the 'run' subcommand first "
            "(or point --out at a prior run's table path)."
        )

    summaries = load_round_summaries(rounds_path=paths["rounds"])
    if not summaries:
        raise RuntimeError(f"{paths['rounds']} has no rounds to fit.")
    offdiag = any(summary.get("frac") is not None for summary in summaries)

    table = build_table_from_summaries(
        summaries=summaries, max_batch_tokens=max_batch_tokens, offdiag=offdiag
    )
    paths["table"].write_text(table.to_json(), encoding="utf-8")
    if offdiag:
        logger.info(
            "Fit SpsAdditiveCostTable (%s bs probes x %s M probes) -> %s",
            len(table.bs_probes),
            len(table.m_probes),
            paths["table"],
        )
    else:
        logger.info(
            "Fit SpsCostTable (%s probes) -> %s",
            len(table.sample_batch_tokens),
            paths["table"],
        )

    if plot:
        plot_fit(
            cells=summaries_to_cells(summaries=summaries),
            table=table,
            plot_path=paths["plot"],
        )

    if self_check:
        run_self_check(out_path=paths["table"], offdiag=offdiag)


def profile_all(
    *,
    base_url: str,
    batch_sizes: list[int],
    settings: RoundSettings,
    out: str,
    max_batch_tokens: Optional[int],
    repeats: int,
    self_check: bool,
    local_tokenizer_path: Optional[str],
    recorder_source: str,
    fracs: Optional[list[float]],
    plot: bool,
) -> None:
    run_profile(
        base_url=base_url,
        batch_sizes=batch_sizes,
        settings=settings,
        out=out,
        repeats=repeats,
        local_tokenizer_path=local_tokenizer_path,
        recorder_source=recorder_source,
        fracs=fracs,
    )
    fit_profile(
        out=out,
        max_batch_tokens=max_batch_tokens,
        self_check=self_check,
        plot=plot,
    )


def load_round_summaries(*, rounds_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in rounds_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summaries_to_cells(*, summaries: list[dict]) -> list[dict]:
    return [
        {
            "bs": summary["batch_size_per_rank"],
            "M": summary["batch_tokens"],
            "T": 1.0 / summary["steps_per_sec"],
            "frac": summary.get("frac"),
        }
        for summary in summaries
    ]


def build_table_from_summaries(
    *, summaries: list[dict], max_batch_tokens: Optional[int], offdiag: bool
):
    if offdiag:
        return build_additive_table_from_cells(
            cells=summaries_to_cells(summaries=summaries)
        )

    by_batch_tokens: dict[int, list[float]] = {}
    for summary in summaries:
        by_batch_tokens.setdefault(summary["batch_tokens"], []).append(
            summary["steps_per_sec"]
        )
    probes = [
        (batch_tokens, statistics.median(values))
        for batch_tokens, values in sorted(by_batch_tokens.items())
    ]
    return profile_sps_table(probes=probes, max_batch_tokens=max_batch_tokens)


def fetch_server_context(
    *,
    base_url: str,
    local_tokenizer_path: Optional[str],
    record_source: RecordSource,
    allowed_modes: tuple[str, ...] = ("static",),
) -> ServerContext:
    response = requests.get(base_url + "/server_info", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    info = response.json()

    speculative_algorithm = info.get("speculative_algorithm")
    if speculative_algorithm != "DSPARK":
        raise ValueError(
            f"Profile against a DSpark server: {base_url} reports "
            f"speculative_algorithm={speculative_algorithm!r}. The SPS table is "
            "measured from real static-mode DSpark verify steps; relaunch with "
            "--speculative-algorithm DSPARK and SGLANG_RAGGED_VERIFY_MODE=static."
        )
    if info.get("disable_cuda_graph"):
        raise ValueError(
            "The server runs with --disable-cuda-graph; an SPS table measured "
            "without cuda graphs is uselessly slow. Relaunch with cuda graphs "
            "enabled."
        )

    internal_states = info.get("internal_states") or []
    if not internal_states:
        raise RuntimeError(f"{base_url}/server_info returned no internal_states.")
    sps_payloads = [state.get(record_source.payload_key) for state in internal_states]
    for rank_index, payload in enumerate(sps_payloads):
        if payload is None:
            raise ValueError(
                f"DP rank {rank_index} reports no {record_source.payload_key}; "
                f"launch the server with {record_source.enable_hint}."
            )
        if payload.get("mode") not in allowed_modes:
            raise ValueError(
                f"{record_source.payload_key}.mode must be one of {allowed_modes}, "
                f"got {payload.get('mode')!r} on DP rank {rank_index}."
            )
        if record_source is INFO_RECORD_SOURCE:
            components = payload.get("components") or []
            missing = {"core", "step_cpu_time"} - set(components)
            if missing:
                raise ValueError(
                    f"DP rank {rank_index} {record_source.payload_key} is missing "
                    f"component(s) {sorted(missing)}; launch with "
                    f"{record_source.enable_hint}."
                )
        if payload.get("simulate_acc_len") != REQUIRED_SIMULATE_ACC_LEN:
            raise ValueError(
                f"DP rank {rank_index} reports simulate_acc_len="
                f"{payload.get('simulate_acc_len')!r}, but SPS profiling "
                f"requires exactly SGLANG_SIMULATE_ACC_LEN="
                f"{REQUIRED_SIMULATE_ACC_LEN} (spec fully ineffective: every "
                "step advances every request by exactly the bonus token, so "
                "the per-step KV conditioning is deterministic instead of "
                "drifting with the model's accept behavior)."
            )
    verify_num_draft_tokens = {
        int(payload["verify_num_draft_tokens"]) for payload in sps_payloads
    }
    simulate_acc_lens = {float(payload["simulate_acc_len"]) for payload in sps_payloads}
    if len(simulate_acc_lens) != 1:
        raise RuntimeError(
            f"DP ranks disagree on simulate_acc_len: {sorted(simulate_acc_lens)}."
        )
    if len(verify_num_draft_tokens) != 1:
        raise RuntimeError(
            "DP ranks disagree on verify_num_draft_tokens: "
            f"{sorted(verify_num_draft_tokens)}."
        )

    tokenizer_path = local_tokenizer_path or info.get("tokenizer_path")
    if not tokenizer_path:
        raise RuntimeError(
            "Could not resolve a tokenizer path from /server_info; pass "
            "--local-tokenizer-path explicitly."
        )

    internal_state = internal_states[0]
    dp_size = int(internal_state.get("dp_size") or 1)
    cuda_graph_max_bs = resolve_cuda_graph_max_bs(internal_state=internal_state)
    max_running_per_dp = internal_state.get("effective_max_running_requests_per_dp", -1)
    if max_running_per_dp and max_running_per_dp > 0:
        skip_max_running = float(max_running_per_dp * dp_size)
    else:
        logger.warning(
            "Server did not report effective_max_running_requests_per_dp (%s); "
            "not clamping the batch-size sweep against the running cap.",
            max_running_per_dp,
        )
        skip_max_running = float("inf")

    skip_token_capacity = 0.0
    for state in internal_states:
        skip_token_capacity += state.get("memory_usage", {}).get(
            "token_capacity", 1_000_000_000
        )

    return ServerContext(
        base_url=base_url,
        tokenizer_path=tokenizer_path,
        tp_size=int(info.get("tp_size", 1) or 1),
        dp_size=dp_size,
        verify_num_draft_tokens=verify_num_draft_tokens.pop(),
        simulate_acc_len=simulate_acc_lens.pop(),
        cuda_graph_max_bs=cuda_graph_max_bs,
        skip_max_running_requests_threshold=skip_max_running,
        skip_token_capacity_threshold=skip_token_capacity,
        record_source=record_source,
    )


def resolve_cuda_graph_max_bs(*, internal_state: dict) -> Optional[int]:
    cuda_graph_config = internal_state.get("cuda_graph_config")
    if not isinstance(cuda_graph_config, dict):
        return None
    decode_config = cuda_graph_config.get("decode")
    if not isinstance(decode_config, dict):
        return None
    captured_bs = decode_config.get("bs")
    if isinstance(captured_bs, list) and captured_bs:
        return int(max(captured_bs))
    max_bs = decode_config.get("max_bs")
    if max_bs is not None:
        return int(max_bs)
    return None


def validate_sweep_against_server(
    *, context: ServerContext, batch_sizes: list[int]
) -> None:
    if context.cuda_graph_max_bs is None:
        raise ValueError(
            "Could not resolve the server's captured cuda-graph max batch size "
            "from /server_info, so the sweep cannot be confirmed to stay inside "
            "the captured decode graphs. Steps that fall back to eager silently "
            "poison the table with a different perf regime. Relaunch the server "
            "so it reports cuda_graph_config in /server_info, or -- if you really "
            "want to profile anyway -- delete this raise, but be careful."
        )
    max_per_rank = max(batch_sizes)
    if max_per_rank > context.cuda_graph_max_bs:
        raise ValueError(
            f"The sweep reaches {max_per_rank} running requests per DP rank but "
            "the server captured decode cuda graphs only up to bs="
            f"{context.cuda_graph_max_bs}; steps beyond it run eager and poison "
            "the table. Relaunch the server with a larger --cuda-graph-max-bs-decode "
            "or shrink --max-batch-size. If you really want to profile anyway, "
            "delete this raise, but be careful."
        )


def build_request_count_sweep(max_num_reqs: int) -> list[int]:
    if max_num_reqs < 1:
        raise ValueError(f"max_num_reqs must be >= 1, got {max_num_reqs}.")
    raw = [
        1,
        2,
        4,
        8,
        *range(16, 64, 8),
        *range(64, 128, 16),
        *range(128, 256, 32),
        *range(256, max_num_reqs + 1, 64),
    ]
    sweep = sorted({value for value in raw if 1 <= value <= max_num_reqs})
    if sweep[-1] != max_num_reqs:
        sweep.append(max_num_reqs)
    return sweep


def round_max_new_tokens(*, settings: RoundSettings, context: ServerContext) -> int:
    commit_tokens_per_step = (
        max(
            0,
            min(
                round(context.simulate_acc_len - 1), context.verify_num_draft_tokens - 1
            ),
        )
        + 1
    )
    steady_steps_budget = max(
        settings.min_steady_steps,
        math.ceil(settings.min_steady_seconds / STEP_TIME_FLOOR_SECONDS),
    )
    total_steps = ROUND_WARMUP_STEPS + steady_steps_budget + ROUND_STEP_SLACK
    return total_steps * commit_tokens_per_step + settings.ramp_token_slack


def run_warmup_round(
    *,
    context: ServerContext,
    vocab_size: int,
    batch_sizes: list[int],
    settings: RoundSettings,
    rng: random.Random,
    frac: Optional[float] = None,
) -> None:
    warmup_settings = RoundSettings(
        input_len=settings.input_len,
        temperature=settings.temperature,
        min_steady_steps=WARMUP_ROUND_STEADY_STEPS,
        min_steady_seconds=0.0,
        round_timeout_seconds=settings.round_timeout_seconds,
        ramp_token_slack=settings.ramp_token_slack,
    )
    try:
        run_one_round(
            context=context,
            vocab_size=vocab_size,
            batch_size_per_rank=min(8, max(batch_sizes)),
            settings=warmup_settings,
            rng=rng,
            frac=frac,
        )
    except Exception:
        logger.warning("Warmup round failed; continuing.", exc_info=True)


def run_one_round(
    *,
    context: ServerContext,
    vocab_size: int,
    batch_size_per_rank: int,
    settings: RoundSettings,
    rng: random.Random,
    frac: Optional[float] = None,
) -> Optional[RoundOutcome]:
    batch_size = batch_size_per_rank * context.dp_size
    max_new_tokens = round_max_new_tokens(settings=settings, context=context)
    if should_skip_due_to_max_running_requests(
        batch_size, context.skip_max_running_requests_threshold
    ) or should_skip_due_to_token_capacity(
        batch_size,
        settings.input_len,
        max_new_tokens,
        context.skip_token_capacity_threshold,
    ):
        return None

    if frac is not None:
        set_forced_budget_frac(base_url=context.base_url, frac=frac)

    flush_cache(base_url=context.base_url)
    watermarks = [
        max((row.forward_ct for row in rows), default=-1)
        for rows in fetch_rank_rows(
            base_url=context.base_url, record_source=context.record_source
        )
    ]

    start_time = time.monotonic()
    load_thread = start_load(
        base_url=context.base_url,
        num_requests=batch_size,
        input_len=settings.input_len,
        max_new_tokens=max_new_tokens,
        temperature=settings.temperature,
        vocab_size=vocab_size,
        rng=rng,
    )
    reached_target = wait_for_aligned_steps(
        context=context,
        watermarks=watermarks,
        batch_size_per_rank=batch_size_per_rank,
        min_steady_steps=settings.min_steady_steps,
        min_steady_seconds=settings.min_steady_seconds,
        timeout_seconds=settings.round_timeout_seconds,
    )
    abort_all_requests(base_url=context.base_url)
    load_thread.join(timeout=LOAD_JOIN_TIMEOUT_SECONDS)
    if load_thread.is_alive():
        logger.warning(
            "Load batch for bs=%s did not return within %.0fs after abort; "
            "continuing with the collected records.",
            batch_size,
            LOAD_JOIN_TIMEOUT_SECONDS,
        )
    wall_seconds = time.monotonic() - start_time
    if not reached_target:
        logger.warning(
            "Round bs=%s hit the %.0fs timeout before both gates (>=%s steady "
            "steps and >=%.1fs) were met; proceeding with what was collected.",
            batch_size,
            settings.round_timeout_seconds,
            settings.min_steady_steps,
            settings.min_steady_seconds,
        )

    rank_rows = fetch_rank_rows(
        base_url=context.base_url, record_source=context.record_source
    )
    if len(rank_rows) != len(watermarks):
        raise RuntimeError(
            f"DP rank count changed mid-profile: {len(watermarks)} -> "
            f"{len(rank_rows)}."
        )
    new_rank_rows = [
        [row for row in rows if row.forward_ct > watermark]
        for rows, watermark in zip(rank_rows, watermarks)
    ]
    return postprocess_round(
        rank_rows=new_rank_rows,
        batch_size_per_rank=batch_size_per_rank,
        dp_size=context.dp_size,
        verify_num_draft_tokens=context.verify_num_draft_tokens,
        min_steady_steps=settings.min_steady_steps,
        load_info=LoadInfo(
            num_requests=batch_size,
            max_new_tokens=max_new_tokens,
            wall_seconds=round(wall_seconds, 3),
            reached_target=reached_target,
        ),
        frac=frac,
    )


def start_load(
    *,
    base_url: str,
    num_requests: int,
    input_len: int,
    max_new_tokens: int,
    temperature: float,
    vocab_size: int,
    rng: random.Random,
) -> threading.Thread:
    token_high = vocab_size - RANDOM_TOKEN_HIGH_MARGIN
    if token_high <= RANDOM_TOKEN_LOW:
        raise ValueError(f"vocab_size={vocab_size} too small for random prompts.")
    input_ids = [
        [rng.randrange(RANDOM_TOKEN_LOW, token_high) for _ in range(input_len)]
        for _ in range(num_requests)
    ]
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "stream": False,
    }

    def _post() -> None:
        try:
            requests.post(base_url + "/generate", json=payload, timeout=DEFAULT_TIMEOUT)
        except Exception:
            logger.warning(
                "Load batch POST /generate failed (expected on abort for some "
                "server versions).",
                exc_info=True,
            )

    thread = threading.Thread(target=_post, daemon=True)
    thread.start()
    return thread


def wait_for_aligned_steps(
    *,
    context: ServerContext,
    watermarks: list[int],
    batch_size_per_rank: int,
    min_steady_steps: int,
    min_steady_seconds: float,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    steady_start: Optional[float] = None
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_SECONDS)
        try:
            rank_rows = fetch_rank_rows(
                base_url=context.base_url, record_source=context.record_source
            )
        except Exception:
            logger.warning("Polling /server_info failed; retrying.", exc_info=True)
            continue
        new_rank_rows = [
            [row for row in rows if row.forward_ct > watermark]
            for rows, watermark in zip(rank_rows, watermarks)
        ]
        if len(new_rank_rows) != len(watermarks):
            continue
        aligned = count_aligned_steps(
            rank_rows=new_rank_rows, batch_size_per_rank=batch_size_per_rank
        )
        if aligned >= ROUND_WARMUP_STEPS and steady_start is None:
            steady_start = time.monotonic()
        steady_steps = max(0, aligned - ROUND_WARMUP_STEPS)
        steady_seconds = (
            time.monotonic() - steady_start if steady_start is not None else 0.0
        )
        logger.debug(
            "Aligned-step poll: %d aligned (%d/%d steady steps, %.1f/%.1fs)",
            aligned,
            steady_steps,
            min_steady_steps,
            steady_seconds,
            min_steady_seconds,
        )
        if steady_steps >= min_steady_steps and steady_seconds >= min_steady_seconds:
            return True
    return False


def count_aligned_steps(
    *, rank_rows: list[list[SpsRow]], batch_size_per_rank: int
) -> int:
    if any(not rows for rows in rank_rows):
        return 0
    by_ct_per_rank = [{row.forward_ct: row for row in rows} for rows in rank_rows]
    common_cts = set(by_ct_per_rank[0])
    for by_ct in by_ct_per_rank[1:]:
        common_cts &= set(by_ct)
    return sum(
        1
        for ct in common_cts
        if all(
            by_ct[ct].num_running_reqs == batch_size_per_rank
            for by_ct in by_ct_per_rank
        )
    )


def abort_all_requests(*, base_url: str) -> None:
    response = requests.post(
        base_url + "/abort_request",
        json={"abort_all": True},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()


def flush_cache(*, base_url: str) -> None:
    try:
        requests.post(base_url + "/flush_cache", timeout=DEFAULT_TIMEOUT)
    except Exception:
        logger.warning("POST /flush_cache failed; continuing.", exc_info=True)


def set_forced_budget_frac(*, base_url: str, frac: Optional[float]) -> None:
    response = requests.post(
        base_url + "/set_internal_state",
        json={"server_args": {"dspark_force_budget_frac": frac}},
        timeout=DEFAULT_TIMEOUT,
    ).json()
    outs = response if isinstance(response, list) else [response]

    def _ok(out) -> bool:
        return bool(out.get("updated")) if isinstance(out, dict) else bool(out)

    if not outs or not all(_ok(out) for out in outs):
        raise RuntimeError(
            f"set dspark_force_budget_frac={frac} rejected by server: {response}"
        )


def fetch_rank_rows(
    *, base_url: str, record_source: RecordSource
) -> list[list[SpsRow]]:
    response = requests.get(base_url + "/server_info", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    internal_states = response.json().get("internal_states") or []
    rank_rows: list[list[SpsRow]] = []
    for state in internal_states:
        payload = state.get(record_source.payload_key) or {}
        rows: list[SpsRow] = []
        for record in payload.get("records", []):
            step_time = _row_step_time(record=record, record_source=record_source)
            if step_time is None:
                continue
            rows.append(
                SpsRow(
                    forward_ct=int(record["forward_ct"]),
                    num_running_reqs=int(record["num_running_reqs"]),
                    num_verify_tokens=int(record["num_verify_tokens"]),
                    step_time=step_time,
                )
            )
        rank_rows.append(rows)
    return rank_rows


def _row_step_time(*, record: dict, record_source: RecordSource) -> Optional[float]:
    if not record_source.step_time_ms:
        return float(record["step_time"])
    value = record.get("step_cpu_ms")
    return None if value is None else float(value) / 1000.0


def postprocess_round(
    *,
    rank_rows: list[list[SpsRow]],
    batch_size_per_rank: int,
    dp_size: int,
    verify_num_draft_tokens: int,
    min_steady_steps: int,
    load_info: LoadInfo,
    frac: Optional[float] = None,
) -> RoundOutcome:
    offdiag = frac is not None
    batch_size = batch_size_per_rank * dp_size
    expected_tokens = batch_size_per_rank * verify_num_draft_tokens

    if len(rank_rows) != dp_size:
        raise RuntimeError(
            f"Expected records from {dp_size} DP ranks, got {len(rank_rows)}."
        )

    by_ct_per_rank: list[dict[int, SpsRow]] = []
    for rank_index, rows in enumerate(rank_rows):
        if not rows:
            raise RuntimeError(
                f"DP rank {rank_index} produced no new decode-step records this "
                "round; the load generator did not reach it (DP imbalance or "
                "the round was too short)."
            )
        by_ct_per_rank.append({row.forward_ct: row for row in rows})

    common_cts = set(by_ct_per_rank[0])
    for by_ct in by_ct_per_rank[1:]:
        common_cts &= set(by_ct)
    if not common_cts:
        raise RuntimeError(
            "DP ranks share no common forward_ct in this round; their step "
            "counters are misaligned, so per-step cross-rank checks are "
            "impossible. This breaks the uniformity assumption of the table."
        )

    aligned_cts: list[int] = []
    aligned_verify_tokens: set[int] = set()
    for ct in sorted(common_cts):
        rows_at_ct = [by_ct[ct] for by_ct in by_ct_per_rank]
        if all(row.num_running_reqs == batch_size_per_rank for row in rows_at_ct):
            for rank_index, row in enumerate(rows_at_ct):
                if not offdiag and row.num_verify_tokens < expected_tokens:
                    raise RuntimeError(
                        f"DP rank {rank_index} at forward_ct={ct} reports "
                        f"num_verify_tokens={row.num_verify_tokens}, expected at "
                        f"least {expected_tokens} (= {batch_size_per_rank} reqs x "
                        f"{verify_num_draft_tokens}); ranks are not running the "
                        "uniform static verify the table assumes. The recorded "
                        "count is the replayed graph tier, which may exceed the "
                        "candidate count when a bs is not an exact capture tier."
                    )
                aligned_verify_tokens.add(row.num_verify_tokens)
            aligned_cts.append(ct)

    if len(aligned_cts) < ROUND_WARMUP_STEPS + min_steady_steps:
        raise RuntimeError(
            f"Round bs={batch_size} never stabilized: only {len(aligned_cts)} "
            f"of {len(common_cts)} common decode steps had every rank at the "
            f"target {batch_size_per_rank} requests (need at least "
            f"{ROUND_WARMUP_STEPS + min_steady_steps}). Increase "
            "--round-timeout / --target-steady-steps, or inspect the raw "
            "records for retraction / DP imbalance."
        )

    window_cts = [
        ct for ct in sorted(common_cts) if aligned_cts[0] <= ct <= aligned_cts[-1]
    ]
    match_fraction = len(aligned_cts) / len(window_cts)
    if match_fraction < MATCH_FRACTION_ERROR:
        raise RuntimeError(
            f"Round bs={batch_size} is unstable mid-round: only "
            f"{match_fraction:.0%} of the {len(window_cts)} decode steps inside "
            "the steady window ran at the target per-rank batch (retraction or "
            "DP imbalance, not just ramp-in/drain). Inspect the raw records."
        )
    if match_fraction < MATCH_FRACTION_WARN:
        logger.warning(
            "Round bs=%s: %.0f%% of %s steady-window decode steps ran at the "
            "target per-rank batch; treat this probe with suspicion.",
            batch_size,
            match_fraction * 100.0,
            len(window_cts),
        )

    steady_cts = aligned_cts[ROUND_WARMUP_STEPS:]
    per_ct_step_times = [
        statistics.fmean(by_ct[ct].step_time for by_ct in by_ct_per_rank)
        for ct in steady_cts
    ]
    per_rank_median_step_time = [
        statistics.median(by_ct[ct].step_time for ct in steady_cts)
        for by_ct in by_ct_per_rank
    ]
    median_step_time = statistics.median(per_ct_step_times)

    if offdiag:
        if len(aligned_verify_tokens) != 1:
            raise RuntimeError(
                f"Round bs={batch_size} frac={frac}: aligned steps ran at "
                f"differing num_verify_tokens {sorted(aligned_verify_tokens)}; the "
                "budget pin did not hold a single graph tier across all "
                "ranks/steps, so the measurement is ambiguous. Inspect the raw "
                "records."
            )
        graph_tier = aligned_verify_tokens.pop()
        budget = int(frac * batch_size_per_rank * (verify_num_draft_tokens - 1))
        batch_tokens = batch_size_per_rank + budget
        if graph_tier < batch_tokens:
            raise RuntimeError(
                f"Round bs={batch_size} frac={frac}: replayed graph tier "
                f"{graph_tier} is smaller than the pinned M={batch_tokens} "
                f"(= {batch_size_per_rank} + int({frac} * {batch_size_per_rank} "
                f"* {verify_num_draft_tokens - 1})); the budget pin did not take."
            )
    else:
        batch_tokens = expected_tokens

    return RoundOutcome(
        batch_size=batch_size,
        batch_size_per_rank=batch_size_per_rank,
        batch_tokens=batch_tokens,
        steps_per_sec=1.0 / median_step_time,
        num_steady_steps=len(steady_cts),
        match_fraction=match_fraction,
        per_rank_median_step_time=per_rank_median_step_time,
        rank_rows=rank_rows,
        load_info=load_info,
        frac=frac,
    )


def fitted_step_time(*, table, bs: int, m: int) -> float:
    if isinstance(table, SpsAdditiveCostTable):
        return table.step_time(num_reqs=bs, budget=max(0, m - bs))
    return 1.0 / table.lookup(m)


def bs_color_map(*, batch_sizes: list[int]) -> dict[int, str]:
    span = max(len(batch_sizes) - 1, 1)
    colors = {}
    for index, bs in enumerate(batch_sizes):
        hue = 240.0 * (1.0 - index / span)
        colors[bs] = f"hsl({hue:.0f}, 70%, 50%)"
    return colors


def plot_fit(*, cells: list[dict], table, plot_path: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning(
            "plotly not installed; skipping the fit plot. Install plotly + "
            "kaleido to render %s.",
            plot_path.name,
        )
        return

    batch_sizes = sorted({cell["bs"] for cell in cells})
    color_of = bs_color_map(batch_sizes=batch_sizes)
    fig = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.07,
        subplot_titles=(
            "step time",
            "throughput = M / T",
            "raw (circle) vs fit (square)",
        ),
    )
    for bs in batch_sizes:
        points = sorted(
            (cell for cell in cells if cell["bs"] == bs), key=lambda c: c["M"]
        )
        m_values = [cell["M"] for cell in points]
        t_ms = [cell["T"] * 1e3 for cell in points]
        color = color_of[bs]
        fig.add_trace(
            go.Scatter(
                x=m_values,
                y=t_ms,
                mode="markers+lines",
                name=f"bs={bs}",
                legendgroup=f"bs={bs}",
                marker=dict(color=color, size=7),
                line=dict(color=color, width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=m_values,
                y=[cell["M"] / cell["T"] for cell in points],
                mode="markers+lines",
                legendgroup=f"bs={bs}",
                showlegend=False,
                marker=dict(color=color, size=7),
                line=dict(color=color, width=1),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=m_values,
                y=t_ms,
                mode="markers",
                legendgroup=f"bs={bs}",
                showlegend=False,
                marker=dict(color=color, size=8, symbol="circle"),
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=m_values,
                y=[
                    fitted_step_time(table=table, bs=bs, m=cell["M"]) * 1e3
                    for cell in points
                ],
                mode="markers",
                legendgroup=f"bs={bs}",
                showlegend=False,
                marker=dict(
                    color=color, size=9, symbol="square-open", line=dict(width=2)
                ),
            ),
            row=1,
            col=3,
        )
    fig.update_xaxes(
        title_text="M = num total verify tokens", rangemode="tozero", row=1, col=1
    )
    fig.update_xaxes(
        title_text="M = num total verify tokens", rangemode="tozero", row=1, col=2
    )
    fig.update_xaxes(
        title_text="M = num total verify tokens", rangemode="tozero", row=1, col=3
    )
    fig.update_yaxes(title_text="T = step time (ms)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(
        title_text="throughput (tokens/s)", rangemode="tozero", row=1, col=2
    )
    fig.update_yaxes(title_text="T = step time (ms)", rangemode="tozero", row=1, col=3)
    fig.update_layout(
        title="DSpark SPS profiler: raw cells vs additive fit",
        legend_title="batch size",
        template="plotly_white",
        width=1700,
        height=640,
    )
    try:
        fig.write_image(str(plot_path), scale=2)
    except Exception:
        logger.warning(
            "Failed to render %s (kaleido missing?); skipping plot.",
            plot_path.name,
            exc_info=True,
        )
        return
    logger.info("Wrote fit plot to %s", plot_path)


def build_additive_table_from_cells(*, cells: list[dict]) -> SpsAdditiveCostTable:
    if len(cells) < 4:
        raise RuntimeError(
            f"Off-diagonal fit needs at least 4 cells, got {len(cells)}."
        )
    bias, alpha, theta, _rel, _stats = ols_resid_backfit(cells)
    bs_probes = sorted(alpha)
    m_probes = sorted(theta)
    return SpsAdditiveCostTable(
        bias_seconds=bias,
        bs_probes=bs_probes,
        alpha_seconds=[alpha[b] for b in bs_probes],
        m_probes=m_probes,
        theta_seconds=[theta[m] for m in m_probes],
    )


def ols_resid_backfit(cells: list, mbin_w: int = 64):
    def mbin(m):
        return round(m / mbin_w) * mbin_w

    bslist = sorted({c["bs"] for c in cells})
    mbins = sorted({mbin(c["M"]) for c in cells})
    bs_col = {b: i for i, b in enumerate(bslist[1:])}
    m_col = {m: i for i, m in enumerate(mbins[1:])}
    num_cols = 1 + len(bs_col) + len(m_col)

    design = np.zeros((len(cells), num_cols))
    target = np.array([c["T"] for c in cells], dtype=float)
    for row, c in enumerate(cells):
        design[row, 0] = 1.0
        if c["bs"] in bs_col:
            design[row, 1 + bs_col[c["bs"]]] = 1.0
        if mbin(c["M"]) in m_col:
            design[row, 1 + len(bs_col) + m_col[mbin(c["M"])]] = 1.0

    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    bias = float(beta[0])
    alpha = {bslist[0]: 0.0}
    for b in bslist[1:]:
        alpha[b] = float(beta[1 + bs_col[b]])
    theta = {mbins[0]: 0.0}
    for m in mbins[1:]:
        theta[m] = float(beta[1 + len(bs_col) + m_col[m]])

    resid = [c["T"] - (bias + alpha[c["bs"]] + theta[mbin(c["M"])]) for c in cells]
    rel = [abs(r) / c["T"] * 100 for r, c in zip(resid, cells)]
    rms = (sum(r * r for r in resid) / len(resid)) ** 0.5
    tbar = statistics.fmean(c["T"] for c in cells)
    ss_tot = sum((c["T"] - tbar) ** 2 for c in cells)
    r2 = 1.0 - sum(r * r for r in resid) / ss_tot if ss_tot > 0 else float("nan")

    def probe_se(pred):
        se = {}
        for key in sorted({pred(c) for c in cells}):
            rs = [r for r, c in zip(resid, cells) if pred(c) == key]
            se[key] = (statistics.pstdev(rs) / (len(rs) ** 0.5)) if len(rs) > 1 else 0.0
        return se

    stats = {
        "rms_ms": rms * 1e3,
        "r2": r2,
        "n": len(cells),
        "alpha_se": probe_se(lambda c: c["bs"]),
        "theta_se": probe_se(lambda c: mbin(c["M"])),
    }
    return bias, alpha, theta, rel, stats


def round_summary_dict(*, outcome: RoundOutcome, repeat: int) -> dict:
    return {
        "repeat": repeat,
        "batch_size": outcome.batch_size,
        "batch_size_per_rank": outcome.batch_size_per_rank,
        "frac": outcome.frac,
        "batch_tokens": outcome.batch_tokens,
        "steps_per_sec": outcome.steps_per_sec,
        "num_steady_steps": outcome.num_steady_steps,
        "match_fraction": outcome.match_fraction,
        "per_rank_median_step_time": outcome.per_rank_median_step_time,
        "load_info": msgspec.to_builtins(outcome.load_info),
    }


def append_round_files(
    *,
    records_path: Path,
    rounds_path: Path,
    outcome: RoundOutcome,
    repeat: int,
) -> None:
    with records_path.open("a", encoding="utf-8") as fout:
        for rank_index, rows in enumerate(outcome.rank_rows):
            for row in rows:
                fout.write(
                    json.dumps(
                        {
                            "repeat": repeat,
                            "batch_size": outcome.batch_size,
                            "batch_size_per_rank": outcome.batch_size_per_rank,
                            "dp_rank": rank_index,
                            "forward_ct": row.forward_ct,
                            "num_running_reqs": row.num_running_reqs,
                            "num_verify_tokens": row.num_verify_tokens,
                            "step_time": row.step_time,
                        }
                    )
                    + "\n"
                )
    with rounds_path.open("a", encoding="utf-8") as fout:
        fout.write(
            json.dumps(round_summary_dict(outcome=outcome, repeat=repeat)) + "\n"
        )


def write_manifest(
    *,
    manifest_path: Path,
    records_path: Path,
    rounds_path: Path,
    context: ServerContext,
    batch_sizes: list[int],
    settings: RoundSettings,
    repeats: int,
    rounds: list[RoundOutcome],
    fracs: Optional[list[float]],
) -> None:
    manifest = {
        "base_url": context.base_url,
        "tp_size": context.tp_size,
        "dp_size": context.dp_size,
        "verify_num_draft_tokens": context.verify_num_draft_tokens,
        "simulate_acc_len": context.simulate_acc_len,
        "batch_size_per_rank_sweep": batch_sizes,
        "fracs": fracs,
        "settings": msgspec.to_builtins(settings),
        "repeats": repeats,
        "seed": PROFILE_SEED,
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "conversion_formula": CONVERSION_FORMULA,
        "static_conditioning_caveat": STATIC_CONDITIONING_CAVEAT,
        "records_jsonl": records_path.name,
        "rounds_jsonl": rounds_path.name,
        "round_summaries": [
            round_summary_dict(outcome=outcome, repeat=0) for outcome in rounds
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_self_check(*, out_path: Path, offdiag: bool) -> None:
    table = load_sps_table_from_path(str(out_path))
    if offdiag:
        run_additive_self_check(table=table)
        return
    if len(table.sample_batch_tokens) != len(table.sample_steps_per_sec):
        raise RuntimeError("Reloaded table has mismatched probe / SPS lengths.")

    previous_sps: Optional[float] = None
    for batch_tokens in table.sample_batch_tokens:
        looked_up = table.lookup(batch_tokens)
        if looked_up <= 0:
            raise RuntimeError(
                f"Reloaded table lookup at batch_tokens={batch_tokens} returned "
                f"non-positive SPS {looked_up}."
            )
        if previous_sps is not None and looked_up > previous_sps * 1.10:
            logger.warning(
                "Non-monotone SPS across probes: batch_tokens=%s SPS=%.3f rose "
                "above the previous probe's SPS=%.3f by >10%%; verify the server "
                "was at steady state (no co-tenants, steady clocks).",
                batch_tokens,
                looked_up,
                previous_sps,
            )
        previous_sps = looked_up

    below_floor = table.lookup(table.sample_batch_tokens[0] - 1)
    if below_floor != table.sample_steps_per_sec[0]:
        raise RuntimeError(
            "Reloaded table lookup below the smallest probe did not clamp to the "
            f"first SPS ({below_floor} != {table.sample_steps_per_sec[0]})."
        )
    logger.info(
        "Self-check passed: reloaded %s probes, all lookups positive and "
        "below-floor clamp holds.",
        len(table.sample_batch_tokens),
    )


def run_additive_self_check(*, table: SpsAdditiveCostTable) -> None:
    for num_reqs in table.bs_probes:
        for budget in (0, max(table.m_probes) - table.bs_probes[0]):
            value = table.step_time(num_reqs=num_reqs, budget=max(0, budget))
            if not value > 0:
                raise RuntimeError(
                    f"Reloaded additive table step_time(num_reqs={num_reqs}, "
                    f"budget={budget}) is non-positive ({value})."
                )
    logger.info(
        "Self-check passed: reloaded additive table (%s bs probes x %s M "
        "probes), all step_time lookups positive.",
        len(table.bs_probes),
        len(table.m_probes),
    )


def add_out_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        help="Output JSON path for the SPS table. Raw per-step records land next "
        "to it as <stem>.records.jsonl, one line per (bs, frac) cell as "
        "<stem>.rounds.jsonl, <out>.manifest.json ties everything together, and "
        "the fit plot as <stem>.plot.png.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Python logging level for the profiler.",
    )


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Base URL of the already-running DSpark server, e.g. "
        "http://localhost:30000. The profiler never launches a server.",
    )
    parser.add_argument(
        "--fracs",
        type=float,
        nargs="+",
        default=None,
        help="Off-diagonal K-fraction sweep in (0, 1]. When given, the server "
        "must run SGLANG_RAGGED_VERIFY_MODE=compact and each (bs, frac) cell "
        "pins dspark_force_budget_frac to profile T(bs, M); the fit is a 2D "
        "SpsAdditiveCostTable. When omitted, the diagonal static sweep runs and "
        "the fit is a 1D SpsCostTable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=None,
        help="Explicit PER-DP-RANK running-request counts to sweep; the load "
        "generator sends value * dp_size requests so every rank (GPU group) "
        "sits at the given batch. Overrides --max-batch-size when given.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
        help="Upper bound of the auto-generated tapered PER-DP-RANK "
        "request-count sweep (used only when --batch-size is not given), so "
        "per-rank token coverage is identical for any dp_size.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=DEFAULT_INPUT_LEN,
        help="Prompt length per request. Short: the table is conditioned on the "
        "decode-heavy regime.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for the load requests; default 1.0 to hit "
        "the same accept/sampling kernels as real serving.",
    )
    parser.add_argument(
        "--min-steady-steps",
        type=int,
        default=DEFAULT_MIN_STEADY_STEPS,
        help="Stop a round only after at least this many aligned steady steps "
        "(and --min-steady-seconds) have been collected, then abort the load "
        "batch. Bounds cheap small-batch rounds; the batch is held at exactly "
        "the target running-request count the whole time (no drain tail).",
    )
    parser.add_argument(
        "--min-steady-seconds",
        type=float,
        default=DEFAULT_MIN_STEADY_SECONDS,
        help="Stop a round only after at least this much steady-state wall time "
        "(and --min-steady-steps) has elapsed. Bounds expensive large-batch "
        "rounds, where a fixed step count would run many slow steps.",
    )
    parser.add_argument(
        "--round-timeout",
        type=float,
        default=DEFAULT_ROUND_TIMEOUT_SECONDS,
        help="Per-round wall-clock budget in seconds to collect the target "
        "steps before giving up and using what was collected.",
    )
    parser.add_argument(
        "--ramp-token-slack",
        type=int,
        default=0,
        help="Extra per-request tokens on top of the step budget so requests "
        "outlive the whole-batch prefill ramp. Required for long --input-len "
        "at high batch sizes, where the ramp exceeds the request lifetime and "
        "full-batch alignment becomes unreachable; size it as roughly "
        "ramp_seconds / step_time.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Times to repeat the whole sweep; per batch_tokens the median "
        "steps_per_sec across repeats is taken.",
    )
    parser.add_argument(
        "--local-tokenizer-path",
        type=str,
        default=None,
        help="Override the tokenizer path (defaults to the one reported by "
        "/server_info).",
    )
    parser.add_argument(
        "--recorder-source",
        type=str,
        choices=sorted(RECORD_SOURCES),
        default=SPS_RECORD_SOURCE.name,
        help="Which server-side per-step record feed to read: 'sps' (legacy "
        "SpsDataRecorder via SGLANG_DSPARK_ENABLE_SPS_RECORD) or 'info' (the "
        "DsparkInfoDumper 'core'+'step_cpu_time' components via "
        "SGLANG_DSPARK_DEBUG_DUMP). Both yield the same steps_per_sec table.",
    )


def add_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=None,
        help="Override the diagonal table's max_batch_tokens metadata (defaults "
        "to the largest probed batch_tokens). Ignored for off-diagonal fits.",
    )
    parser.add_argument(
        "--no-self-check",
        dest="self_check",
        action="store_false",
        help="Skip the read-back + lookup self-check of the written table.",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip the <stem>.plot.png fit plot (needs plotly + kaleido).",
    )


def run_settings(*, args: argparse.Namespace) -> RoundSettings:
    return RoundSettings(
        input_len=args.input_len,
        temperature=args.temperature,
        min_steady_steps=args.min_steady_steps,
        min_steady_seconds=args.min_steady_seconds,
        round_timeout_seconds=args.round_timeout,
        ramp_token_slack=args.ramp_token_slack,
    )


def run_batch_sizes(*, args: argparse.Namespace) -> list[int]:
    if args.batch_size is not None:
        return args.batch_size
    return build_request_count_sweep(args.max_batch_size)


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile a DSpark SPS cost table from an already-running DSpark "
            "server. Subcommands: 'run' collects raw per-cell data, 'fit' builds "
            "the table (and plot) from that data, 'all' does both."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Collect raw sweep data (append one line per cell to jsonl)."
    )
    add_out_arg(run_parser)
    add_run_args(run_parser)

    fit_parser = subparsers.add_parser(
        "fit", help="Fit the table and render the plot from a prior run's jsonl."
    )
    add_out_arg(fit_parser)
    add_fit_args(fit_parser)

    all_parser = subparsers.add_parser("all", help="Run then fit in one shot.")
    add_out_arg(all_parser)
    add_run_args(all_parser)
    add_fit_args(all_parser)

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(message)s",
    )

    if args.command == "run":
        run_profile(
            base_url=args.base_url,
            batch_sizes=run_batch_sizes(args=args),
            settings=run_settings(args=args),
            out=args.out,
            repeats=args.repeats,
            local_tokenizer_path=args.local_tokenizer_path,
            recorder_source=args.recorder_source,
            fracs=args.fracs,
        )
    elif args.command == "fit":
        fit_profile(
            out=args.out,
            max_batch_tokens=args.max_batch_tokens,
            self_check=args.self_check,
            plot=args.plot,
        )
    else:
        profile_all(
            base_url=args.base_url,
            batch_sizes=run_batch_sizes(args=args),
            settings=run_settings(args=args),
            out=args.out,
            max_batch_tokens=args.max_batch_tokens,
            repeats=args.repeats,
            self_check=args.self_check,
            local_tokenizer_path=args.local_tokenizer_path,
            recorder_source=args.recorder_source,
            fracs=args.fracs,
            plot=args.plot,
        )


if __name__ == "__main__":
    cli_main()
