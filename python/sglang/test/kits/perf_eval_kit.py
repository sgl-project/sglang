"""Data-driven performance harness for the PPU benchmark suites.

This is the serving-throughput sibling of :mod:`answer_eval_kit`.  It carries no
grading logic of its own: a measurement produces the numbers
``sglang.bench_serving`` already computes -- time to first token and token
throughput above all -- and the report records them.  Nothing here compares a
number against a threshold, because no threshold has been measured on this
hardware yet; what turns a run red is an inability to measure, not a slow
result.  See :func:`measurement_record` for the exact boundary.

The module is deliberately free of ``torch`` and of anything that touches a
device, so the unit tests can validate every config and render every report on a
laptop.  The half that has to import ``sglang.test.test_utils`` -- and through it
``torch`` -- lives in :mod:`perf_suite_kit`.

The configs this validates are ported from the internal btv1.5 corpus under
``model-test-cases/testcases/btv1.5/llm_infer_sglang/144G/Daily/Prefill``.  Where
a name differs between that corpus and SGLang, the SGLang spelling is what a
config states, and the mapping is recorded in the suite's README: btv1.5 writes
``tp`` and ``ep`` for what SGLang calls ``tp_size`` and ``ep_size``, and states
``mem_fraction_static`` as a string.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Borrowed rather than duplicated.  These four are schema-agnostic -- they take
# explicit arguments and know nothing about Answer's report shape -- so sharing
# them keeps one implementation of the provenance block both suites are compared
# through.  Nothing schema-shaped is imported: the validator, the report and the
# renderers below are this suite's own, because an Answer report and a
# performance report have no fields in common beyond provenance.
from sglang.test.kits.answer_eval_kit import (
    canonical_digest,
    checkpoint_config_digest,
    default_provenance,
    load_json,
)

PERF_CONFIG_SCHEMA_VERSION = "ppu-perf-test-config/v1"
PERF_REPORT_SCHEMA_VERSION = "ppu-perf-report/v1"
# The row shape the trend series is built out of.  Versioned separately from the
# report because the two are read by different things: a report is read once, by
# whoever opens that run's artifact, while a row is read months later alongside
# rows written by every version since, and a reader of that mixture needs to tell
# a row that predates a field from a row that lost one.
TREND_POINT_SCHEMA_VERSION = "ppu-perf-trend-point/v1"


class PerfEvalError(RuntimeError):
    """Base error for a malformed performance config or an unmeasurable run."""


class MeasurementError(PerfEvalError):
    """A single measurement could not be taken."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


# The reviewed server parameters, grouped by how each one is typed and how it
# reaches `sglang serve`.  They are module level because the validator and the
# argument builder have to agree on the set: a name checked here but never
# rendered would be silent decoration, and a name rendered without being checked
# would be the untyped pass-through this schema exists to refuse.
#
# The set is wider than Answer's because these configs describe a serving
# benchmark rather than a correctness probe, and the internal corpus tunes the
# scheduler for it: page size, expert parallelism, the a2a backend, the stream
# interval and the radix cache all appear there and all change the number being
# measured.
REQUIRED_SERVER_PARAMETERS = {
    "trust_remote_code",
    "tp_size",
    "attention_backend",
    "mem_fraction_static",
    "quantization",
    "watchdog_timeout",
}

# `reasoning_parser` is required by the Answer schema and optional here, which is
# a measured difference rather than a relaxation: of the 21 ported cases only the
# GLM-5.2 and MiniMax-M2.7 ones name a parser, and the Kimi-K2.6, Qwen3.5 and
# Qwen3.8 ones state no parser at all.  Requiring it would force nine configs to
# invent a value the case they are ported from does not carry, and the parser
# only shapes a chat completion -- these measurements post raw text to
# `/generate` and ask for one token, so it cannot move a number here either way.
# It is still accepted, so a config stays faithful to the command line it is
# ported from.
SERVER_PARAMETER_POSITIVE_INTEGERS = (
    "tp_size",
    # A checkpoint too large for one board is served across several, and the
    # pipeline degree is how the layers are split over them.
    "pp_size",
    # Expert parallelism, which btv1.5 spells `ep`.  Zero there means "leave it
    # off", and a config states the flag only when the case sets it above one.
    "ep_size",
    "page_size",
    "cuda_graph_max_bs",
    "chunked_prefill_size",
    "max_running_requests",
    "num_continuous_decode_steps",
    "attn_cp_size",
    "max_mamba_cache_size",
    "stream_interval",
)
SERVER_PARAMETER_POSITIVE_NUMBERS = ("watchdog_timeout", "dist_timeout")
SERVER_PARAMETER_STRINGS = (
    "attention_backend",
    # A model whose prefill and decode want different kernels names them apart
    # and leaves the unified backend null; SGLang treats the pair as the override
    # of the unified choice, not as an addition to it.
    "prefill_attention_backend",
    "decode_attention_backend",
    "dsa_prefill_backend",
    "dsa_decode_backend",
    "cp_strategy",
    "reasoning_parser",
    "tool_call_parser",
    "quantization",
    "dtype",
    "moe_a2a_backend",
    "deepep_mode",
)
# Rendered as bare flags, so `true` is the only value they can carry; see the
# check in validate_test_config for why `false` is refused rather than ignored.
SERVER_PARAMETER_STORE_TRUE = (
    "disable_piecewise_cuda_graph",
    "disable_shared_experts_fusion",
    "disable_custom_all_reduce",
    "enforce_disable_flashinfer_allreduce_fusion",
    "enable_prefill_cp",
    # Prefix reuse across the ten identical prompts of one measurement would
    # turn nine prefills into cache hits and report a time to first token that
    # no first request ever saw.  The internal Qwen3.8 cases disable it for that
    # reason, and every config that measures a cold prefill should.
    "disable_radix_cache",
    "disable_overlap_schedule",
    "enable_metrics",
)
# Null means "leave the flag off entirely", which for these two is a reviewed
# instruction rather than an omission: see the quantization comment in
# validate_test_config, and the attention-backend coherence check for the other.
SERVER_PARAMETER_NULLABLE = ("attention_backend", "quantization")
# Only meaningful under the sparse attention backend, which is what the
# coherence check below enforces.
DSA_SERVER_PARAMETERS = (
    "dsa_prefill_backend",
    "dsa_decode_backend",
)
SUPPORTED_SERVER_PARAMETERS = (
    REQUIRED_SERVER_PARAMETERS
    | set(SERVER_PARAMETER_POSITIVE_INTEGERS)
    | set(SERVER_PARAMETER_POSITIVE_NUMBERS)
    | set(SERVER_PARAMETER_STRINGS)
    | set(SERVER_PARAMETER_STORE_TRUE)
)

# The order the value-carrying parameters are rendered in.  Listed explicitly so
# the command line a config produces is stable across dict orderings, which is
# what makes two runs of the same config comparable in the logs -- and, for this
# suite, what makes two runs' numbers comparable at all.
SERVER_PARAMETER_CLI_ORDER = (
    "tp_size",
    "pp_size",
    "ep_size",
    "moe_a2a_backend",
    "deepep_mode",
    "attention_backend",
    "prefill_attention_backend",
    "decode_attention_backend",
    "dsa_prefill_backend",
    "dsa_decode_backend",
    "cp_strategy",
    "attn_cp_size",
    "page_size",
    "cuda_graph_max_bs",
    "chunked_prefill_size",
    "max_running_requests",
    "num_continuous_decode_steps",
    "max_mamba_cache_size",
    "stream_interval",
    "mem_fraction_static",
    "dtype",
    "quantization",
    "reasoning_parser",
    "tool_call_parser",
    "dist_timeout",
    "watchdog_timeout",
)

# The environment variables a config may set around the server, split by what
# this tree can say about them.
#
# These four have a read point in this tree, checked when the name was added:
# `SGLANG_WARMUP_TIMEOUT` is declared in `sglang.srt.environ`;
# `SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8` is read through
# `get_bool_env_var` in `sglang.srt.layers.attention.dsa.utils`; and
# `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` and
# `SGLANG_SAIL_DEEPEP_RECV_HOOK` are read on the DeepEP dispatch path.
SUPPORTED_SERVER_ENVIRONMENT = {
    "SGLANG_WARMUP_TIMEOUT",
    "SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
    "SGLANG_SAIL_DEEPEP_RECV_HOOK",
}

# These three have no read point in this tree, and are accepted anyway, which is
# a departure from the Answer schema's rule and a deliberate one.  The internal
# cases these suites are ported from export them around the server, so a
# performance number measured without them was measured on a different machine
# than the one the number will be compared against; refusing them would make the
# port unfaithful in exactly the dimension being measured.  They are listed
# separately rather than merged into the set above so that the distinction stays
# visible: a name here is honoured by something below this tree -- the PPU SDK,
# the HGGC runtime or a kernel library -- and the evidence for it is the run's
# own provenance block, not a grep of this repository.  A name that turns out to
# be honoured by nobody would silently describe nothing, which is the risk the
# Answer schema declines to take and this one accepts with its eyes open.
EXTERNAL_SERVER_ENVIRONMENT = {
    "DG_USE_MOE_DYNAMIC_TILE",
    "HGGC_EMBEDDED_COPY_THRESHOLD",
    "SGLANG_NSA_DUAL_STREAM",
}

ALLOWED_SERVER_ENVIRONMENT = SUPPORTED_SERVER_ENVIRONMENT | EXTERNAL_SERVER_ENVIRONMENT

# The only dataset shape these suites measure.  A fixed-length prefill
# measurement needs a prompt of exactly the requested token count, which rules
# out every corpus-backed dataset: those report the length distribution of the
# corpus instead.
#
# `random-ids` rather than `random`, though both map to `RandomDataset`, because
# the two differ in where the tokens come from: `random` draws them from
# ShareGPT and downloads that corpus from Hugging Face when `--dataset-path`
# does not already hold it, which the PPU pods cannot do.  `random-ids`
# synthesises the ids locally, which is also the closer analogue of what the
# internal cases do -- they generate their prompts from a length distribution
# rather than from a corpus.  The suite pairs it with `tokenize_prompt`, so the
# prompt reaches `/generate` as ids and the measured input length is the
# requested one exactly; see `PerfSuiteMixin.build_benchmark_args`.
SUPPORTED_WORKLOAD_DATASETS = ("random-ids",)

MEASUREMENT_REQUIRED_KEYS = {
    "id",
    "input_len",
    "output_len",
    "num_prompts",
    "concurrency",
    "source_case",
    "tc_name",
}


def validate_test_config(config: dict[str, Any]) -> None:
    """Validate the public, data-driven performance execution contract."""

    if config.get("schema_version") != PERF_CONFIG_SCHEMA_VERSION:
        raise PerfEvalError("unsupported performance test config schema")
    if not isinstance(config.get("test_id"), str) or not config["test_id"]:
        raise PerfEvalError("test_id must be a non-empty string")
    if set(config) != {
        "schema_version",
        "test_id",
        "hardware",
        "model",
        "server",
        "workload",
    }:
        raise PerfEvalError("the config may only carry the reviewed top-level keys")
    for section in ("hardware", "model", "server", "workload"):
        if not isinstance(config.get(section), dict):
            raise PerfEvalError(f"{section} must be an object")

    _validate_hardware(config["hardware"])
    _validate_model(config["model"])
    _validate_server(config["server"], config["hardware"])
    _validate_workload(config["workload"])


def _validate_hardware(hardware: dict[str, Any]) -> None:
    if hardware.get("platform") != "PPU":
        raise PerfEvalError("hardware.platform must be PPU")
    if not isinstance(hardware.get("generation"), str) or not hardware["generation"]:
        raise PerfEvalError("hardware.generation must be a non-empty string")
    # Per node, not per job: every node of a multi-node config sees the same
    # local device ordinals, and the workflow derives CUDA_VISIBLE_DEVICES from
    # this list inside each pod.  hardware.nnodes carries the second dimension.
    #
    # A config whose tensor-parallel degree is below the board's device count
    # states the shorter list -- the internal MiniMax-M2.7 cases serve at tp 2
    # and the Qwen3.5 and GLM-5.2 mxfp4 ones at tp 4 -- so the job still holds a
    # whole board while the server is handed only the devices the case uses.
    visible_devices = hardware.get("visible_devices")
    if (
        not isinstance(visible_devices, list)
        or not visible_devices
        or any(
            not isinstance(device, int) or isinstance(device, bool) or device < 0
            for device in visible_devices
        )
        or len(set(visible_devices)) != len(visible_devices)
    ):
        raise PerfEvalError(
            "hardware.visible_devices must contain unique non-negative integers"
        )
    memory_gib = hardware.get("memory_gib_per_device")
    if (
        not isinstance(memory_gib, (int, float))
        or isinstance(memory_gib, bool)
        or memory_gib <= 0
    ):
        raise PerfEvalError("hardware.memory_gib_per_device must be positive")
    nnodes = hardware.get("nnodes", 1)
    if not isinstance(nnodes, int) or isinstance(nnodes, bool) or nnodes < 1:
        raise PerfEvalError("hardware.nnodes must be a positive integer")


def _validate_model(model: dict[str, Any]) -> None:
    model_path = model.get("path")
    if not isinstance(model_path, str) or not Path(model_path).is_absolute():
        raise PerfEvalError("model.path must be an absolute path")
    for field in ("checkpoint_name", "served_model_name"):
        if not isinstance(model.get(field), str) or not model[field]:
            raise PerfEvalError(f"model.{field} must be a non-empty string")
    if Path(model_path).name != model["checkpoint_name"]:
        raise PerfEvalError(
            "model.checkpoint_name must match the basename of model.path"
        )
    accepted_model_types = model.get("accepted_model_types")
    if (
        not isinstance(accepted_model_types, list)
        or not accepted_model_types
        or any(
            not isinstance(value, str) or not value for value in accepted_model_types
        )
    ):
        raise PerfEvalError("model.accepted_model_types must contain non-empty strings")


def _validate_server(server: dict[str, Any], hardware: dict[str, Any]) -> None:
    startup_timeout = server.get("startup_timeout_seconds")
    if (
        not isinstance(startup_timeout, (int, float))
        or isinstance(startup_timeout, bool)
        or startup_timeout <= 0
    ):
        raise PerfEvalError("server.startup_timeout_seconds must be positive")
    parameters = server.get("parameters")
    if (
        not isinstance(parameters, dict)
        or not REQUIRED_SERVER_PARAMETERS <= set(parameters)
        or not set(parameters) <= SUPPORTED_SERVER_PARAMETERS
    ):
        raise PerfEvalError(
            "server.parameters must contain exactly the supported SGLang parameters"
        )
    if not isinstance(parameters["trust_remote_code"], bool):
        raise PerfEvalError("server.parameters.trust_remote_code must be boolean")
    for field in SERVER_PARAMETER_POSITIVE_INTEGERS:
        if field not in parameters:
            continue
        if (
            not isinstance(parameters[field], int)
            or isinstance(parameters[field], bool)
            or parameters[field] <= 0
        ):
            raise PerfEvalError(f"server.parameters.{field} must be a positive integer")
    for field in SERVER_PARAMETER_POSITIVE_NUMBERS:
        if field not in parameters:
            continue
        if (
            not isinstance(parameters[field], (int, float))
            or isinstance(parameters[field], bool)
            or parameters[field] <= 0
        ):
            raise PerfEvalError(f"server.parameters.{field} must be positive")
    # `false` is not a second spelling of "absent". Each of these renders as a
    # bare flag, so a false entry would state an intention the command line
    # cannot carry, and the next reader would have to work out whether the
    # default it silently accepted was the reviewed one.
    for field in SERVER_PARAMETER_STORE_TRUE:
        if field not in parameters:
            continue
        if parameters[field] is not True:
            raise PerfEvalError(
                f"server.parameters.{field} may only be set to true; omit it to "
                "keep the server default"
            )
    # SGLang's own check is tp_size * pp_size against the devices it is given, so
    # the product is what has to match the devices this config declares; a config
    # that got this wrong would start a server whose ranks never all arrive, and
    # burn the whole startup budget doing it.
    if parameters["tp_size"] * parameters.get("pp_size", 1) != (
        len(hardware["visible_devices"]) * hardware.get("nnodes", 1)
    ):
        raise PerfEvalError(
            "server.parameters.tp_size times pp_size must equal the visible "
            "device count summed over hardware.nnodes nodes"
        )
    mem_fraction = parameters["mem_fraction_static"]
    if (
        not isinstance(mem_fraction, (int, float))
        or isinstance(mem_fraction, bool)
        or not 0 < mem_fraction <= 1
    ):
        raise PerfEvalError("server.parameters.mem_fraction_static must be in (0, 1]")
    # Two of these may be null, and null is a reviewed instruction rather than an
    # omission.
    #
    # quantization: a checkpoint quantised offline already declares its format in
    # its own config.json, and SGLang reads that when the flag is absent; naming
    # a format here instead overrides that declaration, and several of the
    # spellings argparse accepts -- "fp8" among them -- mean online quantisation
    # of unquantised weights, which is not what such a checkpoint needs.  None of
    # the ported cases states a format, so every config here is null; the key
    # stays required so that is stated rather than inferred from a missing line.
    #
    # attention_backend: a model whose prefill and decode want different kernels
    # names them apart instead, and a unified backend alongside them would be
    # overridden anyway, so stating one would misdescribe the run.
    for field in SERVER_PARAMETER_STRINGS:
        if field not in parameters:
            continue
        value = parameters[field]
        if value is None and field in SERVER_PARAMETER_NULLABLE:
            continue
        if not isinstance(value, str) or not value:
            suffix = " or null" if field in SERVER_PARAMETER_NULLABLE else ""
            raise PerfEvalError(
                f"server.parameters.{field} must be a non-empty string{suffix}"
            )
    if parameters["attention_backend"] is None and not (
        "prefill_attention_backend" in parameters
        or "decode_attention_backend" in parameters
    ):
        raise PerfEvalError(
            "server.parameters.attention_backend may only be null when the "
            "prefill or decode backend is named separately"
        )
    # "nsa" is accepted because SGLang still accepts it, and refusing it here
    # would reject a config copied verbatim from an internal case; "dsa" is the
    # spelling a new config should use.
    if any(field in parameters for field in DSA_SERVER_PARAMETERS) and parameters[
        "attention_backend"
    ] not in ("dsa", "nsa"):
        raise PerfEvalError(
            "the dsa_* server parameters only apply to the sparse attention "
            "backend, so attention_backend must name it"
        )
    if "cp_strategy" in parameters:
        if parameters["cp_strategy"] not in ("zigzag", "interleave"):
            raise PerfEvalError(
                "server.parameters.cp_strategy must be zigzag or interleave"
            )
        if parameters.get("enable_prefill_cp") is not True:
            raise PerfEvalError(
                "server.parameters.cp_strategy requires enable_prefill_cp"
            )
    elif parameters.get("enable_prefill_cp") is True:
        raise PerfEvalError(
            "server.parameters.enable_prefill_cp requires an explicit cp_strategy"
        )
    # deepep_mode names how the DeepEP dispatch runs, and SGLang consults it only
    # on that path, so stating it without the backend would describe a setting no
    # run honours.  The three ported cases that set one set both.
    if "deepep_mode" in parameters and parameters.get("moe_a2a_backend") != "deepep":
        raise PerfEvalError(
            "server.parameters.deepep_mode only applies to the DeepEP all-to-all "
            "backend, so moe_a2a_backend must name it"
        )

    # Optional, and empty is spelled by leaving it out: the variables here are
    # exported around the server the same way the internal cases export them,
    # and SUPPORTED_SERVER_ENVIRONMENT and EXTERNAL_SERVER_ENVIRONMENT say why
    # each name is allowed to be one.
    if "env" in server:
        environment = server["env"]
        if (
            not isinstance(environment, dict)
            or not environment
            or not set(environment) <= ALLOWED_SERVER_ENVIRONMENT
        ):
            raise PerfEvalError(
                "server.env must be a non-empty object of reviewed environment "
                "variables"
            )
        for name, value in environment.items():
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise PerfEvalError(f"server.env.{name} must be a string or a number")
            if isinstance(value, str) and not value:
                raise PerfEvalError(f"server.env.{name} must not be empty")
    if not set(server) <= {"startup_timeout_seconds", "parameters", "env"}:
        raise PerfEvalError("server may only carry the reviewed keys")


def _validate_workload(workload: dict[str, Any]) -> None:
    if not set(workload) <= {
        "dataset",
        "measurements",
        "random_range_ratio",
        "warmup_requests",
        "warmup_passes",
        "seed",
        "flush_cache_timeout_seconds",
    }:
        raise PerfEvalError("workload may only carry the reviewed keys")
    if workload.get("dataset") not in SUPPORTED_WORKLOAD_DATASETS:
        raise PerfEvalError(
            "workload.dataset must name a supported dataset: "
            + ", ".join(SUPPORTED_WORKLOAD_DATASETS)
        )
    # The ratio the random dataset draws its lengths from.  1.0 is what pins
    # every prompt to exactly the requested length: `compute_random_lens` samples
    # from `[int(full_len * range_ratio), full_len]`, so any smaller ratio
    # measures a distribution of lengths rather than the length the case names.
    # The key exists rather than being pinned in code so a future sweep over
    # mixed lengths needs no change here, and every ported config states 1.0.
    ratio = workload.get("random_range_ratio", 1.0)
    if (
        not isinstance(ratio, (int, float))
        or isinstance(ratio, bool)
        or not 0 < ratio <= 1
    ):
        raise PerfEvalError("workload.random_range_ratio must be in (0, 1]")
    warmup_requests = workload.get("warmup_requests", 1)
    if (
        not isinstance(warmup_requests, int)
        or isinstance(warmup_requests, bool)
        or warmup_requests < 0
    ):
        raise PerfEvalError("workload.warmup_requests must be a non-negative integer")
    # How many full passes at a measurement's own shape to run and discard
    # before the recorded one.  Zero is the colocated line's setting and the
    # default: its prefill measurements pay their compilation cost once and the
    # number is the number.  The disaggregated GLM-5.2 case sets one, because the
    # internal harness it is ported from reports a second pass over a first: the
    # boards this line runs on pay a per-batch-shape compilation cost on the
    # first request of each shape -- measured at ~170s for a 4k prefill in run
    # 34317611899 -- and `warmup_requests`, capped at 32 output tokens, cannot
    # reach the decode shapes to absorb it.  A discarded pass does, and the KV
    # cache is flushed after it, so the recorded pass runs warm on compilation
    # and cold on cache.
    warmup_passes = workload.get("warmup_passes", 0)
    if (
        not isinstance(warmup_passes, int)
        or isinstance(warmup_passes, bool)
        or warmup_passes < 0
    ):
        raise PerfEvalError("workload.warmup_passes must be a non-negative integer")
    seed = workload.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise PerfEvalError("workload.seed must be a non-negative integer")
    flush_timeout = workload.get("flush_cache_timeout_seconds", 900)
    if (
        not isinstance(flush_timeout, (int, float))
        or isinstance(flush_timeout, bool)
        or flush_timeout <= 0
    ):
        raise PerfEvalError("workload.flush_cache_timeout_seconds must be positive")

    measurements = workload.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise PerfEvalError("workload.measurements must be a non-empty list")
    seen_ids: set[str] = set()
    for index, measurement in enumerate(measurements):
        if not isinstance(measurement, dict):
            raise PerfEvalError(f"workload.measurements[{index}] must be an object")
        if set(measurement) != MEASUREMENT_REQUIRED_KEYS:
            raise PerfEvalError(
                f"workload.measurements[{index}] must state exactly "
                + ", ".join(sorted(MEASUREMENT_REQUIRED_KEYS))
            )
        for field in ("id", "source_case", "tc_name"):
            if not isinstance(measurement[field], str) or not measurement[field]:
                raise PerfEvalError(
                    f"workload.measurements[{index}].{field} must be a non-empty string"
                )
        # The id names the measurement in the report, in the annotations and in
        # the per-measurement result file, so two measurements sharing one would
        # overwrite each other's numbers.
        if measurement["id"] in seen_ids:
            raise PerfEvalError(
                f"workload.measurements[{index}].id repeats {measurement['id']!r}"
            )
        seen_ids.add(measurement["id"])
        for field in ("input_len", "output_len", "num_prompts", "concurrency"):
            value = measurement[field]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise PerfEvalError(
                    f"workload.measurements[{index}].{field} must be a positive integer"
                )
        # A concurrency above the prompt count would name a parallelism the
        # measurement never reaches, and the reported concurrency would then
        # describe the request count instead of the setting.
        if measurement["concurrency"] > measurement["num_prompts"]:
            raise PerfEvalError(
                f"workload.measurements[{index}].concurrency exceeds its "
                "num_prompts, so the run would never reach it"
            )


def resolve_measurement_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the workload into one fully-specified measurement per entry.

    The workload-level keys are folded into each measurement so a caller never
    has to reach back into the config for a default, and so the report records
    the values a measurement actually ran with rather than the config's shape.
    """

    workload = config["workload"]
    return [
        {
            "id": measurement["id"],
            "dataset": workload["dataset"],
            "input_len": measurement["input_len"],
            "output_len": measurement["output_len"],
            "num_prompts": measurement["num_prompts"],
            "concurrency": measurement["concurrency"],
            "random_range_ratio": workload.get("random_range_ratio", 1.0),
            "warmup_requests": workload.get("warmup_requests", 1),
            "warmup_passes": workload.get("warmup_passes", 0),
            "seed": workload.get("seed", 0),
            "flush_cache_timeout_seconds": workload.get(
                "flush_cache_timeout_seconds", 900
            ),
            "source_case": measurement["source_case"],
            "tc_name": measurement["tc_name"],
        }
        for measurement in workload["measurements"]
    ]


def perf_node_count(config: dict[str, Any]) -> int:
    """The number of nodes the reviewed config serves the checkpoint across."""

    return config["hardware"].get("nnodes", 1)


def resolve_distributed_runtime(
    config: dict[str, Any], environ: dict[str, str] | None = None
) -> dict[str, Any] | None:
    """Bind a multi-node config to the rendezvous this process was handed.

    Returns ``None`` for a single-node config, so a caller can treat the
    single-node path as "no rendezvous" rather than as "one node of one".

    The node rank and the rendezvous address are runtime facts, not reviewed
    ones: they come from whatever launched the pods, so they are read from the
    environment rather than from the config.  ``NODE_RANK``, ``MASTER_ADDR`` and
    ``MASTER_PORT`` are the variables ppu-distributed-action injects per pod in
    gang mode.  ``SGLANG_PPU_PERF_DIST_INIT_ADDR`` overrides the address it
    composes, which is not a convenience: that action leaves ``spec.dnsPolicy``
    at ClusterFirst while asking for host networking, so the pods receive the
    host resolver and the cluster name in ``MASTER_ADDR`` does not resolve --
    measured on all four ranks of run 33750074634.  Until that fix lands, the
    caller has to hand this the rank 0 address it discovered some other way,
    which ``scripts/ci/ppu/answer_rendezvous.sh`` is what publishes.

    Deliberately this suite's own function rather than a call into
    ``answer_eval_kit``: the two lines read different variable names on purpose,
    so a change made for one cannot move the other's rendezvous behaviour, and
    two entries of different suites can share a board without reading each
    other's rank.
    """

    environ = os.environ if environ is None else environ
    nnodes = perf_node_count(config)
    if nnodes == 1:
        return None

    _check_launcher_group_size(environ, nnodes)
    return {
        "nnodes": nnodes,
        "node_rank": _distributed_node_rank(environ, nnodes),
        "dist_init_addr": _distributed_init_addr(environ),
    }


def _check_launcher_group_size(environ: dict[str, str], nnodes: int) -> None:
    """Refuse a launcher that started a different number of nodes than reviewed.

    A launcher that starts too few pods leaves every rank it did start with a
    valid rank, and the shortfall then surfaces only as a rendezvous that never
    completes: the group would sit on the boards it did get for the whole startup
    budget before failing.  Checked rather than trusted because the group size is
    stated twice, once in the reviewed config and once in the workflow that asks
    the cluster for boards, and nothing else compares them.

    Silent when the launcher does not state a group size, which is the bare-metal
    and local case; the rank is what is required there, not this.
    """

    raw = environ.get("NNODES")
    if raw is None or raw == "":
        return
    try:
        launched = int(raw)
    except ValueError:
        raise PerfEvalError(f"NNODES must be an integer, not {raw!r}") from None
    if launched != nnodes:
        raise PerfEvalError(
            f"the launcher started {launched} node(s) and this config is served "
            f"across {nnodes}"
        )


def _distributed_node_rank(environ: dict[str, str], nnodes: int) -> int:
    for name in ("SGLANG_PPU_PERF_NODE_RANK", "NODE_RANK"):
        raw = environ.get(name)
        if raw is None or raw == "":
            continue
        try:
            node_rank = int(raw)
        except ValueError:
            raise PerfEvalError(f"{name} must be an integer, not {raw!r}") from None
        if not 0 <= node_rank < nnodes:
            raise PerfEvalError(
                f"{name} is {node_rank}, outside the [0, {nnodes}) this config declares"
            )
        return node_rank
    raise PerfEvalError(
        "a multi-node performance config needs the rank of this node: set "
        "NODE_RANK or SGLANG_PPU_PERF_NODE_RANK"
    )


def _distributed_init_addr(environ: dict[str, str]) -> str:
    override = environ.get("SGLANG_PPU_PERF_DIST_INIT_ADDR")
    if override:
        host, _, port = override.rpartition(":")
        if not host or not port:
            raise PerfEvalError(
                f"SGLANG_PPU_PERF_DIST_INIT_ADDR must read host:port, not {override!r}"
            )
    else:
        host = environ.get("MASTER_ADDR") or ""
        port = environ.get("MASTER_PORT") or ""
        if not host or not port:
            raise PerfEvalError(
                "a multi-node performance config needs the rendezvous address: "
                "set MASTER_ADDR and MASTER_PORT, or "
                "SGLANG_PPU_PERF_DIST_INIT_ADDR"
            )
    try:
        port_number = int(port)
    except ValueError:
        raise PerfEvalError(
            f"the rendezvous port must be an integer, not {port!r}"
        ) from None
    if not 1 <= port_number <= 65535:
        raise PerfEvalError(f"the rendezvous port {port_number} is out of range")
    return f"{host}:{port_number}"


def build_perf_server_args(
    config: dict[str, Any], *, distributed: dict[str, Any] | None = None
) -> list[str]:
    """Translate a validated performance config into SGLang server arguments.

    ``distributed`` is the result of :func:`resolve_distributed_runtime`, and it
    is required exactly when the config declares more than one node: a caller
    that forgot it would otherwise launch every pod as an independent rank 0
    that tries to fit the whole checkpoint on its own devices.
    """

    parameters = config["server"]["parameters"]
    nnodes = perf_node_count(config)
    if distributed is None and nnodes > 1:
        raise PerfEvalError(
            "a multi-node performance config must be launched with the resolved "
            "rendezvous from resolve_distributed_runtime"
        )
    if distributed is not None and nnodes == 1:
        raise PerfEvalError(
            "a single-node performance config must not be handed a rendezvous"
        )
    if distributed is not None and distributed["nnodes"] != nnodes:
        raise PerfEvalError(
            f"the rendezvous spans {distributed['nnodes']} nodes and the config "
            f"declares {nnodes}"
        )
    args: list[str] = []
    if parameters["trust_remote_code"]:
        args.append("--trust-remote-code")
    for name in SERVER_PARAMETER_CLI_ORDER:
        # An absent optional parameter and an explicit null are the same
        # instruction -- leave the flag off -- so a config that stays on pure
        # tensor parallelism produces the command line it produced before
        # pp_size existed, rather than an explicit --pp-size 1.
        value = parameters.get(name)
        if value is None:
            continue
        args.extend([f"--{name.replace('_', '-')}", str(value)])
    # Validation has already refused any value but true, so presence is the whole
    # instruction here.
    for name in SERVER_PARAMETER_STORE_TRUE:
        if parameters.get(name):
            args.append(f"--{name.replace('_', '-')}")
    args.extend(["--served-model-name", config["model"]["served_model_name"]])
    if distributed is not None:
        args.extend(
            [
                "--nnodes",
                str(distributed["nnodes"]),
                "--node-rank",
                str(distributed["node_rank"]),
                "--dist-init-addr",
                distributed["dist_init_addr"],
            ]
        )
    return args


def perf_server_environment(config: dict[str, Any]) -> dict[str, str]:
    """The environment a config asks to be set around the server.

    Numbers are rendered rather than passed through because an environment is
    strings, and the internal cases state several of these as JSON numbers.
    """

    return {name: str(value) for name, value in config["server"].get("env", {}).items()}


def perf_expected_hardware(config: dict[str, Any]) -> str:
    """Render the reviewed hardware contract for provenance."""

    hardware = config["hardware"]
    memory_gib = hardware["memory_gib_per_device"]
    if isinstance(memory_gib, float) and memory_gib.is_integer():
        memory_gib = int(memory_gib)
    devices_per_node = len(hardware["visible_devices"])
    nnodes = hardware.get("nnodes", 1)
    # A multi-node contract has to state both dimensions, because 4nx8 and 1nx32
    # are not the same machine.
    topology = (
        f"{devices_per_node}x{memory_gib}g"
        if nnodes == 1
        else f"{nnodes}nx{devices_per_node}x{memory_gib}g"
    )
    return f"{hardware['generation']}-{topology}"


# The numbers a measurement records, as (reported name, bench_serving name).
#
# The reported names are spelled out because ``bench_serving``'s own keys are
# ambiguous once they leave that module: ``duration`` is seconds,
# ``input_throughput`` is prompt tokens per second and ``request_throughput`` is
# requests per second, and a reader comparing two reports should not have to know
# which is which.
#
# These are recorded for every measurement, single-token or not.  End-to-end
# latency and both throughput peaks are defined from the first token, so a
# prefill-only case carries them as truthfully as a decode one; only time per
# output token and inter-token latency are undefined for one output token, and
# those are split out into ``DECODE_METRIC_FIELDS`` below rather than recorded
# as the zero ``bench_serving`` computes over an empty list.
#
# P90 time to first token is absent for a different reason: ``BenchmarkMetrics``
# carries p90 for end-to-end latency and for time per output token but not for
# TTFT, and the internal metric list asks for it.  Computing it here would mean
# either duplicating the percentile arithmetic or patching bench_serving, and the
# reviewed decision was to omit it rather than do either.  The per-request TTFTs
# are in the raw result file next to the report, so it can be recovered.
METRIC_FIELDS = (
    ("ttft_mean_ms", "mean_ttft_ms"),
    ("ttft_median_ms", "median_ttft_ms"),
    ("ttft_std_ms", "std_ttft_ms"),
    ("ttft_p99_ms", "p99_ttft_ms"),
    ("e2e_latency_mean_ms", "mean_e2e_latency_ms"),
    ("e2e_latency_median_ms", "median_e2e_latency_ms"),
    ("e2e_latency_p90_ms", "p90_e2e_latency_ms"),
    ("e2e_latency_p99_ms", "p99_e2e_latency_ms"),
    ("request_throughput_req_s", "request_throughput"),
    ("input_token_throughput_tok_s", "input_throughput"),
    ("output_token_throughput_tok_s", "output_throughput"),
    ("output_token_throughput_peak_tok_s", "max_output_tokens_per_s"),
    ("total_token_throughput_tok_s", "total_throughput"),
    ("duration_s", "duration"),
    ("completed", "completed"),
    ("total_input_tokens", "total_input_tokens"),
    ("total_output_tokens", "total_output_tokens"),
    ("concurrency", "concurrency"),
    ("max_concurrent_requests", "max_concurrent_requests"),
)

# The decode-side numbers, recorded only when a measurement asks for more than
# one output token.  Time per output token and inter-token latency are defined
# from the second token onwards, so for a single-token prefill case
# ``bench_serving`` computes them over an empty list and returns zero; recording
# that zero would read as a measurement rather than as an undefined quantity, so
# the colocated prefill line omits these fields entirely while the disaggregated
# line, which decodes 1500 tokens, records them.  ``measurement_record`` picks
# the set from the measurement's own ``output_len``.
DECODE_METRIC_FIELDS = (
    ("tpot_mean_ms", "mean_tpot_ms"),
    ("tpot_median_ms", "median_tpot_ms"),
    ("tpot_std_ms", "std_tpot_ms"),
    ("tpot_p99_ms", "p99_tpot_ms"),
    ("itl_mean_ms", "mean_itl_ms"),
    ("itl_median_ms", "median_itl_ms"),
    ("itl_std_ms", "std_itl_ms"),
    ("itl_p95_ms", "p95_itl_ms"),
    ("itl_p99_ms", "p99_itl_ms"),
)

# How far the prompts the tokenizer actually produced may drift from the length
# the case names before the report says so.  A drift is worth stating -- a
# measurement of 3,900 tokens is not a measurement of 4,000 -- and is not worth
# turning a run red for, because the tokenizer, not the server, decides it.
INPUT_LENGTH_TOLERANCE = 0.01

# The reasons a measurement can fail to produce numbers.  Every one of them is an
# inability to measure rather than a slow result: no threshold is enforced
# anywhere in this module.
REASON_CODES = (
    "server_start_failed",
    "benchmark_crashed",
    "metrics_missing",
    "incomplete_requests",
    "request_errors",
    "cache_flush_failed",
)


def _summarize_lengths(lengths: list[int] | None) -> dict[str, Any] | None:
    if not lengths:
        return None
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
        "count": len(lengths),
    }


def extract_metrics(
    raw: dict[str, Any], *, include_decode: bool = False
) -> dict[str, Any]:
    """Pull the recorded numbers out of what ``run_benchmark`` returned.

    A missing key is an error rather than a null, because ``run_benchmark``
    populates all of them together: an absent one means this tree's benchmark
    result shape moved, and silently recording nulls would turn that into a run
    of blank reports nobody reads twice.

    ``include_decode`` adds the time-per-output-token and inter-token-latency
    fields, and is set from the measurement's ``output_len``: they are defined
    only past the first output token, so a single-token prefill measurement
    leaves them out rather than record the zero ``bench_serving`` computes.
    """

    fields = METRIC_FIELDS + DECODE_METRIC_FIELDS if include_decode else METRIC_FIELDS
    missing = [source for _, source in fields if source not in raw]
    if missing:
        raise MeasurementError(
            "metrics_missing",
            "the benchmark result is missing " + ", ".join(sorted(missing)),
        )
    return {name: raw[source] for name, source in fields}


def observed_workload(raw: dict[str, Any]) -> dict[str, Any]:
    """What the requests actually were, as evidence for what was measured.

    ``random_range_ratio`` at 1.0 is supposed to pin every prompt to the length
    the case names, and this is how a report shows that it did rather than
    asserting it.
    """

    return {
        "input_lengths": _summarize_lengths(raw.get("input_lens")),
        "output_lengths": _summarize_lengths(raw.get("output_lens")),
        "request_errors": [error for error in raw.get("errors") or [] if error],
    }


def measurement_record(
    plan_entry: dict[str, Any], raw: dict[str, Any]
) -> dict[str, Any]:
    """Grade one completed measurement into a report record.

    "Graded" means only this: did the run produce numbers for every request it
    was asked to send.  A measurement that did is ``measured`` whatever the
    numbers say.
    """

    record = _base_record(plan_entry)
    observed = observed_workload(raw)
    record["observed"] = observed
    record["metrics"] = extract_metrics(
        raw, include_decode=plan_entry["output_len"] > 1
    )

    if observed["request_errors"]:
        record["status"] = "failed"
        record["reason_code"] = "request_errors"
        # One error stands for the rest: ten identical prompts against one server
        # fail the same way, and the whole list is in the raw result file.
        record["detail"] = (
            f"{len(observed['request_errors'])} of {plan_entry['num_prompts']} "
            f"requests failed: {observed['request_errors'][0]}"
        )
        return record
    completed = record["metrics"]["completed"]
    if completed != plan_entry["num_prompts"]:
        record["status"] = "failed"
        record["reason_code"] = "incomplete_requests"
        record["detail"] = (
            f"{completed} of {plan_entry['num_prompts']} requests completed"
        )
        return record

    record["status"] = "measured"
    input_lengths = observed["input_lengths"]
    if input_lengths is not None:
        requested = plan_entry["input_len"]
        drift = abs(input_lengths["mean"] - requested) / requested
        if drift > INPUT_LENGTH_TOLERANCE:
            record["warnings"].append(
                {
                    "code": "input_length_drift",
                    "detail": (
                        f"requested {requested} prompt tokens, observed a mean of "
                        f"{input_lengths['mean']:.1f} over {input_lengths['count']} "
                        "requests"
                    ),
                }
            )
    return record


def failed_measurement_record(
    plan_entry: dict[str, Any], reason_code: str, detail: str
) -> dict[str, Any]:
    """Record a measurement that produced no numbers at all."""

    if reason_code not in REASON_CODES:
        raise PerfEvalError(f"unknown reason code {reason_code!r}")
    record = _base_record(plan_entry)
    record["status"] = "failed"
    record["reason_code"] = reason_code
    record["detail"] = detail
    return record


def _base_record(plan_entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": plan_entry["id"],
        "dataset": plan_entry["dataset"],
        "input_len": plan_entry["input_len"],
        "output_len": plan_entry["output_len"],
        "num_prompts": plan_entry["num_prompts"],
        "concurrency": plan_entry["concurrency"],
        "random_range_ratio": plan_entry["random_range_ratio"],
        # Where this measurement came from, so a number in this report can be
        # matched against the internal case it is ported from without going
        # through the README.
        "source_case": plan_entry["source_case"],
        "tc_name": plan_entry["tc_name"],
        "status": "failed",
        "reason_code": None,
        "detail": None,
        "metrics": None,
        "observed": None,
        "warnings": [],
    }


def build_report(
    config: dict[str, Any],
    measurements: list[dict[str, Any]],
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the schema-stable report from the graded measurements."""

    failed = [record for record in measurements if record["status"] == "failed"]
    return {
        "schema_version": PERF_REPORT_SCHEMA_VERSION,
        "test_id": config["test_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "verdict": "failed" if failed else "passed",
            "total": len(measurements),
            "measured": len(measurements) - len(failed),
            "failed": len(failed),
            "warnings": sum(len(record["warnings"]) for record in measurements),
        },
        # The digest of the config that produced these numbers.  Two reports whose
        # digests differ were measured on different settings, whatever their file
        # names say, and that is the first thing a comparison has to check.
        "config_digest": canonical_digest(config),
        "measurements": measurements,
        "provenance": provenance or {},
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_summary(report: dict[str, Any]) -> str:
    """Render the report as the markdown the workflows read annotations out of.

    Three prefixes carry to the run page and nothing else in this document starts
    with any of them, which is what lets a shell with no JSON parser turn them
    into annotations: ``- MEASURED `` becomes a notice, ``- FAIL `` an error and
    ``- WARN `` a warning.  ``test_ppu_perf_eval_unit`` locks those prefixes,
    because a workflow cannot.
    """

    summary = report["summary"]
    served_model_name = report.get("provenance", {}).get("served_model_name")
    lines = [
        f"## PPU serving performance ({served_model_name or 'unknown model'})",
        "",
        f"- Verdict: **{summary['verdict']}**",
        f"- Measurements: {summary['measured']}/{summary['total']} measured",
        "- Thresholds: none. A measurement is red only when it produced no "
        "numbers, never for being slow.",
    ]
    if report["measurements"] and all(
        record["output_len"] == 1 for record in report["measurements"]
    ):
        lines.append(
            "- Every measurement asks for one output token, so time per output "
            "token and inter-token latency are undefined and not reported."
        )
    lines.append("")

    measured = [
        record for record in report["measurements"] if record["status"] == "measured"
    ]
    if measured:
        lines.append("### Measured")
        lines.append("")
        for record in measured:
            metrics = record["metrics"]
            line = (
                f"- MEASURED {record['id']} | "
                f"in={record['input_len']} out={record['output_len']} "
                f"prompts={record['num_prompts']} "
                f"concurrency={record['concurrency']} | "
                f"ttft_mean={_fmt(metrics['ttft_mean_ms'])}ms "
                f"ttft_p50={_fmt(metrics['ttft_median_ms'])}ms "
                f"ttft_p99={_fmt(metrics['ttft_p99_ms'])}ms | "
                f"output={_fmt(metrics['output_token_throughput_tok_s'])}tok/s "
                f"total={_fmt(metrics['total_token_throughput_tok_s'])}tok/s | "
                f"duration={_fmt(metrics['duration_s'])}s"
            )
            # Decode timings only when the measurement asked for more than one
            # output token; a prefill case never carries them (see
            # DECODE_METRIC_FIELDS), so the annotation does not invent a zero.
            if "tpot_mean_ms" in metrics:
                line += (
                    f" | tpot_mean={_fmt(metrics['tpot_mean_ms'])}ms "
                    f"itl_p99={_fmt(metrics['itl_p99_ms'])}ms"
                )
            lines.append(line)
        lines.append("")

    failed = [
        record for record in report["measurements"] if record["status"] == "failed"
    ]
    if failed:
        lines.append("### Not measured")
        lines.append("")
        for record in failed:
            lines.append(
                f"- FAIL {record['id']} | {record['reason_code']} | {record['detail']}"
            )
        lines.append("")

    warnings = [
        (record["id"], warning)
        for record in report["measurements"]
        for warning in record["warnings"]
    ]
    if warnings:
        lines.append("### Warnings")
        lines.append("")
        for identifier, warning in warnings:
            lines.append(
                f"- WARN {identifier} | {warning['code']} | {warning['detail']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def render_junit(report: dict[str, Any]) -> bytes:
    summary = report["summary"]
    suite = ET.Element(
        "testsuite",
        name=f"ppu-perf-{report['test_id']}",
        tests=str(summary["total"]),
        failures=str(summary["failed"]),
        errors="0",
    )
    classname = "ppu.perf." + report["test_id"].replace("-", "_").replace(".", "_")
    for record in report["measurements"]:
        testcase = ET.SubElement(
            suite, "testcase", classname=classname, name=record["id"]
        )
        if record["status"] == "failed":
            failure = ET.SubElement(
                testcase,
                "failure",
                type=record["reason_code"] or "unmeasured",
                message=record["detail"] or "",
            )
            failure.text = json.dumps(
                {"reason_code": record["reason_code"], "detail": record["detail"]}
            )
        output = ET.SubElement(testcase, "system-out")
        # The numbers travel in the JUnit file too, so a reader who has only the
        # test report still has them.
        output.text = json.dumps(
            {
                "metrics": record["metrics"],
                "observed": record["observed"],
                "warnings": record["warnings"],
            },
            sort_keys=True,
        )
    return ET.tostring(suite, encoding="utf-8", xml_declaration=True)


# What a row records about the run that produced it, beyond the numbers.  These
# are the inputs that legitimately move a measurement, and a series carrying only
# the numbers cannot tell a regression from a changed input: a throughput drop on
# the night the base image moved is a different finding from the same drop on an
# unchanged image.  ``package_versions`` is four keys, so it travels whole rather
# than as a chosen subset that would have to be widened the first time a fifth
# one matters.
TREND_PROVENANCE_FIELDS = (
    "source_revision",
    "github_run_id",
    "github_run_url",
    "base_image",
    "base_image_digest",
    "checkpoint_name",
    "checkpoint_config_sha256",
    "expected_hardware",
    "package_versions",
)


def trend_points(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a report into one row per measurement, for the trend series.

    A row is one point in a series keyed by ``(test_id, measurement_id,
    config_digest)``.  The digest belongs to the key rather than riding along as
    a passenger, for the reason ``build_report`` already states about reports:
    two rows whose digests differ were measured on different settings, so a
    comparison that ignores it reads a config change as a performance change.
    Editing a config ends one series and starts another, which is the intended
    behaviour -- the numbers either side of that edit were never comparable.

    A failed measurement is carried too, with ``metrics`` null.  A series has to
    distinguish a night that could not measure from a night that never ran, and
    the internal corpus is the argument for keeping both: most of its
    disaggregated records are failures, and it is counting them that tells a
    reader whether a configuration is measurable at all rather than merely slow.
    """

    provenance = report.get("provenance") or {}
    shared = {
        "schema_version": TREND_POINT_SCHEMA_VERSION,
        "generated_at": report["generated_at"],
        "test_id": report["test_id"],
        "config_digest": report["config_digest"],
        "provenance": {
            field: provenance.get(field) for field in TREND_PROVENANCE_FIELDS
        },
    }
    return [
        {
            **shared,
            "measurement_id": record["id"],
            # The workload shape is repeated into every row, denormalised on
            # purpose: a row has to be readable without resolving its digest back
            # to a config file that may since have been edited or deleted.
            "input_len": record["input_len"],
            "output_len": record["output_len"],
            "num_prompts": record["num_prompts"],
            "concurrency": record["concurrency"],
            "source_case": record["source_case"],
            "tc_name": record["tc_name"],
            "status": record["status"],
            "reason_code": record["reason_code"],
            "metrics": record["metrics"],
        }
        for record in report["measurements"]
    ]


def render_trend_jsonl(report: dict[str, Any]) -> str:
    """The trend rows as one JSON object per line.

    Lines rather than one document, because the series is read back as the
    concatenation of every run's file: a JSON array per run would have to be
    unwrapped before the points could be streamed together.  Keys are sorted and
    the separators are compact so that two runs measuring the same thing produce
    rows that differ only where the numbers do, which is what makes a diff on the
    data branch worth reading.
    """

    return "".join(
        json.dumps(point, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for point in trend_points(report)
    )


def write_report_files(report: dict[str, Any], output_dir: Path) -> None:
    """Write the report next to the raw benchmark output.

    No redaction pass, unlike the Answer report: a measurement records lengths
    and timings, never a generated token, so there is nothing here that a public
    artifact should not carry.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(render_summary(report), encoding="utf-8")
    (output_dir / "junit.xml").write_bytes(render_junit(report))
    # Written here, next to the report, rather than derived later from
    # result.json: the collector copies this directory wholesale, so the rows
    # reach the artifact whether or not anything is yet publishing them to the
    # data branch, and a run measured before that step exists is not lost.
    (output_dir / "trend.jsonl").write_text(
        render_trend_jsonl(report), encoding="utf-8"
    )


def perf_provenance(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """The provenance block, plus the two facts specific to this suite.

    ``default_provenance`` is shared with the Answer evaluator because it is
    schema-agnostic; the workload shape is not, so it is added here rather than
    there.
    """

    provenance = default_provenance(
        config["model"]["served_model_name"],
        config["model"]["path"],
        server_config=config["server"]["parameters"],
        server_environment=perf_server_environment(config),
        expected_hardware=perf_expected_hardware(config),
        **kwargs,
    )
    provenance["workload"] = config["workload"]
    return provenance


# The three sections a disaggregated config shares with a colocated one, exported
# under public names so `pd_perf_eval_kit` can validate them with these functions
# instead of a second copy: what a board is, what a checkpoint is and what a
# measurement asks for are the same reviewed contract on both lines, and a copy
# would let them drift.  Aliases rather than renames, because the callers inside
# this module read better against the private names next to the ones that stay
# private (`_validate_server` has no disaggregated meaning).
validate_hardware = _validate_hardware
validate_model = _validate_model
validate_workload = _validate_workload


__all__ = [
    "ALLOWED_SERVER_ENVIRONMENT",
    "EXTERNAL_SERVER_ENVIRONMENT",
    "INPUT_LENGTH_TOLERANCE",
    "MEASUREMENT_REQUIRED_KEYS",
    "METRIC_FIELDS",
    "DECODE_METRIC_FIELDS",
    "MeasurementError",
    "PERF_CONFIG_SCHEMA_VERSION",
    "PERF_REPORT_SCHEMA_VERSION",
    "PerfEvalError",
    "REASON_CODES",
    "REQUIRED_SERVER_PARAMETERS",
    "SERVER_PARAMETER_CLI_ORDER",
    "SERVER_PARAMETER_NULLABLE",
    "SERVER_PARAMETER_POSITIVE_INTEGERS",
    "SERVER_PARAMETER_POSITIVE_NUMBERS",
    "SERVER_PARAMETER_STORE_TRUE",
    "SERVER_PARAMETER_STRINGS",
    "SUPPORTED_SERVER_ENVIRONMENT",
    "SUPPORTED_SERVER_PARAMETERS",
    "SUPPORTED_WORKLOAD_DATASETS",
    "TREND_POINT_SCHEMA_VERSION",
    "TREND_PROVENANCE_FIELDS",
    "build_perf_server_args",
    "build_report",
    "checkpoint_config_digest",
    "extract_metrics",
    "failed_measurement_record",
    "load_json",
    "measurement_record",
    "observed_workload",
    "perf_expected_hardware",
    "perf_node_count",
    "perf_provenance",
    "perf_server_environment",
    "render_junit",
    "render_summary",
    "render_trend_jsonl",
    "resolve_distributed_runtime",
    "resolve_measurement_plan",
    "trend_points",
    "validate_hardware",
    "validate_model",
    "validate_test_config",
    "validate_workload",
    "write_report_files",
]
