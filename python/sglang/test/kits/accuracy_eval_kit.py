"""The reviewed contract for the registered PPU Accuracy suites.

This is the half of the Accuracy line that needs neither torch nor a live
server: it validates a config, renders the server command line and the
`evalscope` invocation a config asks for, reads the report `evalscope` leaves
behind, and turns it into this repository's report shape.  `accuracy_suite_kit`
is the other half, which owns the server and the subprocess.

The line exists because the internal btv1.5 `llm_infer_sglang_evalscope`
`P0_daily` plan scores its models with EvalScope against gsm8k, ceval, and
ifeval, and nothing in this tree did that.  `answer_eval_kit` grades ten
reviewed questions with rules of our own; these three are public academic
benchmarks with published scoring, so the port runs the same tool the internal
plan runs rather than reimplementing three scorers.  What this module owns is
therefore everything around that tool: which parameters a config may state,
which datasets it may name, and what its number means once it exists.

Two departures from the internal cases are structural rather than incidental,
and both are recorded in the suite README:

* `evalscope` is installed into a virtual environment of its own, because its
  `modelscope[datasets]` dependency asks for `datasets>=4.0.0` while the PPU
  image pins `datasets 3.1.0` behind a `dill<0.3.9` constraint.  It reaches the
  server over HTTP and needs none of that stack, so isolation costs nothing and
  a shared install would silently move the runtime under the server.
* a first run of a dataset has no baseline to be compared against, so a config
  without one is measured rather than judged -- the same rule the perf line
  uses.  A number is published either way; only the verdict differs.
"""

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sglang.test.kits.answer_eval_kit import (
    canonical_digest,
    default_provenance,
    load_json,
)
from sglang.test.kits.quality_trend import write_quality_trend

__all__ = [
    "ACCURACY_REPORT_SCHEMA_VERSION",
    "AccuracyEvalError",
    "DATASET_CONTRACTS",
    "DEFAULT_EVALUATION_TIMEOUT_SECONDS",
    "REPORT_SCHEMA_VERSION",
    "REASON_CODES",
    "SUPPORTED_SERVER_ENVIRONMENT",
    "SUPPORTED_SERVER_PARAMETERS",
    "accuracy_expected_hardware",
    "accuracy_provenance",
    "accuracy_server_environment",
    "build_accuracy_server_args",
    "build_evalscope_command",
    "build_report",
    "canonical_digest",
    "default_provenance",
    "failed_measurement_record",
    "load_json",
    "locate_evalscope_report",
    "measurement_record",
    "read_evalscope_report",
    "render_junit",
    "render_summary",
    "resolve_evaluation_plan",
    "validate_test_config",
    "write_report_files",
]


class AccuracyEvalError(RuntimeError):
    """A config, an invocation, or a report that this contract refuses."""


# What a measurement can say about itself besides "measured".  Named here so a
# workflow reading a report matches on a closed set rather than on prose, and so
# the unit test can assert the set has not quietly grown.
REASON_CODES = (
    "server_start_failed",
    "evalscope_failed",
    "report_missing",
    "report_unreadable",
    "primary_metric_missing",
    "primary_metric_mismatch",
    "incomplete_samples",
    "below_baseline",
    "above_baseline",
)

# Our own report shape, which is what the workflow and any comparison read.
# Bumped when a consumer would have to change, not when a field is added.
ACCURACY_REPORT_SCHEMA_VERSION = 1

# The report shape `evalscope` writes, which this module reads.  Pinned because
# v1 and v2 place the score differently: v2 serializes the metric list plus a
# primary identity and drops the convenience `score` field entirely, so a reader
# that guessed would silently take a diagnostic metric for the conclusion.
REPORT_SCHEMA_VERSION = 2

# The datasets a config may name, and what this repository knows about each.
#
# `dataset_id` is the ModelScope id EvalScope resolves by default, kept here
# because the config states a staged local directory instead: this is what that
# directory has to be a copy of, and it is what the message on a missing dataset
# names.  `primary_metric` is the metric EvalScope marks as the benchmark's
# conclusion, and a config that names a different one is refused rather than
# silently scored on something else.  `samples` is the published size of the
# evaluation split, which is what makes a truncated dataset visible as a number
# instead of as a lower score.
#
# `nltk_resources` names the NLTK corpora the dataset's *scorer* needs, as
# (download id, lookup path) pairs.  It exists because EvalScope fetches them
# lazily, inside the scoring of the first sample that needs one, and swallows the
# failure per sample: on 2026-09-10 `kimi26-mxfp4-ifeval` reached that sample 29
# seconds before the download finished and scored 540 of 541 prompts, while the
# five other IFEval entries won the same race and scored all 541.  A missing
# corpus therefore does not fail a run, it shrinks the denominator -- so the
# suite fetches these before evaluation starts, leaving no race to lose.
DATASET_CONTRACTS = {
    "gsm8k": {
        "dataset_id": "AI-ModelScope/gsm8k",
        "split": "test",
        "primary_metric": "accuracy",
        "default_few_shot_num": 4,
        "samples": 1319,
        "nltk_resources": (),
    },
    "ceval": {
        "dataset_id": "evalscope/ceval",
        "split": "val",
        "primary_metric": "accuracy",
        "default_few_shot_num": 5,
        "samples": 1346,
        "nltk_resources": (),
    },
    "ifeval": {
        "dataset_id": "opencompass/ifeval",
        "split": "train",
        "primary_metric": "prompt_level_strict",
        "default_few_shot_num": 0,
        "samples": 541,
        # Sentence tokenisation, for the instructions that count sentences.
        "nltk_resources": (("punkt_tab", "tokenizers/punkt_tab"),),
    },
}

REQUIRED_SERVER_PARAMETERS = {
    "trust_remote_code",
    "tp_size",
    "mem_fraction_static",
    "quantization",
    "watchdog_timeout",
}

SERVER_PARAMETER_POSITIVE_INTEGERS = (
    "tp_size",
    "pp_size",
    "cuda_graph_max_bs",
    "chunked_prefill_size",
    "max_running_requests",
    "num_continuous_decode_steps",
    "page_size",
    "stream_interval",
    "attn_cp_size",
    # The speculative family.  Present here and absent from the Answer line
    # because every P0_daily case carries MTP: the internal plan's own reason
    # for these cases is the decode path with a draft model attached, so a port
    # that dropped them would score a different serving configuration than the
    # one the baseline was produced on.
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
)
SERVER_PARAMETER_POSITIVE_NUMBERS = ("watchdog_timeout", "dist_timeout")
SERVER_PARAMETER_STRINGS = (
    "attention_backend",
    "prefill_attention_backend",
    "decode_attention_backend",
    # The sparse attention family, spelled `dsa_*`: the tree renamed the whole
    # `nsa_*` family and kept the old spellings only as deprecated aliases, so a
    # config written the old way would work today and stop working without
    # notice.
    "dsa_prefill_backend",
    "dsa_decode_backend",
    "dsa_prefill_cp_mode",
    "reasoning_parser",
    "tool_call_parser",
    "quantization",
    "speculative_algorithm",
    "speculative_attention_mode",
    "speculative_draft_attention_backend",
    "mamba_scheduler_strategy",
)
SERVER_PARAMETER_STORE_TRUE = (
    "disable_piecewise_cuda_graph",
    "disable_shared_experts_fusion",
    "disable_custom_all_reduce",
    "enforce_disable_flashinfer_allreduce_fusion",
    "enable_dsa_prefill_context_parallel",
    "disable_radix_cache",
    "enable_metrics",
)
SERVER_PARAMETER_NULLABLE = ("attention_backend", "quantization")
# Only meaningful under the sparse attention backend, which is what the
# coherence check in `_validate_server` enforces.
DSA_SERVER_PARAMETERS = (
    "dsa_prefill_backend",
    "dsa_decode_backend",
    "dsa_prefill_cp_mode",
)
SUPPORTED_SERVER_PARAMETERS = (
    REQUIRED_SERVER_PARAMETERS
    | set(SERVER_PARAMETER_POSITIVE_INTEGERS)
    | set(SERVER_PARAMETER_POSITIVE_NUMBERS)
    | set(SERVER_PARAMETER_STRINGS)
    | set(SERVER_PARAMETER_STORE_TRUE)
)

# The order value-carrying parameters are rendered in, so two runs of one config
# produce the same command line in the logs regardless of dict ordering.
SERVER_PARAMETER_CLI_ORDER = (
    "tp_size",
    "pp_size",
    "attention_backend",
    "prefill_attention_backend",
    "decode_attention_backend",
    "dsa_prefill_backend",
    "dsa_decode_backend",
    "dsa_prefill_cp_mode",
    "attn_cp_size",
    "page_size",
    "stream_interval",
    "cuda_graph_max_bs",
    "chunked_prefill_size",
    "max_running_requests",
    "num_continuous_decode_steps",
    "mem_fraction_static",
    "quantization",
    "reasoning_parser",
    "tool_call_parser",
    "speculative_algorithm",
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
    "speculative_attention_mode",
    "speculative_draft_attention_backend",
    "mamba_scheduler_strategy",
    "dist_timeout",
    "watchdog_timeout",
)

# The environment a config may set around the server.  `SGLANG_USE_MODELSCOPE`
# is deliberately absent although every P0_daily case sets it: it asks SGLang to
# resolve the model through ModelScope, and these configs name an absolute NAS
# path.  These pods can in fact reach ModelScope, which makes the variable worse
# than useless rather than merely useless: what it can buy is a hub fetch of
# weights that are already on the filesystem the config names.
SUPPORTED_SERVER_ENVIRONMENT = {
    "SGLANG_WARMUP_TIMEOUT",
    "SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8",
}

# The generation keys a config may pin, matching what the internal cases state
# in their own `generation_config`.  `seed` is ours: these cases sample rather
# than decode greedily, and a fixed seed is what keeps two runs of one config
# comparable without changing the distribution the baseline was measured on.
#
# The truncation and penalty keys are here because the cases differ in them and
# a score is a score of a distribution: Qwen3.5 states `top_k` 20 with `min_p`,
# `presence_penalty` and `repetition_penalty`, MiniMax-M2.7 states `top_k` 40,
# and GLM-5.2 and Kimi-K2.6 state neither.  Dropping them would sample a wider
# distribution than the source case and make the resulting number a measurement
# of something else.  They travel to the server untouched -- the block is
# rendered into `--generation-config` and EvalScope forwards it to the
# OpenAI-compatible endpoint, which accepts all four.
SUPPORTED_GENERATION_KEYS = {
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "timeout",
}

_EVALUATION_REQUIRED_KEYS = {
    "dataset",
    "dataset_dir",
    "eval_batch_size",
    "generation",
    "primary_metric",
    "baseline",
    "source_case",
    "tc_name",
}
_EVALUATION_OPTIONAL_KEYS = {
    "few_shot_num",
    "limit",
    "filters",
    "subset_list",
    "min_ratio",
    "max_ratio",
    "timeout_seconds",
}

# The ratio band the internal cases carry: a score at least 98% of the baseline
# passes, and one above twice it is refused as implausible rather than
# celebrated.  Stated as defaults so a config only names them to depart.
DEFAULT_MIN_RATIO = 0.98
DEFAULT_MAX_RATIO = 2.00

# How long a dataset gets before its `evalscope` process is abandoned.  The
# datasets are 541 to 1346 samples against a 32768-token generation budget, so
# this is a guard against a hung run rather than a target; the workflow's pod
# timeout is the real fence.
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 21600


def validate_test_config(config: dict[str, Any]) -> None:
    """Refuse a config this line cannot execute exactly as written."""

    expected_keys = {
        "schema_version",
        "test_id",
        "name",
        "category",
        "framework",
        "test_category",
        "hardware",
        "model",
        "server",
        "evaluation",
    }
    if not isinstance(config, dict):
        raise AccuracyEvalError("an Accuracy test config must be a JSON object")
    missing = sorted(expected_keys - set(config))
    unexpected = sorted(set(config) - expected_keys)
    if missing or unexpected:
        raise AccuracyEvalError(
            f"an Accuracy test config states exactly {sorted(expected_keys)}; "
            f"missing {missing}, unexpected {unexpected}"
        )
    if config["schema_version"] != "ppu-accuracy-test-config/v1":
        raise AccuracyEvalError(
            "unsupported Accuracy schema_version " f"{config['schema_version']!r}"
        )
    if config["test_category"] != "accuracy":
        raise AccuracyEvalError(
            f"test_category must be 'accuracy'; got {config['test_category']!r}"
        )
    _validate_hardware(config["hardware"])
    _validate_model(config["model"])
    _validate_server(config["server"], config["hardware"])
    _validate_evaluation(config["evaluation"])


def _validate_hardware(hardware: Any) -> None:
    if not isinstance(hardware, dict):
        raise AccuracyEvalError("hardware must be a JSON object")
    expected = {"platform", "generation", "visible_devices", "memory_gib_per_device"}
    if set(hardware) != expected:
        raise AccuracyEvalError(f"hardware states exactly {sorted(expected)}")
    if hardware["platform"] != "PPU":
        raise AccuracyEvalError("this line runs on PPU only")
    devices = hardware["visible_devices"]
    if not isinstance(devices, list) or not devices:
        raise AccuracyEvalError("hardware.visible_devices must be a non-empty list")
    if devices != list(range(len(devices))):
        raise AccuracyEvalError(
            "hardware.visible_devices must be a dense range starting at 0; "
            f"got {devices}"
        )


def _validate_model(model: Any) -> None:
    if not isinstance(model, dict):
        raise AccuracyEvalError("model must be a JSON object")
    expected = {"path", "checkpoint_name", "served_model_name", "accepted_model_types"}
    if set(model) != expected:
        raise AccuracyEvalError(f"model states exactly {sorted(expected)}")
    if not str(model["path"]).startswith("/"):
        raise AccuracyEvalError("model.path must be absolute")
    types = model["accepted_model_types"]
    if not isinstance(types, list) or not types:
        raise AccuracyEvalError("model.accepted_model_types must be a non-empty list")


def _validate_server(server: Any, hardware: dict[str, Any]) -> None:
    if not isinstance(server, dict):
        raise AccuracyEvalError("server must be a JSON object")
    if set(server) - {"startup_timeout_seconds", "parameters", "env"}:
        raise AccuracyEvalError(
            "server states startup_timeout_seconds, parameters, and optionally env"
        )
    timeout = server.get("startup_timeout_seconds")
    if not isinstance(timeout, int) or timeout <= 0:
        raise AccuracyEvalError("server.startup_timeout_seconds must be a positive int")
    parameters = server.get("parameters")
    if not isinstance(parameters, dict):
        raise AccuracyEvalError("server.parameters must be a JSON object")
    missing = sorted(REQUIRED_SERVER_PARAMETERS - set(parameters))
    if missing:
        raise AccuracyEvalError(f"server.parameters is missing {missing}")
    unsupported = sorted(set(parameters) - SUPPORTED_SERVER_PARAMETERS)
    if unsupported:
        raise AccuracyEvalError(
            f"server.parameters states {unsupported}, which this schema does not "
            "model; adding one is a schema change with its own review"
        )
    for name in SERVER_PARAMETER_POSITIVE_INTEGERS:
        value = parameters.get(name)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise AccuracyEvalError(f"server.parameters.{name} must be a positive int")
    for name in SERVER_PARAMETER_POSITIVE_NUMBERS:
        value = parameters.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise AccuracyEvalError(
                f"server.parameters.{name} must be a positive number"
            )
    for name in SERVER_PARAMETER_STRINGS:
        value = parameters.get(name)
        if value is None:
            if name in SERVER_PARAMETER_NULLABLE or name not in parameters:
                continue
            raise AccuracyEvalError(f"server.parameters.{name} must not be null")
        if not isinstance(value, str) or not value:
            raise AccuracyEvalError(f"server.parameters.{name} must be a non-empty str")
    for name in SERVER_PARAMETER_STORE_TRUE:
        if name not in parameters:
            continue
        # Rendered as a bare flag, so `false` cannot be expressed on the command
        # line at all.  Refused rather than ignored: a config that says `false`
        # is asking for the opposite of what the run would do.
        if parameters[name] is not True:
            raise AccuracyEvalError(
                f"server.parameters.{name} is a bare flag; state true or omit it"
            )
    if parameters["tp_size"] != len(hardware["visible_devices"]):
        raise AccuracyEvalError(
            f"server.parameters.tp_size {parameters['tp_size']} does not match the "
            f"{len(hardware['visible_devices'])} declared visible devices"
        )
    if parameters.get("attention_backend") is not None and (
        parameters.get("prefill_attention_backend")
        or parameters.get("decode_attention_backend")
    ):
        raise AccuracyEvalError(
            "a unified attention_backend and the prefill/decode pair are "
            "alternatives; state one or the other"
        )
    dsa_stated = [name for name in DSA_SERVER_PARAMETERS if parameters.get(name)]
    if dsa_stated and parameters.get("attention_backend") != "dsa":
        raise AccuracyEvalError(
            f"{dsa_stated} configure the sparse attention backend and mean "
            "nothing without attention_backend 'dsa'"
        )
    if parameters.get("enable_dsa_prefill_context_parallel") and not parameters.get(
        "attn_cp_size"
    ):
        raise AccuracyEvalError(
            "enable_dsa_prefill_context_parallel needs attn_cp_size to say over "
            "how many devices the prefill is split"
        )
    _validate_speculative(parameters)
    environment = server.get("env", {})
    if not isinstance(environment, dict):
        raise AccuracyEvalError("server.env must be a JSON object")
    unsupported_env = sorted(set(environment) - SUPPORTED_SERVER_ENVIRONMENT)
    if unsupported_env:
        raise AccuracyEvalError(
            f"server.env states {unsupported_env}, which nothing in this tree "
            "reads; accepting it would let a config claim a setting no run honours"
        )


def _validate_speculative(parameters: dict[str, Any]) -> None:
    """Keep a speculative configuration whole, and off the radix cache.

    The partial case is the dangerous one: `speculative_algorithm` without its
    step counts launches a server that is not the one the baseline was measured
    on, and the numbers would still look plausible.

    The radix-cache check is a measured incompatibility on this hardware, not a
    theoretical one -- a speculative server with the radix cache left on failed
    to start in the perf line's own bring-up -- so a config that asks for both
    is refused here rather than after a checkpoint load.
    """

    algorithm = parameters.get("speculative_algorithm")
    companions = (
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
    )
    stated = [name for name in companions if parameters.get(name) is not None]
    if algorithm is None:
        if stated:
            raise AccuracyEvalError(
                f"{stated} require speculative_algorithm to be stated as well"
            )
        return
    absent = [name for name in companions if parameters.get(name) is None]
    if absent:
        raise AccuracyEvalError(
            f"speculative_algorithm {algorithm!r} also needs {absent}"
        )
    if not parameters.get("disable_radix_cache"):
        raise AccuracyEvalError(
            "a speculative server on this hardware needs disable_radix_cache "
            "true; the two together fail to start"
        )


def _validate_evaluation(evaluation: Any) -> None:
    if not isinstance(evaluation, dict):
        raise AccuracyEvalError("evaluation must be a JSON object")
    missing = sorted(_EVALUATION_REQUIRED_KEYS - set(evaluation))
    unexpected = sorted(
        set(evaluation) - _EVALUATION_REQUIRED_KEYS - _EVALUATION_OPTIONAL_KEYS
    )
    if missing or unexpected:
        raise AccuracyEvalError(
            f"evaluation requires {sorted(_EVALUATION_REQUIRED_KEYS)} and allows "
            f"{sorted(_EVALUATION_OPTIONAL_KEYS)}; missing {missing}, "
            f"unexpected {unexpected}"
        )
    dataset = evaluation["dataset"]
    contract = DATASET_CONTRACTS.get(dataset)
    if contract is None:
        raise AccuracyEvalError(
            f"dataset {dataset!r} is not one of {sorted(DATASET_CONTRACTS)}"
        )
    if evaluation["primary_metric"] != contract["primary_metric"]:
        raise AccuracyEvalError(
            f"{dataset} concludes on {contract['primary_metric']!r}; the config "
            f"names {evaluation['primary_metric']!r}"
        )
    if not str(evaluation["dataset_dir"]).startswith("/"):
        raise AccuracyEvalError(
            "evaluation.dataset_dir must be an absolute path: the dataset is a "
            "directory staged on shared storage rather than a hub id"
        )
    batch_size = evaluation["eval_batch_size"]
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise AccuracyEvalError("evaluation.eval_batch_size must be a positive int")
    few_shot_num = evaluation.get("few_shot_num")
    if few_shot_num is not None and (
        not isinstance(few_shot_num, int)
        or isinstance(few_shot_num, bool)
        or few_shot_num < 0
    ):
        raise AccuracyEvalError(
            "evaluation.few_shot_num must be a non-negative int when stated"
        )
    limit = evaluation.get("limit")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
    ):
        raise AccuracyEvalError(
            "evaluation.limit must be a positive int when stated; omit it or "
            "state null to evaluate the whole split"
        )
    _validate_generation(evaluation["generation"])
    _validate_baseline(evaluation)
    filters = evaluation.get("filters")
    if filters is not None and not isinstance(filters, dict):
        raise AccuracyEvalError("evaluation.filters must be a JSON object")
    subset_list = evaluation.get("subset_list")
    if subset_list is not None and (
        not isinstance(subset_list, list)
        or not subset_list
        or not all(isinstance(item, str) and item for item in subset_list)
    ):
        raise AccuracyEvalError(
            "evaluation.subset_list must be a non-empty list of strings when stated"
        )
    timeout = evaluation.get("timeout_seconds")
    if timeout is not None and (
        not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0
    ):
        raise AccuracyEvalError(
            "evaluation.timeout_seconds must be a positive int when stated"
        )


def _validate_generation(generation: Any) -> None:
    if not isinstance(generation, dict):
        raise AccuracyEvalError("evaluation.generation must be a JSON object")
    if "max_tokens" not in generation:
        raise AccuracyEvalError("evaluation.generation must state max_tokens")
    unsupported = sorted(set(generation) - SUPPORTED_GENERATION_KEYS)
    if unsupported:
        raise AccuracyEvalError(
            f"evaluation.generation states {unsupported}; this schema models "
            f"{sorted(SUPPORTED_GENERATION_KEYS)}"
        )
    for name in ("max_tokens", "top_k", "seed", "timeout"):
        value = generation.get(name)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise AccuracyEvalError(
                f"evaluation.generation.{name} must be a non-negative int"
            )
    for name in (
        "temperature",
        "top_p",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
    ):
        value = generation.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise AccuracyEvalError(
                f"evaluation.generation.{name} must be a non-negative number"
            )


def _validate_baseline(evaluation: dict[str, Any]) -> None:
    """A baseline is a number or an explicit null, never absent.

    Null is the reviewed way to say "this dataset has no measured baseline on
    this hardware yet", which is the state every entry starts in and the reason
    `measurement_record` can return `measured` without a verdict.  Requiring the
    key means that state is stated rather than inferred from a typo.
    """

    baseline = evaluation["baseline"]
    if baseline is not None:
        if isinstance(baseline, bool) or not isinstance(baseline, (int, float)):
            raise AccuracyEvalError(
                "evaluation.baseline must be a number or null; got "
                f"{type(baseline).__name__}"
            )
        if not 0 < baseline <= 1:
            raise AccuracyEvalError(
                "evaluation.baseline is a score on the 0..1 scale EvalScope "
                f"reports; got {baseline}"
            )
    for name, default in (
        ("min_ratio", DEFAULT_MIN_RATIO),
        ("max_ratio", DEFAULT_MAX_RATIO),
    ):
        value = evaluation.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise AccuracyEvalError(f"evaluation.{name} must be a positive number")
    if evaluation.get("min_ratio", DEFAULT_MIN_RATIO) > evaluation.get(
        "max_ratio", DEFAULT_MAX_RATIO
    ):
        raise AccuracyEvalError("evaluation.min_ratio must not exceed max_ratio")
    if baseline is None and ("min_ratio" in evaluation or "max_ratio" in evaluation):
        raise AccuracyEvalError(
            "a ratio band without a baseline judges nothing; state the baseline "
            "or drop the band"
        )


def resolve_evaluation_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Fold the dataset contract's defaults into the config's own evaluation.

    The result is what the run executes and what the report records, so a
    default that changes later cannot silently reinterpret an existing report.
    """

    evaluation = config["evaluation"]
    contract = DATASET_CONTRACTS[evaluation["dataset"]]
    plan = dict(evaluation)
    plan["few_shot_num"] = evaluation.get(
        "few_shot_num", contract["default_few_shot_num"]
    )
    plan["limit"] = evaluation.get("limit")
    plan["min_ratio"] = evaluation.get("min_ratio", DEFAULT_MIN_RATIO)
    plan["max_ratio"] = evaluation.get("max_ratio", DEFAULT_MAX_RATIO)
    plan["timeout_seconds"] = evaluation.get(
        "timeout_seconds", DEFAULT_EVALUATION_TIMEOUT_SECONDS
    )
    plan["expected_samples"] = plan["limit"] or contract["samples"]
    plan["dataset_id"] = contract["dataset_id"]
    plan["split"] = contract["split"]
    plan["nltk_resources"] = contract["nltk_resources"]
    return plan


def build_accuracy_server_args(config: dict[str, Any]) -> list[str]:
    """Translate a validated config into SGLang server CLI arguments."""

    parameters = config["server"]["parameters"]
    args = []
    if parameters["trust_remote_code"]:
        args.append("--trust-remote-code")
    for name in SERVER_PARAMETER_CLI_ORDER:
        # An absent optional parameter and an explicit null are the same
        # instruction: leave the flag off, so a config produces the command line
        # it produced before the parameter existed.
        value = parameters.get(name)
        if value is None:
            continue
        args.extend([f"--{name.replace('_', '-')}", str(value)])
    for name in SERVER_PARAMETER_STORE_TRUE:
        if parameters.get(name):
            args.append(f"--{name.replace('_', '-')}")
    args.extend(["--served-model-name", config["model"]["served_model_name"]])
    return args


def accuracy_server_environment(config: dict[str, Any]) -> dict[str, str]:
    """The environment a config asks to be set around the server."""

    return {name: str(value) for name, value in config["server"].get("env", {}).items()}


def accuracy_expected_hardware(config: dict[str, Any]) -> str:
    """Render the reviewed hardware contract for provenance."""

    hardware = config["hardware"]
    memory_gib = hardware["memory_gib_per_device"]
    if isinstance(memory_gib, float) and memory_gib.is_integer():
        memory_gib = int(memory_gib)
    return f"{hardware['generation']}-{len(hardware['visible_devices'])}x{memory_gib}g"


def build_evalscope_command(
    config: dict[str, Any],
    plan: dict[str, Any],
    *,
    base_url: str,
    work_dir: Path,
    executable: str = "evalscope",
) -> list[str]:
    """The `evalscope` invocation a config asks for.

    `--eval-type openai_api` is the whole reason this line can run at all: the
    model is already served by the process the suite launched, so EvalScope is a
    client here and needs neither the checkpoint nor a device.

    The dataset is passed as `dataset_id` pointing at a staged directory rather
    than as the hub id in `DATASET_CONTRACTS`, so that the measurement reads the
    same bytes every night instead of whatever a hub serves at the time.
    `local_path` would also work and is what the offline guide still shows, but
    it is documented as deprecated in favour of `dataset_id`.
    """

    generation = dict(plan["generation"])
    dataset = plan["dataset"]
    dataset_args = {
        dataset: {
            "dataset_id": plan["dataset_dir"],
            "few_shot_num": plan["few_shot_num"],
        }
    }
    if plan.get("filters"):
        dataset_args[dataset]["filters"] = plan["filters"]
    if plan.get("subset_list"):
        dataset_args[dataset]["subset_list"] = plan["subset_list"]
    command = [
        executable,
        "eval",
        "--model",
        config["model"]["served_model_name"],
        "--api-url",
        f"{base_url.rstrip('/')}/v1",
        "--api-key",
        "EMPTY",
        "--eval-type",
        "openai_api",
        "--datasets",
        dataset,
        "--dataset-args",
        json.dumps(dataset_args, sort_keys=True),
        "--generation-config",
        json.dumps(generation, sort_keys=True),
        "--eval-batch-size",
        str(plan["eval_batch_size"]),
        "--work-dir",
        str(work_dir),
        # Without this the report lands under a timestamp the caller has to
        # guess at; the suite reads the report back, so it names the directory.
        "--no-timestamp",
    ]
    if plan["limit"] is not None:
        command.extend(["--limit", str(plan["limit"])])
    return command


def locate_evalscope_report(work_dir: Path, dataset: str) -> Path | None:
    """Find the one report `evalscope` wrote for this dataset.

    The path is `reports/<model_id>/<dataset>.json`, and `model_id` is derived
    from the model name rather than stated, so it is discovered instead of
    constructed.  More than one match means the work directory was reused
    across models, which would make "the report" ambiguous; that is an error
    rather than a choice of the first hit.
    """

    reports_dir = work_dir / "reports"
    if not reports_dir.is_dir():
        return None
    matches = sorted(reports_dir.glob(f"*/{dataset}.json"))
    if not matches:
        return None
    if len(matches) > 1:
        raise AccuracyEvalError(
            f"{reports_dir} holds {len(matches)} reports for {dataset}: "
            f"{[str(path) for path in matches]}"
        )
    return matches[0]


def _scalar_key(value: Any) -> tuple[str, Any]:
    """Keep booleans distinct from numbers, the way EvalScope's identity does.

    Mirrors `MetricIdentity.sort_key`: `True` and `1` are equal in Python but
    are different dimension values there, and this reader matches identities by
    equality, so it has to draw the same distinction.
    """

    if isinstance(value, bool):
        return "boolean", value
    if isinstance(value, (int, float)):
        return "number", value
    return "string", value


def _identity_key(identity: Any) -> tuple[Any, ...] | None:
    """A comparable form of a serialized `MetricIdentity`.

    The identity is an object -- `{name, aggregation, dimensions}` -- rather
    than a string, so it cannot be compared as one, and its `dimensions` are a
    mapping whose JSON ordering is not guaranteed.
    """

    if not isinstance(identity, dict):
        return None
    dimensions = identity.get("dimensions") or {}
    if not isinstance(dimensions, dict):
        return None
    return (
        identity.get("name"),
        identity.get("aggregation"),
        tuple(sorted((key, _scalar_key(value)) for key, value in dimensions.items())),
    )


def _identity_display(metric: dict[str, Any]) -> str:
    """What to call a metric in our own report.

    `Metric.name` is a property in EvalScope and so is absent from the file;
    what is written is `legacy_name` plus the identity it is derived from, and
    this reproduces the same fallback rather than reporting `None`.
    """

    legacy_name = metric.get("legacy_name")
    if legacy_name:
        return str(legacy_name)
    identity = metric.get("identity")
    if not isinstance(identity, dict):
        return "unknown"
    dimensions = identity.get("dimensions") or {}
    rendered = ",".join(
        f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
        for key, value in sorted(dimensions.items())
    )
    suffix = f"[{rendered}]" if rendered else ""
    return f"{identity.get('name')}:{identity.get('aggregation')}{suffix}"


def read_evalscope_report(path: Path) -> dict[str, Any]:
    """Read the score and the sample count out of an EvalScope report.

    The score is taken from the metric identified by `primary_metric_identity`
    and from nowhere else.  EvalScope's own reader does the same and says why:
    the first metric in the list may be a diagnostic such as a token count, so a
    report that could not name its conclusion has no score rather than a
    misleading one.

    `metric_name` is the identity's own `name`, not the display name, because
    that is the field a config's `primary_metric` is checked against.
    """

    report = load_json(path)
    schema_version = report.get("schema_version")
    if schema_version != REPORT_SCHEMA_VERSION:
        raise AccuracyEvalError(
            f"{path} declares report schema_version {schema_version!r}; this "
            f"reader understands {REPORT_SCHEMA_VERSION}"
        )
    metrics = [
        metric for metric in report.get("metrics") or [] if isinstance(metric, dict)
    ]
    reading = {
        "score": None,
        "metric_name": None,
        "metric_display_name": None,
        # `num` is a computed field, so it is present even when the primary
        # metric is not: EvalScope falls back to the first metric's subsets for
        # the count alone, which is a sample count and not a score.
        "samples": report.get("num"),
        "execution": report.get("execution_summary"),
        "unavailable_reason": None,
        "metrics": {
            _identity_display(metric): metric.get("score") for metric in metrics
        },
        "subsets": {},
    }
    identity_key = _identity_key(report.get("primary_metric_identity"))
    if identity_key is None:
        reading["unavailable_reason"] = (
            report.get("primary_metric_unavailable_reason")
            or "the report names no primary metric"
        )
        return reading
    matches = [
        metric
        for metric in metrics
        if _identity_key(metric.get("identity")) == identity_key
    ]
    if len(matches) != 1:
        raise AccuracyEvalError(
            f"{path} names a primary metric that matches {len(matches)} of its "
            "own metrics"
        )
    primary = matches[0]
    reading["score"] = primary.get("score")
    reading["metric_name"] = identity_key[0]
    reading["metric_display_name"] = _identity_display(primary)
    reading["subsets"] = {
        subset.get("name"): subset.get("score")
        for category in primary.get("categories") or []
        if isinstance(category, dict)
        for subset in category.get("subsets") or []
        if isinstance(subset, dict) and not subset.get("is_aggregate")
    }
    return reading


def _base_record(plan: dict[str, Any]) -> dict[str, Any]:
    """The part of a record that is the question rather than the answer.

    Populated before the run so a measurement that produced nothing still says
    what it was asked to do, which is the difference between a report that shows
    a gap and a report that is simply shorter than expected.
    """

    return {
        "id": plan["dataset"],
        "dataset": plan["dataset"],
        "split": plan["split"],
        "few_shot_num": plan["few_shot_num"],
        "eval_batch_size": plan["eval_batch_size"],
        "limit": plan["limit"],
        "expected_samples": plan["expected_samples"],
        "primary_metric": plan["primary_metric"],
        "generation": plan["generation"],
        "baseline": plan["baseline"],
        # Only meaningful against a baseline, and reported either way so a
        # reader does not have to know that to interpret the verdict.
        "min_ratio": plan["min_ratio"] if plan["baseline"] is not None else None,
        "max_ratio": plan["max_ratio"] if plan["baseline"] is not None else None,
        # Where this number came from, so it can be matched against the internal
        # case it is ported from without going through the README.
        "source_case": plan["source_case"],
        "tc_name": plan["tc_name"],
        "status": "failed",
        "reason_code": None,
        "detail": None,
        "score": None,
        "ratio": None,
        "metric_name": None,
        "samples": None,
        "metrics": None,
        "subsets": None,
        "execution": None,
        "warnings": [],
    }


def failed_measurement_record(
    plan: dict[str, Any], reason_code: str, detail: str
) -> dict[str, Any]:
    """Record an evaluation that produced no score at all."""

    if reason_code not in REASON_CODES:
        raise AccuracyEvalError(f"unknown reason code {reason_code!r}")
    record = _base_record(plan)
    record["reason_code"] = reason_code
    record["detail"] = detail
    return record


def measurement_record(plan: dict[str, Any], reading: dict[str, Any]) -> dict[str, Any]:
    """Grade one EvalScope report into a record of this line's shape.

    The order of the checks is the point.  Completeness is settled before the
    score is compared against anything, because a run that answered half the
    split scores lower for a reason that has nothing to do with the model, and
    calling that a regression would send someone looking in the wrong place.

    Without a baseline the record is `measured`: the score is published, no
    verdict is drawn, and the missing baseline is stated as a warning rather
    than left to be inferred from the absence of a ratio.
    """

    record = _base_record(plan)
    record["score"] = reading["score"]
    record["metric_name"] = reading["metric_name"]
    record["samples"] = reading["samples"]
    record["metrics"] = reading["metrics"]
    record["subsets"] = reading["subsets"]
    record["execution"] = reading["execution"]

    if reading["score"] is None:
        record["reason_code"] = "primary_metric_missing"
        record["detail"] = (
            reading["unavailable_reason"] or "the report carries no primary metric"
        )
        return record
    if reading["metric_name"] != plan["primary_metric"]:
        # The config asserts which metric is this dataset's conclusion, and the
        # assertion is false.  Refused rather than scored on whatever EvalScope
        # chose, because the baseline was measured against the named metric.
        record["reason_code"] = "primary_metric_mismatch"
        record["detail"] = (
            f"the config expects {plan['primary_metric']!r}; the report concludes "
            f"on {reading['metric_name']!r}"
        )
        return record

    execution = reading["execution"] or {}
    errored = execution.get("errored") or 0
    if execution.get("incomplete") or errored:
        record["reason_code"] = "incomplete_samples"
        record["detail"] = (
            f"{execution.get('succeeded')} of {execution.get('requested')} "
            f"requested samples succeeded, {errored} errored"
        )
        return record
    if reading["samples"] != plan["expected_samples"]:
        record["reason_code"] = "incomplete_samples"
        record["detail"] = (
            f"the report scores {reading['samples']} samples; "
            f"{plan['dataset']} at this limit has {plan['expected_samples']}"
        )
        return record

    record["status"] = "measured"
    baseline = plan["baseline"]
    if baseline is None:
        record["warnings"].append(
            {
                "code": "no_baseline",
                "detail": (
                    f"{plan['dataset']} has no measured baseline on this hardware "
                    "yet, so this score is recorded and not judged"
                ),
            }
        )
        return record

    ratio = reading["score"] / baseline
    record["ratio"] = ratio
    if ratio < plan["min_ratio"]:
        record["status"] = "failed"
        record["reason_code"] = "below_baseline"
        record["detail"] = (
            f"{reading['score']:.4f} is {ratio:.4f} of the {baseline:.4f} baseline, "
            f"below the {plan['min_ratio']} floor"
        )
    elif ratio > plan["max_ratio"]:
        # Refused rather than celebrated: a score that far above the baseline on
        # a fixed public split is a scoring or dataset problem, not a model that
        # doubled overnight.
        record["status"] = "failed"
        record["reason_code"] = "above_baseline"
        record["detail"] = (
            f"{reading['score']:.4f} is {ratio:.4f} of the {baseline:.4f} baseline, "
            f"above the {plan['max_ratio']} ceiling, which is implausible rather "
            "than an improvement"
        )
    return record


def build_report(
    config: dict[str, Any],
    measurements: list[dict[str, Any]],
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the schema-stable report from the graded measurements."""

    failed = [record for record in measurements if record["status"] == "failed"]
    return {
        "schema_version": ACCURACY_REPORT_SCHEMA_VERSION,
        "test_id": config["test_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "verdict": "failed" if failed else "passed",
            "total": len(measurements),
            "measured": len(measurements) - len(failed),
            "failed": len(failed),
            "warnings": sum(len(record["warnings"]) for record in measurements),
        },
        # The digest of the config that produced these numbers.  Two reports
        # whose digests differ were measured on different settings, whatever
        # their file names say, and that is the first thing a comparison checks.
        "config_digest": canonical_digest(config),
        "measurements": measurements,
        "provenance": provenance or {},
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_summary(report: dict[str, Any]) -> str:
    """Render the report as the markdown the workflow reads annotations out of.

    Three prefixes carry to the run page and nothing else in this document
    starts with any of them, which is what lets a shell with no JSON parser turn
    them into annotations: ``- MEASURED `` becomes a notice, ``- FAIL `` an
    error and ``- WARN `` a warning.  `test_ppu_accuracy_eval_unit` locks those
    prefixes, because a workflow cannot.
    """

    summary = report["summary"]
    served_model_name = report.get("provenance", {}).get("served_model_name")
    lines = [
        f"## PPU accuracy ({served_model_name or 'unknown model'})",
        "",
        f"- Verdict: **{summary['verdict']}**",
        f"- Evaluations: {summary['measured']}/{summary['total']} measured",
        "- Scored by EvalScope against the public split, so the number here is "
        "comparable to a published one rather than to a rule of ours.",
    ]
    if all(record["baseline"] is None for record in report["measurements"]):
        lines.append(
            "- No baseline on this hardware yet, so every score below is "
            "recorded and none is judged."
        )
    lines.append("")

    measured = [
        record for record in report["measurements"] if record["status"] == "measured"
    ]
    if measured:
        lines.append("### Measured")
        lines.append("")
        for record in measured:
            against = (
                f" | baseline={_fmt(record['baseline'])} "
                f"ratio={_fmt(record['ratio'])}"
                if record["baseline"] is not None
                else " | baseline=none"
            )
            lines.append(
                f"- MEASURED {record['id']} | "
                f"{record['metric_name']}={_fmt(record['score'])} | "
                f"samples={record['samples']}/{record['expected_samples']} "
                f"shots={record['few_shot_num']}"
                f"{against}"
            )
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
        name=f"ppu-accuracy-{report['test_id']}",
        tests=str(summary["total"]),
        failures=str(summary["failed"]),
        errors="0",
    )
    classname = "ppu.accuracy." + report["test_id"].replace("-", "_").replace(".", "_")
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
        output.text = json.dumps(
            {
                "score": record["score"],
                "metric_name": record["metric_name"],
                "samples": record["samples"],
                "metrics": record["metrics"],
                "warnings": record["warnings"],
            },
            sort_keys=True,
        )
    return ET.tostring(suite, encoding="utf-8", xml_declaration=True)


def write_report_files(report: dict[str, Any], output_dir: Path) -> None:
    """Write the report where the workflow collects it.

    No redaction pass, unlike the Answer report: a record carries scores, counts
    and subset names, never a generated token, so there is nothing here that a
    public artifact should not carry.  EvalScope's own predictions do contain
    model output, and they stay in its work directory rather than being copied
    in here.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(render_summary(report), encoding="utf-8")
    (output_dir / "junit.xml").write_bytes(render_junit(report))
    write_quality_trend(report, output_dir)


def accuracy_provenance(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """The shared provenance block, plus the facts specific to this line.

    `default_provenance` is shared with the Answer and perf evaluators because
    it is schema-agnostic; what was evaluated is not, so it is added here.
    """

    provenance = default_provenance(
        config["model"]["served_model_name"],
        config["model"]["path"],
        server_config=config["server"]["parameters"],
        server_environment=accuracy_server_environment(config),
        generation_config=config["evaluation"]["generation"],
        expected_hardware=accuracy_expected_hardware(config),
        **kwargs,
    )
    provenance["evaluation"] = config["evaluation"]
    provenance["test_config_id"] = config["test_id"]
    provenance["test_config_sha256"] = canonical_digest(config)
    return provenance
