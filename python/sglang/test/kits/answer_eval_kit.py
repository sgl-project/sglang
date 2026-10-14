"""Deterministic, dependency-light evaluator for the PPU Answer MVP.

The module intentionally does not call an embedding model or an LLM judge.  It
implements the L0/L1 gate and emits records that can be labelled later.  The
schema-stable report is redacted; raw candidate answers are written next to it
as separate files when the caller opts in through ``include_raw_outputs``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

from sglang.test.kits.quality_trend import write_quality_trend


class AnswerEvalError(RuntimeError):
    """Base error for malformed test data or candidate API responses."""


class CandidateRequestError(AnswerEvalError):
    """An L0 failure while calling or parsing the candidate endpoint."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise AnswerEvalError(f"{path} must contain a JSON object")
    return value


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_digest(value: str) -> str:
    """Hash even a JSON-escaped unpaired surrogate without losing evidence."""
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def validate_dataset(dataset: dict[str, Any]) -> None:
    if dataset.get("schema_version") != "ppu-answer-cases/v1":
        raise AnswerEvalError("unsupported case schema")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise AnswerEvalError("the Answer MVP dataset must contain exactly 10 cases")
    case_ids = [case.get("id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != 10 or len(set(case_ids)) != 10 or not all(case_ids):
        raise AnswerEvalError("case ids must be ten unique non-empty strings")
    if any(not case.get("prompt") or not case.get("rules") for case in cases):
        raise AnswerEvalError("every case must define a prompt and deterministic rules")


def validate_profile(profile: dict[str, Any]) -> None:
    if profile.get("schema_version") != "ppu-answer-quality-profile/v1":
        raise AnswerEvalError("unsupported quality profile schema")
    repetition = profile.get("repetition", {})
    thresholds = (
        repetition.get("ngram_suspect_coverage"),
        repetition.get("ngram_hard_fail_coverage"),
    )
    if not all(isinstance(value, (int, float)) for value in thresholds):
        raise AnswerEvalError("n-gram coverage thresholds must be numeric")
    if not 0 <= thresholds[0] < thresholds[1] <= 1:
        raise AnswerEvalError("invalid n-gram coverage thresholds")


# The reviewed server parameters, grouped by how each one is typed and how it
# reaches `sglang serve`.  They are module level because the validator and the
# argument builder have to agree on the set: a name checked here but never
# rendered would be silent decoration, and a name rendered without being checked
# would be the untyped pass-through this schema exists to refuse.
REQUIRED_SERVER_PARAMETERS = {
    "trust_remote_code",
    "tp_size",
    "attention_backend",
    "mem_fraction_static",
    "quantization",
    "reasoning_parser",
    "watchdog_timeout",
}

# Everything below is optional because the internal Answer cases these suites are
# ported from do not all serve their model the same way, and a parameter absent
# from a config has to keep producing the command line that config produced
# before the parameter existed.  The names are SGLang's current ones: the tree
# renamed the whole `nsa_*` family to `dsa_*` and kept the old spellings only as
# deprecated aliases, so a config written with `nsa_*` would work today and warn,
# and stop working without notice.
SERVER_PARAMETER_POSITIVE_INTEGERS = (
    "tp_size",
    # A checkpoint whose per-expert intermediate size is not divisible by the
    # tensor-parallel degree times the quantization block size cannot be sharded
    # by TP alone, and pipelining the layers is how the remaining devices are put
    # to work.
    "pp_size",
    "cuda_graph_max_bs",
    "chunked_prefill_size",
    "max_running_requests",
    "num_continuous_decode_steps",
    "attn_cp_size",
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
    "dsa_prefill_cp_mode",
    "reasoning_parser",
    "quantization",
)
# Rendered as bare flags, so `true` is the only value they can carry; see the
# check in validate_test_config for why `false` is refused rather than ignored.
SERVER_PARAMETER_STORE_TRUE = (
    "disable_piecewise_cuda_graph",
    "disable_shared_experts_fusion",
    "disable_custom_all_reduce",
    "enforce_disable_flashinfer_allreduce_fusion",
    "enable_dsa_prefill_context_parallel",
)
# Null means "leave the flag off entirely", which for these two is a reviewed
# instruction rather than an omission: see the quantization comment below, and
# the attention-backend coherence check for the other.
SERVER_PARAMETER_NULLABLE = ("attention_backend", "quantization")
# Only meaningful under the sparse attention backend, which is what the coherence
# check below enforces.
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

# The order the value-carrying parameters are rendered in.  Listed explicitly so
# the command line a config produces is stable across dict orderings, which is
# what makes two runs of the same config comparable in the logs.
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
    "cuda_graph_max_bs",
    "chunked_prefill_size",
    "max_running_requests",
    "num_continuous_decode_steps",
    "mem_fraction_static",
    "quantization",
    "reasoning_parser",
    "dist_timeout",
    "watchdog_timeout",
)

# The environment variables a config may set around the server, and the only
# ones: an entry here is a name this tree actually reads, checked when it was
# added.  `SGLANG_WARMUP_TIMEOUT` is declared in `sglang.srt.environ`, and
# `SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8` is read through
# `get_bool_env_var` in `sglang.srt.layers.attention.dsa.utils`.  The internal
# GLM cases also export `SGLANG_NSA_DUAL_STREAM=0`, which is deliberately not
# here: nothing in this tree reads that name, so accepting it would let a config
# state a setting no run honours.
SUPPORTED_SERVER_ENVIRONMENT = {
    "SGLANG_WARMUP_TIMEOUT",
    "SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8",
}


def validate_test_config(config: dict[str, Any]) -> None:
    """Validate the public, data-driven Answer execution contract."""

    if config.get("schema_version") != "ppu-answer-test-config/v1":
        raise AnswerEvalError("unsupported Answer test config schema")
    if not isinstance(config.get("test_id"), str) or not config["test_id"]:
        raise AnswerEvalError("test_id must be a non-empty string")

    for section in ("hardware", "model", "server", "request", "evaluation"):
        if not isinstance(config.get(section), dict):
            raise AnswerEvalError(f"{section} must be an object")

    hardware = config["hardware"]
    if hardware.get("platform") != "PPU":
        raise AnswerEvalError("hardware.platform must be PPU")
    if not isinstance(hardware.get("generation"), str) or not hardware["generation"]:
        raise AnswerEvalError("hardware.generation must be a non-empty string")
    # Per node, not per job: every node of a multi-node config sees the same
    # local device ordinals, and the workflow derives CUDA_VISIBLE_DEVICES from
    # this list inside each pod.  hardware.nnodes carries the second dimension.
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
        raise AnswerEvalError(
            "hardware.visible_devices must contain unique non-negative integers"
        )
    memory_gib = hardware.get("memory_gib_per_device")
    if (
        not isinstance(memory_gib, (int, float))
        or isinstance(memory_gib, bool)
        or memory_gib <= 0
    ):
        raise AnswerEvalError("hardware.memory_gib_per_device must be positive")
    nnodes = hardware.get("nnodes", 1)
    if not isinstance(nnodes, int) or isinstance(nnodes, bool) or nnodes < 1:
        raise AnswerEvalError("hardware.nnodes must be a positive integer")

    model = config["model"]
    model_path = model.get("path")
    if not isinstance(model_path, str) or not Path(model_path).is_absolute():
        raise AnswerEvalError("model.path must be an absolute path")
    for field in ("checkpoint_name", "served_model_name"):
        if not isinstance(model.get(field), str) or not model[field]:
            raise AnswerEvalError(f"model.{field} must be a non-empty string")
    if Path(model_path).name != model["checkpoint_name"]:
        raise AnswerEvalError(
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
        raise AnswerEvalError(
            "model.accepted_model_types must contain non-empty strings"
        )

    server = config["server"]
    startup_timeout = server.get("startup_timeout_seconds")
    if (
        not isinstance(startup_timeout, (int, float))
        or isinstance(startup_timeout, bool)
        or startup_timeout <= 0
    ):
        raise AnswerEvalError("server.startup_timeout_seconds must be positive")
    parameters = server.get("parameters")
    if (
        not isinstance(parameters, dict)
        or not REQUIRED_SERVER_PARAMETERS <= set(parameters)
        or not set(parameters) <= SUPPORTED_SERVER_PARAMETERS
    ):
        raise AnswerEvalError(
            "server.parameters must contain exactly the supported SGLang parameters"
        )
    if not isinstance(parameters["trust_remote_code"], bool):
        raise AnswerEvalError("server.parameters.trust_remote_code must be boolean")
    for field in SERVER_PARAMETER_POSITIVE_INTEGERS:
        if field not in parameters:
            continue
        if (
            not isinstance(parameters[field], int)
            or isinstance(parameters[field], bool)
            or parameters[field] <= 0
        ):
            raise AnswerEvalError(
                f"server.parameters.{field} must be a positive integer"
            )
    for field in SERVER_PARAMETER_POSITIVE_NUMBERS:
        if field not in parameters:
            continue
        if (
            not isinstance(parameters[field], (int, float))
            or isinstance(parameters[field], bool)
            or parameters[field] <= 0
        ):
            raise AnswerEvalError(f"server.parameters.{field} must be positive")
    # `false` is not a second spelling of "absent". Each of these renders as a
    # bare flag, so a false entry would state an intention the command line
    # cannot carry, and the next reader would have to work out whether the
    # default it silently accepted was the reviewed one.
    for field in SERVER_PARAMETER_STORE_TRUE:
        if field not in parameters:
            continue
        if parameters[field] is not True:
            raise AnswerEvalError(
                f"server.parameters.{field} may only be set to true; omit it to "
                "keep the server default"
            )
    # SGLang's own check is tp_size * pp_size against the node count, so the
    # product is what has to match the devices the job actually holds; a config
    # that got this wrong would start a server whose ranks never all arrive.
    if parameters["tp_size"] * parameters.get("pp_size", 1) != (
        len(visible_devices) * nnodes
    ):
        raise AnswerEvalError(
            "server.parameters.tp_size times pp_size must equal the visible "
            "device count summed over hardware.nnodes nodes"
        )
    mem_fraction = parameters["mem_fraction_static"]
    if (
        not isinstance(mem_fraction, (int, float))
        or isinstance(mem_fraction, bool)
        or not 0 < mem_fraction <= 1
    ):
        raise AnswerEvalError("server.parameters.mem_fraction_static must be in (0, 1]")
    # Two of these may be null, and null is a reviewed instruction rather than an
    # omission.
    #
    # quantization: a checkpoint quantised offline already declares its format in
    # its own config.json, and SGLang reads that when the flag is absent; naming a
    # format here instead overrides that declaration, and several of the spellings
    # argparse accepts -- "fp8" among them -- mean online quantisation of
    # unquantised weights, which is not what such a checkpoint needs. The
    # internal test cases draw the same line: of the Answer cases only the
    # compressed-tensors ones state a format, and the rest leave it unset.
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
            raise AnswerEvalError(
                f"server.parameters.{field} must be a non-empty string{suffix}"
            )
    if parameters["attention_backend"] is None and not (
        "prefill_attention_backend" in parameters
        or "decode_attention_backend" in parameters
    ):
        raise AnswerEvalError(
            "server.parameters.attention_backend may only be null when the "
            "prefill or decode backend is named separately"
        )
    # "nsa" is accepted because SGLang still accepts it, and refusing it here
    # would reject a config copied verbatim from an internal case; "dsa" is the
    # spelling a new config should use.
    if any(field in parameters for field in DSA_SERVER_PARAMETERS) and parameters[
        "attention_backend"
    ] not in ("dsa", "nsa"):
        raise AnswerEvalError(
            "the dsa_* server parameters only apply to the sparse attention "
            "backend, so attention_backend must name it"
        )
    if (
        "dsa_prefill_cp_mode" in parameters
        and parameters.get("enable_dsa_prefill_context_parallel") is not True
    ):
        raise AnswerEvalError(
            "server.parameters.dsa_prefill_cp_mode describes how prefill context "
            "parallelism splits a sequence, so it needs "
            "enable_dsa_prefill_context_parallel"
        )

    # Optional, and empty is spelled by leaving it out: the variables here are
    # exported around the server the same way the internal cases export them, and
    # SUPPORTED_SERVER_ENVIRONMENT says why each name is allowed to be one.
    if "env" in server:
        environment = server["env"]
        if (
            not isinstance(environment, dict)
            or not environment
            or not set(environment) <= SUPPORTED_SERVER_ENVIRONMENT
        ):
            raise AnswerEvalError(
                "server.env must be a non-empty object of supported SGLang "
                "environment variables"
            )
        for name, value in environment.items():
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise AnswerEvalError(f"server.env.{name} must be a string or a number")
            if isinstance(value, str) and not value:
                raise AnswerEvalError(f"server.env.{name} must not be empty")
    if not set(server) <= {"startup_timeout_seconds", "parameters", "env"}:
        raise AnswerEvalError("server may only carry the reviewed keys")

    request = config["request"]
    request_timeout = request.get("timeout_seconds")
    if (
        not isinstance(request_timeout, (int, float))
        or isinstance(request_timeout, bool)
        or request_timeout <= 0
    ):
        raise AnswerEvalError("request.timeout_seconds must be positive")
    generation = request.get("generation")
    required_generation_parameters = {
        "temperature",
        "top_p",
        "max_tokens",
        "separate_reasoning",
        "chat_template_kwargs",
    }
    if (
        not isinstance(generation, dict)
        or set(generation) != required_generation_parameters
    ):
        raise AnswerEvalError(
            "request.generation must contain exactly the reviewed generation parameters"
        )
    if not isinstance(generation["temperature"], (int, float)) or isinstance(
        generation["temperature"], bool
    ):
        raise AnswerEvalError("request.generation.temperature must be numeric")
    if (
        not isinstance(generation["top_p"], (int, float))
        or isinstance(generation["top_p"], bool)
        or not 0 <= generation["top_p"] <= 1
    ):
        raise AnswerEvalError("request.generation.top_p must be in [0, 1]")
    if (
        not isinstance(generation["max_tokens"], int)
        or isinstance(generation["max_tokens"], bool)
        or generation["max_tokens"] <= 0
    ):
        raise AnswerEvalError(
            "request.generation.max_tokens must be a positive integer"
        )
    if not isinstance(generation["separate_reasoning"], bool):
        raise AnswerEvalError("request.generation.separate_reasoning must be boolean")
    chat_template_kwargs = generation["chat_template_kwargs"]
    # The checkpoints disagree on whether the reasoning pass can be steered at
    # all, and on what the switch is called, so the reviewed set is the union of
    # the names their own templates read:
    #
    #   enable_thinking  Qwen3.5/Qwen3.8-27B and GLM-5.2 honour false and emit a
    #                    closed <think></think> in the generation prompt.
    #   reasoning_effort Qwen3.8-2.4T-A95B answers 'Disabling thinking is not
    #                    supported.' and grades the effort instead.
    #   thinking         Kimi-K2.6 spells the same switch this way; SGLang's own
    #                    kimi_k2 detector agrees, declaring reasoning_default
    #                    "thinking".
    #
    # An empty object is the fourth shape: the MiniMax-M2.7 template reads no
    # switch and opens <think> unconditionally, so a config for it asks for
    # nothing.  Empty is not the same as absent -- the key stays required, so a
    # config states which shape it is rather than leaving a reader to infer it
    # from a missing line.  The legal effort names live in the checkpoint's own
    # template, which rejects the rest by name, so only the shape is checked here.
    supported_chat_template_kwargs = {"enable_thinking", "thinking", "reasoning_effort"}
    if (
        not isinstance(chat_template_kwargs, dict)
        or not set(chat_template_kwargs) <= supported_chat_template_kwargs
    ):
        raise AnswerEvalError(
            "request.generation.chat_template_kwargs may only set the reviewed "
            "reasoning switches: enable_thinking, thinking, reasoning_effort"
        )
    for field in ("enable_thinking", "thinking"):
        if field in chat_template_kwargs and not isinstance(
            chat_template_kwargs[field], bool
        ):
            raise AnswerEvalError(
                f"request.generation.chat_template_kwargs.{field} must be boolean"
            )
    if "reasoning_effort" in chat_template_kwargs and (
        not isinstance(chat_template_kwargs["reasoning_effort"], str)
        or not chat_template_kwargs["reasoning_effort"]
    ):
        raise AnswerEvalError(
            "request.generation.chat_template_kwargs.reasoning_effort must be a "
            "non-empty string"
        )

    for field in ("dataset", "quality_profile"):
        value = config["evaluation"].get(field)
        if not isinstance(value, str) or not value:
            raise AnswerEvalError(f"evaluation.{field} must be a non-empty path")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise AnswerEvalError(
                f"evaluation.{field} must stay relative to the Answer data root"
            )

    validate_baseline(config["evaluation"].get("baseline"))


def validate_baseline(baseline: Any) -> None:
    """Validate the optional per-model regression baseline.

    The baseline is what turns the nightly from an absolute pass/fail into a
    regression gate: ``known_failures`` lists case ids the checkpoint is known
    to miss for model-capability (not infrastructure) reasons, and
    ``min_score`` is the floor of passing cases below which the run reddens
    even if every remaining miss was expected.  Absent baseline keeps the old
    strict behaviour (any miss reddens).
    """

    if baseline is None:
        return
    if not isinstance(baseline, dict):
        raise AnswerEvalError("evaluation.baseline must be an object")
    known = baseline.get("known_failures", [])
    if not isinstance(known, list) or any(
        not isinstance(item, str) or not item for item in known
    ):
        raise AnswerEvalError(
            "evaluation.baseline.known_failures must be a list of case ids"
        )
    if len(set(known)) != len(known):
        raise AnswerEvalError(
            "evaluation.baseline.known_failures must not repeat a case id"
        )
    min_score = baseline.get("min_score")
    if min_score is not None and (
        not isinstance(min_score, int) or isinstance(min_score, bool) or min_score < 0
    ):
        raise AnswerEvalError(
            "evaluation.baseline.min_score must be a non-negative integer"
        )


def resolve_evaluation_paths(
    config: dict[str, Any], data_root: Path
) -> tuple[Path, Path]:
    """Locate the dataset and quality profile a test config selects.

    The paths are declared relative to the Answer data root rather than to the
    directory holding the config, because the configs are filed per model family
    while the corpus and the profile are shared: resolving from the config
    directory would put the shared files above the config, reachable only
    through ``..``, which ``validate_test_config`` rejects so that a config
    cannot name inputs outside the reviewed tree.
    """

    evaluation = config["evaluation"]
    return (
        data_root / evaluation["dataset"],
        data_root / evaluation["quality_profile"],
    )


def answer_node_count(config: dict[str, Any]) -> int:
    """The number of nodes the reviewed config serves the checkpoint across."""

    return config["hardware"].get("nnodes", 1)


def resolve_distributed_runtime(
    config: dict[str, Any], environ: dict[str, str] | None = None
) -> dict[str, Any] | None:
    """Bind a multi-node config to the rendezvous this process was handed.

    Returns ``None`` for a single-node config, so a caller can treat the
    single-node path as "no rendezvous" rather than as "one node of one".

    The node rank and the rendezvous address are runtime facts, not reviewed
    ones: they come from whatever launched the four pods, so they are read from
    the environment rather than from the config.  ``NODE_RANK``, ``MASTER_ADDR``
    and ``MASTER_PORT`` are the variables ppu-distributed-action injects per pod
    in gang mode.  ``SGLANG_PPU_ANSWER_DIST_INIT_ADDR`` overrides the address it
    composes, which is not a convenience: that action leaves ``spec.dnsPolicy``
    at ClusterFirst while asking for host networking, so the pods receive the
    host resolver and the cluster name in ``MASTER_ADDR`` does not resolve --
    measured on all four ranks of run 33750074634.  Until that fix lands, the
    caller has to hand this the rank 0 address it discovered some other way.
    """

    environ = os.environ if environ is None else environ
    nnodes = answer_node_count(config)
    if nnodes == 1:
        return None

    _check_launcher_group_size(environ, nnodes)
    node_rank = _distributed_node_rank(environ, nnodes)
    return {
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": _distributed_init_addr(environ),
    }


def _check_launcher_group_size(environ: dict[str, str], nnodes: int) -> None:
    """Refuse a launcher that started a different number of nodes than reviewed.

    A launcher that starts too few pods leaves every rank it did start with a
    valid rank, and the shortfall then surfaces only as a rendezvous that never
    completes: the group would sit on the boards it did get for the whole 5400s
    startup budget before failing.  Checked rather than trusted because the group
    size is stated twice, once in the reviewed config and once in the workflow
    that asks the cluster for boards, and nothing else compares them.

    Silent when the launcher does not state a group size, which is the bare-metal
    and local case; the rank is what is required there, not this.
    """

    raw = environ.get("NNODES")
    if raw is None or raw == "":
        return
    try:
        launched = int(raw)
    except ValueError:
        raise AnswerEvalError(f"NNODES must be an integer, not {raw!r}") from None
    if launched != nnodes:
        raise AnswerEvalError(
            f"the launcher started {launched} node(s) and this config is served "
            f"across {nnodes}"
        )


def _distributed_node_rank(environ: dict[str, str], nnodes: int) -> int:
    for name in ("SGLANG_PPU_ANSWER_NODE_RANK", "NODE_RANK"):
        raw = environ.get(name)
        if raw is None or raw == "":
            continue
        try:
            node_rank = int(raw)
        except ValueError:
            raise AnswerEvalError(f"{name} must be an integer, not {raw!r}") from None
        if not 0 <= node_rank < nnodes:
            raise AnswerEvalError(
                f"{name} is {node_rank}, outside the [0, {nnodes}) this config declares"
            )
        return node_rank
    raise AnswerEvalError(
        "a multi-node Answer config needs the rank of this node: set NODE_RANK "
        "or SGLANG_PPU_ANSWER_NODE_RANK"
    )


def _distributed_init_addr(environ: dict[str, str]) -> str:
    override = environ.get("SGLANG_PPU_ANSWER_DIST_INIT_ADDR")
    if override:
        host, _, port = override.rpartition(":")
        if not host or not port:
            raise AnswerEvalError(
                "SGLANG_PPU_ANSWER_DIST_INIT_ADDR must read host:port, not "
                f"{override!r}"
            )
    else:
        host = environ.get("MASTER_ADDR") or ""
        port = environ.get("MASTER_PORT") or ""
        if not host or not port:
            raise AnswerEvalError(
                "a multi-node Answer config needs the rendezvous address: set "
                "MASTER_ADDR and MASTER_PORT, or "
                "SGLANG_PPU_ANSWER_DIST_INIT_ADDR"
            )
    try:
        port_number = int(port)
    except ValueError:
        raise AnswerEvalError(
            f"the rendezvous port must be an integer, not {port!r}"
        ) from None
    if not 1 <= port_number <= 65535:
        raise AnswerEvalError(f"the rendezvous port {port_number} is out of range")
    return f"{host}:{port_number}"


def build_answer_server_args(
    config: dict[str, Any], *, distributed: dict[str, Any] | None = None
) -> list[str]:
    """Translate a validated Answer config into SGLang server CLI arguments.

    ``distributed`` is the result of :func:`resolve_distributed_runtime`, and it
    is required exactly when the config declares more than one node: a caller
    that forgot it would otherwise launch every pod as an independent rank 0
    that tries to fit the whole checkpoint on its own eight devices.
    """

    parameters = config["server"]["parameters"]
    nnodes = answer_node_count(config)
    if distributed is None and nnodes > 1:
        raise AnswerEvalError(
            "a multi-node Answer config must be launched with the resolved "
            "rendezvous from resolve_distributed_runtime"
        )
    if distributed is not None and nnodes == 1:
        raise AnswerEvalError(
            "a single-node Answer config must not be handed a rendezvous"
        )
    if distributed is not None and distributed["nnodes"] != nnodes:
        raise AnswerEvalError(
            f"the rendezvous spans {distributed['nnodes']} nodes and the config "
            f"declares {nnodes}"
        )
    args = []
    if parameters["trust_remote_code"]:
        args.append("--trust-remote-code")
    for name in SERVER_PARAMETER_CLI_ORDER:
        # An absent optional parameter and an explicit null are the same
        # instruction -- leave the flag off -- so a config that stays on pure
        # tensor parallelism produces the command line it produced before
        # pp_size existed, rather than an explicit --pp-size 1.
        #
        # For quantization that is the difference between letting the checkpoint
        # declare its own format and overriding that declaration; for
        # attention_backend it is the difference between a unified backend and
        # the prefill/decode pair that replaces it.
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


def answer_server_environment(config: dict[str, Any]) -> dict[str, str]:
    """The environment a config asks to be set around the server.

    Empty for a config that names none, which is every entry that predates this
    block, so those keep launching with the inherited environment untouched.
    Numbers are rendered rather than passed through because an environment is
    strings, and the internal cases state these as JSON numbers.
    """

    return {name: str(value) for name, value in config["server"].get("env", {}).items()}


def answer_expected_hardware(config: dict[str, Any]) -> str:
    """Render the reviewed hardware contract for provenance."""

    hardware = config["hardware"]
    memory_gib = hardware["memory_gib_per_device"]
    if isinstance(memory_gib, float) and memory_gib.is_integer():
        memory_gib = int(memory_gib)
    devices_per_node = len(hardware["visible_devices"])
    nnodes = hardware.get("nnodes", 1)
    # A single node keeps the string it has always had, so the contract recorded
    # by the existing entries does not move; a multi-node contract has to state
    # both dimensions, because 4nx8 and 1nx32 are not the same machine.
    topology = (
        f"{devices_per_node}x{memory_gib}g"
        if nnodes == 1
        else f"{nnodes}nx{devices_per_node}x{memory_gib}g"
    )
    return f"{hardware['generation']}-{topology}"


def normalize_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")
    return re.sub(r"\s+", " ", normalized).strip()


def compact_text(text: str) -> str:
    return "".join(char for char in normalize_answer(text) if not char.isspace())


def parse_thinking_blocks(raw_response: str) -> tuple[str, list[dict[str, Any]]]:
    """Remove non-nested ``<think>`` blocks with a small deterministic parser."""

    open_tag = "<think>"
    close_tag = "</think>"
    cursor = 0
    inside = False
    final_parts: list[str] = []
    malformed: list[str] = []

    while cursor < len(raw_response):
        if raw_response.startswith(open_tag, cursor):
            if inside:
                malformed.append("nested_open_tag")
            inside = True
            cursor += len(open_tag)
            continue
        if raw_response.startswith(close_tag, cursor):
            if not inside:
                malformed.append("closing_tag_without_open")
            else:
                inside = False
            cursor += len(close_tag)
            continue
        if not inside:
            final_parts.append(raw_response[cursor])
        cursor += 1

    if inside:
        malformed.append("unclosed_thinking_block")

    final_answer = "".join(final_parts).strip()
    findings = []
    if malformed:
        findings.append(
            _finding(
                "malformed_thinking_block",
                "hard_fail",
                observed=sorted(set(malformed)),
                threshold="balanced, non-nested <think> blocks",
            )
        )
    if not normalize_answer(final_answer):
        findings.append(
            _finding(
                "empty_final_answer",
                "hard_fail",
                observed=0,
                threshold="at least one non-whitespace code point",
            )
        )
    return final_answer, findings


def _finding(
    reason_code: str,
    action: str,
    *,
    observed: Any,
    threshold: Any,
    evidence: str | None = None,
    rule_description: str | None = None,
    severity: str = "critical",
) -> dict[str, Any]:
    # severity separates infrastructure/contract breakage ("critical": bad
    # serving stack, garbled output, wrong finish_reason) from model-capability
    # misses ("model": a fact rule the checkpoint simply gets wrong).  Engine
    # internal checks default to "critical"; fact-rule findings are re-stamped
    # with the rule's severity in evaluate_case.  The per-case verdict ignores
    # severity; only the run-level regression gate reads it.
    result = {
        "reason_code": reason_code,
        "action": action,
        "severity": severity,
        "observed": observed,
        "threshold": threshold,
    }
    if evidence:
        result["evidence"] = evidence[:160]
    if rule_description:
        result["rule_description"] = rule_description
    return result


def _unicode_findings(text: str, profile: dict[str, Any], surface: str) -> list[dict]:
    settings = profile["unicode"]
    invalid = []
    abnormal = []
    non_whitespace = 0
    for index, char in enumerate(text):
        if not char.isspace():
            non_whitespace += 1
        category = unicodedata.category(char)
        if (
            char == settings["replacement_character"]
            or category == "Cs"
            or (category == "Cc" and char not in "\t\n\r")
        ):
            invalid.append((index, f"U+{ord(char):04X}"))
        if category in settings["abnormal_categories"]:
            abnormal.append((index, f"U+{ord(char):04X}"))

    findings = []
    if invalid:
        findings.append(
            _finding(
                "invalid_unicode",
                "hard_fail",
                observed={"surface": surface, "count": len(invalid)},
                threshold="zero replacement, surrogate, or disallowed control code points",
                evidence=", ".join(codepoint for _, codepoint in invalid[:8]),
            )
        )
    abnormal_ratio = len(abnormal) / max(non_whitespace, 1)
    if (
        len(abnormal) >= settings["abnormal_min_count"]
        and abnormal_ratio >= settings["abnormal_ratio"]
    ):
        findings.append(
            _finding(
                "abnormal_codepoint_ratio",
                "hard_fail",
                observed={
                    "surface": surface,
                    "count": len(abnormal),
                    "ratio": round(abnormal_ratio, 6),
                },
                threshold={
                    "min_count": settings["abnormal_min_count"],
                    "min_ratio": settings["abnormal_ratio"],
                },
                evidence=", ".join(codepoint for _, codepoint in abnormal[:8]),
            )
        )
    return findings


def _same_codepoint_run(text: str, threshold: int) -> tuple[int, str]:
    best_count = 0
    best_char = ""
    previous = ""
    current = 0
    for char in text:
        if char.isspace():
            previous = ""
            current = 0
        elif char == previous:
            current += 1
        else:
            previous = char
            current = 1
        if current > best_count:
            best_count = current
            best_char = char
        if best_count >= threshold:
            break
    return best_count, best_char


def _periodic_fragment(text: str, settings: dict[str, Any]) -> dict[str, Any] | None:
    compact = compact_text(text)
    min_unit = settings["periodic_unit_min"]
    max_unit = settings["periodic_unit_max"]
    min_repeats = settings["periodic_min_repeats"]
    min_span = settings["periodic_min_span"]
    for start in range(len(compact)):
        for unit_length in range(min_unit, max_unit + 1):
            if start + unit_length * min_repeats > len(compact):
                break
            unit = compact[start : start + unit_length]
            repeats = 1
            cursor = start + unit_length
            while compact.startswith(unit, cursor):
                repeats += 1
                cursor += unit_length
            span = repeats * unit_length
            if repeats >= min_repeats and span >= min_span:
                return {
                    "unit": unit[:32],
                    "unit_chars": unit_length,
                    "repeats": repeats,
                    "span_chars": span,
                }
    return None


def _sentence_repetition(text: str, settings: dict[str, Any]) -> tuple[int, str]:
    # Preserve line boundaries here.  ``normalize_answer`` intentionally
    # collapses all whitespace for fact matching, but doing that before this
    # split would make three identical newline-delimited answers look like one
    # long sentence.
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")
    segments = re.split(r"[。！？!?\n]+", normalized)
    cleaned = [segment.strip(" ,，、；;：:\"'“”‘’") for segment in segments]
    cleaned = [
        segment
        for segment in cleaned
        if len(compact_text(segment)) >= settings["sentence_min_chars"]
    ]
    if not cleaned:
        return 0, ""
    sentence, count = Counter(cleaned).most_common(1)[0]
    return count, sentence


def _ngram_set(text: str, size: int) -> set[str]:
    compact = compact_text(text)
    return {compact[i : i + size] for i in range(max(0, len(compact) - size + 1))}


def _repeated_ngram_coverage(text: str, settings: dict[str, Any]) -> float:
    compact = compact_text(text)
    size = settings["ngram_size"]
    if len(compact) < settings["ngram_min_text_chars"] or len(compact) < size:
        return 0.0
    grams = [compact[index : index + size] for index in range(len(compact) - size + 1)]
    frequent = {
        gram
        for gram, count in Counter(grams).items()
        if count >= settings["ngram_min_occurrences"]
    }
    covered: set[int] = set()
    for index, gram in enumerate(grams):
        if gram in frequent:
            covered.update(range(index, index + size))
    return len(covered) / len(compact)


def _mojibake_occurrences(text: str, markers: Iterable[str]) -> tuple[int, list[str]]:
    hits: list[str] = []
    for marker in markers:
        hits.extend([marker] * text.count(marker))
    return len(hits), hits


def _cjk_count(text: str) -> int:
    return sum(
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
        for char in text
    )


def _reversible_latin1_utf8_mojibake(text: str) -> list[dict[str, Any]]:
    """Find high-confidence UTF-8 bytes that were decoded as Latin-1.

    A marker list catches common Western mojibake, but Chinese UTF-8 decoded as
    Latin-1 often appears as ``ä½ å¥½`` and contains none of those markers.
    Only flag a segment when a strict Latin-1 -> UTF-8 round trip succeeds and
    materially increases the number of CJK code points.  This keeps ordinary
    Latin-1 words such as ``café`` out of the hard-fail path.
    """

    occurrences = []
    for match in re.finditer(r"[\x00-\xff]{3,}", text):
        segment = match.group(0)
        try:
            repaired = segment.encode("latin-1").decode("utf-8", errors="strict")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        before = _cjk_count(segment)
        after = _cjk_count(repaired)
        if after > before and after >= 1:
            occurrences.append(
                {
                    "offset": match.start(),
                    "encoded_chars": len(segment),
                    "repaired_cjk_chars": after,
                }
            )
    return occurrences


def degradation_findings(
    raw_response: str, final_answer: str, profile: dict[str, Any]
) -> list[dict[str, Any]]:
    findings = []
    findings.extend(_unicode_findings(raw_response, profile, "raw_response"))
    if final_answer != raw_response:
        findings.extend(_unicode_findings(final_answer, profile, "final_answer"))

    settings = profile["repetition"]
    for surface, text in (
        ("raw_response", raw_response),
        ("final_answer", final_answer),
    ):
        run_count, run_char = _same_codepoint_run(
            normalize_answer(text), settings["same_codepoint_run"]
        )
        if run_count >= settings["same_codepoint_run"]:
            findings.append(
                _finding(
                    "same_codepoint_run",
                    "hard_fail",
                    observed={
                        "surface": surface,
                        "codepoint": f"U+{ord(run_char):04X}",
                        "run": run_count,
                    },
                    threshold=settings["same_codepoint_run"],
                )
            )

        periodic = _periodic_fragment(text, settings)
        if periodic:
            findings.append(
                _finding(
                    "periodic_fragment_repeat",
                    "hard_fail",
                    observed={"surface": surface, **periodic},
                    threshold={
                        "min_repeats": settings["periodic_min_repeats"],
                        "min_span": settings["periodic_min_span"],
                    },
                    evidence=periodic["unit"],
                )
            )

        sentence_count, sentence = _sentence_repetition(text, settings)
        if sentence_count >= settings["sentence_min_repeats"]:
            findings.append(
                _finding(
                    "exact_sentence_repeat",
                    "hard_fail",
                    observed={"surface": surface, "repeats": sentence_count},
                    threshold=settings["sentence_min_repeats"],
                    evidence=sentence,
                )
            )

        coverage = _repeated_ngram_coverage(text, settings)
        if coverage >= settings["ngram_hard_fail_coverage"]:
            findings.append(
                _finding(
                    "repeated_4gram_coverage",
                    "hard_fail",
                    observed={"surface": surface, "coverage": round(coverage, 6)},
                    threshold=settings["ngram_hard_fail_coverage"],
                )
            )
        elif coverage >= settings["ngram_suspect_coverage"]:
            findings.append(
                _finding(
                    "suspicious_4gram_coverage",
                    "suspect",
                    observed={"surface": surface, "coverage": round(coverage, 6)},
                    threshold=settings["ngram_suspect_coverage"],
                )
            )

    mojibake_count, mojibake_hits = _mojibake_occurrences(
        raw_response, profile["mojibake"]["markers"]
    )
    if mojibake_count >= profile["mojibake"]["min_occurrences"]:
        findings.append(
            _finding(
                "mojibake_marker",
                "hard_fail",
                observed={"count": mojibake_count},
                threshold=profile["mojibake"]["min_occurrences"],
                evidence=" ".join(mojibake_hits[:4]),
            )
        )
    reversible_mojibake = _reversible_latin1_utf8_mojibake(raw_response)
    if reversible_mojibake:
        findings.append(
            _finding(
                "reversible_utf8_mojibake",
                "hard_fail",
                observed={"count": len(reversible_mojibake)},
                threshold="zero strict Latin-1 to UTF-8 repairs with a CJK gain",
            )
        )
    return _deduplicate_findings(findings)


def _deduplicate_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = []
    seen = set()
    for finding in findings:
        key = (finding["reason_code"], json.dumps(finding["observed"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    return unique


_CLAUSE_BOUNDARIES = "，,。；;！？!?\n"
_NEGATION_BEFORE_ALIAS = re.compile(
    r"(?:不是|并不是|并非是|并非|非是|不为|不应是|不属于|不包括|不含|不与|没与|未与)"
    r"[^，,。；;！？!?\n]{0,12}$"
)
_NEGATION_AFTER_ALIAS = re.compile(
    r"^(?:并不是|不是|并非是|并非|非是|不为|不应是|不属于|不包括|不含|不与|没与|未与)"
)


def _is_negated_span(text: str, start: int, end: int) -> bool:
    clause_start = max((text.rfind(mark, 0, start) for mark in _CLAUSE_BOUNDARIES)) + 1
    clause_ends = [text.find(mark, end) for mark in _CLAUSE_BOUNDARIES]
    clause_ends = [position for position in clause_ends if position >= 0]
    clause_end = min(clause_ends) if clause_ends else len(text)
    before = text[clause_start:start]
    after = text[end:clause_end]
    return bool(
        _NEGATION_BEFORE_ALIAS.search(before) or _NEGATION_AFTER_ALIAS.search(after)
    )


def _contains_alias(text: str, aliases: Iterable[str]) -> tuple[bool, str | None]:
    folded = normalize_answer(text).casefold()
    for alias in aliases:
        normalized_alias = normalize_answer(alias).casefold()
        cursor = 0
        while True:
            position = folded.find(normalized_alias, cursor)
            if position < 0:
                break
            if not _is_negated_span(folded, position, position + len(normalized_alias)):
                return True, alias
            cursor = position + len(normalized_alias)
    return False, None


def _parse_chinese_integer(token: str) -> int:
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    units = {"十": 10, "百": 100, "千": 1000, "万": 10_000, "亿": 100_000_000}
    if not any(char in units for char in token):
        return int("".join(str(digits[char]) for char in token))

    total = 0
    section = 0
    number = 0
    for char in token:
        if char in digits:
            number = digits[char]
            continue
        unit = units[char]
        if unit < 10_000:
            section += (number or 1) * unit
        else:
            total += (section + number) * unit
            section = 0
        number = 0
    return total + section + number


def _extract_numbers(text: str) -> list[float]:
    values = []
    for match in re.finditer(r"(?<![\d.])[-+]?\d[\d,]*(?:\.\d+)?(?![\d.])", text):
        token = match.group(0).replace(",", "")
        try:
            values.append(float(token))
        except ValueError:
            continue
    for token in re.findall(r"[零〇一二两三四五六七八九十百千万亿]+", text):
        values.append(float(_parse_chinese_integer(token)))
    return values


def _number_mentions(text: str) -> list[tuple[float, bool, int, int]]:
    """Return numeric values, negation state, and source spans."""

    mentions = []
    for match in re.finditer(r"(?<![\d.])[-+]?\d[\d,]*(?:\.\d+)?(?![\d.])", text):
        try:
            value = float(match.group(0).replace(",", ""))
        except ValueError:
            continue
        mentions.append(
            (
                value,
                _is_negated_span(text, match.start(), match.end()),
                match.start(),
                match.end(),
            )
        )
    for match in re.finditer(r"[零〇一二两三四五六七八九十百千万亿]+", text):
        mentions.append(
            (
                float(_parse_chinese_integer(match.group(0))),
                _is_negated_span(text, match.start(), match.end()),
                match.start(),
                match.end(),
            )
        )
    return mentions


_ASSERTED_NUMBER_PREFIX = re.compile(
    r"(?:(?:正确答案|答案|最终答案|最终结果|实际答案|实际结果)"
    r"(?:也)?(?:应当|应该|应)?(?:是|为|等于)|"
    r"(?:而|但|却)(?:实际|其实)?(?:也)?(?:是|为|等于)|"
    r"也(?:是|为|等于))\s*$"
)


def _is_asserted_number(text: str, start: int) -> bool:
    """Detect a nearby assertion that makes a second value contradictory."""

    clause_start = max((text.rfind(mark, 0, start) for mark in _CLAUSE_BOUNDARIES)) + 1
    return bool(_ASSERTED_NUMBER_PREFIX.search(text[clause_start:start]))


def _numeric_rule_matches(
    text: str,
    mentions: list[tuple[float, bool, int, int]],
    target: float,
    tolerance: float,
) -> bool:
    has_target = any(
        not negated and math.isclose(value, target, abs_tol=tolerance, rel_tol=0)
        for value, negated, _, _ in mentions
    )
    has_asserted_conflict = any(
        not negated
        and not math.isclose(value, target, abs_tol=tolerance, rel_tol=0)
        and _is_asserted_number(text, start)
        for value, negated, start, _ in mentions
    )
    return has_target and not has_asserted_conflict


_SEQUENCE_LABEL = re.compile(
    r"(?:第\s*)?[0-9零〇一二两三四五六七八九十]+\s*(?:句|位|项|条|个|名)?"
)


def _sequence_gap_is_decorative(gap: str) -> bool:
    """Allow numbering/labels between expected items, but no extra content."""

    undecorated = unicodedata.normalize("NFKC", gap)
    undecorated = _SEQUENCE_LABEL.sub("", undecorated).replace("姓", "")
    return all(
        char.isspace() or unicodedata.category(char).startswith(("P", "Z"))
        for char in undecorated
    )


def _sequence_has_adjacent_extra_item(
    text: str, start: int, end: int, max_item_chars: int
) -> bool:
    """Reject short list items immediately before or after the expected run.

    Prefix and suffix prose is allowed, but an adjacent delimiter plus a short
    alphanumeric token is another enumerated item.  This closes the otherwise
    invisible ``刘、赵...陈`` and ``赵...陈、刘`` cases without rejecting
    introductions such as ``前十个姓是：`` or a sentence after the list.
    """

    before = re.search(r"([^、,，。；;！？!?：:\n]+)[、,，]\s*$", text[:start])
    after = re.match(r"\s*[、,，]\s*([^、,，。；;！？!?：:\n]+)", text[end:])
    candidates = [match.group(1) for match in (before, after) if match]
    for candidate in candidates:
        cleaned = _SEQUENCE_LABEL.sub("", candidate)
        cleaned = "".join(char for char in cleaned if char.isalnum())
        if 0 < len(cleaned) <= max_item_chars:
            return True
    return False


# How a bare number in an answer may be read by a probability rule, as a divisor
# that maps it onto a probability.  The divisor doubles as the largest value the
# unit admits, because a well-formed magnitude in either unit lands in [0, 1]
# once divided: a bare number above it is not a quantity in that unit at all and
# must stay out of the candidate set, the way an item count or an intermediate
# step in a derivation does.
_BARE_PROBABILITY_UNITS = {"probability": 1.0, "percent": 100.0}


def _bare_probability_units(rule: dict[str, Any]) -> list[str]:
    """Return the readings a probability rule admits for a bare number.

    A bare number carries no unit, so which readings are correct depends on what
    the prompt asked for; the reviewed case declares them rather than the
    evaluator guessing.  The default keeps the probability reading alone, so a
    prompt that asks for a percentage has to say so: inferring ``percent`` from a
    value above 1 would admit a rounded integer such as ``14`` against a target
    of 1/7 whenever the tolerance is loose enough, which is a false pass.
    """

    units = rule.get("bare_number_units", ["probability"])
    if not isinstance(units, list) or not units:
        raise AnswerEvalError("bare_number_units must be a non-empty list")
    unsupported = sorted(set(units) - set(_BARE_PROBABILITY_UNITS))
    if unsupported:
        raise AnswerEvalError(f"unsupported bare number units: {unsupported}")
    return units


def _evaluate_rule(rule: dict[str, Any], text: str) -> dict[str, Any] | None:
    rule_type = rule["type"]
    description = rule.get("description")
    if rule_type == "contains_all":
        missing = []
        for aliases in rule["items"]:
            found, _ = _contains_alias(text, aliases)
            if not found:
                missing.append(aliases)
        if missing:
            return _finding(
                "fact_rule_failed",
                "hard_fail",
                observed={"rule": rule_type, "missing": missing},
                threshold="all fact groups present",
                rule_description=description,
            )
    elif rule_type == "contains_any":
        found, _ = _contains_alias(text, rule["items"])
        if not found:
            return _finding(
                "fact_rule_failed",
                "hard_fail",
                observed={"rule": rule_type, "missing_all_of": rule["items"]},
                threshold="at least one term present",
                rule_description=description,
            )
    elif rule_type == "ordered_items":
        folded = text.casefold()
        cursor = 0
        missing_or_unordered = []
        matches = []
        for aliases in rule["items"]:
            positioned_aliases = [
                (folded.find(normalize_answer(alias).casefold(), cursor), alias)
                for alias in aliases
            ]
            positioned_aliases = [
                (position, alias)
                for position, alias in positioned_aliases
                if position >= 0
            ]
            if not positioned_aliases:
                missing_or_unordered.append(aliases)
                continue
            position = min(position for position, _ in positioned_aliases)
            matched_alias = max(
                (
                    normalize_answer(alias).casefold()
                    for candidate_position, alias in positioned_aliases
                    if candidate_position == position
                ),
                key=len,
            )
            matches.append((position, matched_alias))
            cursor = position + len(matched_alias)
        if missing_or_unordered:
            return _finding(
                "fact_rule_failed",
                "hard_fail",
                observed={
                    "rule": rule_type,
                    "missing_or_unordered": missing_or_unordered,
                },
                threshold="all fact groups present in order",
                rule_description=description,
            )
        if rule.get("exact_sequence"):
            gaps = [
                folded[
                    matches[index][0] + len(matches[index][1]) : matches[index + 1][0]
                ]
                for index in range(len(matches) - 1)
            ]
            max_item_chars = max(
                len(normalize_answer(alias))
                for aliases in rule["items"]
                for alias in aliases
            )
            has_extra_edge_item = rule.get(
                "strict_list_edges", False
            ) and _sequence_has_adjacent_extra_item(
                folded, matches[0][0], cursor, max_item_chars
            )
            if (
                any(not _sequence_gap_is_decorative(gap) for gap in gaps)
                or has_extra_edge_item
            ):
                return _finding(
                    "fact_rule_failed",
                    "hard_fail",
                    observed={
                        "rule": rule_type,
                        "unexpected_content_between_or_adjacent_to_items": True,
                    },
                    threshold=(
                        "only punctuation, whitespace, numbering, or item labels "
                        "within the exact ordered sequence"
                    ),
                    rule_description=description,
                )
    elif rule_type == "exact_entity_set":
        observed = set()
        for canonical, aliases in rule["items"].items():
            found, _ = _contains_alias(text, aliases)
            if found:
                observed.add(canonical)
        for canonical, aliases in rule.get("forbidden_items", {}).items():
            found, _ = _contains_alias(text, aliases)
            if found:
                observed.add(canonical)
        expected = set(rule["expected"])
        if observed != expected:
            return _finding(
                "fact_rule_failed",
                "hard_fail",
                observed={
                    "rule": rule_type,
                    "missing": sorted(expected - observed),
                    "unexpected": sorted(observed - expected),
                },
                threshold={"exact_set": sorted(expected)},
                rule_description=description,
            )
    elif rule_type == "number":
        mentions = _number_mentions(text)
        values = [value for value, _, _, _ in mentions]
        target = float(rule["target"])
        tolerance = float(rule.get("tolerance", 0))
        if not _numeric_rule_matches(text, mentions, target, tolerance):
            return _finding(
                "fact_rule_failed",
                "hard_fail",
                observed={"rule": rule_type, "numbers": values},
                threshold={"target": target, "absolute_tolerance": tolerance},
                rule_description=description,
            )
    elif rule_type == "probability":
        target = float(rule["target"])
        tolerance = float(rule["tolerance"])
        units = _bare_probability_units(rule)
        mentions = []
        compound_spans = []
        for match in re.finditer(r"(\d+)\s*/\s*(\d+)", text):
            numerator, denominator = match.groups()
            if int(denominator):
                compound_spans.append(match.span())
                mentions.append(
                    (
                        int(numerator) / int(denominator),
                        _is_negated_span(text, match.start(), match.end()),
                        match.start(),
                        match.end(),
                    )
                )
        for match in re.finditer(r"([-+]?\d+(?:\.\d+)?)\s*%", text):
            compound_spans.append(match.span())
            mentions.append(
                (
                    float(match.group(1)) / 100,
                    _is_negated_span(text, match.start(), match.end()),
                    match.start(),
                    match.end(),
                )
            )
        for value, negated, start, end in _number_mentions(text):
            # A bare match that begins inside a fraction or a percentage is one of
            # its digits. Containment alone would miss it: an integer directly
            # before a comma absorbs the comma as a thousands separator, so the
            # denominator of ``1/7,`` reaches one character past the fraction.
            overlaps_compound = any(
                compound_start <= start < compound_end
                for compound_start, compound_end in compound_spans
            )
            if overlaps_compound:
                continue
            readings = [
                value / _BARE_PROBABILITY_UNITS[unit]
                for unit in units
                if 0 <= value <= _BARE_PROBABILITY_UNITS[unit]
            ]
            # The readings of one bare number are alternative interpretations of
            # a single claim, not several claims, so keep the one that matches
            # when there is one. Feeding all of them to the conflict check would
            # let a value contradict itself: ``0.1429`` matches as a probability
            # while its percentage reading does not, and the asserting prefix
            # would then turn a correct answer into a contradiction.
            matching = [
                reading
                for reading in readings
                if math.isclose(reading, target, abs_tol=tolerance, rel_tol=0)
            ]
            for reading in matching or readings:
                mentions.append((reading, negated, start, end))
        if not _numeric_rule_matches(text, mentions, target, tolerance):
            return _finding(
                "fact_rule_failed",
                "hard_fail",
                observed={
                    "rule": rule_type,
                    "probability_candidates": [value for value, _, _, _ in mentions],
                },
                threshold={"target": target, "absolute_tolerance": tolerance},
                rule_description=description,
            )
    elif rule_type == "length":
        observed = len(compact_text(text))
        minimum = int(rule["min_chars"])
        maximum = int(rule["max_chars"])
        if not minimum <= observed <= maximum:
            return _finding(
                "fact_rule_failed",
                "hard_fail",
                observed={"rule": rule_type, "non_whitespace_chars": observed},
                threshold={"min_chars": minimum, "max_chars": maximum},
                rule_description=description,
            )
    else:
        raise AnswerEvalError(f"unsupported rule type: {rule_type}")
    return None


def _sample_id(case_id: str, case_revision: int, normalized_answer: str) -> str:
    material = f"{case_id}\0{case_revision}\0{normalized_answer}"
    return _text_digest(material)


def _set_verdict(result: dict[str, Any]) -> None:
    failed = any(item["action"] == "hard_fail" for item in result["findings"])
    result["verdict"] = "failed" if failed else "passed"
    if failed:
        result.setdefault("failure_class", "candidate_failed")
    else:
        result.pop("failure_class", None)


def evaluate_case(
    case: dict[str, Any],
    raw_response: str,
    finish_reason: str | None,
    profile: dict[str, Any],
    case_revision: int,
) -> dict[str, Any]:
    if not isinstance(raw_response, str):
        raise AnswerEvalError("raw_response must be a string")
    final_answer, findings = parse_thinking_blocks(raw_response)
    normalized = normalize_answer(final_answer)
    findings.extend(degradation_findings(raw_response, final_answer, profile))
    if finish_reason != "stop":
        reason_code = (
            "finish_reason_length"
            if finish_reason == "length"
            else "finish_reason_incomplete"
        )
        findings.append(
            _finding(
                reason_code,
                "hard_fail",
                observed=finish_reason,
                threshold="finish_reason must be stop",
            )
        )
    for rule in case["rules"]:
        finding = _evaluate_rule(rule, normalized)
        if finding:
            # A fact rule reflects model capability by default; a rule may opt
            # into "critical" when its miss would instead signal a broken
            # contract that must always gate.
            finding["severity"] = rule.get("severity", "model")
            findings.append(finding)

    result = {
        "case_id": case["id"],
        "kind": case["kind"],
        "sample_id": _sample_id(case["id"], case_revision, normalized),
        "answer_sha256": _text_digest(normalized),
        "raw_response": raw_response,
        "final_answer": final_answer,
        "normalized_answer": normalized,
        "finish_reason": finish_reason,
        "findings": _deduplicate_findings(findings),
        "judge": {"mode": "disabled", "status": "deferred"},
        "semantic_coverage": (
            "deterministic_facts"
            if case["kind"] == "objective"
            else "hard_constraints_only"
        ),
    }
    _set_verdict(result)
    return result


def apply_cross_case_checks(
    results: list[dict[str, Any]], profile: dict[str, Any]
) -> None:
    settings = profile["repetition"]
    ngram_size = settings["ngram_size"]
    for left_index, left in enumerate(results):
        if not left.get("normalized_answer"):
            continue
        left_compact = compact_text(left["normalized_answer"])
        for right in results[left_index + 1 :]:
            if not right.get("normalized_answer"):
                continue
            right_compact = compact_text(right["normalized_answer"])
            reason_code = None
            action = None
            observed: dict[str, Any] = {}
            if (
                left_compact == right_compact
                and len(left_compact) >= settings["cross_case_exact_min_chars"]
            ):
                reason_code = "cross_case_duplicate"
                action = "hard_fail"
                observed = {"exact": True, "chars": len(left_compact)}
            elif (
                min(len(left_compact), len(right_compact))
                >= settings["cross_case_near_min_chars"]
            ):
                left_grams = _ngram_set(left_compact, ngram_size)
                right_grams = _ngram_set(right_compact, ngram_size)
                union = left_grams | right_grams
                jaccard = len(left_grams & right_grams) / len(union) if union else 1.0
                observed = {"jaccard": round(jaccard, 6)}
                if jaccard >= settings["cross_case_hard_fail_jaccard"]:
                    reason_code = "cross_case_duplicate"
                    action = "hard_fail"
                elif jaccard >= settings["cross_case_suspect_jaccard"]:
                    reason_code = "cross_case_near_duplicate"
                    action = "suspect"
            if reason_code:
                for current, peer in ((left, right), (right, left)):
                    current["findings"].append(
                        _finding(
                            reason_code,
                            action,
                            observed={"peer_case_id": peer["case_id"], **observed},
                            threshold=(
                                settings["cross_case_hard_fail_jaccard"]
                                if action == "hard_fail"
                                else settings["cross_case_suspect_jaccard"]
                            ),
                        )
                    )
    for result in results:
        result["findings"] = _deduplicate_findings(result["findings"])
        _set_verdict(result)


def _normal_sample_selected(sample_id: str, profile: dict[str, Any]) -> bool:
    selection = profile["selection"]
    digest = hashlib.sha256(f"{selection['seed']}:{sample_id}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return bucket < int(selection["normal_sample_percent"])


def validate_annotation_record(record: dict[str, Any]) -> None:
    """Validate annotation lifecycle invariants not expressible in JSON Schema."""

    if record.get("schema_version") != "ppu-answer-annotation/v1":
        raise AnswerEvalError("unsupported annotation schema")
    answer_fields = [
        key
        for key in ("candidate_answer", "candidate_answer_ref")
        if isinstance(record.get(key), str) and record[key]
    ]
    if len(answer_fields) != 1:
        raise AnswerEvalError(
            "exactly one non-empty candidate answer field is required"
        )

    annotation = record.get("annotation")
    if not isinstance(annotation, dict):
        raise AnswerEvalError("annotation must be an object")
    status = annotation.get("status")
    reviews = annotation.get("reviews")
    adjudication = annotation.get("adjudication")
    partition = annotation.get("partition")
    if not isinstance(reviews, list) or len(reviews) > 2:
        raise AnswerEvalError("annotation reviews must contain at most two entries")
    reviewers = [
        review.get("reviewer") for review in reviews if isinstance(review, dict)
    ]
    if len(reviewers) != len(reviews) or not all(
        isinstance(reviewer, str) and reviewer.strip() for reviewer in reviewers
    ):
        raise AnswerEvalError("every review must identify its reviewer")
    canonical_reviewers = {reviewer.strip().casefold() for reviewer in reviewers}
    if len(reviewers) != len(canonical_reviewers):
        raise AnswerEvalError("the two reviews must come from different reviewers")

    if status == "pending":
        valid_state = not reviews and adjudication is None and partition is None
    elif status == "reviewed":
        valid_state = len(reviews) == 2 and adjudication is None and partition is None
    elif status == "adjudicated":
        valid_state = (
            len(reviews) == 2
            and isinstance(adjudication, dict)
            and partition in {"calibration", "holdout"}
        )
    else:
        raise AnswerEvalError("unsupported annotation status")
    if not valid_state:
        raise AnswerEvalError(f"invalid annotation lifecycle state: {status}")


def build_label_candidates(
    results: list[dict[str, Any]],
    profile: dict[str, Any],
    provenance: dict[str, Any],
    *,
    case_revision: int,
) -> list[dict[str, Any]]:
    candidates = []
    for result in results:
        # Request/infrastructure failures do not contain an answer that a human
        # can label and their null identifiers would violate the annotation
        # schema.  Keep them in the run report only.
        if not result.get("sample_id") or not result.get("answer_sha256"):
            continue
        # An empty final answer is a graded hard failure (``empty_final_answer``)
        # and stays in the run report, but there is no answer text for a human to
        # review and a null ``candidate_answer`` would fail the annotation
        # invariant below, so it is excluded from the label set for the same
        # reason as the infrastructure failures above.
        if not result.get("final_answer"):
            continue
        reasons = sorted(
            {
                finding["reason_code"]
                for finding in result["findings"]
                if finding["action"] in ("hard_fail", "suspect")
            }
        )
        if result["kind"] == "open_ended":
            reasons.append("open_ended_pending_judge")
        if not reasons and _normal_sample_selected(result["sample_id"], profile):
            reasons.append("stable_normal_sample")
        if not reasons:
            continue
        candidate = {
            "schema_version": "ppu-answer-annotation/v1",
            "sample_id": result["sample_id"],
            "case_id": result["case_id"],
            "case_revision": case_revision,
            "answer_sha256": result["answer_sha256"],
            "quality_profile": {
                "id": profile["profile_id"],
                "sha256": canonical_digest(profile),
            },
            "selection_reasons": sorted(set(reasons)),
            "source_type": "github_nightly_output",
            "provenance": provenance,
            "candidate_answer": result["final_answer"],
            "annotation": {
                "status": "pending",
                "reviews": [],
                "adjudication": None,
                "partition": None,
            },
        }
        validate_annotation_record(candidate)
        candidates.append(candidate)
    return candidates


def _evaluate_regression_gate(
    results: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    """Decide whether a run regressed, given a per-model baseline.

    The per-case verdict stays strict (any hard_fail -> failed); this gate is a
    separate, run-level judgement layered on top so a known model-capability
    miss no longer reddens the nightly on every run.  A run regresses when:
      * a failure carries a "critical" severity finding (broken serving stack,
        garbled output, wrong finish_reason) -- never tolerated;
      * a "model" failure lands on a case that is NOT in known_failures -- a
        genuinely new miss;
      * the count of passing cases drops below min_score (the floor guard).
    Cases listed in known_failures that now pass are reported as
    unexpected_passes so the baseline can be tightened.  With no baseline the
    allowlist is empty and the floor is unset, so any miss regresses -- exactly
    the old strict behaviour.
    """

    baseline = baseline or {}
    known = set(baseline.get("known_failures", []))
    min_score = baseline.get("min_score")

    critical_failures: list[str] = []
    new_regressions: list[str] = []
    known_failures_hit: list[str] = []
    unexpected_passes: list[str] = []

    for result in results:
        case_id = result["case_id"]
        if result["verdict"] == "failed":
            has_critical = any(
                finding["action"] == "hard_fail"
                and finding.get("severity", "critical") == "critical"
                for finding in result["findings"]
            )
            if has_critical:
                critical_failures.append(case_id)
            elif case_id in known:
                known_failures_hit.append(case_id)
            else:
                new_regressions.append(case_id)
        elif case_id in known:
            unexpected_passes.append(case_id)

    passed = sum(result["verdict"] == "passed" for result in results)
    score_below_baseline = min_score is not None and passed < int(min_score)
    regressed = bool(critical_failures or new_regressions) or score_below_baseline
    return {
        "baseline_applied": bool(baseline),
        "min_score": min_score,
        "passed": passed,
        "regressed": regressed,
        "critical_failures": sorted(critical_failures),
        "new_regressions": sorted(new_regressions),
        "known_failures_hit": sorted(known_failures_hit),
        "unexpected_passes": sorted(unexpected_passes),
        "score_below_baseline": score_below_baseline,
    }


def _reasoning_capture_needed(result: dict[str, Any]) -> bool:
    """Whether a case's chain-of-thought is worth persisting for diagnosis.

    True for the exhaustion signature -- a ``length`` finish_reason or an empty
    final answer -- where the reasoning text is the only surface that explains
    why the model produced nothing usable, letting a reader tell a runaway
    reasoning loop from a genuinely long answer clipped at the token ceiling.
    Every other case keeps just the ``reasoning_sha256`` digest so the artifact
    stays lean and the disclosure surface stays minimal.
    """

    if result.get("finish_reason") == "length":
        return True
    return not normalize_answer(result.get("final_answer") or "")


def build_report(
    dataset: dict[str, Any],
    profile: dict[str, Any],
    responses: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_dataset(dataset)
    validate_profile(profile)
    results = []
    expected_model = provenance.get("served_model_name")
    for case in dataset["cases"]:
        response = responses.get(case["id"])
        if response is None or response.get("error"):
            reason_code = (response or {}).get("reason_code", "request_error")
            message = (response or {}).get("error", "missing response record")
            failure_class = (response or {}).get("failure_class", "server_error")
            if failure_class not in {"server_error", "runner_error"}:
                failure_class = "server_error"
            result = {
                "case_id": case["id"],
                "kind": case["kind"],
                "sample_id": None,
                "answer_sha256": None,
                "raw_response": "",
                "final_answer": "",
                "normalized_answer": "",
                "finish_reason": None,
                "findings": [
                    _finding(
                        reason_code,
                        "hard_fail",
                        observed=message,
                        threshold="successful, valid candidate response",
                    )
                ],
                "judge": {"mode": "disabled", "status": "deferred"},
                "semantic_coverage": "none",
                "verdict": "failed",
                "failure_class": failure_class,
            }
        else:
            result = evaluate_case(
                case,
                response["content"],
                response.get("finish_reason"),
                profile,
                int(dataset["revision"]),
            )
            result["returned_model"] = response.get("model")
            result["usage"] = response.get("usage")
            reasoning_text = response.get("reasoning_content")
            result["reasoning_sha256"] = (
                _text_digest(reasoning_text) if reasoning_text else None
            )
            if reasoning_text and _reasoning_capture_needed(result):
                # Keep the chain-of-thought only for the exhaustion cases whose
                # empty or truncated answer leaves it as the sole diagnostic
                # surface; it rides the same disclosure switch as raw_response
                # (kept in result.raw.json, stripped from the public report).
                result["reasoning_content"] = reasoning_text
            if expected_model and response.get("model") != expected_model:
                result["findings"].append(
                    _finding(
                        "unexpected_model_name",
                        "hard_fail",
                        observed=response.get("model"),
                        threshold=expected_model,
                    )
                )
                _set_verdict(result)
        results.append(result)

    apply_cross_case_checks(results, profile)
    hard_failed = sum(result["verdict"] == "failed" for result in results)
    suspects = sum(
        any(finding["action"] == "suspect" for finding in result["findings"])
        for result in results
    )
    gate = _evaluate_regression_gate(results, baseline)
    report = {
        "schema_version": "ppu-answer-result/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": dataset["dataset_id"],
            "revision": dataset["revision"],
            "sha256": canonical_digest(dataset),
        },
        "quality_profile": {
            "id": profile["profile_id"],
            "sha256": canonical_digest(profile),
        },
        "judge": {"mode": "disabled", "status": "deferred"},
        "provenance": {
            **provenance,
            "python_version": platform.python_version(),
            "unicode_version": unicodedata.unidata_version,
        },
        "summary": {
            "total": len(results),
            "passed": len(results) - hard_failed,
            "failed": hard_failed,
            "suspect": suspects,
            "verdict": "passed" if hard_failed == 0 else "failed",
            "gate": gate,
            "semantic_coverage": "L0/L1 only; LLM-as-Judge deferred",
        },
        "cases": results,
    }
    report["label_candidates"] = build_label_candidates(
        results,
        profile,
        report["provenance"],
        case_revision=int(dataset["revision"]),
    )
    return report


def request_chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    request_suffix: str,
    *,
    timeout_seconds: float,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    """Call an OpenAI-compatible endpoint and decode its body as strict UTF-8."""

    import requests

    if {"model", "messages"} & generation_config.keys():
        raise AnswerEvalError("generation_config must not override model or messages")

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": "Bearer EMPTY"},
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": f"{prompt}\n\n{request_suffix}"}
                ],
                **generation_config,
            },
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise CandidateRequestError("request_error", type(exc).__name__) from exc
    if response.status_code != 200:
        # Carry a slice of the body: the status code alone cannot distinguish a
        # rejected sampling parameter from a chat template that refuses the
        # request, and the server log only records the status line.
        detail = response.text.strip().replace("\n", " ")[:400]
        raise CandidateRequestError(
            "request_error",
            f"candidate endpoint returned HTTP {response.status_code}: {detail}",
        )
    try:
        body = response.content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CandidateRequestError(
            "response_decode_error", "response is not UTF-8"
        ) from exc
    try:
        payload = json.loads(body)
        choice = payload["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise CandidateRequestError(
            "response_schema_error",
            "response does not match the chat completion schema",
        ) from exc
    if not isinstance(content, str):
        raise CandidateRequestError(
            "response_decode_error", "chat completion content must be a string"
        )
    reasoning_content = message.get("reasoning_content")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise CandidateRequestError(
            "response_schema_error", "reasoning_content must be a string or null"
        )
    return {
        "content": content,
        "reasoning_content": reasoning_content,
        "finish_reason": choice.get("finish_reason"),
        "model": payload.get("model"),
        "usage": payload.get("usage"),
    }


def redact_report(report: dict[str, Any]) -> dict[str, Any]:
    public = copy.deepcopy(report)
    expected_model = public.get("provenance", {}).get("served_model_name")
    for result in public["cases"]:
        result.pop("sample_id", None)
        result.pop("answer_sha256", None)
        result.pop("reasoning_sha256", None)
        result.pop("reasoning_content", None)
        result.pop("raw_response", None)
        result.pop("final_answer", None)
        result.pop("normalized_answer", None)
        returned_model = result.get("returned_model")
        if returned_model and returned_model != expected_model:
            result["returned_model"] = "<redacted-unexpected-model-name>"
        for finding in result["findings"]:
            finding.pop("evidence", None)
            finding.pop("observed", None)
    public.pop("label_candidates", None)
    return public


def describe_findings(result: dict[str, Any]) -> str:
    """Say in one line why a case did not pass.

    ``reason_code`` names the class of failure and nothing more, so on its own it
    cannot tell a reader which answer was wrong.  The dataset already carries a
    sentence per rule that states what the answer was measured against, and the
    redaction in :func:`_public_report` keeps it, so it belongs here: it is the
    part that makes a red suite legible without opening the report.
    """

    details = []
    for finding in result["findings"]:
        detail = finding["reason_code"]
        description = finding.get("rule_description")
        if description:
            detail = f"{detail} ({description})"
        details.append(detail)
    # dict.fromkeys rather than sorted(set(...)): a repeated rule says nothing
    # twice, but the order the rules were evaluated in is the order the dataset
    # declares them, which is the order a reader of the case file expects.
    return "; ".join(dict.fromkeys(details)) or "no finding recorded"


def render_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    # Two suites now publish into the same step summary, so the heading has to
    # name the model that produced the table rather than one particular one.
    served_model_name = report.get("provenance", {}).get("served_model_name")
    lines = [
        f"## PPU Answer MVP ({served_model_name or 'unknown model'})",
        "",
        f"- Verdict: **{summary['verdict']}**",
        f"- Cases: {summary['passed']}/{summary['total']} passed",
        f"- Suspect cases: {summary['suspect']}",
        "- Semantic coverage: L0/L1 deterministic checks; LLM-as-Judge deferred",
        "",
    ]
    failing = [result for result in report["cases"] if result["verdict"] == "failed"]
    if failing:
        # The table below carries every case, which means finding the three that
        # broke is a scan of ten rows; this list is the answer to "which question
        # failed, and against what" on its own. The workflows also read it: each
        # bullet becomes one annotation, matched on the leading "- `", which is a
        # prefix no other line in this document has. test_ppu_answer_eval_unit
        # locks that shape, because a workflow cannot.
        lines.append("### Failing cases")
        lines.append("")
        for result in failing:
            lines.append(
                f"- `{result['case_id']}` ({result['kind']}): {describe_findings(result)}"
            )
        lines.append("")
    lines.append("| Case | Kind | Verdict | Findings |")
    lines.append("| --- | --- | --- | --- |")
    for result in report["cases"]:
        reasons = ", ".join(
            sorted({finding["reason_code"] for finding in result["findings"]})
        )
        # The verdict of a failing row is emphasised so the rows that matter are
        # visible at a glance rather than read word by word.
        verdict = result["verdict"]
        if verdict == "failed":
            verdict = f"**{verdict}**"
        lines.append(
            f"| `{result['case_id']}` | {result['kind']} | {verdict} | {reasons or '-'} |"
        )
    return "\n".join(lines) + "\n"


def _log_safe(text: str) -> str:
    """Escape what UTF-8 cannot encode, an unpaired surrogate above all.

    Candidate text arrives straight from the model, so it may contain a lone
    surrogate that no UTF-8 stream accepts.  Escaping it here keeps a malformed
    character from raising while the log block is being assembled.
    """

    return text.encode("utf-8", errors="backslashreplace").decode("utf-8")


def render_candidates(
    report: dict[str, Any], dataset: dict[str, Any] | None = None
) -> str:
    """Render the candidate answers as plain text for the job log.

    ``render_summary`` carries verdicts only, so explaining a red nightly means
    downloading the artifact and decoding JSON.  This block puts the answers in
    the log itself.  It discloses exactly what ``result.raw.json`` does, so
    callers must gate it on the same ``include_raw_outputs`` switch.

    Passing ``dataset`` prepends each prompt, which makes the block readable
    without opening the case file alongside it.
    """

    prompts = (
        {case["id"]: case["prompt"] for case in dataset["cases"]}
        if dataset is not None
        else {}
    )
    lines = ["=== Answer candidates (verbatim, one block per case) ==="]
    for result in report["cases"]:
        lines.append("")
        lines.append(f"[{result['verdict']}] {result['case_id']} ({result['kind']})")
        prompt = prompts.get(result["case_id"])
        if prompt:
            lines.append(f"  prompt : {prompt}")
        answer = result.get("final_answer")
        if answer:
            # Indent continuations so a multi-line poem or essay still reads as
            # one block in a log that carries no other structure.
            lines.append("  answer : " + answer.replace("\n", "\n           "))
        else:
            lines.append("  answer : <empty>")
        for finding in result["findings"]:
            detail = [f"{finding['action']}: {finding['reason_code']}"]
            if finding.get("rule_description"):
                detail.append(f"rule={finding['rule_description']}")
            if finding.get("observed") is not None:
                detail.append(
                    "observed="
                    + json.dumps(
                        finding["observed"], ensure_ascii=False, sort_keys=True
                    )
                )
            lines.append("  " + " | ".join(detail))
    return _log_safe("\n".join(lines) + "\n")


def render_junit(report: dict[str, Any]) -> bytes:
    summary = report["summary"]
    suite = ET.Element(
        "testsuite",
        name="ppu-qwen35-answer",
        tests=str(summary["total"]),
        failures=str(summary["failed"]),
        errors="0",
    )
    for result in report["cases"]:
        testcase = ET.SubElement(
            suite,
            "testcase",
            classname="ppu.answer.qwen3_5_397b_a17b",
            name=result["case_id"],
        )
        if result["verdict"] == "failed":
            reasons = sorted(
                {
                    finding["reason_code"]
                    for finding in result["findings"]
                    if finding["action"] == "hard_fail"
                }
            )
            failure = ET.SubElement(
                testcase,
                "failure",
                type=result.get("failure_class", "candidate_failed"),
                message=", ".join(reasons),
            )
            failure.text = json.dumps(reasons)
        output = ET.SubElement(testcase, "system-out")
        output.text = json.dumps({"semantic_coverage": result.get("semantic_coverage")})
    return ET.tostring(suite, encoding="utf-8", xml_declaration=True)


def write_report_files(
    report: dict[str, Any], output_dir: Path, *, include_raw_outputs: bool = False
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    public_report = redact_report(report)
    (output_dir / "result.json").write_text(
        json.dumps(public_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        render_summary(public_report), encoding="utf-8"
    )
    (output_dir / "junit.xml").write_bytes(render_junit(public_report))
    if include_raw_outputs:
        # ensure_ascii escapes candidate text rather than encoding it, which is
        # what keeps these two writes from raising UnicodeEncodeError when a
        # model emits an unpaired surrogate.  Any JSON reader (jq, json.load)
        # decodes the escapes back into the original characters, so the files
        # stay readable.
        (output_dir / "label_candidates.jsonl").write_text(
            "".join(
                json.dumps(item, ensure_ascii=True) + "\n"
                for item in report["label_candidates"]
            ),
            encoding="utf-8",
        )
        (output_dir / "result.raw.json").write_text(
            json.dumps(report, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    write_quality_trend(report, output_dir)


def checkpoint_config_digest(model_path: str) -> str | None:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return None
    digest = hashlib.sha256()
    with config_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Strings that mean "nobody supplied a version" rather than naming one. Bare
# ``0.0.0`` is what setuptools writes when a ``dynamic`` version has no
# provider; ``0.0.0.dev0`` is this project's configured setuptools-scm
# fallback_version. Both look like versions and compare unequal to every real
# one, which is the same failure mode as an empty base_image_digest: recorded,
# plausible, and wrong. Absent is what they actually are.
_PLACEHOLDER_VERSIONS = frozenset({"0.0.0", "0.0.0.dev0"})


def _real_version(version: object) -> str | None:
    """Return ``version`` only when it names a build, else None."""

    if not isinstance(version, str) or not version:
        return None
    return None if version in _PLACEHOLDER_VERSIONS else version


def _sglang_version() -> str | None:
    """Return the version the imported sglang reports for itself.

    Distribution metadata is not trustworthy for this package: CI imports
    sglang from the checked-out tree while ``importlib.metadata`` resolves
    whichever dist-info comes first on ``sys.path``, which is how run
    33587101038 recorded ``0.0.0`` for a 0.5.13 image.  ``__version__``
    describes the package that was actually imported, so prefer it.

    Prefer it, but do not trust it blindly.  Until the install line composed a
    version for itself, ``__version__`` was itself ``0.0.0`` -- the editable
    install had written that into the metadata this reads through -- so this
    preference was a no-op dressed as a safeguard.  Screening placeholders is
    what makes it one again if the install line ever stops supplying one.
    """

    try:
        import sglang

        version = sglang.__version__
    except Exception:
        return None
    return _real_version(version)


def _installed_package_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("sglang", "sglang-kernel", "torch", "transformers"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    # Only sglang is screened: it is the one built here, and so the only one
    # whose version can come out a placeholder.  The rest arrive as wheels.
    versions["sglang"] = _sglang_version() or _real_version(versions["sglang"])
    return versions


def default_provenance(
    served_model_name: str,
    model_path: str,
    *,
    server_config: dict[str, Any] | None = None,
    server_environment: dict[str, str] | None = None,
    generation_config: dict[str, Any] | None = None,
    expected_hardware: str | None = None,
    accelerator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repository = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_url = (
        f"{server_url}/{repository}/actions/runs/{run_id}"
        if repository and run_id
        else None
    )
    return {
        "source_revision": os.environ.get("SGLANG_PPU_SOURCE_REVISION")
        or os.environ.get("GITHUB_SHA"),
        "github_run_id": run_id,
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "github_run_url": run_url,
        "base_image": os.environ.get("PPU_BASE_IMAGE"),
        # ``or None`` because the K8s lines set this to an empty string when the
        # registry could not be asked what the tag resolves to.  An empty string
        # would read as a recorded value and compare unequal to every real digest;
        # absent is what it actually is.
        "base_image_digest": os.environ.get("PPU_BASE_IMAGE_DIGEST") or None,
        "served_model_name": served_model_name,
        "checkpoint_name": Path(model_path).name if model_path else None,
        "checkpoint_config_sha256": checkpoint_config_digest(model_path),
        "expected_hardware": expected_hardware,
        "accelerator": accelerator,
        "server_config": server_config,
        # Recorded separately from server_config because it is a different
        # surface: two runs of one config can differ here only through the
        # launcher, and a reader comparing two reports needs to see that.
        "server_environment": server_environment,
        "generation_config": generation_config,
        "package_versions": _installed_package_versions(),
    }
