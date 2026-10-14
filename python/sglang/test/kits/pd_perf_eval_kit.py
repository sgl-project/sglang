"""Data-driven performance harness for the PPU prefill/decode-disaggregated suites.

The disaggregated sibling of :mod:`perf_eval_kit`, and it borrows that module's
whole second half: a measurement here produces the same numbers, is graded by the
same rule -- red only when it produced no numbers, never for being slow -- and is
reported through the same record shape, so ``collect_perf_evidence.sh`` reads a PD
report without knowing one exists.  What cannot be borrowed is the first half.  A
disaggregated run is not one server with more arguments:

  * Two servers, launched with different arguments from one checkpoint.  The
    btv1.5 PD cases state them apart -- ``prefill_args`` and ``decode_args`` --
    and the two disagree on the attention backend, the memory fraction, the
    scheduler and the parallelism, so a single ``server.parameters`` block could
    not describe the run.
  * A router in front of them, which is what the benchmark posts to.
  * A KV path between them over RDMA, which the flat schema has no field for.
  * Whole boards per role, so the tensor-parallel degree is checked against the
    devices of that role's nodes rather than against the group's.

Hence a second schema rather than a wider one: ``ppu-pd-perf-test-config/v1``.
The reviewed sections a PD config shares with a colocated one -- hardware, model,
workload -- are validated by that module's own functions, so the two lines cannot
drift on what a checkpoint or a measurement is.

Like :mod:`perf_eval_kit` this module is free of ``torch`` and of anything that
touches a device, so every config can be validated and every command line
rendered on a laptop; the half that launches the three processes lives in
:mod:`pd_perf_suite_kit`.

The configs this validates are ported from
``model-test-cases/testcases/btv1.5/llm_infer_sglang/144G/Daily/PD-Disaggregation/notune``
and their launch commands from the ``PD Disaggregation`` section of the matching
``model-site-cases/server_cmds/LLM_Serving/BTV1.5`` page.  The suite's README
records every departure from those commands.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sglang.test.kits.answer_eval_kit import default_provenance
from sglang.test.kits.perf_eval_kit import (
    ALLOWED_SERVER_ENVIRONMENT,
    DSA_SERVER_PARAMETERS,
    REQUIRED_SERVER_PARAMETERS,
    SERVER_PARAMETER_CLI_ORDER,
    SERVER_PARAMETER_NULLABLE,
    SERVER_PARAMETER_POSITIVE_INTEGERS,
    SERVER_PARAMETER_POSITIVE_NUMBERS,
    SERVER_PARAMETER_STORE_TRUE,
    SERVER_PARAMETER_STRINGS,
    PerfEvalError,
    validate_hardware,
    validate_model,
    validate_workload,
)

PD_PERF_CONFIG_SCHEMA_VERSION = "ppu-pd-perf-test-config/v1"

# The two roles, in the order a group assigns them to node ranks: the prefill
# nodes take the low ranks and the decode nodes the rest.  Rank 0 is therefore
# always a prefill node, and it is the node that also runs the router and the
# benchmark -- see :mod:`pd_perf_suite_kit`.
PD_ROLES = ("prefill", "decode")

# What the parameter tables of the colocated schema do not cover, because a
# colocated server has no use for them.  Grouped the same way and for the same
# reason: a name checked but never rendered would be decoration, and a name
# rendered but never checked would be the untyped pass-through the schema exists
# to refuse.
#
# `dp_size` and the two `enable_dp_*` flags are the decode side's data-parallel
# attention, which every ported PD case turns on; `moe_dense_tp_size` is how the
# dense layers are kept off that split.  `context_length` appears on the MiniMax
# cases.  The `speculative_*` family is how the GLM-5.2 MXFP4, MiniMax-M2.7 and
# Qwen3.5 PD cases decode, and `mamba_scheduler_strategy` is Qwen3.5's.
PD_PARAMETER_POSITIVE_INTEGERS = SERVER_PARAMETER_POSITIVE_INTEGERS + (
    "dp_size",
    "context_length",
    "moe_dense_tp_size",
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
)
PD_PARAMETER_POSITIVE_NUMBERS = SERVER_PARAMETER_POSITIVE_NUMBERS
PD_PARAMETER_STRINGS = SERVER_PARAMETER_STRINGS + (
    "speculative_algorithm",
    "speculative_attention_mode",
    "speculative_draft_attention_backend",
    "mamba_scheduler_strategy",
)
PD_PARAMETER_STORE_TRUE = SERVER_PARAMETER_STORE_TRUE + (
    "enable_dp_attention",
    "enable_dp_lm_head",
    # Distinct from `disable_piecewise_cuda_graph`: the prefill side of every
    # ported PD case runs with no graph capture at all.
    "disable_cuda_graph",
    "enable_cache_report",
    "enable_request_time_stats_logging",
)
PD_SUPPORTED_SERVER_PARAMETERS = (
    REQUIRED_SERVER_PARAMETERS
    | set(PD_PARAMETER_POSITIVE_INTEGERS)
    | set(PD_PARAMETER_POSITIVE_NUMBERS)
    | set(PD_PARAMETER_STRINGS)
    | set(PD_PARAMETER_STORE_TRUE)
)

# Refused inside a role's parameters even though SGLang accepts them, because
# this schema states each of them once, in the `disaggregation` section, and
# renders it for both roles.  A second copy could contradict the first, and the
# contradiction would surface as a KV handshake that never completes.
PD_RESERVED_SERVER_PARAMETERS = (
    "disaggregation_mode",
    "disaggregation_transfer_backend",
    "disaggregation_bootstrap_port",
    "disaggregation_ib_device",
)

# Appended rather than interleaved, so a parameter shared with the colocated
# schema renders in the position it renders there and the two command lines can
# be diffed.
PD_PARAMETER_CLI_ORDER = SERVER_PARAMETER_CLI_ORDER + (
    "dp_size",
    "moe_dense_tp_size",
    "context_length",
    "speculative_algorithm",
    "speculative_attention_mode",
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
    "speculative_draft_attention_backend",
    "mamba_scheduler_strategy",
)

# The environment the PD cases export around each role, on top of what the
# colocated schema already admits.
#
# `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER` has a read point in this tree --
# it is declared in `sglang.srt.environ` and consulted in
# `sglang.srt.disaggregation.common.conn` -- so it belongs with the supported
# names.  `SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED` is likewise declared in
# `sglang.srt.environ` and consulted by the PPU DeepEP hook
# (`_is_mnnvl_fabric_supported`), which gates whether `use_fabric` reaches the
# DeepEP `Buffer`; the PD roles set it to 0 on the btv1.5 image whose DeepEP
# build rejects that kwarg.
PD_SUPPORTED_SERVER_ENVIRONMENT = {
    "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER",
    "SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED",
}

# These have no read point in this tree and are accepted anyway, for the reason
# the colocated schema's EXTERNAL_SERVER_ENVIRONMENT records: the internal cases
# export them around the servers, and a number measured without them was
# measured on a different machine than the one it will be compared against.  The
# `MC_*` names are the Mooncake transfer engine's -- the KV path this line
# measures runs through it -- and the `NCCL_*` ones are the collective library's.
PD_EXTERNAL_SERVER_ENVIRONMENT = {
    "MC_LOG_LEVEL",
    "MC_NUM_QP_PER_EP",
    "MC_TE_METRIC",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "SAIL_SGL_DEEPEP_ICN",
    "SAIL_SGL_DEEPEP_RECV_HOOK",
    "SGL_DEEP_EP_RECV_HOOK",
    "SGL_ENABLE_JIT_DEEPGEMM",
}

PD_ALLOWED_SERVER_ENVIRONMENT = (
    ALLOWED_SERVER_ENVIRONMENT
    | PD_SUPPORTED_SERVER_ENVIRONMENT
    | PD_EXTERNAL_SERVER_ENVIRONMENT
)

PD_TRANSFER_BACKENDS = ("mooncake", "mooncake_tcp", "nixl", "ascend", "mori", "fake")

# SGLang's own default for `--disaggregation-bootstrap-port`, and the port the
# router assumes when it is handed a prefill endpoint without one.  The red-zone
# router command states no port, so the handshake works only while the prefill
# server listens on this one; the config restates it rather than inheriting it so
# the number is visible, and this is the check that keeps the two in step.
PD_ROUTER_ASSUMED_BOOTSTRAP_PORT = 8998

# The highest port this line's servers may listen on.  Above it lie two ranges
# neither of which a server can hold on the CI nodes: Kubernetes hands out node
# ports from 30000, and kube-proxy's rules rewrite traffic to <node ip>:<node
# port> on the node itself, so a server that binds one loads a checkpoint, comes
# up, and then answers nothing -- its own warmup reaches whatever backs the
# service instead.  The ephemeral range starts just above that, where an
# outbound connection can take the port before the server asks for it.  The
# red-zone commands use 30000 and 40000 and are fine doing so, because they do
# not run under Kubernetes; this line cannot copy them.
PD_HIGHEST_BINDABLE_PORT = 29999

PD_DISAGGREGATION_KEYS = {
    "prefill_nodes",
    "decode_nodes",
    "transfer_backend",
    "ib_devices",
    "bootstrap_port",
    "prefill_port",
    "decode_port",
    "router_port",
}
PD_DISAGGREGATION_OPTIONAL_KEYS = {
    "peer_wait_timeout_seconds",
    "router_startup_timeout_seconds",
}

PD_HARDWARE_KEYS = {
    "platform",
    "generation",
    "visible_devices",
    "memory_gib_per_device",
}

DEFAULT_PEER_WAIT_TIMEOUT_SECONDS = 5400
DEFAULT_ROUTER_STARTUP_TIMEOUT_SECONDS = 600


def validate_pd_test_config(config: dict[str, Any]) -> None:
    """Validate the public, data-driven disaggregated performance contract."""

    if config.get("schema_version") != PD_PERF_CONFIG_SCHEMA_VERSION:
        raise PerfEvalError("unsupported disaggregated performance config schema")
    if not isinstance(config.get("test_id"), str) or not config["test_id"]:
        raise PerfEvalError("test_id must be a non-empty string")
    if set(config) != {
        "schema_version",
        "test_id",
        "hardware",
        "model",
        "disaggregation",
        "prefill",
        "decode",
        "workload",
    }:
        raise PerfEvalError("the config may only carry the reviewed top-level keys")
    for section in ("hardware", "model", "disaggregation", "prefill", "decode"):
        if not isinstance(config.get(section), dict):
            raise PerfEvalError(f"{section} must be an object")
    if not isinstance(config.get("workload"), dict):
        raise PerfEvalError("workload must be an object")

    hardware = config["hardware"]
    # The colocated schema's own checks, so the two lines cannot drift on what a
    # board or a checkpoint is.  The key set is narrowed here because that one
    # admits `nnodes`, and a PD group states its node count per role instead --
    # a second statement of it here could contradict the first.
    validate_hardware(hardware)
    if not set(hardware) <= PD_HARDWARE_KEYS:
        raise PerfEvalError(
            "hardware may only carry the reviewed keys; a PD group states its "
            "node count per role in disaggregation, not as hardware.nnodes"
        )
    validate_model(config["model"])
    validate_workload(config["workload"])
    _validate_disaggregation(config["disaggregation"])
    for role in PD_ROLES:
        _validate_role(role, config[role], hardware, config["disaggregation"])


def _validate_disaggregation(disaggregation: dict[str, Any]) -> None:
    if (
        not set(disaggregation)
        <= PD_DISAGGREGATION_KEYS | PD_DISAGGREGATION_OPTIONAL_KEYS
    ):
        raise PerfEvalError("disaggregation may only carry the reviewed keys")
    if not PD_DISAGGREGATION_KEYS <= set(disaggregation):
        raise PerfEvalError(
            "disaggregation must state " + ", ".join(sorted(PD_DISAGGREGATION_KEYS))
        )
    for field in ("prefill_nodes", "decode_nodes"):
        value = disaggregation[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise PerfEvalError(f"disaggregation.{field} must be a positive integer")
    if disaggregation["transfer_backend"] not in PD_TRANSFER_BACKENDS:
        raise PerfEvalError(
            "disaggregation.transfer_backend must name a backend SGLang accepts: "
            + ", ".join(PD_TRANSFER_BACKENDS)
        )
    ib_devices = disaggregation["ib_devices"]
    if (
        not isinstance(ib_devices, list)
        or not ib_devices
        or any(not isinstance(device, str) or not device for device in ib_devices)
        or len(set(ib_devices)) != len(ib_devices)
    ):
        raise PerfEvalError(
            "disaggregation.ib_devices must list the RDMA devices the KV path "
            "uses, as unique non-empty strings"
        )
    ports = {}
    for field in ("bootstrap_port", "prefill_port", "decode_port", "router_port"):
        value = disaggregation[field]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= 65535
        ):
            raise PerfEvalError(f"disaggregation.{field} must be a TCP port")
        if value > PD_HIGHEST_BINDABLE_PORT:
            raise PerfEvalError(
                f"disaggregation.{field} must be at most "
                f"{PD_HIGHEST_BINDABLE_PORT}: above it are the Kubernetes node "
                "port range, which the CI nodes rewrite out from under a "
                "listening server, and the ephemeral range, which an outbound "
                "connection can take first"
            )
        ports[field] = value
    # The prefill and decode servers of a 1p1d run land on different nodes, but
    # nothing in this schema says they have to, and two roles sharing a port on
    # one node would fail at bind time after both had loaded a checkpoint.
    if len(set(ports.values())) != len(ports):
        raise PerfEvalError("the disaggregation ports must differ from each other")
    if ports["bootstrap_port"] != PD_ROUTER_ASSUMED_BOOTSTRAP_PORT:
        raise PerfEvalError(
            "disaggregation.bootstrap_port must be "
            f"{PD_ROUTER_ASSUMED_BOOTSTRAP_PORT}: the router is handed the "
            "prefill endpoint without a bootstrap port, exactly as the red-zone "
            "command does, so it assumes that one"
        )
    for field, default in (
        ("peer_wait_timeout_seconds", DEFAULT_PEER_WAIT_TIMEOUT_SECONDS),
        ("router_startup_timeout_seconds", DEFAULT_ROUTER_STARTUP_TIMEOUT_SECONDS),
    ):
        value = disaggregation.get(field, default)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise PerfEvalError(f"disaggregation.{field} must be positive")


def _validate_role(
    role: str,
    section: dict[str, Any],
    hardware: dict[str, Any],
    disaggregation: dict[str, Any],
) -> None:
    if not set(section) <= {"startup_timeout_seconds", "parameters", "env"}:
        raise PerfEvalError(f"{role} may only carry the reviewed keys")
    startup_timeout = section.get("startup_timeout_seconds")
    if (
        not isinstance(startup_timeout, (int, float))
        or isinstance(startup_timeout, bool)
        or startup_timeout <= 0
    ):
        raise PerfEvalError(f"{role}.startup_timeout_seconds must be positive")

    parameters = section.get("parameters")
    if not isinstance(parameters, dict):
        raise PerfEvalError(f"{role}.parameters must be an object")
    # Before the set check below, and not after it: those names are outside the
    # supported set, so that check would already refuse them -- with a message
    # that reads as "unknown parameter" rather than as "stated in the wrong
    # place", which is the mistake actually being made.
    for field in PD_RESERVED_SERVER_PARAMETERS:
        if field in parameters:
            raise PerfEvalError(
                f"{role}.parameters.{field} is stated once in the disaggregation "
                "section and rendered for both roles"
            )
    if (
        not REQUIRED_SERVER_PARAMETERS <= set(parameters)
        or not set(parameters) <= PD_SUPPORTED_SERVER_PARAMETERS
    ):
        raise PerfEvalError(
            f"{role}.parameters must contain exactly the supported SGLang parameters"
        )
    if not isinstance(parameters["trust_remote_code"], bool):
        raise PerfEvalError(f"{role}.parameters.trust_remote_code must be boolean")
    for field in PD_PARAMETER_POSITIVE_INTEGERS:
        if field not in parameters:
            continue
        if (
            not isinstance(parameters[field], int)
            or isinstance(parameters[field], bool)
            or parameters[field] <= 0
        ):
            raise PerfEvalError(f"{role}.parameters.{field} must be a positive integer")
    for field in PD_PARAMETER_POSITIVE_NUMBERS:
        if field not in parameters:
            continue
        if (
            not isinstance(parameters[field], (int, float))
            or isinstance(parameters[field], bool)
            or parameters[field] <= 0
        ):
            raise PerfEvalError(f"{role}.parameters.{field} must be positive")
    for field in PD_PARAMETER_STORE_TRUE:
        if field not in parameters:
            continue
        if parameters[field] is not True:
            raise PerfEvalError(
                f"{role}.parameters.{field} may only be set to true; omit it to "
                "keep the server default"
            )
    # Per role and per role's nodes: a 1p1d run of two boards is two servers of
    # eight devices each, not one of sixteen, so the group's device count is the
    # wrong thing to check a tensor-parallel degree against.
    role_nodes = disaggregation[f"{role}_nodes"]
    if parameters["tp_size"] * parameters.get("pp_size", 1) != (
        len(hardware["visible_devices"]) * role_nodes
    ):
        raise PerfEvalError(
            f"{role}.parameters.tp_size times pp_size must equal the visible "
            f"device count summed over the {role_nodes} node(s) this role holds"
        )
    mem_fraction = parameters["mem_fraction_static"]
    if (
        not isinstance(mem_fraction, (int, float))
        or isinstance(mem_fraction, bool)
        or not 0 < mem_fraction <= 1
    ):
        raise PerfEvalError(f"{role}.parameters.mem_fraction_static must be in (0, 1]")
    for field in PD_PARAMETER_STRINGS:
        if field not in parameters:
            continue
        value = parameters[field]
        if value is None and field in SERVER_PARAMETER_NULLABLE:
            continue
        if not isinstance(value, str) or not value:
            suffix = " or null" if field in SERVER_PARAMETER_NULLABLE else ""
            raise PerfEvalError(
                f"{role}.parameters.{field} must be a non-empty string{suffix}"
            )
    if parameters["attention_backend"] is None and not (
        "prefill_attention_backend" in parameters
        or "decode_attention_backend" in parameters
    ):
        raise PerfEvalError(
            f"{role}.parameters.attention_backend may only be null when the "
            "prefill or decode backend is named separately"
        )
    if any(field in parameters for field in DSA_SERVER_PARAMETERS) and parameters[
        "attention_backend"
    ] not in ("dsa", "nsa"):
        raise PerfEvalError(
            "the dsa_* server parameters only apply to the sparse attention "
            f"backend, so {role}.parameters.attention_backend must name it"
        )
    if (
        "dsa_prefill_cp_mode" in parameters
        and parameters.get("enable_dsa_prefill_context_parallel") is not True
    ):
        raise PerfEvalError(
            f"{role}.parameters.dsa_prefill_cp_mode describes how prefill context "
            "parallelism splits a sequence, so it needs "
            "enable_dsa_prefill_context_parallel"
        )
    if "deepep_mode" in parameters and parameters.get("moe_a2a_backend") != "deepep":
        raise PerfEvalError(
            f"{role}.parameters.deepep_mode only applies to the DeepEP all-to-all "
            "backend, so moe_a2a_backend must name it"
        )
    # Data-parallel attention is what `dp_size` sizes, and SGLang reads the
    # degree only on that path: stating it alone would describe a setting no run
    # honours.  The degree also has to divide the tensor-parallel one, which is
    # what SGLang itself requires of the pair.
    if "dp_size" in parameters:
        if parameters.get("enable_dp_attention") is not True:
            raise PerfEvalError(
                f"{role}.parameters.dp_size sizes data-parallel attention, so it "
                "needs enable_dp_attention"
            )
        if parameters["tp_size"] % parameters["dp_size"]:
            raise PerfEvalError(f"{role}.parameters.dp_size must divide its tp_size")
    if parameters.get("enable_dp_lm_head") is True and (
        parameters.get("enable_dp_attention") is not True
    ):
        raise PerfEvalError(
            f"{role}.parameters.enable_dp_lm_head replicates the head across the "
            "data-parallel attention groups, so it needs enable_dp_attention"
        )
    # The speculative family is one setting stated in several keys, and SGLang
    # consults every one of them only once an algorithm is named.
    speculative = [field for field in parameters if field.startswith("speculative_")]
    if speculative and "speculative_algorithm" not in parameters:
        raise PerfEvalError(
            f"the speculative_* parameters of {role} describe how a named "
            "algorithm decodes, so speculative_algorithm must name one"
        )

    if "env" in section:
        environment = section["env"]
        if (
            not isinstance(environment, dict)
            or not environment
            or not set(environment) <= PD_ALLOWED_SERVER_ENVIRONMENT
        ):
            raise PerfEvalError(
                f"{role}.env must be a non-empty object of reviewed environment "
                "variables"
            )
        for name, value in environment.items():
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise PerfEvalError(f"{role}.env.{name} must be a string or a number")
            if isinstance(value, str) and not value:
                raise PerfEvalError(f"{role}.env.{name} must not be empty")


def pd_node_count(config: dict[str, Any]) -> int:
    """The number of boards the group holds, across both roles."""

    disaggregation = config["disaggregation"]
    return disaggregation["prefill_nodes"] + disaggregation["decode_nodes"]


def pd_topology(config: dict[str, Any]) -> str:
    """The topology as btv1.5 names it, e.g. ``1p1d``."""

    disaggregation = config["disaggregation"]
    return f"{disaggregation['prefill_nodes']}p{disaggregation['decode_nodes']}d"


def pd_role_for_node_rank(config: dict[str, Any], node_rank: int) -> str:
    """Which role the node of this rank serves.

    The prefill nodes take the low ranks.  Rank 0 is therefore a prefill node,
    which is also the node that runs the router and the benchmark: the router has
    to live somewhere, and putting it on a node that already holds the role the
    benchmark's first token comes from keeps the group to the boards the config
    asks for.
    """

    prefill_nodes = config["disaggregation"]["prefill_nodes"]
    if not 0 <= node_rank < pd_node_count(config):
        raise PerfEvalError(
            f"node rank {node_rank} is outside the [0, {pd_node_count(config)}) "
            "this config declares"
        )
    return "prefill" if node_rank < prefill_nodes else "decode"


def resolve_pd_runtime(
    config: dict[str, Any], environ: dict[str, str] | None = None
) -> dict[str, Any]:
    """Bind the config to the node this process was handed.

    Which node this is, and how many the launcher started, are runtime facts
    rather than reviewed ones, so they come from the environment.  ``NODE_RANK``
    and ``NNODES`` are what ppu-distributed-action injects per pod in gang mode;
    ``SGLANG_PPU_PD_PERF_NODE_RANK`` overrides the rank for a bare-metal run.

    Unlike the colocated line there is no rendezvous address here: the two
    servers of a PD group do not join one process group -- each is a rank 0 of
    its own -- and what they exchange instead is an HTTP endpoint, published
    through the results directory by :mod:`pd_perf_suite_kit`.
    """

    environ = os.environ if environ is None else environ
    nnodes = pd_node_count(config)
    _check_launcher_group_size(environ, nnodes)
    node_rank = _pd_node_rank(environ, nnodes)
    role = pd_role_for_node_rank(config, node_rank)
    prefill_nodes = config["disaggregation"]["prefill_nodes"]
    return {
        "nnodes": nnodes,
        "node_rank": node_rank,
        "role": role,
        # The index of this node within its own role, which is what a role's own
        # rendezvous would be built from once a role spans more than one board.
        "role_rank": node_rank if role == "prefill" else node_rank - prefill_nodes,
        "topology": pd_topology(config),
    }


def _check_launcher_group_size(environ: dict[str, str], nnodes: int) -> None:
    """Refuse a launcher that started a different number of nodes than reviewed.

    A group short of a node still hands every pod it did start a valid rank, and
    the shortfall then surfaces only as a peer endpoint that never appears: the
    nodes that did arrive would sit on their boards for the whole wait budget
    first.  Checked rather than trusted because the count is stated twice, once
    in the reviewed config and once in the workflow that asks for boards.
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


def _pd_node_rank(environ: dict[str, str], nnodes: int) -> int:
    for name in ("SGLANG_PPU_PD_PERF_NODE_RANK", "NODE_RANK"):
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
        "a disaggregated performance config needs the rank of this node: set "
        "NODE_RANK or SGLANG_PPU_PD_PERF_NODE_RANK"
    )


def build_pd_server_args(config: dict[str, Any], role: str) -> list[str]:
    """Translate one role of a validated PD config into SGLang server arguments.

    The disaggregation section is rendered after the role's own parameters, and
    identically for both roles apart from the mode: the KV path is one setting of
    the run, not of a server, and a reader comparing the two command lines should
    see it in the same place in each.
    """

    if role not in PD_ROLES:
        raise PerfEvalError(f"unknown disaggregation role {role!r}")
    parameters = config[role]["parameters"]
    disaggregation = config["disaggregation"]
    args: list[str] = []
    if parameters["trust_remote_code"]:
        args.append("--trust-remote-code")
    for name in PD_PARAMETER_CLI_ORDER:
        value = parameters.get(name)
        if value is None:
            continue
        args.extend([f"--{name.replace('_', '-')}", str(value)])
    for name in PD_PARAMETER_STORE_TRUE:
        if parameters.get(name):
            args.append(f"--{name.replace('_', '-')}")
    args.extend(["--served-model-name", config["model"]["served_model_name"]])
    args.extend(
        [
            "--disaggregation-mode",
            role,
            "--disaggregation-transfer-backend",
            disaggregation["transfer_backend"],
            "--disaggregation-bootstrap-port",
            str(disaggregation["bootstrap_port"]),
            "--disaggregation-ib-device",
            ",".join(disaggregation["ib_devices"]),
        ]
    )
    return args


def build_router_args(
    config: dict[str, Any],
    *,
    host: str,
    prefill_urls: list[str],
    decode_urls: list[str],
) -> list[str]:
    """The router command, as the red-zone PD cases state it.

    ``--mini-lb`` is what those commands run and what
    ``sglang.test.server_fixtures.disaggregation_fixture`` launches, so it is
    what this line measures; the full router's own scheduling would be a
    different thing to measure and is not what the ported numbers came from.

    The prefill endpoint is passed without a bootstrap port, exactly as the
    red-zone command does, which is why the schema pins that port to the one the
    router assumes.
    """

    if not prefill_urls or not decode_urls:
        raise PerfEvalError("the router needs at least one endpoint of each role")
    args = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--mini-lb",
        "--host",
        host,
        "--port",
        str(config["disaggregation"]["router_port"]),
    ]
    for url in prefill_urls:
        args.extend(["--prefill", url])
    for url in decode_urls:
        args.extend(["--decode", url])
    return args


def pd_role_environment(config: dict[str, Any], role: str) -> dict[str, str]:
    """The environment a config asks to be set around one role's server."""

    if role not in PD_ROLES:
        raise PerfEvalError(f"unknown disaggregation role {role!r}")
    return {name: str(value) for name, value in config[role].get("env", {}).items()}


def pd_expected_hardware(config: dict[str, Any]) -> str:
    """Render the reviewed hardware contract for provenance.

    Both dimensions and the topology, because ``1p1d`` on two boards of eight and
    ``2p4d`` on six are not the same machine, and neither is a colocated
    sixteen-device run.
    """

    hardware = config["hardware"]
    memory_gib = hardware["memory_gib_per_device"]
    if isinstance(memory_gib, float) and memory_gib.is_integer():
        memory_gib = int(memory_gib)
    devices_per_node = len(hardware["visible_devices"])
    return (
        f"{hardware['generation']}-{pd_topology(config)}x{devices_per_node}"
        f"x{memory_gib}g"
    )


def pd_provenance(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """The provenance block of a disaggregated run.

    The server configuration and the environment are recorded per role, and the
    KV path alongside them: two runs of one checkpoint that differ in the
    transfer backend or in the RDMA devices are not comparable measurements, and
    nothing else in the report would say so.
    """

    provenance = default_provenance(
        config["model"]["served_model_name"],
        config["model"]["path"],
        server_config={
            "disaggregation": dict(config["disaggregation"]),
            "prefill": config["prefill"]["parameters"],
            "decode": config["decode"]["parameters"],
        },
        server_environment={
            role: pd_role_environment(config, role) for role in PD_ROLES
        },
        expected_hardware=pd_expected_hardware(config),
        **kwargs,
    )
    provenance["workload"] = config["workload"]
    provenance["topology"] = pd_topology(config)
    return provenance


def pd_endpoint_url(host: str, port: int) -> str:
    """The URL a role publishes for its peers, and the router for the benchmark."""

    return f"http://{host}:{int(port)}"


def pd_model_path(config: dict[str, Any]) -> Path:
    return Path(config["model"]["path"])


__all__ = [
    "DEFAULT_PEER_WAIT_TIMEOUT_SECONDS",
    "DEFAULT_ROUTER_STARTUP_TIMEOUT_SECONDS",
    "PD_ALLOWED_SERVER_ENVIRONMENT",
    "PD_DISAGGREGATION_KEYS",
    "PD_EXTERNAL_SERVER_ENVIRONMENT",
    "PD_PARAMETER_CLI_ORDER",
    "PD_PARAMETER_POSITIVE_INTEGERS",
    "PD_PARAMETER_STORE_TRUE",
    "PD_PARAMETER_STRINGS",
    "PD_PERF_CONFIG_SCHEMA_VERSION",
    "PD_RESERVED_SERVER_PARAMETERS",
    "PD_ROLES",
    "PD_ROUTER_ASSUMED_BOOTSTRAP_PORT",
    "PD_SUPPORTED_SERVER_ENVIRONMENT",
    "PD_SUPPORTED_SERVER_PARAMETERS",
    "PD_TRANSFER_BACKENDS",
    "build_pd_server_args",
    "build_router_args",
    "pd_endpoint_url",
    "pd_expected_hardware",
    "pd_model_path",
    "pd_node_count",
    "pd_provenance",
    "pd_role_environment",
    "pd_role_for_node_rank",
    "pd_topology",
    "resolve_pd_runtime",
    "validate_pd_test_config",
]
