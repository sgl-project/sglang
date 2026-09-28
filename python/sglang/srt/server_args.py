# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Server argument construction, resolution, CLI registration, and network ports."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import importlib
import logging
import sys
import tempfile
import uuid
from typing import Any, NoReturn

import msgspec

from sglang.srt.arg_groups.arg_utils import (
    add_cli_args_from_dataclass,
    is_record,
    record_fields,
)
from sglang.srt.arg_groups.argparse_actions import (
    DeprecatedStoreTrueAction,
)
from sglang.srt.arg_groups.model_override_base import ep_joiner_of, ep_scale_joiner_of
from sglang.srt.arg_groups.overrides import (
    remote_instance_transfer_engine_of,
    resolution_result,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform, publish
from sglang.srt.speculative.decoupled_spec_io import DecoupledSpecIpcConfig
from sglang.srt.utils.network import NetworkAddress, get_free_port, wait_port_available

logger = logging.getLogger(__name__)


def _reasoning_parser_choices():
    # Importing the registry here costs seconds in every process that parses
    # arguments; a plugin that registered a parser has already imported it.
    module = sys.modules.get("sglang.srt.parser.reasoning_parser")
    if module is not None:
        return list(module.ReasoningParser.DetectorMap)
    from sglang.srt.parser.reasoning_parser_names import REASONING_PARSER_NAMES

    return list(REASONING_PARSER_NAMES)


def _tool_call_parser_choices():
    module = sys.modules.get("sglang.srt.function_call.function_call_parser")
    if module is not None:
        return list(module.FunctionCallParser.ToolCallParserEnum)
    from sglang.srt.function_call.parser_names import TOOL_CALL_PARSER_NAMES

    return list(TOOL_CALL_PARSER_NAMES)


def _real_kv_hash_modes():
    # Lazy: this pulls the whole sglang.kernels package (~2 s) into every
    # process that imports server_args, most of which never use it.
    from sglang.kernels.ops.kv_canary.consts import RealKvHashMode

    return list(RealKvHashMode)


# Compatibility re-exports for callers importing through server_args.
from sglang.srt.arg_groups.arg_utils import NS, A, Arg  # noqa: F401
from sglang.srt.arg_groups.argparse_actions import LoRAPathAction  # noqa: F401

# Public choice lists and plugin registration helpers.
from sglang.srt.arg_groups.choices import (  # noqa: F401
    ATTENTION_BACKEND_CHOICES,
    CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS,
    DETERMINISTIC_ATTENTION_BACKEND_CHOICES,
    DISAGG_TRANSFER_BACKEND_CHOICES,
    DRAFT_ATTENTION_BACKEND_CHOICES,
    FP4_GEMM_RUNNER_BACKEND_CHOICES,
    FP8_GEMM_RUNNER_BACKEND_CHOICES,
    GRAMMAR_BACKEND_CHOICES,
    LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
    LOAD_FORMAT_CHOICES,
    MOE_RUNNER_BACKEND_CHOICES,
    MXFP8_MOE_RUNNER_BACKEND_CHOICES,
    QUANTIZATION_CHOICES,
    RADIX_EVICTION_POLICY_CHOICES,
    RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND,
    RL_ON_POLICY_TARGET_CHOICES,
    SAMPLING_BACKEND_CHOICES,
    add_attention_backend_choices,
    add_chunked_prefix_cache_attention_backend,
    add_deterministic_attention_backend_choices,
    add_disagg_transfer_backend_choices,
    add_draft_attention_backend_choices,
    add_fp4_gemm_runner_backend_choices,
    add_fp8_gemm_runner_backend_choices,
    add_grammar_backend_choices,
    add_linear_attn_kernel_backend_choices,
    add_load_format_choices,
    add_moe_runner_backend_choices,
    add_mxfp8_moe_runner_backend_choices,
    add_quantization_method_choices,
    add_radix_eviction_policy_choices,
    add_radix_supported_deterministic_attention_backend_choices,
    add_rl_on_policy_target_choices,
)
from sglang.srt.arg_groups.fields import collect_input_fields
from sglang.srt.arg_groups.fields.device import (
    Device,
)
from sglang.srt.arg_groups.fields.disagg import (
    Disagg,
)
from sglang.srt.arg_groups.fields.exec_ import (
    ExecComm,
    ExecDeterministic,
    ExecDllm,
    ExecFeatures,
    ExecGraph,
    ExecKernel,
    ExecMamba,
    ExecMoe,
    ExecOffload,
    ExecOverlap,
)
from sglang.srt.arg_groups.fields.lora import (
    Lora,
)
from sglang.srt.arg_groups.fields.memory import (
    Memory,
)
from sglang.srt.arg_groups.fields.mm import (
    Mm,
)
from sglang.srt.arg_groups.fields.model import (
    Model,
)
from sglang.srt.arg_groups.fields.observability import (
    Observability,
)
from sglang.srt.arg_groups.fields.parallel import (
    Parallel,
)
from sglang.srt.arg_groups.fields.schedule import (
    Schedule,
)
from sglang.srt.arg_groups.fields.serving import (
    Serving,
)
from sglang.srt.arg_groups.fields.spec import (
    Spec,
)
from sglang.srt.lora.lora_registry import LoRARef  # noqa: F401
from sglang.srt.model_executor.cuda_graph_config import (  # noqa: F401
    CudaGraphConfig,
    parse_cuda_graph_config_arg,
)
from sglang.srt.utils.common import (  # noqa: F401
    LORA_TARGET_ALL_MODULES,
    SUPPORTED_LORA_TARGET_MODULES,
    human_readable_int,
    json_list_type,
    nullable_str,
)

# Re-exported like the imports above, but resolved on first use: importing them
# eagerly is what the choices helpers avoid, and most processes never read them.
_LAZY_REEXPORTS = {
    "FunctionCallParser": "sglang.srt.function_call.function_call_parser",
    "ReasoningParser": "sglang.srt.parser.reasoning_parser",
    "RealKvHashMode": "sglang.kernels.ops.kv_canary.consts",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_REEXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def _plain(value: Any) -> Any:
    """Convert nested Structs and dataclasses to dicts, copying all values."""
    if not isinstance(value, type) and is_record(value):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in record_fields(type(value))
        }
    if isinstance(value, tuple) and hasattr(value, "_fields"):  # namedtuple
        return type(value)(*(_plain(item) for item in value))
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    if isinstance(value, dict):
        return type(value)((_plain(k), _plain(v)) for k, v in value.items())
    return copy.deepcopy(value)


class ServerArgs:
    """Raw server configuration, sealed when resolution starts.

    Add fields to the matching namespace in ``arg_groups/fields/`` using
    ``A[T, help]`` or ``A[T, Arg(...)]``; see ``arg_utils.Arg`` for CLI metadata.
    Only dynamic choices, deprecated flags, and ``--config`` need manual
    registration in ``add_cli_args``. ``POSITIONAL_FIELD_ORDER`` preserves
    the positional constructor signature when these namespaces are assembled.
    """

    def __post_init__(self):
        """Leave construction unresolved; launchers and publishers call ``resolve_once``."""

    def resolve_once(self) -> None:
        """Resolve once, preserving declarations across pickling to child processes.

        Handlers are not idempotent over their own output. A failed resolution
        cannot be retried on the same record.
        """
        if getattr(self, "_resolution_finished", False):
            return
        if getattr(self, "_resolution_failed", False):
            raise RuntimeError(
                "resolution already failed on this ServerArgs; the handlers that "
                "ran left their writes on the record, and a second pass would "
                "read that partial output as fresh input. Build a new record "
                "from the corrected arguments."
            )
        from sglang.srt.arg_groups.pipeline import run_resolution_pipeline

        self._input_frozen = True
        try:
            run_resolution_pipeline(self)
        except BaseException:
            self._resolution_failed = True
            raise
        finally:
            self._input_frozen = False
        # Also mark the dummy/absent-model path, which returns early from the pipeline.
        self._resolution_finished = True

    @property
    def launch_command(self) -> str | None:
        """Original CLI arguments or Engine constructor call; ``None`` for direct construction."""
        return getattr(self, "_launch_command", None)

    def resolved_dict(self) -> dict[str, Any]:
        """Serialize resolved field values, expanding nested records and excluding bookkeeping."""

        return {
            field.name: _plain(resolution_result(self, field.name))
            for field in record_fields(type(self))
        }

    LANGUAGE_MODEL_ONLY_ARCHITECTURES = (
        "MuseGlimmerForConditionalGeneration",
        "Cosmos3ForConditionalGeneration",
        "Cosmos3EdgeForConditionalGeneration",
    )

    # The attention-backend allow-list is enforced via
    # --enable-page-major-kv-layout (implied by the unified pool in
    # _handle_page_major_kv_layout); the model-family gate is enforced at pool
    # construction in model_runner_kv_cache_mixin._init_pools.

    def _unified_memory_pd_transfer_backends(self) -> set[str]:
        return {"mooncake"}

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):

        add_cli_args_from_dataclass(parser, ServerArgs)

        # --- Fields with dynamic choices (computed at add_cli_args time) ---
        sampling_backend_choices = set(SAMPLING_BACKEND_CHOICES)
        if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get():
            sampling_backend_choices.add("token_oracle")
        parser.add_argument(
            "--sampling-backend",
            type=str,
            choices=sampling_backend_choices,
            default=_declared_default("sampling_backend"),
            help="Choose the kernels for sampling layers.",
        )

        reasoning_parser_choices = _reasoning_parser_choices()
        parser.add_argument(
            "--reasoning-parser",
            type=str,
            choices=["auto"] + reasoning_parser_choices,
            default=_declared_default("reasoning_parser"),
            help=f"Specify the parser for reasoning models. "
            f"Use 'auto' to detect from chat template. "
            f"Options include: {reasoning_parser_choices}.",
        )
        tool_call_parser_choices = _tool_call_parser_choices()
        parser.add_argument(
            "--tool-call-parser",
            type=str,
            choices=["auto"] + tool_call_parser_choices,
            default=_declared_default("tool_call_parser"),
            help=f"Specify the parser for handling tool-call interactions. "
            f"Use 'auto' to detect from chat template. "
            f"Options include: {tool_call_parser_choices}.",
        )
        parser.add_argument(
            "--kv-canary-real-data",
            type=str,
            default=_declared_default("kv_canary_real_data"),
            choices=[m.name.lower() for m in _real_kv_hash_modes()],
            help=(
                "Check the real KV-cache in the canary. "
                "'none' (default) disables the feature. "
                "'partial' checks the first 16 bytes of each real-KV slot. "
                "'all' checks the full real-KV slot."
            ),
        )

        # --- Configuration file support ---
        parser.add_argument(
            "--config",
            type=str,
            help="Read CLI options from a config file. Must be a YAML file with configuration options.",
        )

        # --- Deprecated argument registrations ---
        # `disable_cuda_graph` is `no_cli=True`, so this deprecated spelling is
        # its only command-line entry point.
        parser.add_argument(
            "--disable-cuda-graph",
            action=DeprecatedStoreTrueAction,
            new_flag="--cuda-graph-backend-{decode,prefill}=disabled",
            help="Deprecated. Use --cuda-graph-backend-{decode,prefill}=disabled instead.",
        )
        parser.add_argument(
            "--enable-flashinfer-allreduce-fusion",
            action="store_true",
            help="(Deprecated: use --flashinfer-allreduce-fusion-backend=auto) "
            "Enable FlashInfer allreduce fusion with Residual RMSNorm.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Some dataclass fields (e.g. stat_loggers) intentionally have no CLI
        # surface and won't appear on the argparse Namespace. Skip them so the
        # dataclass default applies.
        attrs = [attr.name for attr in record_fields(cls) if hasattr(args, attr.name)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def get_tokenizer_worker_class(self):
        from sglang.srt.managers.multi_tokenizer_mixin import TokenizerWorker

        return TokenizerWorker

    def url(self, port: int | None = None):
        scheme = "https" if self.ssl_certfile else "http"
        # When binding to all interfaces, use loopback for internal requests.
        host = self.host
        if not host or host == "0.0.0.0":
            host = "127.0.0.1"
        elif host == "::":
            host = "::1"
        return NetworkAddress(host, port if port is not None else self.port).to_url(
            scheme
        )

    @property
    def engine_info_bootstrap_url(self):
        return self.url(port=self.engine_info_bootstrap_port)

    def __setattr__(self, name, value):
        # Seal configuration fields, including underscore-prefixed ones, but allow bookkeeping.
        if not name.startswith("_") or name in _underscore_field_names():
            if getattr(self, "_input_frozen", False):
                raise AttributeError(
                    f"server_args.{name} assigned during resolution; the record "
                    "is the operator's input and resolution does not write it -- "
                    "declare the decision with declare_resolution(server_args, "
                    "source, **fields) so it carries a source and leaves the "
                    "input intact."
                )
            if getattr(self, "_resolution_finished", False):
                raise AttributeError(
                    f"server_args.{name} assigned after resolution; server_args is "
                    "read-only -- use get_context().override(source, ...) to change "
                    "resolved config; a value one runner owns travels as a "
                    "constructor argument."
                )
        # The Struct's own setter, spelled explicitly: this method is copied
        # into the class `defstruct` builds, so a zero-argument `super()` would
        # still close over the class it was written in. `object.__setattr__`
        # does not reach a Struct's fields at all.
        msgspec.Struct.__setattr__(self, name, value)

    def __reduce__(self):
        """Preserve resolution bookkeeping as well as Struct fields when pickling.

        Restore fields before bookkeeping so the write seal is re-armed last.
        """
        return (
            _rebuild_server_args,
            (type(self), msgspec.structs.asdict(self), dict(self.__dict__)),
        )

    def check_server_args(self):
        from sglang.srt.arg_groups.validation_hook import check_server_args

        check_server_args(self)

    def remote_instance_weight_loader_use_transfer_engine(self, load_format=None):
        """``load_format`` overrides the seed's: a draft runner loading under
        ``--speculative-draft-load-format`` needs its own transfer engine."""
        return remote_instance_transfer_engine_of(resolving_view(self), load_format)


# Collect input fields only; field_order.py preserves positional argument order.


_INPUT_NAMESPACES = [
    Model,
    ExecDeterministic,
    ExecDllm,
    ExecOffload,
    ExecOverlap,
    Lora,
    Disagg,
    Spec,
    ExecMoe,
    ExecComm,
    ExecGraph,
    ExecMamba,
    ExecKernel,
    Observability,
    Device,
    Parallel,
    Memory,
    Schedule,
    ExecFeatures,
    Mm,
    Serving,
]

_annotations, _defaults, _namespaces = collect_input_fields(_INPUT_NAMESPACES)
ServerArgs.__annotations__ = {**_annotations, **ServerArgs.__annotations__}
ServerArgs._NS_BY_FIELD = _namespaces
ServerArgs._NAMESPACES = _INPUT_NAMESPACES
# dict=True retains resolution bookkeeping and memoized values outside the fields.
ServerArgs = msgspec.defstruct(
    "ServerArgs",
    [
        (_name, _ann, _defaults[_name]) if _name in _defaults else (_name, _ann)
        for _name, _ann in ServerArgs.__annotations__.items()
    ],
    namespace={
        _k: _v
        for _k, _v in vars(ServerArgs).items()
        if _k not in ("__dict__", "__weakref__", "__annotations__")
    },
    dict=True,
)


# --------------------------------------------------------------------------
# Module-level ServerArgs helpers and runtime shims.
# --------------------------------------------------------------------------


def resolve_encoder_transfer_backend(
    backend: str, model_arch: str, tp_size: int
) -> str:
    if backend != "auto":
        return backend
    if model_arch == "KimiK3ForConditionalGeneration" and tp_size > 1:
        return "zmq_to_tokenizer"
    return "zmq_to_scheduler"


def compute_world_size(
    *, enable_dp_attention: bool, dp_size: int, tp_size: int, pp_size: int
) -> int:
    """Total GPU count across all data-parallel replicas.

    Takes the values rather than a config object: the two sizes are the widths
    the launch asked for, which the Ray driver needs before any process group
    exists, and passing a context would hand it the live groups instead.
    """
    return (1 if enable_dp_attention else dp_size) * tp_size * pp_size


def m3_fp8_attn_gemm_enabled(args) -> bool:
    """Whether MiniMax-M3 attention GEMMs run in fp8 (no opt-in flag; active
    whenever possible): fp8_e4m3 main + index KV caches, fp8-cast q, fp8
    sparse/MSA kernels, with dense layers on trtllm_mha's fp8-q path. Needs
    kv_cache_dtype fp8_e4m3 (e5m2 would silently mis-dispatch fmha_sm100's
    e4m3 kernel), the trtllm_mha backend (the only dense backend with fp8-q
    GEMMs), and SM100 (MSA fp8 variants and trtllm-gen fp8 dense kernels are
    sm100-only). SGLANG_DISABLE_M3_FP8_ATTN_GEMM=1 is the kill switch:
    it forces the pre-fp8 numerics (bf16 indexer + widening sparse path,
    bf16 q) without having to move off trtllm_mha.
    """
    from sglang.srt.environ import envs

    return (
        args.kv_cache_dtype == "fp8_e4m3"
        and args.attention_backend == "trtllm_mha"
        and get_platform().is_sm100
        and not envs.SGLANG_DISABLE_M3_FP8_ATTN_GEMM.get()
    )


@functools.lru_cache(maxsize=1)
def _underscore_field_names() -> frozenset:
    """Configuration fields that must stay sealed despite their underscore prefix."""
    return frozenset(
        field.name for field in record_fields(ServerArgs) if field.name.startswith("_")
    )


# NOTE: The process-wide ServerArgs is owned by the runtime context
# (sglang.srt.runtime_context). The two publish functions below are LEGACY
# shims kept for the existing call-sites; they hand over the same live object
# by reference. Do not add new call-sites. The third function is retired and
# only raises.
# Imports are in-function so the two modules stay cycle-free at import time.
def set_global_server_args_for_scheduler(server_args: ServerArgs):
    """Legacy publish shim (role=scheduler) — prefer
    ``runtime_context.publish(server_args, role=...)`` in new code."""

    publish(server_args, role="scheduler")


def set_global_server_args_for_tokenizer(server_args: ServerArgs):
    """Legacy publish shim (role=tokenizer). Not aliased to the scheduler shim:
    the process role differs."""

    publish(server_args, role="tokenizer")


def get_global_server_args() -> NoReturn:
    """Retired accessor retained to raise a migration error for existing imports."""
    raise RuntimeError(
        "get_global_server_args() is retired. Read the value that is in effect "
        "from its namespace bag -- `get_exec().kernel.attention_backend`, "
        "`get_schedule().max_running_requests`, and so on "
        "(sglang.srt.runtime_context). For the operator's raw input, which is a "
        "different question, `get_server_args()` still answers it."
    )


def _rebuild_server_args(cls, fields, bookkeeping):
    """Rebuild a pickled record: fields through the constructor, the rest after."""
    record = cls(**fields)
    record.__dict__.update(bookkeeping)
    return record


def _declared_default(name: str):
    """Return a field default; Struct class attributes are slot descriptors."""
    return next(
        field.default
        for field in msgspec.structs.fields(ServerArgs)
        if field.name == name
    )


def prepare_server_args(argv: list[str]) -> ServerArgs:
    """
    Prepare the server arguments from the command line arguments.

    Args:
        args: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The server arguments.
    """
    parser = argparse.ArgumentParser(prog="sglang serve")
    ServerArgs.add_cli_args(parser)

    # Check for config file and merge arguments if present
    if "--config" in argv:
        # Import here to avoid circular imports
        from sglang.srt.utils.server_args_config_parser import ConfigArgumentMerger

        # Extract boolean actions from the parser to handle them correctly
        config_merger = ConfigArgumentMerger(parser)
        argv = config_merger.merge_config_with_args(argv)

    radix_eviction_policy_explicitly_set = any(
        arg == "--radix-eviction-policy" or arg.startswith("--radix-eviction-policy=")
        for arg in argv
    )

    raw_args = parser.parse_args(argv)
    raw_args._radix_eviction_policy_explicitly_set = (
        radix_eviction_policy_explicitly_set
    )

    # Set up basic logging before ServerArgs.__post_init__ so that
    # logger.info / logger.warning calls there are properly formatted.
    logging.basicConfig(
        level=getattr(logging, raw_args.log_level.upper()),
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    server_args = ServerArgs.from_cli_args(raw_args)
    # Not a field: the record's fields are the configuration, and this is how
    # the configuration was asked for. It rides along on the record so a
    # subprocess copy can answer the same question the launcher can.
    server_args._launch_command = " ".join(argv)
    return server_args


# --------------------------------------------------------------------------
# Networking constants and PortArgs.
# --------------------------------------------------------------------------


ZMQ_TCP_PORT_DELTA = 233
DP_ATTENTION_HANDSHAKE_PORT_DELTA = 13


@dataclasses.dataclass
class PortArgs:
    # The ipc filename for tokenizer to receive inputs from detokenizer (zmq)
    tokenizer_ipc_name: str
    # The ipc filename for scheduler (rank 0) to receive inputs from tokenizer (zmq)
    scheduler_input_ipc_name: str
    # The ipc filename for detokenizer to receive inputs from scheduler (zmq)
    detokenizer_ipc_name: str

    # The port for nccl initialization (torch.dist)
    nccl_port: int

    # The ipc filename for rpc call between Engine and Scheduler
    rpc_ipc_name: str

    # The ipc filename for Scheduler to send metrics
    metrics_ipc_name: str

    # The ipc filename for MultiTokenizerRouter to receive inputs from TokenizerWorker processes (zmq)
    tokenizer_worker_ipc_name: str | None

    # The ipc endpoints between verifier scheduler and drafter scheduler
    decoupled_spec_ipc_config: DecoupledSpecIpcConfig | None

    # zmq address for load snapshot PUSH/PULL (dp-attention TCP mode only;
    # empty when IPC mode derives the address from instance_id).
    load_collector_ipc_name: str = ""

    # Stable token shared by all processes in one server instance, used to
    # derive the /dev/shm path for load snapshots.
    instance_id: str = ""

    @staticmethod
    def init_new(
        server_args: ServerArgs,
        dp_rank: int | None = None,
        worker_ports: list[int] | None = None,
    ) -> PortArgs:
        cfg = resolving_view(server_args)
        if server_args.nccl_port is None:
            nccl_port = get_free_port()
        else:
            nccl_port = server_args.nccl_port

        if server_args.tokenizer_worker_num == 1:
            tokenizer_worker_ipc_name = None
        else:
            tokenizer_worker_ipc_name = (
                f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
            )

        instance_id = uuid.uuid4().hex[:12]

        decoupled_spec_ipc_config = None
        if server_args.decoupled_spec_role != "null":
            if (
                server_args.decoupled_spec_bind_endpoint is None
                or server_args.decoupled_spec_connect_endpoints is None
                or server_args.decoupled_spec_rank is None
            ):
                raise ValueError(
                    "--decoupled-spec-bind-endpoint, "
                    "--decoupled-spec-connect-endpoints, and "
                    "--decoupled-spec-rank are required for decoupled speculative decoding."
                )
            decoupled_spec_ipc_config = DecoupledSpecIpcConfig(
                bind_endpoint=server_args.decoupled_spec_bind_endpoint,
                connect_endpoints=tuple(server_args.decoupled_spec_connect_endpoints),
                rank=int(server_args.decoupled_spec_rank),
            )

        if not cfg.enable_dp_attention:
            # Normal case, use IPC within a single node
            return PortArgs(
                tokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                detokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                nccl_port=nccl_port,
                rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                tokenizer_worker_ipc_name=tokenizer_worker_ipc_name,
                decoupled_spec_ipc_config=decoupled_spec_ipc_config,
                instance_id=instance_id,
            )
        else:
            # DP attention. Use TCP + port to handle both single-node and multi-node.
            if server_args.nnodes == 1 and server_args.dist_init_addr is None:
                derived_port = server_args.port + ZMQ_TCP_PORT_DELTA
                if derived_port > 65535:
                    derived_port = server_args.port - ZMQ_TCP_PORT_DELTA
                na = NetworkAddress("127.0.0.1", derived_port)
            else:
                na = NetworkAddress.parse(server_args.dist_init_addr)

            dist_init_host = na.host
            dist_init_port = na.port

            # Reserve port_base+0..NUM_DERIVED_PORTS-1 (6 fixed ports + dp_size
            # rust-path slots); derive from server_args only (never dp_rank) so
            # every init_new call agrees, decrementing below dist_init_port on
            # overflow.
            is_rust_server = envs.SGLANG_RUST_SERVER.get()
            NUM_DERIVED_PORTS = 6 if not is_rust_server else 6 + cfg.dp_size
            if ep_scale_joiner_of(resolving_view(server_args)):
                port_base = server_args.port + ZMQ_TCP_PORT_DELTA
                if port_base + NUM_DERIVED_PORTS > 65535:
                    port_base = server_args.port - ZMQ_TCP_PORT_DELTA
            elif dist_init_port + NUM_DERIVED_PORTS > 65535:
                port_base = dist_init_port - NUM_DERIVED_PORTS - 1
            else:
                port_base = dist_init_port + 1

            detokenizer_port = port_base + 1
            rpc_port = port_base + 2
            metrics_port = port_base + 3
            load_collector_port = port_base + 5
            if dp_rank is None:
                # TokenizerManager to DataParallelController
                scheduler_input_port = port_base + 4
            elif is_rust_server:
                # Rust server path (SGLANG_RUST_SERVER + dp attention): there is no
                # DataParallelController allocating worker ports.
                scheduler_input_port = port_base + 6 + dp_rank
            else:
                assert worker_ports is not None
                scheduler_input_port = worker_ports[dp_rank]

            is_joiner = ep_joiner_of(resolving_view(server_args))
            # Under SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE, SGLang never binds
            # dist_init_port / nccl_port (rendezvous uses the externally-managed
            # store; see distributed/bootstrap.py:_resolve_dist_init_method), so
            # their prechecks could only false-positive and are skipped.
            dist_init_overridden = bool(
                envs.SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE.get()
            )
            try:
                if dp_rank is None:
                    if not (is_joiner or dist_init_overridden):
                        wait_port_available(dist_init_port, "dist_init_port")
                    wait_port_available(port_base, "port_base")
                    wait_port_available(detokenizer_port, "detokenizer_port")
                    if not dist_init_overridden:
                        wait_port_available(nccl_port, "nccl_port")
                    wait_port_available(rpc_port, "rpc_port")
                    wait_port_available(metrics_port, "metrics_port")
                    if server_args.nnodes > 1:
                        wait_port_available(load_collector_port, "load_collector_port")
                # Check scheduler_input_port only for dp.
                # Skip check when using worker_ports since the port is already bound by our ZMQ socket
                if dp_rank is None or worker_ports is None:
                    wait_port_available(scheduler_input_port, "scheduler_input_port")
            except ValueError:
                logger.exception(
                    f"Port is already in use. {dist_init_port=} {port_base=} {detokenizer_port=} {nccl_port=} {scheduler_input_port=}"
                )
                raise

            return PortArgs(
                tokenizer_ipc_name=NetworkAddress(dist_init_host, port_base).to_tcp(),
                scheduler_input_ipc_name=NetworkAddress(
                    dist_init_host, scheduler_input_port
                ).to_tcp(),
                detokenizer_ipc_name=NetworkAddress(
                    dist_init_host, detokenizer_port
                ).to_tcp(),
                nccl_port=nccl_port,
                rpc_ipc_name=NetworkAddress(dist_init_host, rpc_port).to_tcp(),
                metrics_ipc_name=NetworkAddress(dist_init_host, metrics_port).to_tcp(),
                tokenizer_worker_ipc_name=tokenizer_worker_ipc_name,
                decoupled_spec_ipc_config=decoupled_spec_ipc_config,
                load_collector_ipc_name=NetworkAddress(
                    dist_init_host, load_collector_port
                ).to_tcp(),
                instance_id=instance_id,
            )
