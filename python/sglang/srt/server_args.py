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
"""Server argument declarations, resolution, and CLI registration.

Keep this file in the following top-level order:

1. Imports and the module logger.
2. Public extension-point choice lists, with each legacy ``add_*`` alias
   immediately below the choice list it extends.
3. Shared (non-extensible) choice lists, scalar defaults, and deprecated
   aliases. A choice list used by only one field belongs inline in that field.
4. ``ServerArgs``: fields first, then resolution/validation helpers, then CLI
   registration and small query helpers. New resolution steps are appended at
   the end of ``arg_groups.pipeline.run_resolution_pipeline``, immediately
   marked complete, unless an earlier dependency is documented explicitly.
5. Module-level ``ServerArgs`` construction/runtime shims.
6. Networking constants and ``PortArgs``.

Model- or vendor-specific utilities belong in ``sglang.srt.arg_groups`` (or
their owning subsystem), not before ``ServerArgs`` in this module.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import logging
import tempfile
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from sglang.kernels.ops.kv_canary.consts import RealKvHashMode
from sglang.srt.arg_groups.arg_utils import NS, A, Arg, add_cli_args_from_dataclass
from sglang.srt.arg_groups.argparse_actions import (
    DeprecatedAction,
    DeprecatedAliasStoreAction,
    DeprecatedStoreConstAction,
    DeprecatedStoreTrueAction,
    LoRAPathAction,
)
from sglang.srt.arg_groups.model_override_base import ep_joiner_of, ep_scale_joiner_of
from sglang.srt.arg_groups.overrides import (
    remote_instance_transfer_engine_of,
    resolution_projection,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    parse_cuda_graph_config_arg,
)
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.runtime_context import (
    get_context,
    get_platform,
    publish,
)
from sglang.srt.speculative.decoupled_spec_io import DecoupledSpecIpcConfig
from sglang.srt.utils.common import (
    LORA_TARGET_ALL_MODULES,
    SUPPORTED_LORA_TARGET_MODULES,
    human_readable_int,
    json_list_type,
    nullable_str,
)
from sglang.srt.utils.network import NetworkAddress, get_free_port, wait_port_available

logger = logging.getLogger(__name__)

# Re-exported. These were importable from this module while the field
# declarations that used them lived here; the declarations moved to
# `arg_groups/fields/` but out-of-tree code -- and `tokenizer_control_mixin`
# for `LoRARef` -- still reaches them through `sglang.srt.server_args`.
from sglang.srt.arg_groups.arg_utils import NS, A, Arg  # noqa: F401
from sglang.srt.arg_groups.argparse_actions import LoRAPathAction  # noqa: F401

# Re-exported for out-of-tree plugins, which have always reached these
# through `sglang.srt.server_args`. The lists and their adders moved to
# `arg_groups/choices.py` with the field declarations that name them.
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


class ServerArgs:
    """Server-wide configuration for SGLang.

    Adding new arguments
    --------------------
    1. **Place the field in the right section.** Arguments are grouped by
       comment blocks (``# Model and tokenizer``, ``# LoRA``, etc.).
       Add new fields to the matching section, or create a new section
       with a ``# ---`` banner when none fits.

    2. **Use the ``A[T, ...]`` annotation.**  ``A`` is an alias for
       ``typing.Annotated``.  The primary CLI flag is auto-derived from the
       field name (``tp_size`` → ``--tp-size``).  Use ``aliases`` for
       longer alternate names
       (``aliases=["--tensor-parallel-size"]``)::

           # Bare string — simplest form (just help text):
           host: A[str, "The host of the HTTP server."] = "127.0.0.1"
           trust_remote_code: A[bool, "Whether to allow custom models."] = False

           # Arg(...) — when you need choices, aliases, type_parser, etc.:
           load_format: A[str, Arg(help="...", choices=CHOICES)] = "auto"
           model_path: A[str, Arg(help="...", aliases=["--model"])]

       See ``Arg`` in ``arg_groups/arg_utils.py`` for the full list of
       supported metadata (``choices``, ``aliases``, ``type_parser``,
       ``nargs``, ``const``, ``action``, ``no_cli``, …).

    3. **Manual entries in ``add_cli_args`` — only for special cases.**
       A few arguments cannot use the annotation style and must be
       registered manually in ``add_cli_args``:

       - **Deprecated flags** that redirect to another field via
         ``DeprecatedAction`` / ``DeprecatedAliasStoreAction`` / etc.
       - **Dynamic choices** computed at runtime (e.g. ``reasoning_parser``
         whose choices come from a plugin registry).
       - The ``--config`` meta-argument (not a dataclass field).

       Everything else should use the ``A[T, ...]`` annotation.

    The fields live in ``arg_groups/fields/`` -- one class per config
    namespace, composed here. Reverse-MRO order puts the last base's fields
    first, which is why ``Model`` is last: ``model_path`` is the one field
    with no default, so it has to stay the first positional argument.
    """

    def __post_init__(self):
        """Construction leaves the record at what the caller asked for.

        Resolution is a separate act, entered through ``resolve_once``: the
        launcher runs it once per engine, and every publishing process asks the
        gate on the way in. A record that is only constructed -- a fixture, a
        config being inspected, one being handed to a subprocess that will
        resolve it itself -- stays raw.
        """

    def resolve_once(self) -> None:
        """Run the resolution pipeline, unless this record has been through it.

        Resolution is a deterministic function of the raw inputs -- two records
        built from the same arguments declare the same things -- but the
        handlers do not survive a second pass over their own output: DP
        attention halves ``chunked_prefill_size`` again on every re-entry.

        The publishing entry of every process calls this. In a child the record
        arrived by pickle and brought its declarations along, so the child has
        nothing left to derive and projects what the parent decided.
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

        # Sealed for the duration, not just afterwards: everything below this
        # line reads the input and declares against it, and the one channel
        # that still writes the record (`declare_direct_writes`, for
        # out-of-tree platform plugins) asks for the seal to be lifted by name.
        self._input_frozen = True
        try:
            run_resolution_pipeline(self)
        except BaseException:
            # The handlers that ran already declared, and they are not
            # idempotent over their own output.
            self._resolution_failed = True
            raise
        finally:
            self._input_frozen = False
        # Set here too, because the dummy/absent-model path returns before the
        # end of the pipeline that normally sets it: the gate is about whether
        # the handlers ran, not how far they got.
        self._resolution_finished = True

    @property
    def launch_command(self) -> Optional[str]:
        """How this record was created, verbatim.

        `resolved_dict` answers with what resolution decided; this answers with
        what the operator asked for, which is a different question and the one
        "why is this server configured like this?" usually means. The two are
        not derivable from each other: a field the operator never set reads the
        same as one they set to the value resolution would have picked anyway.

        The launcher stores the arguments it parsed; the in-process `Engine`
        stores the call that built the record, since there was no command line.
        `None` on a record built directly (a fixture, a subprocess copy that
        predates this, a config being inspected).
        """
        return getattr(self, "_launch_command", None)

    def resolved_dict(self) -> Dict[str, Any]:
        """This configuration as a plain dict of resolved field values.

        What the whole-object readbacks report (`/server_info` and its gRPC and
        in-process twins). `dataclasses.asdict(self)` reads the fields, which
        carry the raw input; this reads the declarations, so it answers with what
        resolution decided. Nested dataclass fields are expanded
        the way `asdict` expands them; the private resolution bookkeeping and the
        `model_config` memo are not fields and do not appear.
        """

        return resolution_projection(self)

    def replace_resolved(self, source: str, **changes: Any) -> ServerArgs:
        """A copy of this record that stays resolved, and says what it changed.

        `dataclasses.replace` builds a new instance, so the copy carries none of
        what makes a record resolved: no raw snapshot, no declarations, no
        finished flag. The next publish therefore resolves it again, which
        drops every decision the stash held -- the late ones (the auto-detected
        parsers) and the direct ones alike -- and re-runs the device probes in
        whatever process opened the copy. The Ray paths replace
        `dist_init_addr` on a resolved record, which is how they reach this.

        The change is appended to the stash rather than left on the field: the
        projection reads the raw snapshot plus the declarations, so a field the
        copy set on its own would publish the parent's raw value instead.

        The carry is shallow. The containers are copied so the copy's own
        declaration does not travel back into the parent, but everything inside
        them -- the stash entries, the raw-input values, the memoized
        `ModelConfig` -- is shared. That is fine for what this is for: a copy
        that immediately crosses a process boundary (Ray actors, the gateway's
        workers), where pickling severs the sharing. A caller that mutates the
        copy's deep structure in-process mutates the parent's too.
        """
        replacement = dataclasses.replace(self, **changes)
        # Provenance, not resolution state: a copy was still launched by
        # whatever launched its parent, resolved or not.
        object.__setattr__(replacement, "_launch_command", self.launch_command)
        if not getattr(self, "_resolution_finished", False):
            # Not resolved yet: the copy goes through the gate itself.
            return replacement

        # Everything outside the fields, enumerated from the instance: the raw
        # snapshot, the stash, and what resolution memoized -- including the
        # model-configuration memo, which the copy carries over rather than
        # rebuild.
        field_names = {field.name for field in dataclasses.fields(self)}
        for name, value in vars(self).items():
            if name in field_names or name == "_resolution_finished":
                continue
            if isinstance(value, (dict, list, set)):
                value = copy.copy(value)
            object.__setattr__(replacement, name, value)
        stash = getattr(replacement, "_resolved_overrides", None)
        if stash is None:
            stash = []
            object.__setattr__(replacement, "_resolved_overrides", stash)
        if changes:
            stash.append((source, dict(changes)))
        object.__setattr__(replacement, "_resolution_finished", True)
        return replacement

    # ------------------------------------------------------------------
    # CUDA graph configuration resolution
    # ------------------------------------------------------------------

    # ===== END TO BE REFACTORED ====

    LANGUAGE_MODEL_ONLY_ARCHITECTURES = (
        "MuseGlimmerForConditionalGeneration",
        "Cosmos3ForConditionalGeneration",
        "Cosmos3EdgeForConditionalGeneration",
    )

    # The attention-backend allow-list is enforced via
    # --enable-page-major-kv-layout (implied by the unified pool in
    # _handle_page_major_kv_layout); the model-family gate is enforced at pool
    # construction in model_runner_kv_cache_mixin._init_pools.

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):

        # Auto-derived from Annotated[..., Arg(...)] field metadata.
        add_cli_args_from_dataclass(parser, ServerArgs)

        # --- Fields with dynamic choices (computed at add_cli_args time) ---
        sampling_backend_choices = set(SAMPLING_BACKEND_CHOICES)
        if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get():
            sampling_backend_choices.add("token_oracle")
        parser.add_argument(
            "--sampling-backend",
            type=str,
            choices=sampling_backend_choices,
            default=ServerArgs.sampling_backend,
            help="Choose the kernels for sampling layers.",
        )

        reasoning_parser_choices = list(ReasoningParser.DetectorMap.keys())
        parser.add_argument(
            "--reasoning-parser",
            type=str,
            choices=["auto"] + reasoning_parser_choices,
            default=ServerArgs.reasoning_parser,
            help=f"Specify the parser for reasoning models. "
            f"Use 'auto' to detect from chat template. "
            f"Options include: {reasoning_parser_choices}.",
        )
        tool_call_parser_choices = list(FunctionCallParser.ToolCallParserEnum.keys())
        parser.add_argument(
            "--tool-call-parser",
            type=str,
            choices=["auto"] + tool_call_parser_choices,
            default=ServerArgs.tool_call_parser,
            help=f"Specify the parser for handling tool-call interactions. "
            f"Use 'auto' to detect from chat template. "
            f"Options include: {tool_call_parser_choices}.",
        )
        parser.add_argument(
            "--kv-canary-real-data",
            type=str,
            default=ServerArgs.kv_canary_real_data,
            choices=[m.name.lower() for m in RealKvHashMode],
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
        parser.add_argument(
            "--enable-expert-distribution-metrics",
            action=DeprecatedAction,
            error_message=(
                "--enable-expert-distribution-metrics is no longer supported. Use "
                "--expert-balancedness-report-mode with one of: off, server_log, "
                "prometheus, both."
            ),
            help=(
                "Removed. Use --expert-balancedness-report-mode with one of: "
                "off, server_log, prometheus, both."
            ),
        )
        parser.add_argument(
            "--stream-output",
            action=DeprecatedStoreTrueAction,
            dest="incremental_streaming_output",
            new_flag="--incremental-streaming-output",
            help="[Deprecated] Use --incremental-streaming-output instead.",
        )
        parser.add_argument(
            "--prefill-round-robin-balance",
            action=DeprecatedAction,
            help="Note: --prefill-round-robin-balance is deprecated now.",
        )
        parser.add_argument(
            "--collect-tokens-histogram",
            action=DeprecatedAction,
            help="Deprecated. Token histograms are now automatically collected when --enable-metrics is set.",
        )
        parser.add_argument(
            "--nsa-prefill-backend",
            dest="dsa_prefill_backend",
            action=DeprecatedAliasStoreAction,
            new_flag="--dsa-prefill-backend",
            default=argparse.SUPPRESS,
            type=str,
            choices=[
                "flashmla_sparse",
                "flashmla_sparse_q8",
                "flashmla_kv",
                "flashmla_auto",
                "flashinfer_sparse_mla",
                "fa3",
                "tilelang",
                "aiter",
                "trtllm",
            ],
            help="[Deprecated] Use --dsa-prefill-backend instead.",
        )
        parser.add_argument(
            "--nsa-decode-backend",
            dest="dsa_decode_backend",
            action=DeprecatedAliasStoreAction,
            new_flag="--dsa-decode-backend",
            default=argparse.SUPPRESS,
            type=str,
            choices=[
                "flashmla_sparse",
                "flashmla_sparse_q8",
                "flashmla_kv",
                "flashmla_auto",
                "flashinfer_sparse_mla",
                "fa3",
                "tilelang",
                "aiter",
                "trtllm",
            ],
            help="[Deprecated] Use --dsa-decode-backend instead.",
        )
        parser.add_argument(
            "--speculative-dflash-draft-window-size",
            type=int,
            dest="speculative_draft_window_size",
            action=DeprecatedAliasStoreAction,
            new_flag="--speculative-draft-window-size",
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--mamba-scheduler-strategy",
            dest="mamba_radix_cache_strategy",
            type=str,
            action=DeprecatedAliasStoreAction,
            new_flag="--mamba-radix-cache-strategy",
            default=ServerArgs.mamba_radix_cache_strategy,
            help="Deprecated alias for --mamba-radix-cache-strategy.",
        )
        parser.add_argument(
            "--cuda-graph-max-bs",
            type=int,
            action=DeprecatedAliasStoreAction,
            new_flag="--cuda-graph-max-bs-decode",
            dest="cuda_graph_max_bs_decode",
            help="Deprecated alias for --cuda-graph-max-bs-decode.",
        )
        parser.add_argument(
            "--cuda-graph-bs",
            type=int,
            nargs="+",
            action=DeprecatedAliasStoreAction,
            new_flag="--cuda-graph-bs-decode",
            dest="cuda_graph_bs_decode",
            help="Deprecated alias for --cuda-graph-bs-decode.",
        )
        parser.add_argument(
            "--disable-cuda-graph",
            action=DeprecatedStoreTrueAction,
            new_flag="--cuda-graph-backend-{decode,prefill}=disabled",
            help="Deprecated. Use --cuda-graph-backend-{decode,prefill}=disabled instead.",
        )
        parser.add_argument(
            "--enable-breakable-cuda-graph",
            action=DeprecatedStoreConstAction,
            dest="cuda_graph_backend_prefill",
            const_value=Backend.BREAKABLE,
            new_flag="--cuda-graph-backend-prefill=breakable",
            help="Deprecated alias for --cuda-graph-backend-prefill=breakable.",
        )
        parser.add_argument(
            "--disable-piecewise-cuda-graph",
            action=DeprecatedStoreConstAction,
            dest="cuda_graph_backend_prefill",
            const_value=Backend.DISABLED,
            new_flag="--cuda-graph-backend-prefill=disabled",
            help="Deprecated alias for --cuda-graph-backend-prefill=disabled.",
        )
        parser.add_argument(
            "--enforce-piecewise-cuda-graph",
            action=DeprecatedStoreConstAction,
            dest="cuda_graph_backend_prefill",
            const_value=Backend.TC_PIECEWISE,
            new_flag="--cuda-graph-backend-prefill=tc_piecewise",
            help="Deprecated alias for --cuda-graph-backend-prefill=tc_piecewise. "
            "Explicitly setting the prefill backend now skips the auto-disable "
            "cascade automatically.",
        )
        parser.add_argument(
            "--piecewise-cuda-graph-tokens",
            type=int,
            nargs="+",
            action=DeprecatedAliasStoreAction,
            new_flag="--cuda-graph-bs-prefill",
            dest="cuda_graph_bs_prefill",
            help="Deprecated alias for --cuda-graph-bs-prefill.",
        )
        parser.add_argument(
            "--piecewise-cuda-graph-compiler",
            type=str,
            choices=["eager", "inductor"],
            action=DeprecatedAliasStoreAction,
            new_flag="--cuda-graph-tc-compiler",
            dest="cuda_graph_tc_compiler",
            help="Deprecated alias for --cuda-graph-tc-compiler.",
        )
        parser.add_argument(
            "--piecewise-cuda-graph-max-tokens",
            type=int,
            action=DeprecatedAliasStoreAction,
            new_flag="--cuda-graph-max-bs-prefill",
            dest="cuda_graph_max_bs_prefill",
            help="Deprecated alias for --cuda-graph-max-bs-prefill.",
        )
        parser.add_argument(
            "--enable-nsa-prefill-context-parallel",
            dest="enable_dsa_prefill_context_parallel",
            action=DeprecatedStoreTrueAction,
            new_flag="--enable-prefill-cp",
            help="[Deprecated] Use --enable-prefill-cp instead.",
        )
        parser.add_argument(
            "--enable-gdn-replayssm-spec",
            dest="enable_linear_replayssm_spec",
            action=DeprecatedStoreTrueAction,
            new_flag="--enable-linear-replayssm-spec",
            help="[Deprecated] Use --enable-linear-replayssm-spec instead.",
        )
        parser.add_argument(
            "--enable-prefill-context-parallel",
            dest="enable_prefill_context_parallel",
            action=DeprecatedStoreTrueAction,
            new_flag="--enable-prefill-cp",
            help="[Deprecated] Use --enable-prefill-cp instead.",
        )
        parser.add_argument(
            "--nsa-prefill-cp-mode",
            dest="dsa_prefill_cp_mode",
            action=DeprecatedAliasStoreAction,
            new_flag="--cp-strategy",
            type=str,
            default=argparse.SUPPRESS,
            choices=["in-seq-split", "round-robin-split"],
            help="[Deprecated] Use --cp-strategy instead.",
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
        attrs = [
            attr.name for attr in dataclasses.fields(cls) if hasattr(args, attr.name)
        ]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def get_tokenizer_worker_class(self):
        from sglang.srt.managers.multi_tokenizer_mixin import TokenizerWorker

        return TokenizerWorker

    def url(self, port: Optional[int] = None):
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
        # The record holds the operator's input. It is writable while the
        # caller is still assembling it and sealed from the moment resolution
        # starts: a resolver that writes a field would overwrite the very thing
        # the record exists to remember, and the decision it meant to record
        # belongs in the stash, where it carries a source and does not destroy
        # the input it was derived from.
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
        object.__setattr__(self, name, value)

    def check_server_args(self):
        from sglang.srt.arg_groups.validation_hook import check_server_args

        check_server_args(self)

    def remote_instance_weight_loader_use_transfer_engine(self, load_format=None):
        """``load_format`` overrides the seed's: a draft runner loading under
        ``--speculative-draft-load-format`` needs its own transfer engine."""
        return remote_instance_transfer_engine_of(resolving_view(self), load_format)


# ---------------------------------------------------------------------------
# The record is assembled, not inherited.
#
# Its fields are exactly the operator's input, and the classes below are the
# statement of that: a namespace declares its input fields and its derived ones
# side by side, and only the input half is collected here. Inheriting the
# namespace classes would have made the record's contents a property of which
# classes happen to appear in a base list -- correct today, and silently wrong
# the first time a derived field is declared in an inherited class.
#
# The order is the order the fields were declared in before the split, so the
# CLI registers its options in the same sequence and `model_path` stays the
# first positional argument.
# ---------------------------------------------------------------------------


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
# The assembled record has no base classes, so it carries the map the classes
# used to answer through their `_NS_PATH`.
ServerArgs._NS_BY_FIELD = _namespaces
# The classes themselves, so the bag projection can find the declarations
# that are not fields -- the derived half of each namespace.
ServerArgs._NAMESPACES = _INPUT_NAMESPACES
for _name, _value in _defaults.items():
    setattr(ServerArgs, _name, _value)
ServerArgs = dataclasses.dataclass(ServerArgs)


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


# NOTE: The process-wide ServerArgs is owned by the runtime context
# (sglang.srt.runtime_context). The two functions below are LEGACY shims kept
# for the existing call-sites; they publish/read the same live object by
# reference. Do not add new call-sites — the counts are ratcheted
# (decrease-only) by test/registered/unit/test_legacy_global_ratchet.py.
# Imports are in-function so the two modules stay cycle-free at import time.
@functools.lru_cache(maxsize=1)
def _underscore_field_names() -> frozenset:
    """Real dataclass fields whose names start with an underscore.

    The read-only guard exempts underscore names because they are the record's
    own bookkeeping (the stash, the flags, the cache keys). A *field* that
    happens to start with an underscore is still resolved configuration --
    `_speculative_draft_quantization_explicitly_set` is one -- and exempting it
    by spelling would leave exactly one leaf writable on a read-only record.
    """
    return frozenset(
        field.name
        for field in dataclasses.fields(ServerArgs)
        if field.name.startswith("_")
    )


def set_global_server_args_for_scheduler(server_args: ServerArgs):
    """Legacy publish shim (role=scheduler) — prefer
    ``runtime_context.publish(server_args, role=...)`` in new code."""

    publish(server_args, role="scheduler")


def set_global_server_args_for_tokenizer(server_args: ServerArgs):
    """Legacy publish shim (role=tokenizer). Not aliased to the scheduler shim:
    the process role differs."""

    publish(server_args, role="tokenizer")


def get_global_server_args() -> ServerArgs:
    """Legacy accessor shim — prefer ``get_server_args()`` from
    ``sglang.srt.runtime_context`` in new code."""

    return get_context().server_args


@contextmanager
def record_writable(server_args: Any):
    """Lift the input seal for a resolver that genuinely writes the record.

    There is exactly one: `declare_direct_writes`, which hands the record to an
    out-of-tree platform plugin that sets fields on it. Those implementations
    live outside this tree and cannot be converted by editing a resolver here,
    so the write stays and is captured into the stash afterwards. Naming the
    exception is the point -- an in-tree resolver that reaches for this is
    doing something it should be declaring instead.
    """
    frozen = getattr(server_args, "_input_frozen", False)
    if frozen:
        object.__setattr__(server_args, "_input_frozen", False)
    try:
        yield
    finally:
        if frozen:
            object.__setattr__(server_args, "_input_frozen", True)


def prepare_server_args(argv: List[str]) -> ServerArgs:
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

    raw_args = parser.parse_args(argv)

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
    object.__setattr__(server_args, "_launch_command", " ".join(argv))
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
    tokenizer_worker_ipc_name: Optional[str]

    # The ipc endpoints between verifier scheduler and drafter scheduler
    decoupled_spec_ipc_config: Optional[DecoupledSpecIpcConfig]

    # zmq address for load snapshot PUSH/PULL (dp-attention TCP mode only;
    # empty when IPC mode derives the address from instance_id).
    load_collector_ipc_name: str = ""

    # Stable token shared by all processes in one server instance, used to
    # derive the /dev/shm path for load snapshots.
    instance_id: str = ""

    @staticmethod
    def init_new(
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        worker_ports: Optional[List[int]] = None,
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
