"""Configuration and startup validation for the AFD runtime."""

from __future__ import annotations

from enum import Enum
from typing import Any

import msgspec

from .contracts import M2N_FFN_LANES, M2N_STAGE_ROWS, AFDError, AFDRole


class AFDExecutionMode(str, Enum):
    OFF = "off"
    ATTENTION = "attention"
    FFN = "ffn"

    @property
    def role(self) -> AFDRole | None:
        if self == AFDExecutionMode.ATTENTION:
            return AFDRole.ATTENTION
        if self == AFDExecutionMode.FFN:
            return AFDRole.FFN
        return None


def _admitted_attention_backends() -> tuple[str, ...]:
    """Union over registered profiles; the per-profile gate stays authoritative."""

    from .profiles import registered_capability_profiles

    return tuple(
        sorted(
            {
                backend
                for profile in registered_capability_profiles()
                for backend in profile.fa_backends
            }
        )
    )


def _max_admitted_lanes() -> int:
    """Upper bound over registered profiles; the per-profile gate stays authoritative."""

    from .profiles import registered_capability_profiles

    return max(
        (
            lanes
            for profile in registered_capability_profiles()
            for lanes in profile.lanes
        ),
        default=0,
    )


class AFDConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """The sole AFD feature config; all retained allocation is bounded here."""

    stages: int = 2
    experimental_m2n: bool = False
    experimental_m2n_stage_rows: int = 16
    experimental_direct_transport: str = "nccl"
    # None is the capability-based default. An explicit legacy false still
    # selects the fixed F path, including when the new selector is omitted.
    experimental_direct: bool | None = None
    ffn_backend: str = "auto"
    experimental_direct_stage_rows: int = 32
    max_hbm_bytes: int = 8 * 1024 * 1024 * 1024
    attention_backend: str = "trtllm_mha"
    lanes: int = 1
    # FFN ranks are `lanes`; attention ranks are independent positive integers.
    # None selects the symmetric layout. Model/parallelism checks still apply.
    attention_lanes: int | None = None
    rendezvous_host: str = "127.0.0.1"
    rendezvous_port: int = 1239
    # Each role reaches the rendezvous only after loading its own weights, so this
    # must cover the skew between them; a 705 GB model over NFS blows past the
    # 300 s TCPStore default long before the peer arrives.
    rendezvous_timeout_seconds: int = 1800
    # ...but once both roles are up, that same store carries the FFN role's wait
    # for the next step descriptor, and an idle server must not die just because
    # no request arrived. Rendezvous wants to give up (a peer that never appears
    # is a misconfiguration); serving wants to wait (an empty queue is normal),
    # so the two phases get their own knobs. Not unbounded: a peer that crashed
    # has to eventually release its GPUs rather than hold them until reboot.
    idle_timeout_seconds: int = 86400
    close_timeout_seconds: int = 30
    # Every NCCL channel costs one CTA that spins for the whole transfer, so an
    # uncapped dispatch/return communicator starves the co-resident attention
    # and MoE kernels of SMs.
    nccl_num_channels: int = 8

    @property
    def attention_lane_count(self) -> int:
        """Attention ranks: `attention_lanes` when set, else symmetric."""

        return self.lanes if self.attention_lanes is None else self.attention_lanes

    @property
    def lanes_per_ffn(self) -> int:
        """Maximum A ingress group width; 1 in the symmetric case."""

        return (self.attention_lane_count + self.lanes - 1) // self.lanes

    @property
    def direct_mode(self) -> str:
        if self.experimental_direct is not None:
            return "direct" if self.experimental_direct else "fixed"
        return self.ffn_backend

    @classmethod
    def from_json(cls, value: str) -> AFDConfig:
        try:
            config = msgspec.json.decode(value, type=cls, strict=True)
        except (msgspec.DecodeError, msgspec.ValidationError) as exc:
            raise ValueError(f"AFD_GRAPH_CONFIG_INVALID: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if self.experimental_direct_transport not in ("nccl",) or (
            self.experimental_direct_transport != "nccl"
            and not self.experimental_direct
        ):
            raise AFDError("AFD_DIRECT_TRANSPORT_UNSUPPORTED")
        if (
            self.experimental_direct is not None
            and type(self.experimental_direct) is not bool
        ):
            raise AFDError("AFD_DIRECT_CONFIG_INVALID")
        if self.ffn_backend not in ("auto", "fixed", "direct"):
            raise AFDError("AFD_FFN_BACKEND_INVALID")
        if (
            type(self.experimental_direct_stage_rows) is not int
            or self.experimental_direct_stage_rows < 1
        ):
            raise AFDError("AFD_DIRECT_CAPACITY_INVALID")
        if self.experimental_m2n and self.direct_mode == "direct":
            raise AFDError("AFD_DIRECT_SCOPE_UNSUPPORTED")
        if type(self.experimental_m2n) is not bool:
            raise AFDError("AFD_M2N_CONFIG_INVALID")
        if (
            type(self.experimental_m2n_stage_rows) is not int
            or self.experimental_m2n_stage_rows not in M2N_STAGE_ROWS
        ):
            raise AFDError("AFD_M2N_STAGE_ROWS_UNSUPPORTED", str(M2N_STAGE_ROWS))
        if self.experimental_m2n and (
            self.lanes not in M2N_FFN_LANES
            or self.attention_lane_count != 8
            or self.stages != 2
        ):
            raise AFDError("AFD_M2N_TOPOLOGY_UNSUPPORTED", "requires 8A{2,4,8}F/S2")
        if self.stages != 2:
            raise AFDError(
                "AFD_GRAPH_STAGE_COUNT_UNSUPPORTED",
                f"stages={self.stages}",
            )
        if self.max_hbm_bytes < 1:
            raise AFDError(
                "AFD_GRAPH_HBM_LIMIT_INVALID",
                f"max_hbm_bytes={self.max_hbm_bytes}",
            )
        if self.attention_backend not in _admitted_attention_backends():
            raise AFDError(
                "AFD_GRAPH_FA_BACKEND_INVALID",
                f"attention_backend={self.attention_backend!r}",
            )
        if type(self.lanes) is not int or self.lanes < 1:
            raise AFDError(
                "AFD_GRAPH_LANE_COUNT_INVALID",
                f"lanes={self.lanes}",
            )
        if self.attention_lanes is not None and (
            type(self.attention_lanes) is not int or self.attention_lanes < 1
        ):
            raise AFDError(
                "AFD_GRAPH_ATTENTION_LANE_COUNT_INVALID",
                f"attention_lanes={self.attention_lanes} lanes={self.lanes}",
            )
        if (
            not self.rendezvous_host
            or self.rendezvous_port < 1
            or self.rendezvous_port + self.lanes + 1 > 65535
        ):
            raise AFDError(
                "AFD_GRAPH_RENDEZVOUS_INVALID",
                f"host={self.rendezvous_host!r} port={self.rendezvous_port} "
                f"lanes={self.lanes}",
            )
        if self.close_timeout_seconds < 1 or self.close_timeout_seconds > 300:
            raise AFDError(
                "AFD_GRAPH_CLOSE_TIMEOUT_INVALID",
                f"seconds={self.close_timeout_seconds}",
            )
        if (
            self.rendezvous_timeout_seconds < 1
            or self.rendezvous_timeout_seconds > 7200
        ):
            raise AFDError(
                "AFD_GRAPH_RENDEZVOUS_TIMEOUT_INVALID",
                f"seconds={self.rendezvous_timeout_seconds}",
            )
        # Floor at the close timeout's own ceiling: the shutdown handshake narrows
        # this same store down to `close_timeout_seconds`, so a serving timeout
        # below that would be the tighter of the two and would make close's
        # explicit deadline unreachable.
        if self.idle_timeout_seconds < 300 or self.idle_timeout_seconds > 604800:
            raise AFDError(
                "AFD_GRAPH_IDLE_TIMEOUT_INVALID",
                f"seconds={self.idle_timeout_seconds}",
            )
        if self.nccl_num_channels < 1 or self.nccl_num_channels > 32:
            raise AFDError(
                "AFD_GRAPH_NCCL_CHANNELS_INVALID",
                f"nccl_num_channels={self.nccl_num_channels}",
            )


def execution_mode_from_server_args(server_args: Any) -> AFDExecutionMode:
    """Parse a role without mutating process state."""

    try:
        return AFDExecutionMode(server_args.afd_execution_mode)
    except ValueError as exc:
        raise AFDError(
            "AFD_EXECUTION_MODE_INVALID",
            f"mode={server_args.afd_execution_mode!r}",
        ) from exc


def validate_afd_server_args(server_args: Any) -> None:
    """Validate topology and exclusions before either role starts."""

    mode = execution_mode_from_server_args(server_args)
    config = server_args.afd_config
    if mode == AFDExecutionMode.OFF:
        if config is not None:
            raise AFDError(
                "AFD_GRAPH_WITHOUT_EXECUTION_MODE",
                "set --afd-execution-mode attention or ffn",
            )
        return
    if config is None:
        raise AFDError(
            "AFD_GRAPH_CONFIG_REQUIRED",
            f"mode={mode.value}",
        )
    config.validate()
    if mode == AFDExecutionMode.FFN:
        # These loaders need ModelRunner-owned transport/cache lifecycle which
        # the standalone F service does not create. Reject before CUDA startup.
        if (
            getattr(server_args, "load_format", "auto") == "remote_instance"
            or getattr(
                server_args,
                "remote_instance_weight_loader_start_seed_via_transfer_engine",
                False,
            )
            or getattr(server_args, "weight_cache_mode", "off") != "off"
        ):
            raise AFDError("AFD_FFN_LOADER_LIFECYCLE_UNSUPPORTED")
    from .profiles import validate_startup_capabilities

    validate_startup_capabilities(
        server_args=server_args,
        config=config,
    )
    # CPU-overlap scheduling used to be refused outright. It is admitted now
    # because the scheduler no longer runs forwards on a second thread: the
    # overlap loop keeps a one-deep result queue and issues every forward into
    # one stream, so exactly one forward is ever in flight. AFD's per-step state
    # and the graph's static input buffers are therefore ordered by the stream,
    # not by the absence of overlap. What overlap changes is only *when* the
    # host does the next step's work -- before the current step's device work
    # drains instead of after it.
    if server_args.enable_two_batch_overlap:
        raise AFDError("AFD_TBO_UNSUPPORTED")
    if server_args.enable_single_batch_overlap:
        raise AFDError("AFD_SINGLE_BATCH_OVERLAP_UNSUPPORTED")
    dsv4_ffn_megamoe = (
        mode == AFDExecutionMode.FFN
        and config.attention_backend == "dsv4"
        and server_args.moe_a2a_backend == "megamoe"
        and getattr(server_args, "moe_runner_backend", None) == "deep_gemm"
    )
    if server_args.moe_a2a_backend != "none" and not dsv4_ffn_megamoe:
        raise AFDError(
            "AFD_INTERNAL_COLLECTIVE_UNSUPPORTED",
            f"role={mode.value} moe_a2a_backend={server_args.moe_a2a_backend!r} "
            "expected='none' or the DSV4 FFN MegaMoE/DeepGEMM row adapter",
        )
    if (
        server_args.speculative_algorithm is not None
        or server_args.speculative_num_steps is not None
    ):
        raise AFDError("AFD_MTP_SPECULATIVE_UNSUPPORTED")
