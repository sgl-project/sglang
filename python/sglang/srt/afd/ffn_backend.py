"""Routed FFN execution plugins for AFD.

Boundary: A supplies hidden states and global top-k IDs/weights; dispatch writes
weighted expert outputs in the SAME source-row order into caller-owned storage
and returns an event that the framework waits before consuming that storage.
F executes one complete decoder step, with identical layer/stage order on A/F.
A backend may use per-layer calls or one fused cross-layer persistent kernel.

Owned here: backend selection/qualification, A/F data exchange, expert execution,
private buffers, prepared weight views and backend resources. Not owned here:
request scheduling, KV/attention, router/top-k, residual, STEP/control messages,
shape/graph management or capture/replay policy. Model loading stays native.

Only qualified routed-decode steps use these plugins. The existing native AFD
path remains the fallback (including F-side routing) for prefill/tails. Selection
is peer-visible in the existing step descriptor; never fall back rank-locally.
Preparation happens once. Execution must preserve stream/event dependencies and
stable graph storage, without host synchronization during capture/replay.
"""

from __future__ import annotations

from typing import Any, Protocol

from .contracts import AFDError


class FFNExecutionBackend(Protocol):
    """A/F expert data plane, not a per-layer GEMM or a service scheduler.

    The transport owns bounded abort of ``communicators`` after ``close``; the
    backend closes its other resources while the shared control store is alive.
    ``hbm_bytes`` preserves each backend's existing graph-budget accounting.
    """

    first_layer: int  # Earlier dense layers use the ordinary AFD path.
    hbm_bytes: int
    communicators: Any
    needs_warmup_barrier: bool

    def dispatch(self, *, hidden, ids, weights, stage: int, output) -> Any:
        """A: enqueue one expert request; return completion of output writes."""
        ...

    def run_ffn_step(self) -> tuple:
        """F: enqueue all layers/stages; return written, live graph outputs."""
        ...

    def finish_warmup(self) -> None:
        """Optional cross-role drain before capture, never inside capture."""
        ...

    def close(self, *, seconds: float) -> None:
        """Release backend resources; transport aborts exported communicators."""
        ...


def direct_step_reason(config, lane_rows, is_extend, *, backend=None):
    """One shared row matrix decides the data plane; never truncate a lane."""
    if getattr(config, "experimental_m2n", False):
        return "explicit_m2n"
    if config.direct_mode == "fixed":
        return "fixed_policy"
    if backend is None:
        return "backend_not_qualified"
    if is_extend:
        return "prefill"
    if len(lane_rows) != config.attention_lane_count or any(
        len(rows) != 2 or any(type(n) is not int or n < 0 for n in rows)
        for rows in lane_rows
    ):
        raise AFDError("AFD_DIRECT_STEP_SHAPE_INVALID")
    if not any(any(rows) for rows in lane_rows):
        return "empty_step"
    if max(max(rows) for rows in lane_rows) > backend.max_rows:
        return "capacity_overflow"
    return "direct"


def select_direct_step(config, lane_rows, is_extend, *, backend=None):
    return direct_step_reason(config, lane_rows, is_extend, backend=backend) == "direct"


def negotiate_direct_capabilities(descriptors):
    """Resolve role-local offers once; A owns the request/graph size plan."""
    from .contracts import AFDRole

    attention = [d for d in descriptors if d["role"] == AFDRole.ATTENTION.value]
    offers = [d["runtime_contract"]["direct_execution"] for d in descriptors]
    plans = {tuple(d["runtime_contract"]["capture_stage_sizes"]) for d in attention}
    if len(plans) != 1:
        raise AFDError("AFD_ATTENTION_CAPTURE_PLAN_MISMATCH")
    sizes = next(iter(plans))
    modes = {o["mode"] for o in offers}
    if "fixed" in modes and "direct" in modes:
        raise AFDError("AFD_FFN_BACKEND_POLICY_CONFLICT")
    enabled = "fixed" not in modes and all(o["available"] for o in offers)
    reasons = tuple(sorted({o["reason"] for o in offers if not o["available"]}))
    if "direct" in modes and not enabled:
        raise AFDError("AFD_DIRECT_PEER_UNSUPPORTED", str(reasons))
    if enabled and any(o["geometry"] != offers[0]["geometry"] for o in offers):
        raise AFDError("AFD_DIRECT_GEOMETRY_MISMATCH")
    eager_capacities = {
        d["runtime_contract"]["direct_execution"]["eager_capacity"] for d in attention
    }
    if len(eager_capacities) != 1:
        raise AFDError("AFD_ATTENTION_DIRECT_CAPACITY_MISMATCH")
    return {
        "enabled": enabled,
        "reason": "qualified" if enabled else ",".join(reasons),
        "capture_sizes": sizes,
        "buffer_sizes": sizes or (next(iter(eager_capacities)),),
    }


def select_m2n_step(config: Any, lane_rows: tuple, is_extend: bool) -> bool:
    return bool(
        config.experimental_m2n
        and not is_extend
        and lane_rows == ((config.experimental_m2n_stage_rows,) * 2,) * 8
    )


def validate_ffn_backend(*, config, adapter, model_family, dtype, capture_sizes):
    """Advertise local capability, including the native expert placement on F.

    Policy and graph capacities may differ between roles. The startup exchange
    resolves one capability/capture plan, and the STEP descriptor decides use.
    """
    if config.experimental_m2n:
        from .m2n import m2n_model_spec

        spec = m2n_model_spec(adapter)
        if (
            model_family not in ("qwen3_moe", "glm_moe_dsa")
            or str(dtype) != "torch.bfloat16"
            or capture_sizes != (config.experimental_m2n_stage_rows,)
            or (spec["identity"] == "glm5" and config.lanes != 8)
        ):
            raise AFDError(
                "AFD_M2N_SCOPE_UNSUPPORTED",
                "requires an admitted model/topology and its exact capture stage width",
            )

    capability = {
        "mode": config.direct_mode,
        "available": False,
        "reason": "fixed_policy",
        "geometry": None,
    }
    if config.direct_mode == "fixed":
        return capability
    reason = None
    if getattr(config, "experimental_m2n", False):
        reason = "explicit_m2n"
    elif model_family != "qwen3_moe":
        reason = "model_family"
    elif str(dtype) != "torch.bfloat16":
        reason = "hidden_dtype"
    if reason is None:
        from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend

        if not get_moe_a2a_backend().is_none():
            reason = "explicit_moe_a2a"
        elif not (
            get_moe_runner_backend().is_triton()
            or get_moe_runner_backend().is_deep_gemm()
        ):
            reason = "moe_runner"
    if reason is None:
        native = adapter.inner.config
        geometry = {
            "hidden_size": adapter.hidden_size,
            "num_layers": adapter.num_layers,
            "num_experts": native.num_experts,
            "top_k": native.num_experts_per_tok,
        }
        if (
            not all(type(n) is int and n > 0 for n in geometry.values())
            or geometry["top_k"] > geometry["num_experts"]
        ):
            reason = "model_geometry"
        else:
            capability["geometry"] = geometry
            from .contracts import AFDRole

            for layer in adapter.inner.layers:
                mlp = getattr(layer, "mlp", None)
                if adapter.role == AFDRole.ATTENTION and not callable(
                    getattr(mlp, "route", None)
                ):
                    reason = "non_routed_layer"
                    break
                if adapter.role == AFDRole.FFN:
                    experts = getattr(mlp, "experts", None)
                    if (
                        getattr(experts, "moe_tp_size", None) != 1
                        or getattr(experts, "moe_ep_size", None) != config.lanes
                        or getattr(experts, "num_local_experts", 0) < 1
                        or getattr(experts, "reduce_results", True)
                    ):
                        reason = "native_expert_placement"
                        break
    if reason is not None:
        capability["reason"] = reason
        if config.direct_mode == "direct" and reason != "explicit_m2n":
            raise AFDError("AFD_DIRECT_SCOPE_UNSUPPORTED", reason)
        return capability
    capability.update(available=True, reason="qualified")
    return capability


def build_ffn_backend(
    *, transport, adapter, config, device
) -> FFNExecutionBackend | None:
    """Select once; import optional implementations only when explicitly enabled."""
    if config.experimental_m2n:
        from .m2n import AFDM2NDataPlane

        transport.ffn_backend = AFDM2NDataPlane(
            transport=transport, adapter=adapter, config=config, device=device
        )
    elif getattr(transport, "direct_enabled", config.experimental_direct is True):
        from .expert_exchange import AFDDirectExchange

        # Register partial ownership so constructor failure still aborts every
        # communicator already created, through the normal transport cleanup.
        backend = AFDDirectExchange.__new__(AFDDirectExchange)
        backend.communicators = []
        transport.ffn_backend = backend
        backend.__init__(
            transport=transport, adapter=adapter, config=config, device=device
        )
    else:
        transport.ffn_backend = None
    return transport.ffn_backend
