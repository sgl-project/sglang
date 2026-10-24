"""Bounded NCCL direct exchange for native Qwen routed decode.

Routing changes GPU tensor values, not graph edges or NCCL counts. Pack each
source token once per destination F and return one local weighted sum per
token. The native F dispatcher masks non-local experts. Fixed-capacity payloads
keep source row order, so packing and inverse permutations are unnecessary.
"""

from __future__ import annotations

from .contracts import AFDError, AFDRole


def direct_step_reason(config, lane_rows, is_extend, *, backend=None):
    """One shared row matrix decides the data plane; never truncate a lane."""
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


def validate_direct_backend(*, config, adapter, model_family, dtype, capture_sizes):
    """Advertise local capability, including the native expert placement on F.

    Policy and graph capacities may differ between roles. The startup exchange
    resolves one capability/capture plan, and the STEP descriptor decides use.
    """
    capability = {
        "mode": config.direct_mode,
        "available": False,
        "reason": "fixed_policy",
        "geometry": None,
    }
    if config.direct_mode == "fixed":
        return capability
    reason = None
    if model_family != "qwen3_moe":
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
        if config.direct_mode == "direct":
            raise AFDError("AFD_DIRECT_SCOPE_UNSUPPORTED", reason)
        return capability
    capability.update(available=True, reason="qualified")
    return capability


def build_direct_exchange(
    *, transport, adapter, config, device
) -> AFDDirectExchange | None:
    """Select once; import optional implementations only when explicitly enabled."""
    if getattr(transport, "direct_enabled", config.experimental_direct is True):
        from .expert_exchange import AFDDirectExchange

        # Register partial ownership so constructor failure still aborts every
        # communicator already created, through the normal transport cleanup.
        backend = AFDDirectExchange.__new__(AFDDirectExchange)
        backend.communicators = []
        transport.direct = backend
        backend.__init__(
            transport=transport, adapter=adapter, config=config, device=device
        )
    else:
        transport.direct = None
    return transport.direct


def direct_payload(hidden, ids, weights):
    """Reuse source rows; the F dispatcher maps global IDs to local or -1."""
    import torch

    return (
        hidden.contiguous(),
        ids.to(torch.int64).contiguous(),
        weights.float().contiguous(),
    )


def combine_partials(partials, output):
    """Each F returns source rows in the same order, already weighted locally."""
    import torch

    output.copy_(partials[:, : output.shape[0]].sum(dim=0, dtype=torch.float32))


class AFDDirectExchange:
    first_layer = 0
    needs_warmup_barrier = False

    def __init__(self, *, transport, adapter, config, device):
        import torch

        self.transport = transport
        self.adapter = adapter
        self.device = transport._device
        self.role = transport.role
        self.widths = tuple(
            getattr(
                transport,
                "direct_buffer_sizes",
                (config.experimental_direct_stage_rows,),
            )
        )
        self.max_rows = max(self.widths)
        self.rows = self.max_rows
        self.stages = config.stages
        self.num_a = config.attention_lane_count
        self.num_f = config.lanes
        self.hidden = adapter.hidden_size
        self.top_k = adapter.inner.config.num_experts_per_tok
        self.num_experts = adapter.inner.config.num_experts
        self.bindings = []
        self.communicators = []
        self.bucket_buffers = {}
        self.active_stages = (True,) * self.stages
        self.hbm_bytes = 0
        if self.role == AFDRole.FFN:
            for layer in adapter.inner.layers:
                experts = layer.mlp.experts
                if (
                    experts.moe_tp_size != 1
                    or experts.moe_ep_size != self.num_f
                    or experts.num_local_experts < 1
                    or experts.reduce_results
                ):
                    raise AFDError("AFD_DIRECT_EXPERT_AXIS_REQUIRED")
        # F-first coordination ranks. Prefix stores share the existing TCP port.
        # Separate stage/direction communicators avoid cross-stage NCCL cycles.
        for stage in range(self.stages):
            pair = []
            for direction in ("dispatch", "combine"):
                pair.append(
                    self._make_binding(stage=stage, direction=direction, config=config)
                )
            self.bindings.append(pair)
        # One stable allocation per negotiated graph width. Values may vary,
        # while counts/pointers for a given bucket never change under replay.
        for rows in self.widths:
            buckets = []
            for stage in range(self.stages):
                count = self.num_a if self.role == AFDRole.FFN else self.num_f
                result = torch.empty(
                    (count, rows, self.hidden), device=self.device, dtype=torch.bfloat16
                )
                if self.role == AFDRole.FFN:
                    ids = torch.empty(
                        (count, rows, self.top_k), device=self.device, dtype=torch.int64
                    )
                    weights = torch.empty_like(ids, dtype=torch.float32)
                    fields = (result, ids, weights)
                else:
                    hidden = torch.empty(
                        (rows, self.hidden), device=self.device, dtype=torch.bfloat16
                    )
                    ids = torch.empty(
                        (rows, self.top_k), device=self.device, dtype=torch.int64
                    )
                    weights = torch.empty_like(ids, dtype=torch.float32)
                    fields = (result, hidden, ids, weights)
                buckets.append(fields)
                self.hbm_bytes += sum(x.numel() * x.element_size() for x in fields)
            self.bucket_buffers[rows] = buckets
        self.buffers = self.bucket_buffers[self.rows]

    def _make_binding(self, *, stage, direction, config):
        import torch

        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )
        from sglang.srt.distributed.utils import StatelessProcessGroup

        from .transport import _BoundNCCL

        group = StatelessProcessGroup(
            rank=self.transport._coordination_rank,
            world_size=self.num_a + self.num_f,
            store=torch.distributed.PrefixStore(
                f"afd-direct-v1/{stage}/{direction}", self.transport._control.store
            ),
        )
        comm = PyNcclCommunicator(
            group=group, device=self.device, max_ctas=config.nccl_num_channels
        )
        self.communicators.append((f"direct-{stage}-{direction}", comm))
        return _BoundNCCL(
            comm=comm, stream=torch.cuda.Stream(device=self.device), device=self.device
        )

    def prepare_step(self, *, shape):
        """Consume the shared shape before graph selection, not during capture."""
        requested = max(max(vector) for vector in shape.lane_bucket_rows)
        rows = next((width for width in self.widths if width >= requested), None)
        if rows is None:
            raise AFDError(
                "AFD_DIRECT_BUFFER_CAPACITY", f"rows={rows} capacities={self.widths}"
            )
        self.rows = rows
        self.buffers = self.bucket_buffers[rows]
        self.active_stages = tuple(
            any(vector[s] for vector in shape.lane_stage_rows)
            for s in range(self.stages)
        )

    def _exchange(self, stage, direction, tensors, *, send):
        import torch

        binding = self.bindings[stage][direction]
        stream = binding._stream
        self.transport._fork(stream=stream)
        peers = tuple(
            range(self.num_f)
            if self.role == AFDRole.ATTENTION
            else range(self.num_f, self.num_f + self.num_a)
        )
        tensors = tuple(tensors)
        if len(tensors) != len(peers):
            raise AFDError("AFD_DIRECT_PEER_PAYLOAD_COUNT")

        def operation(comm):
            comm.group_start()
            try:
                for peer, fields in zip(peers, tensors):
                    for tensor in fields:
                        if send:
                            comm.send(tensor, dst=peer)
                        else:
                            comm.recv(tensor, src=peer)
                        tensor.record_stream(stream)
            finally:
                comm.group_end()

        binding.invoke(torch=torch, operation=operation)
        event = torch.cuda.Event()
        event.record(stream)
        return event

    def dispatch(self, *, hidden, ids, weights, stage, output):
        import torch

        if (
            self.role != AFDRole.ATTENTION
            or hidden.ndim != 2
            or hidden.shape[0] > self.rows
            or hidden.shape[1] != self.hidden
            or output.shape != hidden.shape
            or ids.shape != (hidden.shape[0], self.top_k)
            or weights.shape != ids.shape
            or hidden.dtype != torch.bfloat16
            or output.dtype != hidden.dtype
        ):
            raise AFDError("AFD_DIRECT_A_INPUT_INVALID")
        _, packed_hidden, packed_ids, packed_weights = self.buffers[stage]
        # Empty/partial lanes still post the same peer-visible fixed-capacity
        # messages. Invalid padded routes are explicit, never torch.empty data.
        n = hidden.shape[0]
        packed_hidden.zero_()
        packed_ids.fill_(-1)
        packed_weights.zero_()
        packed_hidden[:n].copy_(hidden)
        packed_ids[:n].copy_(ids)
        packed_weights[:n].copy_(weights)
        payload = (packed_hidden, packed_ids, packed_weights)
        self._exchange(stage, 0, [payload] * self.num_f, send=True)
        partials = self.buffers[stage][0]
        self._exchange(stage, 1, [(p,) for p in partials], send=False)
        stream = self.bindings[stage][1]._stream
        with torch.cuda.stream(stream):
            combine_partials(partials, output)
            output.record_stream(stream)
            event = torch.cuda.Event()
            event.record(stream)
        return event

    def run_ffn_step(self) -> tuple:
        from .layer_executor import execute_layer_operations

        outputs = [()] * self.stages

        def run_layer(*, state, layer, index):
            if self.active_stages[index]:
                outputs[index] = self.run_layer(layer=layer, stage=index)

        execute_layer_operations(self.adapter.num_layers, run_layer)
        return tuple(outputs)

    def finish_warmup(self) -> None:
        pass

    def close(self, *, seconds: float) -> None:
        # Communicators are aborted by the transport's bounded close service.
        pass

    def run_layer(self, *, layer, stage):
        if self.role != AFDRole.FFN:
            raise AFDError("AFD_DIRECT_F_ROLE_REQUIRED")
        hidden, ids, weights = self.buffers[stage]
        ready = self._exchange(stage, 0, list(zip(hidden, ids, weights)), send=False)
        self.transport.wait(ready)
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        from .model_adapters.base import stage_forward_context

        with stage_forward_context(None):
            output = self.adapter.inner.layers[layer].mlp.experts(
                hidden.flatten(0, 1),
                StandardTopKOutput(weights.flatten(0, 1), ids.flatten(0, 1), None),
            )
        output = output.view(self.num_a, self.rows, self.hidden)
        done = self._exchange(stage, 1, [(p,) for p in output], send=True)
        # Native runners can alias the receive buffer in their output.
        self.transport.wait(done)
        return (output,)
