"""Translate the native AFD configuration surface without importing its runtime.

Only the topology and execution policy select a simulator adapter. Rendezvous
and lifecycle fields describe the CPU task service, not measured kernel costs.
"""

import json
from dataclasses import dataclass

# Fields shared by the pinned native configurations. Unsupported execution
# families are rejected, rather than silently choosing a different cost path.
_FIELDS = {
    "stages",
    "lanes",
    "attention_lanes",
    "ffn_backend",
    "experimental_direct",
    "experimental_direct_transport",
    "experimental_direct_stage_rows",
    "attention_backend",
    "max_hbm_bytes",
    "rendezvous_host",
    "rendezvous_port",
    "rendezvous_timeout_seconds",
    "idle_timeout_seconds",
    "close_timeout_seconds",
    "nccl_num_channels",
    "experimental_m2n",
    "experimental_m2n_stage_rows",
}


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"AFD_CONFIG_INVALID: {name} must be a positive integer")
    return value


@dataclass(frozen=True)
class NativeAFDConfig:
    raw: dict
    routing: str
    ffn_ranks: int
    attention_ranks: int
    stages: int
    stage_capacity: int

    @classmethod
    def parse(cls, value):
        raw = json.loads(value) if isinstance(value, str) else dict(value)
        if not isinstance(raw, dict) or raw.keys() - _FIELDS:
            raise ValueError("AFD_CONFIG_UNSUPPORTED_FIELDS")
        policy = raw.get("ffn_backend", "auto")
        if policy not in ("auto", "fixed", "direct"):
            raise ValueError("AFD_FFN_BACKEND_INVALID")
        legacy = raw.get("experimental_direct")
        if legacy is not None:
            if type(legacy) is not bool:
                raise ValueError("AFD_DIRECT_CONFIG_INVALID")
            # Matches native direct_mode: an explicit legacy value takes priority.
            policy = "direct" if legacy else "fixed"
        if policy == "auto":
            raise ValueError(
                "AFD_AUTO_REQUIRES_NATIVE_CAPABILITY_NEGOTIATION: select fixed or direct for simulation"
            )
        if raw.get("experimental_m2n", False) is not False:
            raise ValueError("AFD_M2N_SIMULATION_UNSUPPORTED")
        if raw.get("experimental_direct_transport", "nccl") != "nccl":
            raise ValueError("AFD_DIRECT_TRANSPORT_UNSUPPORTED")
        ffn = _positive_int(raw.get("lanes", 1), "lanes")
        attention = raw.get("attention_lanes")
        attention = (
            ffn if attention is None else _positive_int(attention, "attention_lanes")
        )
        stages = _positive_int(raw.get("stages", 2), "stages")
        capacity = _positive_int(
            raw.get("experimental_direct_stage_rows", 32),
            "experimental_direct_stage_rows",
        )
        if attention != 8 or stages != 2:
            raise ValueError(
                "AFD_SIMULATOR_LAYOUT_UNSUPPORTED: current profiles require 8A/S2"
            )
        if raw.get("attention_backend", "trtllm_mha") != "trtllm_mha":
            raise ValueError("AFD_SIMULATOR_ATTENTION_PROFILE_UNSUPPORTED")
        for key in (
            "max_hbm_bytes",
            "rendezvous_port",
            "rendezvous_timeout_seconds",
            "idle_timeout_seconds",
            "close_timeout_seconds",
            "nccl_num_channels",
        ):
            if key in raw:
                _positive_int(raw[key], key)
        if raw.get("rendezvous_port", 1239) > 65535:
            raise ValueError("AFD_CONFIG_INVALID: rendezvous_port")
        return cls(
            raw,
            "direct" if policy == "direct" else "f-side",
            ffn,
            attention,
            stages,
            capacity,
        )

    def check(self, deployment):
        if deployment["mode"] != "afd" or deployment["ffn_ranks"] != self.ffn_ranks:
            raise ValueError("AFD_NATIVE_DEPLOYMENT_MISMATCH")
        if deployment.get("routing", "f-side") != self.routing:
            raise ValueError("AFD_NATIVE_ROUTING_MISMATCH")
        if (
            self.routing == "direct"
            and deployment["batch_size"] // 16 > self.stage_capacity
        ):
            raise ValueError("AFD_DIRECT_BUFFER_CAPACITY")


def select_native_config(value=None, *, routing=None, ffn_ranks=None):
    native = NativeAFDConfig.parse(value) if value is not None else None
    if native and routing is not None and routing != native.routing:
        raise ValueError("AFD_ROUTING_ARGUMENT_CONFLICT")
    if native and ffn_ranks is not None and ffn_ranks != native.ffn_ranks:
        raise ValueError("AFD_FFN_RANK_ARGUMENT_CONFLICT")
    return (
        native,
        native.routing if native else (routing or "f-side"),
        native.ffn_ranks if native else (8 if ffn_ranks is None else ffn_ranks),
    )
