"""Opt-in capture configuration and explicit runtime capability gates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

import msgspec
from sglang.srt.training_capture.protocol import (
    ContractError,
    Digest,
    Identifier,
    Nonnegative,
    Positive,
    StrictStruct,
    Text,
    canonical_bytes,
    digest_bytes,
)


class StoreSetup(StrictStruct):
    local_hostname: Text
    master_server_addr: Text
    protocol: Literal["tcp", "rdma"] = "tcp"
    metadata_server: Text = "P2PHANDSHAKE"
    global_segment_size: Nonnegative = 0
    local_buffer_size: Positive = 16 << 20
    rdma_devices: str = ""


class CaptureLatencyConfig(StrictStruct):
    ttft_seconds: Annotated[float, msgspec.Meta(gt=0)] | None = None
    tpot_seconds: Annotated[float, msgspec.Meta(gt=0)] | None = None
    window_seconds: Annotated[float, msgspec.Meta(gt=0)] = 30.0
    min_observations: Annotated[int, msgspec.Meta(ge=1)] = 16
    max_observations: Annotated[int, msgspec.Meta(ge=1, le=65536)] = 2048
    percentile: Annotated[float, msgspec.Meta(gt=0, le=1)] = 0.95
    recovery_fraction: Annotated[float, msgspec.Meta(gt=0, lt=1)] = 0.8

    def validate(self):
        if self.ttft_seconds is None and self.tpot_seconds is None:
            raise ContractError("latency control requires a TTFT or TPOT budget")
        if not all(
            value is None or math.isfinite(value)
            for value in msgspec.to_builtins(self).values()
        ):
            raise ContractError("capture latency limits must be finite")
        if self.min_observations > self.max_observations:
            raise ContractError("latency minimum observations exceeds buffer capacity")


class AdaptiveCaptureConfig(StrictStruct):
    interval_seconds: Annotated[float, msgspec.Meta(gt=0)] = 1.0
    low_watermark: Annotated[float, msgspec.Meta(ge=0, lt=1)] = 0.25
    high_watermark: Annotated[float, msgspec.Meta(gt=0, le=1)] = 0.75
    writer_stall_seconds: Annotated[float, msgspec.Meta(gt=0)] = 10.0
    cooldown_seconds: Annotated[float, msgspec.Meta(gt=0)] = 5.0
    latency: CaptureLatencyConfig | None = None

    def validate(self):
        if not all(
            math.isfinite(value)
            for value in (
                self.interval_seconds,
                self.low_watermark,
                self.high_watermark,
                self.writer_stall_seconds,
                self.cooldown_seconds,
            )
        ):
            raise ContractError("adaptive capture limits must be finite")
        if self.low_watermark >= self.high_watermark:
            raise ContractError(
                "adaptive capture requires low_watermark < high_watermark"
            )
        if self.latency is not None:
            self.latency.validate()


class CaptureConfig(StrictStruct):
    dataset_id: Identifier
    model_id: Text
    producer_revision: Text
    selected_layer_ids: list[Nonnegative]
    catalog_endpoint: Text
    journal_directory: Text
    store: StoreSetup
    contract_id: Identifier = "maas-target-kv-top128-v1"
    expected_weights_revision: Digest | None = None
    expected_tokenizer_revision: Digest | None = None
    catalog_token_env: str | None = None
    sample_ratio: Annotated[float, msgspec.Meta(ge=0, le=1)] = 0.01
    sample_seed: int = 0
    adaptive: AdaptiveCaptureConfig | None = None
    max_sample_tokens: Annotated[int, msgspec.Meta(ge=2, le=2147483647)] = 8192
    max_inflight_samples: Positive = 4
    max_host_bytes: Positive = 512 << 20
    kv_d2h_batch_tokens: Positive = 1
    kv_export_backend: Literal["torch", "hicache"] = "torch"
    teacher_d2h_batch_tokens: Positive = 1
    teacher_topk_backend: Literal["torch", "flashinfer"] = "torch"
    payload_hash_workers: Annotated[int, msgspec.Meta(ge=1, le=8)] = 1
    max_device_bytes: Nonnegative = 0
    manifest_buffer_bytes: Positive = 1 << 20
    storage_chunk_tokens: Positive = 256
    replica_num: Positive = 1
    capture_lease_seconds: Annotated[float, msgspec.Meta(gt=0)] = 120.0
    max_capture_seconds: Annotated[float, msgspec.Meta(gt=0)] = 1800.0
    http_timeout_seconds: Annotated[float, msgspec.Meta(gt=0)] = 5.0
    http_attempts: Annotated[int, msgspec.Meta(ge=1, le=10)] = 3

    @classmethod
    def load(cls, path: str):
        source = Path(path)
        if source.stat().st_size > 1 << 20:
            raise ContractError("capture configuration is too large")
        config = msgspec.json.decode(source.read_bytes(), type=cls)
        if not config.selected_layer_ids or len(config.selected_layer_ids) != len(
            set(config.selected_layer_ids)
        ):
            raise ContractError("selected layers must be nonempty and unique")
        if not all(
            math.isfinite(value)
            for value in (
                config.sample_ratio,
                config.capture_lease_seconds,
                config.max_capture_seconds,
                config.http_timeout_seconds,
            )
        ):
            raise ContractError("capture limits must be finite")
        if not config.catalog_endpoint.startswith(("http://", "https://")):
            raise ContractError("capture requires a Catalog HTTP endpoint")
        if not Path(config.journal_directory).is_absolute():
            raise ContractError("capture journal directory must be absolute")
        if config.kv_d2h_batch_tokens > 1 and not config.max_device_bytes:
            raise ContractError("batched KV D2H requires a device staging budget")
        if config.kv_export_backend == "hicache" and not config.max_device_bytes:
            raise ContractError("HiCache KV export requires a device metadata budget")
        if config.teacher_d2h_batch_tokens > 1 and not config.max_device_bytes:
            raise ContractError("batched teacher D2H requires a device staging budget")
        if config.adaptive is not None:
            config.adaptive.validate()
        return config

    @property
    def fingerprint(self):
        return digest_bytes(canonical_bytes(self))

    @property
    def startup_policy(self):
        """Common request/storage policy, excluding rank-local capacity and paths."""
        policy = msgspec.to_builtins(self)
        for name in ("journal_directory", "max_host_bytes", "max_device_bytes"):
            del policy[name]
        for name in (
            "local_hostname",
            "local_buffer_size",
            "global_segment_size",
            "rdma_devices",
        ):
            del policy["store"][name]
        return policy


def validate_capture_server_args(args) -> None:
    """Called during ServerArgs validation, before loading target weights."""
    if args.training_capture_config is None:
        return
    CaptureConfig.load(args.training_capture_config)
    from sglang.srt.environ import envs

    unsupported = {
        "DP or context parallelism": args.dp_size != 1
        or args.attn_cp_size != 1
        or args.dcp_size != 1
        or args.enable_dp_attention,
        "speculative algorithm": args.speculative_algorithm not in (None, "DSPARK"),
        "simulated speculative acceptance": args.speculative_algorithm is not None
        and envs.SGLANG_SIMULATE_ACC_LEN.get() > 0,
        "PD capture topology or backend": args.disaggregation_mode != "null"
        and (
            args.disaggregation_transfer_backend != "mooncake"
            or args.optimistic_prefill_attempts > 0
        ),
        "mixed-chunk speculative decoding": args.enable_mixed_chunk
        and args.speculative_algorithm is not None,
        "PDMux": args.enable_pdmux,
        "diffusion language models": args.dllm_algorithm is not None,
        "model overlap": args.enable_two_batch_overlap
        or args.enable_single_batch_overlap,
        "unified or sparse KV memory": args.enable_unified_memory
        or args.enable_hisparse,
        "LoRA": bool(args.enable_lora) or bool(args.lora_paths),
        "quantized weights": args.quantization is not None,
        "unverifiable weight loading": args.load_format not in ("auto", "safetensors"),
        "custom weight or forward hooks": bool(args.custom_weight_loader)
        or bool(args.forward_hooks),
        "embedding/encoder serving": args.is_embedding or args.encoder_only,
    }
    rejected = [name for name, enabled in unsupported.items() if enabled]
    if rejected:
        raise ValueError(
            "training capture does not yet support: " + ", ".join(rejected)
        )
