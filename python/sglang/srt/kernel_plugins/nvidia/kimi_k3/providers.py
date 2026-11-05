# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Cache-independent Kimi K3 megakernel resources and runtime-pool binding.

SGLang's megakernel framework imports this module only after configuration and
all-rank validation.  It prepares the frozen B1-B8/B16/B32/B64 TP8 provider set
and later binds it to SGLang-owned typed cache views.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
)
from sglang.srt.kernel_plugins.nvidia.kimi_k3.projection_storage import (
    coalesce_kda_projection_storage,
    coalesce_mla_projection_storage,
)
from sglang.srt.runtime_context import get_parallel

from .source_manifest import source_paths

try:
    from sglang.srt.kernel_plugins.nvidia.binding import HybridPoolView
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.binding import HybridPoolView

PAYLOAD_PACKAGE = "sglang.srt.kernel_plugins.nvidia.kimi_k3.cutedsl"
PAYLOAD_MODULES = {
    suffix: f"{PAYLOAD_PACKAGE}.{suffix}"
    for suffix in (
        "gemm.projection",
        "mla.prepared",
        "kda.prepared",
        "attn_res.prepared",
        "oproj.prepared",
        "common.twoshot",
        "moe.dense",
        "moe.experts_runtime",
        "moe.front",
        "moe.prepared",
        "moe.small_batch.chain",
        "moe.tail_runtime",
        "mla.dcp",
        "common.embedding",
    )
}
_BATCHES = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64)
_MAX_BATCH = max(_BATCHES)

# Names read from SGLang internals. A contract test checks these before GPU use.
SGLANG_READS = {
    "srt/distributed/communication_op.py": ("tensor_model_parallel_all_reduce",),
    "srt/distributed/parallel_state.py": ("device_group", "cpu_group"),
    "srt/arg_groups/fields/parallel.py": (
        "tp_group",
        "tp_rank",
        "tp_size",
        "attn_dcp_size",
        "attn_dcp_rank",
        "dcp_group",
    ),
    "srt/layers/vocab_parallel_embedding.py": (
        "num_added_embeddings",
        "tp_size",
        "shard_indices",
        "org_vocab_start_index",
        "org_vocab_size",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "HybridPoolView",
        "cache",
        "capacity",
        "conv",
        "kda",
        "layer_mapping",
        "logical_dtype",
        "mla",
        "page_size",
        "post_capture_active",
        "reserved_capacity",
        "serving_capacity",
        "state",
        "storage_dtype",
        "temporal",
        "write_shard_size",
        "write_shard_rank",
    ),
    "srt/runtime_context.py": ("get_parallel",),
    "srt/models/deepseek_v2.py": ("q_b_proj_qrep_weight", "w_kc_qrep"),
    "srt/model_executor/model_runner.py": ("_prepare_replicated_q_proj",),
}


def _validate_pool_tensor(
    name: str,
    value: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} is not a tensor")
    if value.device.type != "cuda" or value.device != device:
        raise ValueError(
            f"{name} must be on CUDA device {device}, found {value.device}"
        )
    if value.dtype != dtype:
        raise ValueError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(
            f"{name} has stride {tuple(value.stride())}, expected {stride}"
        )
    if not value.is_contiguous():
        raise ValueError(f"{name} is not contiguous")
    if int(value.data_ptr()) % 16:
        raise ValueError(f"{name} is not 16-byte aligned")


def _digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"expected a scalar tensor, got {tuple(value.shape)}")
        return float(value.detach().item())
    return float(value)


def _module(suffix: str, origins: dict[str, str]) -> Any:
    name = PAYLOAD_MODULES[suffix]
    module = importlib.import_module(name)
    origins[name] = str(Path(inspect.getsourcefile(module) or "").resolve())
    return module


def _cuda_memory() -> dict[str, int]:
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def _tensor_row(kind: str, name: str, value: torch.Tensor) -> dict[str, Any]:
    return {
        "kind": kind,
        "name": name,
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        # Device ordinals are process-local and therefore are not part of the
        # all-rank resource identity.
        "device": value.device.type,
        "nbytes": int(value.untyped_storage().nbytes()),
        "data_ptr": int(value.untyped_storage().data_ptr()),
    }


def _resource_kind(name: str) -> str:
    if ".boxes." in name:
        return "mailbox"
    if ".scratch." in name:
        return "scratch"
    if ".workspaces." in name:
        return "workspace"
    if ".projection_weight" in name or ".moe_front_weights." in name:
        return "projection"
    return "provider"


def _handle_row(kind: str, name: str, value: Any) -> dict[str, Any]:
    if kind == "stream":
        handle = int(value.cuda_stream)
        device = value.device.type
    else:
        handle = int(value.cuda_event)
        device = "cuda"
    if handle <= 0:
        raise RuntimeError(f"uninitialized CUDA {kind}: {name}")
    return {
        "kind": kind,
        "name": name,
        "shape": [],
        "stride": [],
        "dtype": type(value).__name__,
        "device": device,
        "nbytes": 0,
        "data_ptr": handle,
    }


def _owned_resource_inventory(root: Any, model: Any) -> list[dict[str, Any]]:
    """Inventory unique non-model storages and explicit CUDA handles.

    The inventory is ownership-based rather than allocator-delta-based: compiler
    and CUDA runtime allocations remain visible in the separate memory snapshots.
    """

    model_storages: set[int] = set()
    model_modules = {id(module) for module in model.modules()}
    for value in (*model.parameters(), *model.buffers()):
        if value.device.type == "cuda":
            model_storages.add(int(value.untyped_storage().data_ptr()))
    # Kimi's post-load transforms also retain a few deliberately unregistered
    # tensor views/copies (for example absorbed MLA W_UK/W_UV and fused KDA
    # views).  They remain model-owned even though state_dict does not list
    # them; exclude direct tensor attributes on every model module as well.
    for module in model.modules():
        for value in vars(module).values():
            if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                model_storages.add(int(value.untyped_storage().data_ptr()))

    rows: list[dict[str, Any]] = []
    seen_objects: set[int] = set()
    seen_storages: set[int] = set()
    # Embedding and DCP borrow established framework collectives. Their retained
    # coordinators/process groups and communicator buffers are not plugin-owned.
    external_owners = {
        id(getattr(getattr(root, "embedding", None), "group", None)),
        *(id(getattr(box, "group", None)) for box in root.boxes.values()),
    }

    def visit(
        value: Any, name: str, depth: int = 0, forced_kind: str | None = None
    ) -> None:
        if depth > 12:
            return
        if isinstance(value, torch.Tensor):
            if value.device.type != "cuda":
                return
            pointer = int(value.untyped_storage().data_ptr())
            if pointer in model_storages or pointer in seen_storages:
                return
            seen_storages.add(pointer)
            rows.append(_tensor_row(forced_kind or _resource_kind(name), name, value))
            return
        if isinstance(value, torch.cuda.Stream):
            if id(value) in seen_objects:
                return
            seen_objects.add(id(value))
            rows.append(_handle_row("stream", name, value))
            return
        if isinstance(value, torch.cuda.Event):
            if id(value) in seen_objects:
                return
            seen_objects.add(id(value))
            rows.append(_handle_row("event", name, value))
            return
        if value is None or isinstance(value, (str, bytes, int, float, bool, type)):
            return
        identity = id(value)
        if identity in model_modules or identity in external_owners:
            return
        if identity in seen_objects:
            return
        seen_objects.add(identity)
        if isinstance(value, dict):
            for key in sorted(value, key=lambda item: str(item)):
                visit(value[key], f"{name}.{key}", depth + 1, forced_kind)
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{name}[{index}]", depth + 1, forced_kind)
            return
        values = getattr(value, "__dict__", None)
        if values is not None:
            for key in sorted(values):
                # Compiled code objects may retain a large Python compiler graph;
                # their device storage is accounted by CUDA memory, not tensor
                # inventory.  Provider-owned tensor attributes remain traversed.
                if key in {
                    "activation",
                    "code",
                    "codes",
                    "kernels",
                    "layers",
                    "model",
                }:
                    continue
                visit(values[key], f"{name}.{key}", depth + 1, forced_kind)

    # Register shared roots before walking providers that hold views of them so
    # classification never depends on dataclass attribute order.
    visit(root.boxes, "resources.boxes", forced_kind="mailbox")
    visit(root.scratch, "resources.scratch", forced_kind="scratch")
    visit(root.workspaces, "resources.workspaces", forced_kind="workspace")
    visit(root.runtime_handles, "resources.runtime_handles")
    for layer_id in sorted(root.attention):
        visit(
            root.attention[layer_id].projection_weight,
            f"resources.attention.{layer_id}.projection_weight",
            forced_kind="projection",
        )
    for layer_id in sorted(root.moe_front_weights):
        visit(
            root.moe_front_weights[layer_id],
            f"resources.moe_front_weights.{layer_id}",
            forced_kind="projection",
        )
    visit(root, "resources")
    rows.sort(key=lambda row: (row["kind"], row["name"]))
    return rows


def _pointer_free_digest(rows: Iterable[dict[str, Any]]) -> str:
    return _digest([{k: v for k, v in row.items() if k != "data_ptr"} for row in rows])


def _layer_proxy(layer: Any) -> Any:
    """Supply the one source-only AttnRes index from SGLang's equivalent facts."""

    return SimpleNamespace(
        self_attention_res_norm=layer.self_attention_res_norm,
        self_attention_res_proj=layer.self_attention_res_proj,
        input_layernorm=layer.input_layernorm,
        prev_valid_blocks=int(layer.prev_valid_blocks),
        is_block_write_layer=bool(layer.is_block_write_layer),
        block_write_idx=int(layer.layer_idx // layer.attn_res_block_size),
    )


def _final_proxy(backbone: Any) -> Any:
    return SimpleNamespace(
        output_attn_res_norm=backbone.output_attn_res_norm,
        output_attn_res_proj=backbone.output_attn_res_proj,
        norm=backbone.norm,
        num_attn_res_blocks=math.ceil(
            int(backbone.config.num_hidden_layers)
            / int(backbone.config.attn_res_block_size)
        ),
    )


def _validate_mla_query_shard(attention: Any) -> None:
    """Keep prepared Q local even when the stock path replicates projection.

    SGLang's replicated-Q preparation adds full-head buffers and can re-alias
    ``q_b_proj.weight`` into its local slice. PreparedMla retains the Parameter
    and binds its final storage after backend preparation. The kernels still
    project twelve-head shards and gather the resulting Q once across DCP.
    The native fallback may independently use its replicas. Never
    feed a 96-head replica into the fixed 12-head query/value kernels or switch
    the native configuration to make this path fit.
    """
    qb, wk = attention.q_b_proj.weight, attention.w_kc
    if not isinstance(qb, torch.Tensor) or tuple(qb.shape) != (2304, 1536):
        raise ValueError(
            "prepared MLA Q projection needs the original [2304, 1536] shard"
        )
    if qb.dtype != torch.bfloat16 or qb.stride() != (1536, 1):
        raise ValueError("prepared MLA Q projection needs a row-major BF16 shard")
    if qb.data_ptr() % 16:
        raise ValueError("prepared MLA Q projection shard is not 16-byte aligned")
    if not isinstance(wk, torch.Tensor) or wk.dtype != torch.bfloat16:
        raise ValueError("prepared MLA absorbed Q needs the original BF16 head shard")

    qrep = getattr(attention, "q_b_proj_qrep_weight", None)
    krep = getattr(attention, "w_kc_qrep", None)
    if qrep is None and krep is None:
        # Native preparation may not have run yet, or may retain its own
        # sharded-Q fallback. This provider never depends on these buffers.
        return
    if not isinstance(qrep, torch.Tensor) or not isinstance(krep, torch.Tensor):
        raise ValueError("native MLA replicated-Q buffers are incomplete")
    parallel = get_parallel()
    dcp, rank = int(parallel.attn_dcp_size), int(parallel.attn_dcp_rank)
    if dcp != 8 or not 0 <= rank < dcp:
        raise ValueError(
            "native MLA replicated-Q buffers require the TP8/DCP8 topology"
        )
    for replica, shape, source, name in (
        (qrep, (dcp * 2304, 1536), qb, "Q projection"),
        (krep, (dcp * 12, 128, 512), wk, "absorbed Q"),
    ):
        if (
            tuple(replica.shape) != shape
            or replica.dtype != torch.bfloat16
            or replica.device != source.device
            or not replica.is_contiguous()
        ):
            raise ValueError(f"native MLA replicated {name} has incompatible storage")
        width = shape[0] // dcp
        if not torch.equal(replica.narrow(0, rank * width, width), source):
            raise ValueError(
                f"native MLA replicated {name} differs from its local shard"
            )


def _mla_proxy(attention: Any) -> tuple[Any, tuple[torch.Tensor, ...]]:
    _validate_mla_query_shard(attention)
    source_w_uk = attention.w_kc
    if source_w_uk.shape != (12, 128, 512):
        raise ValueError(f"unsupported MLA W_UK_T shape: {source_w_uk.shape}")
    if source_w_uk.stride() != (65536, 1, 128):
        raise ValueError(
            f"unsupported MLA source W_UK_T stride: {source_w_uk.stride()}"
        )
    kernel_w_uk = source_w_uk.contiguous()
    if kernel_w_uk.stride() != (65536, 512, 1):
        raise ValueError(f"unsupported MLA W_UK_T stride: {kernel_w_uk.stride()}")
    if kernel_w_uk.data_ptr() % 16:
        raise ValueError("MLA kernel W_UK_T is not 16-byte aligned")
    if not torch.equal(kernel_w_uk, source_w_uk):
        raise ValueError("MLA W_UK_T layout normalization changed values")
    if attention.w_vc.shape != (12, 512, 128):
        raise ValueError(f"unsupported MLA W_UV shape: {attention.w_vc.shape}")
    if attention.w_vc.stride() != (65536, 1, 512):
        raise ValueError(f"unsupported MLA W_UV stride: {attention.w_vc.stride()}")
    if attention.w_vc.data_ptr() % 16:
        raise ValueError("MLA W_UV is not 16-byte aligned")
    q_scale = getattr(attention.attn_mqa, "q_scale_float", None)
    k_scale = getattr(attention.attn_mqa, "k_scale_float", None)
    q_scale = 1.0 if q_scale is None else _float(q_scale)
    k_scale = 1.0 if k_scale is None else _float(k_scale)
    if (q_scale, k_scale) != (1.0, 1.0):
        raise ValueError(f"unsupported MLA activation scales: {(q_scale, k_scale)}")
    device = attention.q_b_proj.weight.device
    q_inv = torch.tensor([1.0], dtype=torch.float32, device=device)
    k_inv = torch.tensor([1.0], dtype=torch.float32, device=device)
    proxy = SimpleNamespace(
        q_a_layernorm=attention.q_a_layernorm,
        kv_a_layernorm=attention.kv_a_layernorm,
        q_b_proj=attention.q_b_proj,
        W_UK_T=kernel_w_uk,
        W_UV=attention.w_vc,
        _q_scale_inv=q_inv,
        _k_scale_inv=k_inv,
        _q_scale_float=q_scale,
        _k_scale_float=k_scale,
        scale=_float(attention.scaling),
        rms_norm_eps=_float(attention.q_a_layernorm.variance_epsilon),
    )
    if _float(attention.w_scale) != 1.0:
        raise ValueError("unsupported absorbed MLA weight scale")
    if abs(proxy.scale - 192.0**-0.5) >= 1e-12:
        raise ValueError(f"unsupported MLA attention scale: {proxy.scale}")
    if proxy.rms_norm_eps != 1e-5:
        raise ValueError(f"unsupported MLA RMS epsilon: {proxy.rms_norm_eps}")
    return proxy, (kernel_w_uk, q_inv, k_inv)


def _kda_proxy(attention: Any) -> Any:
    lower = getattr(getattr(attention, "attn", None), "lower_bound", None)
    if lower is None:
        raise ValueError("KDA gate lower bound is absent")
    if _float(lower) != -5.0:
        raise ValueError(f"unsupported KDA gate lower bound: {_float(lower)}")
    if attention.qkv_conv1d.bias is not None:
        raise ValueError("KDA convolution bias is unsupported")
    return SimpleNamespace(
        conv1d=attention.qkv_conv1d,
        dt_bias=attention.dt_bias,
        A_log=attention.A_log.reshape(-1),
        f_b_proj=attention.f_b_proj,
        o_norm=attention.o_norm,
        gate_lower_bound=_float(lower),
    )


def _prepare_mla_workspace(
    module: Any, device: torch.device, heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the payload's cache-independent FlashInfer workspace early."""

    key = (device, heads)
    if key in module._FLASHINFER:
        return module._FLASHINFER[key]
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    count = get_trtllm_gen_multi_ctas_kv_counter_bytes(64, heads, sms)
    workspace = torch.zeros(128 << 20, dtype=torch.int8, device=device)
    counter = torch.zeros(count, dtype=torch.uint8, device=device)
    module._FLASHINFER[key] = workspace, counter
    return workspace, counter


@dataclass
class PreparedAttention:
    layer_id: int
    kind: str
    projection_weight: torch.Tensor
    projection: Any
    core: Any
    output: Any
    cache: torch.Tensor | None = None
    caches: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass
class ResourceSet:
    model: Any
    layers: tuple[Any, ...]
    attention: dict[int, PreparedAttention]
    feed_forward: dict[int, Any]
    moe_front_weights: dict[int, torch.Tensor]
    before: dict[int, Any]
    final: Any
    fused_before: dict[int, Any]
    boxes: dict[str, Any]
    scratch: dict[str, Any]
    workspaces: dict[str, torch.Tensor]
    runtime_handles: dict[str, Any]
    payload_modules: dict[str, str]
    compilation_inventory: list[dict[str, Any]]
    memory_before: dict[str, int]
    memory_after: dict[str, int]
    deadline_ns: int
    batches: tuple[int, ...] = _BATCHES
    native_mla: bool = False
    full_decode: bool = False
    prefetch: Any = None
    embedding: Any = None
    resource_inventory: list[dict[str, Any]] = field(default_factory=list)
    resource_digest: str = ""
    binding: dict[str, Any] | None = None
    binding_fingerprint: str | None = None
    memory_before_bind: dict[str, int] | None = None
    memory_after_bind: dict[str, int] | None = None

    @property
    def persistent_resource_bytes(self) -> int:
        return sum(int(row["nbytes"]) for row in self.resource_inventory)

    @property
    def owned_tensor_bytes(self) -> int:
        return self.persistent_resource_bytes

    @property
    def cuda_allocated_delta(self) -> int:
        return max(
            0,
            int(self.memory_after["allocated"]) - int(self.memory_before["allocated"]),
        )

    def refresh_inventory(self) -> None:
        self.resource_inventory = _owned_resource_inventory(self, self.model)
        self.resource_digest = _pointer_free_digest(self.resource_inventory)

    def _binding_candidate(self, hybrid: HybridPoolView) -> dict[str, Any]:
        full_ids = [int(v) for v in self.model.config.full_attention_layer_ids]
        linear_ids = [int(v) for v in self.model.config.linear_layer_ids]
        full_mapping = {
            int(layer_id): int(index)
            for layer_id, index in hybrid.mla.layer_mapping.items()
        }
        mamba_mapping = {
            int(layer_id): int(index)
            for layer_id, index in hybrid.kda.layer_mapping.items()
        }
        if set(full_mapping) != set(full_ids):
            raise ValueError("full-attention pool mapping differs from Kimi config")
        if set(mamba_mapping) != set(linear_ids):
            raise ValueError("linear-attention pool mapping differs from Kimi config")
        if set(full_mapping.values()) != set(range(len(full_ids))):
            raise ValueError("full-attention pool mapping is not dense")
        if set(mamba_mapping.values()) != set(range(len(linear_ids))):
            raise ValueError("linear-attention pool mapping is not dense")

        page_size = int(hybrid.mla.page_size)
        if self.native_mla or self.full_decode:
            first_core = self.attention[full_ids[0]].core
            if (
                hybrid.mla.write_shard_size != first_core.kv_shards
                or hybrid.mla.write_shard_rank != first_core.kv_rank
            ):
                raise ValueError(
                    "Kimi native MLA write-slot topology differs from build"
                )
        if page_size != 64:
            raise ValueError(f"unsupported MLA page size: {page_size}")
        if hybrid.mla.logical_dtype != torch.float8_e4m3fn:
            raise ValueError(
                "unsupported MLA logical dtype: "
                f"{hybrid.mla.logical_dtype}; expected {torch.float8_e4m3fn}"
            )
        if hybrid.mla.storage_dtype != torch.uint8:
            raise ValueError(
                "unsupported MLA storage dtype: "
                f"{hybrid.mla.storage_dtype}; expected {torch.uint8}"
            )
        reserved_capacity = int(hybrid.mla.reserved_capacity)
        serving_capacity = int(hybrid.mla.serving_capacity)
        if reserved_capacity <= 0:
            raise ValueError("Kimi K3 MLA reserved capacity is empty")
        if reserved_capacity % page_size:
            raise ValueError(
                "Kimi K3 MLA reserved capacity is not page aligned: "
                f"{reserved_capacity}"
            )
        if serving_capacity <= 0 or serving_capacity > reserved_capacity:
            raise ValueError(
                "Kimi K3 MLA serving capacity is outside its reservation: "
                f"{serving_capacity} versus {reserved_capacity}"
            )
        kda_capacity = int(hybrid.kda.capacity)
        if kda_capacity <= 0:
            raise ValueError("Kimi K3 KDA state capacity is empty")

        first_cache = hybrid.mla.cache(full_ids[0])
        first_parameter = next(iter(self.model.parameters()), None)
        device = (
            first_cache.device if first_parameter is None else first_parameter.device
        )
        expected_mla_shape = (
            reserved_capacity // page_size + 1,
            page_size,
            576,
        )
        expected_conv_shape = (kda_capacity + 1, 3, 4608)
        expected_temporal_shape = (kda_capacity + 1, 12, 128, 128)

        local_identities: dict[str, int] = {}
        local_layouts: list[dict[str, Any]] = []
        for layer_id in full_ids:
            cache_view = hybrid.mla.cache(layer_id)
            _validate_pool_tensor(
                f"layer {layer_id} MLA cache",
                cache_view,
                dtype=torch.uint8,
                shape=expected_mla_shape,
                stride=(page_size * 576, 576, 1),
                device=device,
            )
            local_identities[f"mla.{layer_id}"] = int(cache_view.data_ptr())
            local_layouts.append(
                {
                    "kind": "mla",
                    "layer_id": layer_id,
                    "shape": list(cache_view.shape),
                    "stride": list(cache_view.stride()),
                    "dtype": str(cache_view.dtype),
                }
            )
        for layer_id in linear_ids:
            state = hybrid.kda.state(layer_id)
            if len(state.conv) != 1:
                raise ValueError(
                    f"layer {layer_id} has {len(state.conv)} KDA convolution states"
                )
            conv_view = state.conv[0]
            temporal_view = state.temporal
            _validate_pool_tensor(
                f"layer {layer_id} KDA convolution state",
                conv_view,
                dtype=torch.bfloat16,
                shape=expected_conv_shape,
                stride=(3 * 4608, 4608, 1),
                device=device,
            )
            _validate_pool_tensor(
                f"layer {layer_id} KDA temporal state",
                temporal_view,
                dtype=torch.float32,
                shape=expected_temporal_shape,
                stride=(12 * 128 * 128, 128 * 128, 128, 1),
                device=device,
            )
            local_identities[f"kda_conv.{layer_id}"] = int(conv_view.data_ptr())
            local_identities[f"kda_temporal.{layer_id}"] = int(temporal_view.data_ptr())
            local_layouts.extend(
                (
                    {
                        "kind": "kda_conv",
                        "layer_id": layer_id,
                        "shape": list(conv_view.shape),
                        "stride": list(conv_view.stride()),
                        "dtype": str(conv_view.dtype),
                    },
                    {
                        "kind": "kda_temporal",
                        "layer_id": layer_id,
                        "shape": list(temporal_view.shape),
                        "stride": list(temporal_view.stride()),
                        "dtype": str(temporal_view.dtype),
                    },
                )
            )

        binding = {
            "prepared_batches": list(self.batches),
            "native_mla": self.native_mla,
            "full_decode": self.full_decode,
            "write_shard_size": getattr(hybrid.mla, "write_shard_size", 1),
            "write_shard_rank": getattr(hybrid.mla, "write_shard_rank", 0),
            # The live serving capacity may shrink after graph capture. The
            # reservation and the tensor identities/layouts are immutable.
            "kv_reserved_capacity": reserved_capacity,
            "kv_post_capture_active": bool(hybrid.mla.post_capture_active),
            "kda_capacity": kda_capacity,
            "page_size": page_size,
            "logical_dtype": str(hybrid.mla.logical_dtype),
            "storage_dtype": str(hybrid.mla.storage_dtype),
            "full_attention_layer_ids": full_ids,
            "linear_layer_ids": linear_ids,
            "full_layer_dense_indices": {
                str(layer_id): int(full_mapping[layer_id]) for layer_id in full_ids
            },
            "linear_layer_dense_indices": {
                str(layer_id): int(mamba_mapping[layer_id]) for layer_id in linear_ids
            },
            "local_identities": local_identities,
            "local_layouts": local_layouts,
        }
        return binding

    def bind(self, hybrid: HybridPoolView) -> None:
        candidate = self._binding_candidate(hybrid)
        fingerprint = _digest(candidate)
        if self.binding_fingerprint is not None:
            if fingerprint != self.binding_fingerprint or candidate != self.binding:
                raise RuntimeError("Kimi K3 megakernel pool binding changed")
            return

        self.memory_before_bind = _cuda_memory()
        for layer_id, provider in self.attention.items():
            if provider.kind == "mla":
                cache = hybrid.mla.cache(layer_id)
                provider.core.prepare_cache(cache)
                provider.cache = cache
            else:
                state = hybrid.kda.state(layer_id)
                caches = (state.conv[0], state.temporal)
                provider.core.prepare_caches(caches)
                provider.caches = caches
            self.compilation_inventory.append(
                {"kind": "attention_core", "layer_id": layer_id, "name": provider.kind}
            )
        if self.prefetch is not None:
            self.prefetch.bind()
        torch.cuda.synchronize()
        after = self._binding_candidate(hybrid)
        if after != candidate:
            raise RuntimeError("Kimi K3 pool identity changed during binding")
        self.binding = candidate
        self.binding_fingerprint = fingerprint
        self.memory_after_bind = _cuda_memory()


def build_resources(
    model: Any,
    *,
    context_length: int,
    memory_before: dict[str, int],
    deadline_ns: int,
    native_mla: bool = False,
    full_decode: bool = False,
    batches: tuple[int, ...] = _BATCHES,
) -> ResourceSet:
    """Prepare each admitted TP8 token-row width after all-rank consensus."""

    batches = tuple(batches)
    extended = native_mla or full_decode
    if 48 in batches and not extended:
        raise ValueError("Kimi K3 Graph48 requires an explicit extended profile")
    supported = range(1, _MAX_BATCH + 1) if extended else _BATCHES
    if not batches or any(
        type(batch) is not int or batch not in supported for batch in batches
    ):
        raise ValueError("Kimi K3 resources need supported prepared rows within 1-64")
    batches = tuple(sorted(set(batches)))
    memory_before = dict(memory_before)
    origins: dict[str, str] = source_paths()
    projection_module = _module("gemm.projection", origins)
    mla_module = _module("mla.prepared", origins)
    kda_module = _module("kda.prepared", origins)
    attnres_module = _module("attn_res.prepared", origins)
    oproj_module = _module("oproj.prepared", origins)
    twoshot_module = _module("common.twoshot", origins)
    dense_module = _module("moe.dense", origins)
    experts_module = _module("moe.experts_runtime", origins)
    front_module = _module("moe.front", origins)
    prepared_moe_module = _module("moe.prepared", origins)
    chain_module = _module("moe.small_batch.chain", origins)
    tail_module = _module("moe.tail_runtime", origins)

    parallel = get_parallel()
    tp, rank = int(parallel.tp_size), int(parallel.tp_rank)
    if tp != 8:
        raise ValueError(f"Kimi K3 megakernel initial resources require TP8, got {tp}")
    group = parallel.tp_group.device_group
    device = next(model.parameters()).device
    owner = (group, device, tp, rank)
    # All local GPU writes are synchronized by each box before this CPU barrier.
    # Validate both groups before constructing any mailbox so no partial setup can
    # enter a barrier with a mismatched rank order or a CUDA-only backend.
    from sglang.srt.kernel_plugins.nvidia.kimi_k3.cutedsl.common.host import (
        validate_init_barrier_group,
    )

    init_group = validate_init_barrier_group(group, parallel.tp_group.cpu_group)
    dcp_init_group = None
    if full_decode and int(parallel.attn_dcp_size) > 1:
        dcp_init_group = validate_init_barrier_group(
            parallel.dcp_group.device_group, parallel.dcp_group.cpu_group
        )
    init_kwargs = {"init_barrier_group": init_group}
    if native_mla and full_decode:
        raise ValueError("K3 native MLA and full decode profiles are exclusive")
    backbone = model.model
    layers = tuple(backbone.layers[backbone.start_layer : backbone.end_layer])
    if len(layers) != 93:
        raise ValueError(f"expected the full 93-layer Kimi K3 model, got {len(layers)}")
    full_ids = [int(value) for value in model.config.full_attention_layer_ids]
    linear_ids = [int(value) for value in model.config.linear_layer_ids]
    observed_full = [
        int(layer.layer_idx) for layer in layers if hasattr(layer.self_attn, "q_b_proj")
    ]
    observed_linear = [
        int(layer.layer_idx)
        for layer in layers
        if not hasattr(layer.self_attn, "q_b_proj")
    ]
    if len(full_ids) != 24 or len(linear_ids) != 69:
        raise ValueError(
            "unexpected Kimi attention split: "
            f"MLA={len(full_ids)} KDA={len(linear_ids)}"
        )
    if observed_full != full_ids or observed_linear != linear_ids:
        raise ValueError(
            "Kimi layer modules disagree with the configured attention split"
        )

    mailbox = oproj_module.OprojMailbox(*owner, **init_kwargs)
    oproj_twoshot = twoshot_module.TwoShot(*owner, **init_kwargs)
    front_box = oproj_module.OprojMailbox(
        *owner, (1, 64, 3584 + 2 * 896), **init_kwargs
    )
    small_box = chain_module.SmallBatchBox(*owner, **init_kwargs)
    tail_box = tail_module.MoeTailBox(*owner, **init_kwargs)
    boxes = {
        "oproj": mailbox,
        "oproj_twoshot": oproj_twoshot,
        "moe_front": front_box,
        "small_batch": small_box,
        "moe_tail": tail_box,
    }
    dcp_box = None
    if full_decode and int(parallel.attn_dcp_size) > 1:
        dcp_module = _module("mla.dcp", origins)
        dcp_box = dcp_module.DcpBox(
            parallel.dcp_group.device_group,
            device,
            int(parallel.attn_dcp_size),
            int(parallel.attn_dcp_rank),
            12,
            interleave=1,
            deadline_ns=deadline_ns,
            init_barrier_group=dcp_init_group,
        )
        boxes["dcp"] = dcp_box
    moe_state = prepared_moe_module.MoEScratch(tp, device, batches)
    # Materialize the state-owned event handles before resource inventory.  The
    # prepared MoE records them again at each execution boundary, so completing
    # an initialization record on its own stream does not alter runtime order.
    for event in (moe_state.begin, moe_state.shared_done, moe_state.gate_done):
        event.record(moe_state.stream)
    moe_state.stream.synchronize()
    scratch = {
        # PreparedMla.prepare_cache performs one bounded warm call.  Zero is a
        # finite FP8 encoding, unlike arbitrary uninitialized byte patterns.
        "mla.query": torch.zeros((64, 12, 576), dtype=torch.uint8, device=device),
        "mla.projected": torch.empty(
            (_MAX_BATCH, 3648), dtype=torch.bfloat16, device=device
        ),
        "kda.projected": torch.empty(
            (_MAX_BATCH, 6288), dtype=torch.bfloat16, device=device
        ),
        "attention.local": torch.empty(
            (_MAX_BATCH, 1536), dtype=torch.bfloat16, device=device
        ),
        "hidden.0": torch.empty(
            (_MAX_BATCH, 7168), dtype=torch.bfloat16, device=device
        ),
        "hidden.1": torch.empty(
            (_MAX_BATCH, 7168), dtype=torch.bfloat16, device=device
        ),
        "hidden.out": torch.empty(
            (_MAX_BATCH, 7168), dtype=torch.bfloat16, device=device
        ),
        "prefix": torch.empty((_MAX_BATCH, 7168), dtype=torch.bfloat16, device=device),
        "blocks": torch.empty(
            (_MAX_BATCH, 8, 7168), dtype=torch.bfloat16, device=device
        ),
        "moe": moe_state,
    }
    if full_decode:
        scratch["embedding.out"] = torch.empty(
            (_MAX_BATCH, 7168), dtype=torch.bfloat16, device=device
        )
    # Explicitly own the side-stream/event lifecycle used by the future executor.
    # Recording the event materializes its CUDA handle without entering a model
    # forward path.
    provider_stream = torch.cuda.Stream(device=device)
    provider_ready = torch.cuda.Event()
    provider_ready.record(provider_stream)
    provider_stream.synchronize()
    runtime_handles = {"provider": provider_stream, "ready": provider_ready}
    workspaces: dict[str, torch.Tensor] = {}
    if full_decode:
        dcp_module = _module("mla.dcp", origins)
        # Size every prepared token-row width before the serving KV budget is
        # computed. DCP processes all ranks' heads over this rank's local shard.
        heads = 12 if dcp_box is None else 12 * dcp_box.size
        max_seq = (
            int(context_length)
            if dcp_box is None
            else dcp_module.local_max(int(context_length), dcp_box.size)
        )
        workspaces["mla.tokenspeed.workspace"] = dcp_module.prepare_workspace(
            device, heads, max_seq, batches=batches
        )

    beta = float(backbone.config.activation_situ_beta)
    linear_beta = float(backbone.config.activation_situ_linear_beta)
    moe_layers = [layer for layer in layers if bool(layer._is_moe_layer)]
    attention: dict[int, PreparedAttention] = {}
    feed_forward: dict[int, Any] = {}
    moe_front_weights: dict[int, torch.Tensor] = {}
    compilation: list[dict[str, Any]] = []

    for layer in layers:
        layer_id = int(layer.layer_idx)
        module = layer.self_attn
        is_mla = hasattr(module, "q_b_proj")
        if is_mla:
            proxy, _owned = _mla_proxy(module)
            projection_weight = coalesce_mla_projection_storage(module)
            if tuple(projection_weight.shape) != (3648, 7168):
                raise ValueError(
                    f"layer {layer_id} MLA projection shape changed: "
                    f"{tuple(projection_weight.shape)}"
                )
            projection = projection_module.PreparedProjection(
                projection_weight,
                {
                    batch: (
                        (64, 64, 2, 5)
                        if batch > 32
                        else (64, 32, 2, 8)
                        if batch > 16
                        else (64, 16, 2, 10)
                        if batch > 8
                        else (64, 8, 4, 8)
                    )
                    for batch in batches
                },
            ).prepare()
            core = mla_module.PreparedMla(
                proxy,
                {batch: 3648 for batch in batches},
                scratch["mla.query"],
                max_context=int(context_length),
                core="auto" if full_decode else "trtllm",
                dcp=dcp_box,
                parity=full_ids.index(layer_id) % 2,
                queries=(1,),
                native_mla=native_mla,
                kv_shards=int(parallel.attn_dcp_size)
                if native_mla or full_decode
                else 1,
                kv_rank=int(parallel.attn_dcp_rank) if native_mla or full_decode else 0,
                deadline_ns=deadline_ns if full_decode else 0,
            )
            if (
                "mla.flashinfer.workspace" not in workspaces
                and not native_mla
                and dcp_box is None
            ):
                workspace, counter = _prepare_mla_workspace(
                    mla_module, device, int(core.heads)
                )
                workspaces.update(
                    {
                        "mla.flashinfer.workspace": workspace,
                        "mla.flashinfer.counter": counter,
                    }
                )
            kind = "mla"
        else:
            proxy = _kda_proxy(module)
            projection_weight = coalesce_kda_projection_storage(module)
            if tuple(projection_weight.shape) != (6288, 7168):
                raise ValueError(
                    f"layer {layer_id} KDA projection shape changed: "
                    f"{tuple(projection_weight.shape)}"
                )
            projection = projection_module.PreparedProjection(
                projection_weight,
                {
                    batch: (
                        (64, 64, 1, 6)
                        if batch > 32
                        else (128, 32, 2, 5)
                        if batch > 16
                        else (128, 16, 2, 6)
                        if batch > 8
                        else (128, 8, 2, 6)
                    )
                    for batch in batches
                },
            ).prepare()
            core = kda_module.PreparedKda(
                proxy,
                {
                    batch: (
                        6288,
                        0 if batch > 8 else 1 if batch == 1 else 2,
                    )
                    for batch in batches
                },
                False,
            )
            kind = "kda"

        front = None
        if layer._is_moe_layer:
            parity = layer_id % 2
            front = (
                small_box.front_views[parity],
                small_box.hint_views[parity],
                (1,),
            )
        output = oproj_module.PreparedOproj(
            layer,
            mailbox,
            batches,
            is_mla,
            front,
            oproj_twoshot,
            deadline_ns=deadline_ns,
        )
        attention[layer_id] = PreparedAttention(
            layer_id, kind, projection_weight, projection, core, output
        )
        if full_decode:
            from sglang.srt.kernel_plugins.nvidia.kimi_k3.skinny_projection import (
                SkinnyProjection,
            )

            attention[layer_id].projection = SkinnyProjection(
                projection_weight, projection
            ).prepare()
        compilation.extend(
            (
                {
                    "kind": "projection",
                    "layer_id": layer_id,
                    "name": kind,
                    "batches": list(batches),
                },
                {
                    "kind": "oproj",
                    "layer_id": layer_id,
                    "name": kind,
                    "batches": list(batches),
                },
            )
        )

        if not layer._is_moe_layer:
            feed_forward[layer_id] = dense_module.PreparedDenseMLP(
                layer.mlp,
                batches,
                reduce=tensor_model_parallel_all_reduce,
            )
            compilation.append(
                {"kind": "dense_mlp", "layer_id": layer_id, "name": "dense"}
            )
            continue

        routed = layer.mlp.experts
        experts = experts_module.PersistentExperts(
            routed.w13_weight,
            routed.w13_weight_scale,
            routed.w2_weight,
            routed.w2_weight_scale,
            beta=beta,
            linear_beta=linear_beta,
            cooperative=not extended,
            resident_sms=(
                torch.cuda.get_device_properties(device).multi_processor_count
                if native_mla or full_decode
                else None
            ),
        )
        op = SimpleNamespace(
            rank=rank,
            contract=SimpleNamespace(
                tp_size=tp, rms_eps=float(model.config.rms_norm_eps)
            ),
        )
        ordinal = moe_layers.index(layer)
        tail = tail_module.PreparedTail(
            op,
            layer.mlp.routed_expert_norm.weight,
            layer.mlp.routed_expert_up_proj.weight,
            batches,
            (tail_box, ordinal, len(moe_layers)),
            deadline_ns=deadline_ns,
        )
        moe = SimpleNamespace(
            shared_experts=layer.mlp.shared_experts,
            gate=layer.mlp.gate,
            routed_expert_down_proj=layer.mlp.routed_expert_down_proj,
            routed_scaling_factor=layer.mlp.routed_scaling_factor,
            moe_renormalize=bool(model.config.moe_renormalize),
        )
        router, down = moe.gate.weight, moe.routed_expert_down_proj.weight
        router_rows, down_rows = 896 // tp, 3584 // tp
        router = router[rank * router_rows : (rank + 1) * router_rows]
        down = down[rank * down_rows : (rank + 1) * down_rows]
        packed_front = torch.cat(
            (router, router.new_zeros((-router_rows % 16, 7168)), down)
        ).detach()
        moe_front_weights[layer_id] = packed_front
        generic_front = front_module.SharedFront(
            moe,
            packed_front,
            front_box,
            moe_state.front,
            batches,
            beta,
            linear_beta,
            plan=moe_state.plan,
            counters=moe_state.shared,
            q=1,
            deadline_ns=deadline_ns,
        )
        chain = chain_module.SmallBatchChain(
            small_box,
            moe,
            experts,
            tail,
            layer_id,
            beta,
            linear_beta,
            None,
            deadline_ns=deadline_ns,
        )
        prepared_moe = prepared_moe_module.PreparedMoE(
            generic_front,
            experts,
            tail,
            moe_state,
            prefetch=None,
            layer_idx=layer_id,
        )
        prepared_moe.small_batch = chain
        feed_forward[layer_id] = prepared_moe
        compilation.extend(
            (
                {"kind": "moe_experts", "layer_id": layer_id, "name": "mxfp4"},
                {
                    "kind": "moe_front",
                    "layer_id": layer_id,
                    "name": "prepared_rows",
                    "batches": list(batches),
                },
                {
                    "kind": "moe_tail",
                    "layer_id": layer_id,
                    "name": "prepared_rows",
                    "batches": list(batches),
                },
                {"kind": "moe_small_batch", "layer_id": layer_id, "name": "b1"},
                {
                    "kind": "moe_prepared",
                    "layer_id": layer_id,
                    "name": "prepared_rows",
                    "batches": list(batches),
                },
            )
        )

    prefetch = embedding = None
    if full_decode:
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.prefetch import PrefetchController

        prefetch = PrefetchController(
            SimpleNamespace(device=device, tp=tp), layers, moe_front_weights, attention
        )
        for layer_id, provider in feed_forward.items():
            if not layers[layer_id]._is_moe_layer:
                continue
            provider.prefetch = prefetch
            provider.small_batch.prefetch = prefetch
        embed_module = backbone.embed_tokens
        indices = embed_module.shard_indices
        if (
            embed_module.num_added_embeddings != 0
            or embed_module.tp_size != tp
            or indices.org_vocab_start_index != rank * embed_module.weight.shape[0]
        ):
            raise ValueError(
                "K3 prepared embedding requires ordinary contiguous TP shards"
            )
        embedding_module = _module("common.embedding", origins)
        embedding = embedding_module.PreparedEmbedding(embed_module, tp, rank)
        compilation.append(
            {"kind": "embedding", "name": "local_gather_native_tp_reduce"}
        )
        compilation.append({"kind": "prefetch", "name": "vllm_tp8_weight_overlap"})

    before = {
        int(layer.layer_idx): attnres_module.PreparedAttnRes.for_target(
            _layer_proxy(layer),
            has_delta=layer is not layers[0],
            deadline_ns=deadline_ns,
        )
        for layer in layers
    }
    final = attnres_module.PreparedAttnRes.for_target(
        _final_proxy(backbone), final=True, deadline_ns=deadline_ns
    )
    compilation.extend(
        {"kind": "attn_res", "layer_id": int(layer.layer_idx), "name": "input"}
        for layer in layers
    )
    compilation.append({"kind": "attn_res", "layer_id": 93, "name": "final"})
    fused_before: dict[int, Any] = {}
    followers = [before[int(layer.layer_idx)] for layer in layers[1:]] + [final]
    for layer, consumer in zip(layers, followers):
        if not layer._is_moe_layer:
            continue
        consumer.bind_ring(tail_box.out.buffer)
        fused_before[int(layer.layer_idx)] = consumer

    torch.cuda.synchronize()
    resources = ResourceSet(
        model=model,
        layers=layers,
        attention=attention,
        feed_forward=feed_forward,
        moe_front_weights=moe_front_weights,
        before=before,
        final=final,
        fused_before=fused_before,
        boxes=boxes,
        scratch=scratch,
        workspaces=workspaces,
        runtime_handles=runtime_handles,
        payload_modules=origins,
        compilation_inventory=compilation,
        memory_before=memory_before,
        memory_after=_cuda_memory(),
        deadline_ns=deadline_ns,
        batches=batches,
        native_mla=native_mla,
        full_decode=full_decode,
        prefetch=prefetch,
        embedding=embedding,
    )
    resources.refresh_inventory()
    return resources
