"""Versioned tensor descriptors and semantic validation for MaaS snapshots.

The manifest carries metadata only. Tensor objects contain little-endian,
contiguous bytes, never pickle. Validation runs in the writer/reader threads.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from typing import Annotated, Literal, Mapping

import msgspec
import torch

Identifier = Annotated[
    str,
    msgspec.Meta(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$", min_length=1, max_length=160),
]
Digest = Annotated[str, msgspec.Meta(pattern=r"^[a-f0-9]{64}$")]
Positive = Annotated[int, msgspec.Meta(ge=1)]
Nonnegative = Annotated[int, msgspec.Meta(ge=0)]
Text = Annotated[str, msgspec.Meta(min_length=1)]
DType = Literal["int32", "int64", "uint8", "float32", "float16", "bfloat16"]
DTYPES = {
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
ELEMENT_BYTES = {
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}
OWNER = "dp0-pp0-tp0"


class CaptureError(RuntimeError):
    """A sample cannot be published; serving may continue."""


class ContractError(CaptureError):
    pass


class StrictStruct(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    pass


class TeacherIdentity(StrictStruct):
    model_id: Text
    weights_revision: Text
    adapter_revision: Text | None
    tokenizer_revision: Text
    fingerprint_sha256: Digest
    vocab_size: Annotated[int, msgspec.Meta(ge=128, le=2147483647)]
    output_transform: Text


class LayerGeometry(StrictStruct):
    layer_id: Nonnegative
    num_kv_heads: Positive
    key_head_dim: Positive
    value_head_dim: Positive


class KVSpec(StrictStruct):
    codec: Literal[
        "dense_bf16_post_rope_v1",
        "dense_fp16_post_rope_v1",
        "dense_bf16_pre_rope_v1",
        "dense_fp16_pre_rope_v1",
    ]
    dtype: Literal["bfloat16", "float16"]
    selected_layer_ids: list[Nonnegative]
    layers: list[LayerGeometry]
    source_k_stage: Literal["pre_rope", "post_rope"]
    source_k_norm: Text
    rope_config: dict
    rope_config_sha256: Digest
    storage_chunk_tokens: Positive
    source_page_size: Positive
    layout: Literal["token_head_dim"] = "token_head_dim"
    layer_numbering: Literal["target_attention_layer_zero_based"] = (
        "target_attention_layer_zero_based"
    )
    validity_policy: Literal["all_selected_layers_per_valid_token"] = (
        "all_selected_layers_per_valid_token"
    )


class SequenceInfo(StrictStruct):
    prompt_length: Positive
    response_length: Positive
    total_length: Annotated[int, msgspec.Meta(ge=2, le=2147483647)]
    stop_reason: Literal["eos", "stop_token", "stop_string", "length"]
    loss_mask_policy: Literal[
        "current_response_include_eos", "current_response_exclude_eos"
    ] = "current_response_include_eos"
    stop_token_policy: Literal["preserve_internal_accepted_tokens"] = (
        "preserve_internal_accepted_tokens"
    )
    position_ids_semantics: Literal["actual_target_positions"] = (
        "actual_target_positions"
    )


class LogitsSpec(StrictStruct):
    top_k: Literal[128] = 128
    dtype: Literal["float32"] = "float32"
    semantics: Literal["model_output_before_serving_processors"] = (
        "model_output_before_serving_processors"
    )
    normalization: Literal["full_vocabulary_logsumexp"] = "full_vocabulary_logsumexp"
    lse_temperature: float = 1.0
    row_alignment: Literal["predicts_token_at_logits_position"] = (
        "predicts_token_at_logits_position"
    )
    vocab_ids: Literal["global_unpadded"] = "global_unpadded"


class Topology(StrictStruct):
    tp_size: Positive = 1
    pp_size: Positive = 1
    aux_owner: Identifier = OWNER
    owners: list[Identifier] = msgspec.field(default_factory=lambda: [OWNER])


class Provenance(StrictStruct):
    capture_mode: Literal[
        "autoregressive",
        "speculative_accepted_target_path",
        "pd_autoregressive",
        "pd_speculative_accepted_target_path",
    ]
    producer_revision: Text
    capture_config_sha256: Digest
    sampling_config: dict
    trace_id: Identifier


class TensorDescriptor(StrictStruct, omit_defaults=True):
    object_id: Identifier
    name: Text
    kind: Literal["aux", "kv"]
    key: Text
    dtype: DType
    shape: list[Positive]
    nbytes: Positive
    sha256: Digest
    owner_id: Identifier
    byte_order: Literal["little"]
    contiguous: bool
    layer_id: Nonnegative | None = None
    component: Literal["k", "v"] | None = None
    token_range: tuple[Nonnegative, Positive] | None = None
    head_range: tuple[Nonnegative, Positive] | None = None


class Manifest(StrictStruct):
    dataset_id: Identifier
    sample_id: Identifier
    generation_id: Identifier
    created_at: Text
    teacher: TeacherIdentity
    sequence: SequenceInfo
    kv: KVSpec
    provenance: Provenance
    objects: list[TensorDescriptor]
    total_tensor_bytes: Positive
    contract_id: Identifier = "maas-target-kv-top128-v1"
    schema_version: Literal[1] = 1
    payload_format: Literal["maas_target_kv_v1"] = "maas_target_kv_v1"
    state: Literal["READY"] = "READY"
    input_mode: Literal["target_kv"] = "target_kv"
    logits: LogitsSpec = LogitsSpec()
    topology: Topology = msgspec.field(default_factory=Topology)
    extensions: dict = {}

    @property
    def key_prefix(self) -> str:
        return f"draft-data/{self.dataset_id}/{self.sample_id}/{self.generation_id}/"


def canonical_bytes(value) -> bytes:
    return json.dumps(
        msgspec.to_builtins(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest_bytes(value) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> memoryview:
    if (
        sys.byteorder != "little"
        or tensor.device.type != "cpu"
        or not tensor.is_contiguous()
    ):
        raise ContractError(
            "tensor transport requires little-endian contiguous CPU storage"
        )
    return memoryview(tensor.detach().view(torch.uint8).numpy()).cast("B")


def aux_specs(n: int, r: int) -> dict[str, tuple[str, list[int]]]:
    return {
        "token_ids": ("int32", [n]),
        "position_ids": ("int64", [n]),
        "loss_mask": ("uint8", [n]),
        "kv_valid": ("uint8", [n]),
        "logits_positions": ("int32", [r]),
        "teacher_topk_ids": ("int32", [r, 128]),
        "teacher_topk_logits": ("float32", [r, 128]),
        "teacher_logsumexp": ("float32", [r]),
    }


def validate_kv_spec(kv: KVSpec) -> None:
    layer_ids = kv.selected_layer_ids
    if not layer_ids or len(set(layer_ids)) != len(layer_ids):
        raise ContractError("selected layers must be nonempty and unique")
    if [layer.layer_id for layer in kv.layers] != layer_ids:
        raise ContractError("layer geometry must preserve selected layer order")
    prefix = "bf16" if kv.dtype == "bfloat16" else "fp16"
    if kv.codec != f"dense_{prefix}_{kv.source_k_stage}_v1":
        raise ContractError("KV codec disagrees with dtype or K stage")
    if digest_bytes(canonical_bytes(kv.rope_config)) != kv.rope_config_sha256:
        raise ContractError("RoPE configuration digest mismatch")


def _coverage(objects: list[TensorDescriptor], n: int, heads: int) -> int:
    if not objects:
        raise ContractError("missing selected layer/component")
    end = max(o.token_range[1] for o in objects)
    if end not in (n - 1, n):
        raise ContractError("only the final token may lack KV")
    boundaries = sorted({0, end} | {x for o in objects for x in o.token_range})
    for lo, hi in zip(boundaries, boundaries[1:]):
        intervals = sorted(
            o.head_range
            for o in objects
            if o.token_range[0] <= lo and o.token_range[1] >= hi
        )
        cursor = 0
        for start, stop in intervals:
            if start != cursor:
                raise ContractError("overlapping or missing KV heads/tokens")
            cursor = stop
        if cursor != heads:
            raise ContractError("incomplete KV head coverage")
    return end


def validate_manifest(manifest: Manifest, *, max_tensor_bytes: int = 2 << 30) -> int:
    """Validate before allocating or reading tensor objects; return KV coverage."""
    # Struct constructors are intentionally cheap; decoding checks annotated types.
    try:
        msgspec.json.decode(msgspec.json.encode(manifest), type=Manifest)
    except (msgspec.ValidationError, TypeError, ValueError) as error:
        raise ContractError("invalid manifest field") from error
    if manifest.extensions.get("example_only"):
        raise ContractError("example manifests are not training samples")
    try:
        created = datetime.fromisoformat(manifest.created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            raise ValueError("missing timezone")
    except ValueError as error:
        raise ContractError("created_at must be an RFC3339 timestamp") from error
    if manifest.logits.lse_temperature != 1.0:
        raise ContractError("teacher LSE must use temperature 1")
    n, p, r = (
        manifest.sequence.total_length,
        manifest.sequence.prompt_length,
        manifest.sequence.response_length,
    )
    if n != p + r:
        raise ContractError("N must equal P + R")
    if (
        len(manifest.objects) > 16384
        or not 0 < manifest.total_tensor_bytes <= max_tensor_bytes
    ):
        raise ContractError("sample exceeds the reader allocation budget")
    validate_kv_spec(manifest.kv)
    owners = manifest.topology.owners
    if len(set(owners)) != len(owners) or manifest.topology.aux_owner not in owners:
        raise ContractError("invalid owner set")
    specs = aux_specs(n, r)
    seen_ids, seen_keys, seen_aux = set(), set(), set()
    groups = defaultdict(list)
    geometries = {g.layer_id: g for g in manifest.kv.layers}
    total = 0
    for obj in manifest.objects:
        if not obj.contiguous:
            raise ContractError("wire tensors must be contiguous")
        if obj.object_id in seen_ids or obj.key in seen_keys:
            raise ContractError("duplicate object identity")
        seen_ids.add(obj.object_id)
        seen_keys.add(obj.key)
        if obj.owner_id not in owners:
            raise ContractError("unregistered object owner")
        if not obj.key.startswith(manifest.key_prefix) or ".." in obj.key.split("/"):
            raise ContractError("object key escapes the sample namespace")
        expected_bytes = math.prod(obj.shape) * ELEMENT_BYTES[obj.dtype]
        if not obj.shape or obj.nbytes != expected_bytes:
            raise ContractError("tensor byte count disagrees with shape")
        total += obj.nbytes
        if total > max_tensor_bytes:
            raise ContractError("tensor allocation budget exceeded")
        if obj.kind == "aux":
            if obj.name not in specs or obj.name in seen_aux:
                raise ContractError("unknown or duplicate aux field")
            if (obj.dtype, obj.shape) != specs[
                obj.name
            ] or obj.owner_id != manifest.topology.aux_owner:
                raise ContractError("aux field dtype, shape or owner mismatch")
            if any(
                x is not None
                for x in (obj.layer_id, obj.component, obj.token_range, obj.head_range)
            ):
                raise ContractError("aux objects cannot carry KV ranges")
            if obj.key != manifest.key_prefix + "aux/" + obj.name:
                raise ContractError("noncanonical aux key")
            seen_aux.add(obj.name)
        else:
            if (
                obj.layer_id not in geometries
                or obj.component not in ("k", "v")
                or obj.token_range is None
                or obj.head_range is None
            ):
                raise ContractError("invalid KV descriptor")
            geometry = geometries[obj.layer_id]
            t0, t1 = obj.token_range
            h0, h1 = obj.head_range
            dim = (
                geometry.key_head_dim
                if obj.component == "k"
                else geometry.value_head_dim
            )
            if not (0 <= t0 < t1 <= n and 0 <= h0 < h1 <= geometry.num_kv_heads):
                raise ContractError("KV range out of bounds")
            if (
                obj.shape != [t1 - t0, h1 - h0, dim]
                or obj.dtype != manifest.kv.dtype
                or obj.name != f"target_{obj.component}.{obj.layer_id}"
            ):
                raise ContractError("KV geometry mismatch")
            stem = manifest.key_prefix + f"kv/{obj.layer_id}/{obj.owner_id}/"
            suffix = obj.key.removeprefix(stem).split("/")
            if (
                not obj.key.startswith(stem)
                or len(suffix) != 2
                or not suffix[0].isdigit()
                or suffix[1] != obj.component
            ):
                raise ContractError("noncanonical KV key")
            groups[(obj.layer_id, obj.component)].append(obj)
    if seen_aux != set(specs) or total != manifest.total_tensor_bytes:
        raise ContractError("missing aux fields or inconsistent total bytes")
    if {o.owner_id for o in manifest.objects} != set(owners):
        raise ContractError("missing owner objects")
    coverage = {
        _coverage(groups[(g.layer_id, c)], n, g.num_kv_heads)
        for g in manifest.kv.layers
        for c in ("k", "v")
    }
    if len(coverage) != 1:
        raise ContractError("selected layers have inconsistent KV validity")
    return coverage.pop()


def decode_manifest(
    data: bytes, *, max_manifest_bytes: int = 8 << 20, max_tensor_bytes: int = 2 << 30
) -> Manifest:
    if len(data) > max_manifest_bytes:
        raise ContractError("manifest exceeds metadata budget")
    try:
        manifest = msgspec.json.decode(data, type=Manifest)
        raw = msgspec.json.decode(data)
    except msgspec.DecodeError as error:
        raise ContractError("invalid manifest JSON") from error
    required = set(Manifest.__struct_fields__) - {"extensions"}
    if not required.issubset(raw):
        raise ContractError("missing required manifest fields")
    for name, cls in (
        ("teacher", TeacherIdentity),
        ("sequence", SequenceInfo),
        ("kv", KVSpec),
        ("logits", LogitsSpec),
        ("topology", Topology),
        ("provenance", Provenance),
    ):
        if not set(cls.__struct_fields__).issubset(raw[name]):
            raise ContractError(f"missing required {name} fields")
    for obj in raw["objects"]:
        required = set(TensorDescriptor.__struct_fields__)
        if obj["kind"] == "aux":
            required -= {"layer_id", "component", "token_range", "head_range"}
        if not required.issubset(obj):
            raise ContractError("missing required tensor descriptor fields")
    validate_manifest(manifest, max_tensor_bytes=max_tensor_bytes)
    return manifest


def validate_tensors(
    manifest: Manifest,
    tensors: Mapping[str, torch.Tensor],
    *,
    max_tensor_bytes: int = 2 << 30,
) -> None:
    nv = validate_manifest(manifest, max_tensor_bytes=max_tensor_bytes)
    if set(tensors) != {o.key for o in manifest.objects}:
        raise ContractError("tensor object set differs from the manifest")
    aux = {}
    for obj in manifest.objects:
        tensor = tensors[obj.key]
        if list(tensor.shape) != obj.shape or tensor.dtype != DTYPES[obj.dtype]:
            raise ContractError("received tensor shape/dtype mismatch")
        if digest_bytes(tensor_bytes(tensor)) != obj.sha256:
            raise ContractError("tensor checksum mismatch")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ContractError("nonfinite captured values")
        if obj.kind == "aux":
            aux[obj.name] = tensor
    n, p = manifest.sequence.total_length, manifest.sequence.prompt_length
    vocab = manifest.teacher.vocab_size
    if not ((aux["token_ids"] >= 0) & (aux["token_ids"] < vocab)).all():
        raise ContractError("token IDs outside teacher vocabulary")
    if not torch.equal(aux["logits_positions"], torch.arange(p, n, dtype=torch.int32)):
        raise ContractError("teacher rows must cover every accepted response token")
    if not (
        (aux["position_ids"] >= 0).all()
        and (aux["position_ids"][1:] > aux["position_ids"][:-1]).all()
    ):
        raise ContractError("positions must be nonnegative and strictly increasing")
    mask = aux["loss_mask"]
    if mask[:p].any() or (mask > 1).any():
        raise ContractError("invalid response loss mask")
    if (
        manifest.sequence.loss_mask_policy == "current_response_include_eos"
        and not mask[p:].all()
    ):
        raise ContractError("response mask has missing supervision")
    expected_valid = torch.zeros(n, dtype=torch.uint8)
    expected_valid[:nv] = 1
    if not torch.equal(aux["kv_valid"], expected_valid):
        raise ContractError("KV validity differs from object coverage")
    ids = aux["teacher_topk_ids"]
    if not ((ids >= 0) & (ids < vocab)).all():
        raise ContractError("top-k IDs outside teacher vocabulary")
    ordered_ids = ids.sort(dim=-1).values
    if (ordered_ids[:, 1:] == ordered_ids[:, :-1]).any():
        raise ContractError("duplicate IDs within a teacher row")
    logits = aux["teacher_topk_logits"]
    if (logits[:, 1:] > logits[:, :-1]).any():
        raise ContractError("top-k scores are not sorted")
    mass = torch.exp(logits - aux["teacher_logsumexp"].unsqueeze(-1)).sum(-1)
    if (mass > 1.00002).any():
        raise ContractError("top-k mass exceeds full-vocabulary normalization")
