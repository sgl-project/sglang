"""Bind sample semantics to the loaded attention implementation and artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import msgspec
import torch
from sglang.srt.training_capture.config import CaptureConfig
from sglang.srt.training_capture.protocol import (
    ContractError,
    KVSpec,
    LayerGeometry,
    Nonnegative,
    Positive,
    StrictStruct,
    TeacherIdentity,
    Text,
    canonical_bytes,
    digest_bytes,
    validate_kv_spec,
)
from sglang.srt.training_capture.topology import CaptureLayout, plan_capture_layout


class LocalLayerContract(StrictStruct):
    geometry: LayerGeometry
    head_range: tuple[Nonnegative, Positive]
    rope_config: dict
    source_k_norm: Text


class RankTargetContract(StrictStruct):
    teacher: TeacherIdentity
    tp_rank: Nonnegative
    tp_size: Positive
    pp_rank: Nonnegative
    pp_size: Positive
    dp_rank: Nonnegative
    pp_layer_range: tuple[Nonnegative, Positive]
    num_attention_layers: Positive
    selected_layer_ids: list[Nonnegative]
    dtype: Literal["bfloat16", "float16"]
    source_page_size: Positive
    storage_chunk_tokens: Positive
    layers: list[LocalLayerContract]


def artifact_digest(root: Path, names: list[str]) -> str:
    if not root.is_dir() or not names:
        raise ContractError("capture identity requires local model/tokenizer artifacts")
    records = []
    for name in sorted(set(names)):
        if Path(name).name != name:
            raise ContractError("artifact must be directly inside its model directory")
        path = root / name
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 << 20):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise ContractError(
                "model/tokenizer artifact changed during identity binding"
            )
        records.append(
            {"name": name, "size": after.st_size, "sha256": digest.hexdigest()}
        )
    return digest_bytes(canonical_bytes(records))


def local_safetensors_digest(root: Path) -> str:
    index = root / "model.safetensors.index.json"
    if index.exists():
        index_data = json.loads(index.read_bytes())
        weights = list(set(index_data["weight_map"].values())) + [index.name]
    else:
        weights = [p.name for p in root.glob("*.safetensors")]
    if not weights:
        raise ContractError("identity binding requires local safetensors artifacts")
    return artifact_digest(root, weights + ["config.json"])


def bind_contract(
    *, config: CaptureConfig, model, model_config, tokenizer_path: str, pool
) -> tuple[TeacherIdentity, KVSpec]:
    return bind_target_contract(
        model_id=config.model_id,
        selected_layer_ids=config.selected_layer_ids,
        storage_chunk_tokens=config.storage_chunk_tokens,
        expected_weights_revision=config.expected_weights_revision,
        expected_tokenizer_revision=config.expected_tokenizer_revision,
        model=model,
        model_config=model_config,
        tokenizer_path=tokenizer_path,
        pool=pool,
    )


def bind_target_contract(
    *,
    model_id: str,
    selected_layer_ids: list[int],
    storage_chunk_tokens: int,
    model,
    model_config,
    tokenizer_path: str,
    pool,
    expected_weights_revision: str | None = None,
    expected_tokenizer_revision: str | None = None,
) -> tuple[TeacherIdentity, KVSpec]:
    """Bind a complete single-rank target; distributed callers collect all ranks."""
    rank = bind_rank_target_contract(
        model_id=model_id,
        selected_layer_ids=selected_layer_ids,
        storage_chunk_tokens=storage_chunk_tokens,
        model=model,
        model_config=model_config,
        tokenizer_path=tokenizer_path,
        pool=pool,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        expected_weights_revision=expected_weights_revision,
        expected_tokenizer_revision=expected_tokenizer_revision,
    )
    teacher, kv, _ = assemble_target_contract([rank], tp_size=1, pp_size=1)
    return teacher, kv


def bind_rank_target_contract(
    *,
    model_id: str,
    selected_layer_ids: list[int],
    storage_chunk_tokens: int,
    model,
    model_config,
    tokenizer_path: str,
    pool,
    tp_rank: int,
    tp_size: int,
    pp_rank: int,
    pp_size: int,
    dp_rank: int = 0,
    expected_weights_revision: str | None = None,
    expected_tokenizer_revision: str | None = None,
) -> RankTargetContract:
    """Inspect local attention and buffers, including noncanonical KV replicas."""
    from sglang.srt.layers.linear import QKVParallelLinear
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    if (
        any(type(x) is not int or x < 0 for x in (tp_rank, pp_rank, dp_rank))
        or any(type(x) is not int or x < 1 for x in (tp_size, pp_size))
        or tp_rank >= tp_size
        or pp_rank >= pp_size
    ):
        raise ContractError("invalid target identity rank")
    architecture = type(model).__name__
    if architecture not in ("Qwen3ForCausalLM", "Qwen2ForCausalLM", "LlamaForCausalLM"):
        raise ContractError(
            f"training capture has no validated codec for {architecture}"
        )
    hf = model_config.hf_text_config.to_dict()
    if (
        model_config.is_multimodal
        or model_config.quantization is not None
        or hf.get("quantization_config")
        or hf.get("sliding_window")
    ):
        raise ContractError("capture requires unquantized text-only full attention")
    if (
        not isinstance(pool, MHATokenToKVPool)
        or pool.is_quantized_kv_cache
        or pool.use_hnd
        or pool.dtype not in (torch.bfloat16, torch.float16)
    ):
        raise ContractError("capture supports unquantized BF16/FP16 NHD KV only")
    if model_config.vocab_size < 128:
        raise ContractError("capture requires at least 128 vocabulary entries")
    num_layers = hf["num_hidden_layers"]
    start, end = model.model.start_layer, model.model.end_layer
    if (
        model.pp_group.world_size != pp_size
        or model.pp_group.rank_in_group != pp_rank
        or len(model.model.layers) != num_layers
        or not 0 <= start < end <= num_layers
        or pool.start_layer != start
        or pool.layer_num != end - start
    ):
        raise ContractError("model and KV pool disagree with the PP stage")
    if (
        not selected_layer_ids
        or any(
            type(x) is not int or not 0 <= x < num_layers for x in selected_layer_ids
        )
        or len(set(selected_layer_ids)) != len(selected_layer_ids)
        or type(storage_chunk_tokens) is not int
        or storage_chunk_tokens < 1
    ):
        raise ContractError("invalid selected layers or storage chunk size")

    def inspect_projection(attention):
        projection = attention.qkv_proj
        if (
            not isinstance(projection, QKVParallelLinear)
            or (projection.tp_rank, projection.tp_size) != (tp_rank, tp_size)
            or (projection.kv_tp_rank, projection.kv_tp_size) != (tp_rank, tp_size)
            or projection.total_num_kv_heads != attention.total_num_kv_heads
            or projection.num_kv_heads != attention.num_kv_heads
            or projection.head_size != attention.head_dim
            or projection.v_head_size != pool.v_head_dim
        ):
            raise ContractError(
                "attention projection differs from ordinary TP geometry"
            )
        return projection

    # Even a stage with no selected layers must bind its actual TP placement.
    inspect_projection(model.model.layers[start].self_attn)
    layers = []
    for layer_id in selected_layer_ids:
        if not start <= layer_id < end:
            continue
        attention = model.model.layers[layer_id].self_attn
        projection = inspect_projection(attention)
        for buffer, dim in (
            (pool.get_key_buffer(layer_id), projection.head_size),
            (pool.get_value_buffer(layer_id), projection.v_head_size),
        ):
            if (
                buffer.dtype != pool.dtype
                or buffer.ndim != 3
                or tuple(buffer.shape[1:]) != (projection.num_kv_heads, dim)
            ):
                raise ContractError("local KV buffer differs from attention geometry")
        rope = attention.rotary_emb
        if type(rope) is not RotaryEmbedding:
            raise ContractError("capture currently requires standard, unscaled RoPE")
        norm = (
            canonical_bytes(
                {
                    "type": "rmsnorm",
                    "epsilon": attention.k_norm.variance_epsilon,
                    "stage": "before_rope",
                }
            ).decode()
            if architecture == "Qwen3ForCausalLM"
            else "none"
        )
        first = (
            projection.kv_tp_rank // projection.num_kv_head_replicas
        ) * projection.num_kv_heads
        layers.append(
            LocalLayerContract(
                geometry=LayerGeometry(
                    layer_id=layer_id,
                    num_kv_heads=projection.total_num_kv_heads,
                    key_head_dim=projection.head_size,
                    value_head_dim=projection.v_head_size,
                ),
                head_range=(first, first + projection.num_kv_heads),
                rope_config={
                    "type": "default",
                    "theta": float(rope.base),
                    "rotary_dim": rope.rotary_dim,
                    "interleaved": not rope.is_neox_style,
                    "scaling": None,
                },
                source_k_norm=norm,
            )
        )
    root = Path(model_config.model_path)
    weights_revision = local_safetensors_digest(root)
    tokenizer_root = Path(tokenizer_path)
    tokenizer_names = [
        name
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
        )
        if (tokenizer_root / name).is_file()
    ]
    if not any(
        name in tokenizer_names for name in ("tokenizer.json", "tokenizer.model")
    ):
        raise ContractError("capture requires local tokenizer artifacts")
    tokenizer_revision = artifact_digest(tokenizer_root, tokenizer_names)
    if expected_weights_revision not in (
        None,
        weights_revision,
    ) or expected_tokenizer_revision not in (None, tokenizer_revision):
        raise ContractError(
            "model/tokenizer artifacts differ from the configured immutable revisions"
        )
    processor = model.logits_processor
    if (
        tp_size > 1
        and pp_rank == pp_size - 1
        and not getattr(processor, "do_tensor_parallel_all_gather", False)
    ):
        raise ContractError("teacher capture requires global TP logits before sampling")
    transform = canonical_bytes(
        {
            "logit_scale": processor.logit_scale,
            "final_logit_softcapping": processor.final_logit_softcapping,
        }
    ).decode()
    fingerprint = digest_bytes(
        canonical_bytes(
            {
                "weights_revision": weights_revision,
                "tokenizer_revision": tokenizer_revision,
                "architecture": architecture,
                "resolved_config": hf,
                "model_dtype": str(model_config.dtype),
                "output_transform": transform,
            }
        )
    )
    teacher = TeacherIdentity(
        model_id=model_id,
        weights_revision=weights_revision,
        adapter_revision=None,
        tokenizer_revision=tokenizer_revision,
        fingerprint_sha256=fingerprint,
        vocab_size=model_config.vocab_size,
        output_transform=transform,
    )
    return RankTargetContract(
        teacher=teacher,
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        dp_rank=dp_rank,
        pp_layer_range=(start, end),
        num_attention_layers=num_layers,
        selected_layer_ids=list(selected_layer_ids),
        dtype=str(pool.dtype).removeprefix("torch."),
        source_page_size=pool.page_size,
        storage_chunk_tokens=storage_chunk_tokens,
        layers=layers,
    )


def assemble_target_contract(
    ranks: list[RankTargetContract],
    *,
    tp_size: int,
    pp_size: int,
    dp_rank: int = 0,
    aux_tp_rank: int = 0,
) -> tuple[TeacherIdentity, KVSpec, CaptureLayout]:
    """Validate every serving rank, including replicas with no payload ownership."""
    if (
        not ranks
        or any(type(x) is not int or x < 1 for x in (tp_size, pp_size))
        or type(dp_rank) is not int
        or dp_rank < 0
    ):
        raise ContractError("invalid or empty target identity topology")
    try:
        ranks = msgspec.convert(
            msgspec.to_builtins(ranks), type=list[RankTargetContract]
        )
    except (msgspec.ValidationError, TypeError) as error:
        raise ContractError("invalid rank target contract") from error
    reference = ranks[0]
    common = (
        "teacher",
        "num_attention_layers",
        "selected_layer_ids",
        "dtype",
        "source_page_size",
        "storage_chunk_tokens",
    )
    seen, stage_ranges, layers = set(), {}, {}
    for rank in ranks:
        identity = (rank.pp_rank, rank.tp_rank)
        if (
            (rank.tp_size, rank.pp_size, rank.dp_rank) != (tp_size, pp_size, dp_rank)
            or rank.tp_rank >= tp_size
            or rank.pp_rank >= pp_size
            or identity in seen
            or any(getattr(rank, name) != getattr(reference, name) for name in common)
        ):
            raise ContractError("duplicate rank or inconsistent target identity")
        seen.add(identity)
        start, end = rank.pp_layer_range
        if (
            not start < end <= rank.num_attention_layers
            or stage_ranges.setdefault(rank.pp_rank, (start, end)) != (start, end)
            or [layer.geometry.layer_id for layer in rank.layers]
            != [i for i in rank.selected_layer_ids if start <= i < end]
        ):
            raise ContractError("rank target contract has inconsistent PP layers")
        for layer in rank.layers:
            previous = layers.setdefault(layer.geometry.layer_id, layer)
            if (
                layer.geometry != previous.geometry
                or layer.rope_config != previous.rope_config
                or layer.source_k_norm != previous.source_k_norm
            ):
                raise ContractError("ranks disagree on selected layer semantics")
    if seen != {(pp, tp) for pp in range(pp_size) for tp in range(tp_size)}:
        raise ContractError("target identity requires every serving rank")
    ranges = [stage_ranges[pp] for pp in range(pp_size)]
    if ranges[-1][1] != reference.num_attention_layers:
        raise ContractError("PP stages do not cover the target model")
    selected = reference.selected_layer_ids
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(layers) != set(selected)
    ):
        raise ContractError("incomplete or duplicate selected layer identity")
    codec = layers[selected[0]]
    if any(
        layer.rope_config != codec.rope_config
        or layer.source_k_norm != codec.source_k_norm
        for layer in layers.values()
    ):
        raise ContractError("selected layers require different RoPE/norm codecs")
    kv = KVSpec(
        codec=f"dense_{'bf16' if reference.dtype == 'bfloat16' else 'fp16'}_post_rope_v1",
        dtype=reference.dtype,
        selected_layer_ids=selected,
        layers=[layers[i].geometry for i in selected],
        source_k_stage="post_rope",
        source_k_norm=codec.source_k_norm,
        rope_config=codec.rope_config,
        rope_config_sha256=digest_bytes(canonical_bytes(codec.rope_config)),
        storage_chunk_tokens=reference.storage_chunk_tokens,
        source_page_size=reference.source_page_size,
    )
    validate_kv_spec(kv)
    layout = plan_capture_layout(
        kv,
        tp_size=tp_size,
        pp_layer_ranges=ranges,
        dp_rank=dp_rank,
        aux_tp_rank=aux_tp_rank,
    )
    for rank in ranks:
        for layer in rank.layers:
            replicas = max(1, tp_size // layer.geometry.num_kv_heads)
            canonical_rank = rank.tp_rank // replicas * replicas
            partition = layout.partition(
                f"dp{dp_rank}-pp{rank.pp_rank}-tp{canonical_rank}"
            )
            expected = next(
                p for p in partition.heads if p.layer_id == layer.geometry.layer_id
            )
            if layer.head_range != (expected.start, expected.end):
                raise ContractError(
                    "actual KV heads differ from canonical TP placement"
                )
    return reference.teacher, kv, layout
