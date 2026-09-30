"""Bind sample semantics to the loaded attention implementation and artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from sglang.srt.training_capture.config import CaptureConfig
from sglang.srt.training_capture.protocol import (
    ContractError,
    KVSpec,
    LayerGeometry,
    TeacherIdentity,
    canonical_bytes,
    digest_bytes,
)


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
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

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
    if pool.dtype not in (torch.bfloat16, torch.float16):
        raise ContractError("capture supports BF16/FP16 dense KV only")
    if model_config.vocab_size < 128:
        raise ContractError("capture requires at least 128 vocabulary entries")
    geometries, rope_configs, norms = [], [], []
    for layer_id in selected_layer_ids:
        if not 0 <= layer_id < len(model.model.layers):
            raise ContractError("selected layer outside target model")
        attention = model.model.layers[layer_id].self_attn
        rope = attention.rotary_emb
        if type(rope) is not RotaryEmbedding:
            raise ContractError("capture currently requires standard, unscaled RoPE")
        rope_configs.append(
            {
                "type": "default",
                "theta": float(rope.base),
                "rotary_dim": rope.rotary_dim,
                "interleaved": not rope.is_neox_style,
                "scaling": None,
            }
        )
        norms.append(
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
        geometries.append(
            LayerGeometry(
                layer_id=layer_id,
                num_kv_heads=attention.num_kv_heads,
                key_head_dim=attention.head_dim,
                value_head_dim=pool.v_head_dim,
            )
        )
    if any(value != rope_configs[0] for value in rope_configs) or len(set(norms)) != 1:
        raise ContractError("selected layers require different RoPE/norm codecs")
    dtype = str(pool.dtype).removeprefix("torch.")
    kv = KVSpec(
        codec=f"dense_{'bf16' if dtype == 'bfloat16' else 'fp16'}_post_rope_v1",
        dtype=dtype,
        selected_layer_ids=selected_layer_ids,
        layers=geometries,
        source_k_stage="post_rope",
        source_k_norm=norms[0],
        rope_config=rope_configs[0],
        rope_config_sha256=digest_bytes(canonical_bytes(rope_configs[0])),
        storage_chunk_tokens=storage_chunk_tokens,
        source_page_size=pool.page_size,
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
    return teacher, kv
