# SPDX-License-Identifier: Apache-2.0
"""Exact non-routed GGUF loader for the Kimi-K3 expert-pack runtime."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path

import torch

_LAYER_RE = re.compile(r"^blk\.(?P<layer>\d+)\.(?P<suffix>.+)$")
_ROUTED_EXPERT_RE = re.compile(r"^blk\.\d+\.ffn_(?:gate|up|down)_exps\.weight$")

_TOP_LEVEL_NAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}

_COMMON_LAYER_NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_routed_down.weight": "mlp.routed_expert_down_proj.weight",
    "ffn_routed_norm.weight": "mlp.routed_expert_norm.weight",
    "ffn_routed_up.weight": "mlp.routed_expert_up_proj.weight",
    "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight",
}

_KDA_NAMES = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "ssm_g.weight": "self_attn.g_proj.weight",
    "ssm_beta.weight": "self_attn.b_proj.weight",
    "ssm_f_a.weight": "self_attn.f_a_proj.weight",
    "ssm_f_b.weight": "self_attn.f_b_proj.weight",
    "ssm_conv1d_q.weight": "self_attn.q_conv1d.weight",
    "ssm_conv1d_k.weight": "self_attn.k_conv1d.weight",
    "ssm_conv1d_v.weight": "self_attn.v_conv1d.weight",
    "ssm_a": "self_attn.A_log",
    "ssm_dt.bias": "self_attn.dt_bias",
    "ssm_norm.weight": "self_attn.o_norm.weight",
}

_MLA_NAMES = {
    "attn_q_a.weight": "self_attn.q_a_proj.weight",
    "attn_q_a_norm.weight": "self_attn.q_a_layernorm.weight",
    "attn_q_b.weight": "self_attn.q_b_proj.weight",
    "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
    "attn_gate.weight": "self_attn.g_proj.weight",
    # K and V use different GGUF types and must remain separate.
    "attn_k_b.weight": "self_attn.k_b_qweight",
    "attn_v_b.weight": "self_attn.v_b_qweight",
}


def routed_expert_tensor(name: str) -> bool:
    return _ROUTED_EXPERT_RE.fullmatch(name) is not None


def kimi_k3_checkpoint_targets(source_name: str) -> tuple[str, ...]:
    """Map one llama.cpp Kimi-K3 tensor to exact SGLang parameter names."""

    if source_name == "output_res_score.weight":
        return (
            "model.output_attn_res_proj.weight",
            "model.output_attn_res_norm.weight",
        )
    if source_name in _TOP_LEVEL_NAMES:
        return (_TOP_LEVEL_NAMES[source_name],)

    match = _LAYER_RE.fullmatch(source_name)
    if match is None:
        raise KeyError(f"unsupported Kimi-K3 GGUF tensor name: {source_name}")
    layer = int(match.group("layer"))
    suffix = match.group("suffix")
    prefix = f"model.layers.{layer}."
    if suffix == "attn_res_score.weight":
        return (
            prefix + "self_attention_res_proj.weight",
            prefix + "self_attention_res_norm.weight",
        )
    if suffix == "ffn_res_score.weight":
        return (
            prefix + "mlp_res_proj.weight",
            prefix + "mlp_res_norm.weight",
        )
    target = _COMMON_LAYER_NAMES.get(suffix)
    if target is None:
        target = _KDA_NAMES.get(suffix)
    if target is None:
        target = _MLA_NAMES.get(suffix)
    if target is None:
        raise KeyError(f"unsupported Kimi-K3 GGUF tensor name: {source_name}")
    return (prefix + target,)


def _runtime_name(checkpoint_name: str, quantized: bool) -> str:
    if not quantized or not checkpoint_name.endswith(".weight"):
        return checkpoint_name
    return checkpoint_name.removesuffix("weight") + "qweight"


def _residual_target_value(raw: torch.Tensor, target_index: int) -> torch.Tensor:
    if raw.ndim != 1:
        raise ValueError(
            f"Kimi-K3 attention-residual score must be a vector, got {tuple(raw.shape)}"
        )
    if target_index == 0:
        return raw.unsqueeze(0)
    if target_index == 1:
        return torch.ones_like(raw)
    raise ValueError(f"invalid Kimi-K3 attention-residual target {target_index}")


def _kda_a_log_target_value(raw: torch.Tensor) -> torch.Tensor:
    """Undo llama.cpp's GGUF-time ``A_log -> -exp(A_log)`` transform."""
    if not raw.is_floating_point() or not torch.isfinite(raw).all():
        raise ValueError("Kimi-K3 GGUF ssm_a must contain finite floating values")
    if not torch.all(raw < 0):
        raise ValueError("Kimi-K3 GGUF ssm_a must contain only -exp(A_log) values")
    return torch.log(-raw)


def kimi_k3_nonexpert_weights_iterator(
    manifest_path: str | os.PathLike[str],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Stream non-routed tensors shard by shard without reading routed payloads."""

    import gguf

    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("format") != "SGLANG-KIMI-GGMLMOEPACK-ADAPTER-v1":
        raise ValueError("Kimi-K3 manifest format is unsupported")
    if not manifest.get("complete"):
        raise ValueError("Kimi-K3 manifest is incomplete")

    records_by_shard: dict[int, list[dict]] = defaultdict(list)
    for record in manifest["source"]["tensors"]:
        records_by_shard[int(record["shard_index"])].append(record)

    emitted: set[str] = set()
    for shard in manifest["source"]["shards"]:
        shard_index = int(shard["index"])
        shard_path = Path(shard["path"]).resolve()
        if not shard_path.is_file() or shard_path.stat().st_size != int(shard["size"]):
            raise FileNotFoundError(
                f"Kimi-K3 GGUF shard is missing or changed: {shard_path}"
            )
        reader = gguf.GGUFReader(str(shard_path), mode="r")
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        expected = {record["name"]: record for record in records_by_shard[shard_index]}
        if set(tensors) != set(expected):
            raise ValueError(f"Kimi-K3 GGUF shard inventory changed: {shard_path}")

        for source_name, tensor in tensors.items():
            record = expected[source_name]
            if tensor.tensor_type.name != record["dtype"]:
                raise ValueError(f"Kimi-K3 GGUF tensor type changed: {source_name}")
            if routed_expert_tensor(source_name):
                continue

            targets = kimi_k3_checkpoint_targets(source_name)
            quantized = tensor.tensor_type.name not in ("F32", "F16", "BF16")
            raw = torch.tensor(tensor.data)
            for target_index, checkpoint_name in enumerate(targets):
                if source_name.endswith(".ssm_a"):
                    value = _kda_a_log_target_value(raw)
                elif len(targets) == 2:
                    value = _residual_target_value(raw, target_index)
                else:
                    value = raw
                runtime_name = _runtime_name(checkpoint_name, quantized)
                if runtime_name in emitted:
                    raise ValueError(
                        f"duplicate Kimi-K3 target parameter: {runtime_name}"
                    )
                if quantized:
                    type_name = runtime_name.removesuffix("qweight") + "qweight_type"
                    if type_name in emitted:
                        raise ValueError(
                            f"duplicate Kimi-K3 target parameter: {type_name}"
                        )
                    emitted.add(type_name)
                    yield type_name, torch.tensor(
                        int(tensor.tensor_type), dtype=torch.uint8
                    )
                emitted.add(runtime_name)
                yield runtime_name, value
