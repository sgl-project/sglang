"""Export consolidated KV-input DSpark training weights for SGLang serving."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import msgspec
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.dspark_components.dspark_target_kv_contract import (
    read_target_kv_draft_contract,
)
from sglang.srt.training_capture.protocol import (
    ContractError,
    canonical_bytes,
)
from transformers import AutoConfig

_DTYPES = {name: getattr(torch, name) for name in ("bfloat16", "float16", "float32")}


def _positive(config, name, default=None):
    value = config.get(name, default)
    if type(value) is not int or value <= 0:
        raise ContractError(f"export requires a positive integer {name}")
    return value


def _layout(config):
    contract = read_target_kv_draft_contract(config)
    if contract is None:
        raise ContractError("export requires explicit target_kv input mode")
    hidden = _positive(config, "hidden_size")
    intermediate = _positive(config, "intermediate_size")
    layers = _positive(config, "num_hidden_layers")
    heads = _positive(config, "num_attention_heads")
    kv_heads = _positive(config, "num_key_value_heads", heads)
    dimension = _positive(config, "head_dim", hidden // heads)
    vocab = _positive(config, "vocab_size")
    rank = _positive(config, "markov_rank")
    if layers > 4096 or dimension % 2 or heads % kv_heads:
        raise ContractError("invalid draft layer/head geometry")
    if vocab != contract.teacher.vocab_size:
        raise ContractError("draft vocabulary differs from its teacher contract")
    if config.get("quantization_config") is not None:
        raise ContractError("KV draft export requires unquantized weights")
    if config.get("hidden_act", "silu") != "silu":
        raise ContractError("KV draft export requires SiLU")
    epsilon = config.get("rms_norm_eps", 1e-6)
    if type(epsilon) not in (float, int) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ContractError("draft RMSNorm epsilon must be finite and positive")
    attention_bias = config.get("attention_bias", False)
    if type(attention_bias) is not bool:
        raise ContractError("attention_bias must be a boolean")
    head_type = config.get("markov_head_type")
    if head_type not in ("vanilla", "gated", "rnn"):
        raise ContractError("export requires a supported explicit Markov head")
    parsed = parse_dspark_draft_config(draft_hf_config=config)
    if (
        parsed.gamma != contract.sequence.prediction_count
        or parsed.mask_token_id != contract.sequence.mask_token_id
        or parsed.markov_rank != rank
        or parsed.markov_head_type != head_type
        or _positive(config, "block_size") != contract.sequence.prediction_count
        or type(config.get("mask_token_id")) is not int
        or config["mask_token_id"] != contract.sequence.mask_token_id
    ):
        raise ContractError("draft config aliases disagree with its serving contract")
    dtype = config.get("dtype") or config.get("torch_dtype")
    if dtype not in _DTYPES or any(
        config.get(key) is not None and config[key] != dtype
        for key in ("dtype", "torch_dtype")
    ):
        raise ContractError("export requires one explicit FP32/FP16/BF16 dtype")
    shapes = {
        "kv_encoder.projection.weight": (hidden, contract.feature_size),
        "kv_encoder.norm_weight": (hidden,),
        "norm.weight": (hidden,),
        "markov_head.markov_w1.weight": (vocab, rank),
        "markov_head.markov_w2.weight": (vocab, rank),
    }
    if head_type == "gated":
        shapes.update(
            {
                "markov_head.gate_proj.weight": (rank, hidden + rank),
                "markov_head.gate_proj.bias": (rank,),
            }
        )
    elif head_type == "rnn":
        shapes.update(
            {
                "markov_head.joint_proj.weight": (3 * rank, hidden + 2 * rank),
                "markov_head.joint_proj.bias": (3 * rank,),
            }
        )
    fused = {}
    for index in range(layers):
        prefix = f"layers.{index}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            shapes[prefix + name + ".weight"] = (hidden,)
        for name in ("q_norm", "k_norm"):
            shapes[prefix + "self_attn." + name + ".weight"] = (dimension,)
        for name, rows in (
            ("q", heads * dimension),
            ("k", kv_heads * dimension),
            ("v", kv_heads * dimension),
        ):
            shapes[prefix + f"self_attn.{name}_proj.weight"] = (rows, hidden)
            if attention_bias:
                shapes[prefix + f"self_attn.{name}_proj.bias"] = (rows,)
        shapes[prefix + "self_attn.o_proj.weight"] = (hidden, heads * dimension)
        if attention_bias:
            shapes[prefix + "self_attn.o_proj.bias"] = (hidden,)
        for name in ("gate", "up"):
            shapes[prefix + f"mlp.{name}_proj.weight"] = (intermediate, hidden)
        shapes[prefix + "mlp.down_proj.weight"] = (hidden, intermediate)
        for suffix in ("weight", "bias") if attention_bias else ("weight",):
            fused[prefix + "self_attn.qkv_proj." + suffix] = tuple(
                prefix + f"self_attn.{part}_proj.{suffix}" for part in ("q", "k", "v")
            )
        fused[prefix + "mlp.gate_up_proj.weight"] = tuple(
            prefix + f"mlp.{part}_proj.weight" for part in ("gate", "up")
        )
    return contract, _DTYPES[dtype], shapes, fused


@torch.no_grad()
def _cpu_weights(weights, dtype, shapes, fused):
    output = {}
    for original_name, weight in (
        weights.items() if isinstance(weights, Mapping) else weights
    ):
        if not isinstance(original_name, str):
            raise ContractError("checkpoint weight names must be strings")
        name = original_name.removeprefix("model.")
        if name not in shapes and name not in fused:
            raise ContractError(f"unexpected KV draft weight: {original_name}")
        if (
            not isinstance(weight, torch.Tensor)
            or weight.layout != torch.strided
            or weight.is_meta
            or weight.dtype not in _DTYPES.values()
        ):
            raise ContractError(
                f"export requires dense floating weights: {original_name}"
            )
        parts = fused.get(name, (name,))
        expected = (sum(shapes[part][0] for part in parts), *shapes[parts[0]][1:])
        if tuple(weight.shape) != expected:
            raise ContractError(
                f"weight shape mismatch for {original_name}: expected {expected}, got {tuple(weight.shape)}"
            )
        extrema = torch.stack(torch.aminmax(weight)).to(dtype=dtype)
        if not torch.isfinite(extrema).all().item():
            raise ContractError(f"weight is nonfinite in {dtype}: {original_name}")
        slices = weight.detach().split([shapes[part][0] for part in parts], dim=0)
        for part, value in zip(parts, slices, strict=True):
            if part in output:
                raise ContractError(f"duplicate or mixed packed/split weight: {part}")
            # Own each CPU storage even when trainer parameters share storage.
            output[part] = value.to(device="cpu", dtype=dtype, copy=True).contiguous()
    missing = shapes.keys() - output.keys()
    if missing:
        raise ContractError(f"missing KV draft weights: {sorted(missing)}")
    return output


def _digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def export_target_kv_checkpoint(
    config, weights, *, golden_fixture, output_dir, acceptance_report=None
):
    """Export a quiescent, consolidated draft state; never copy a parity pass.

    ``config`` must already declare the KV-input architecture, teacher/codec,
    sequence/objective and exact golden-fixture digest. ``weights`` may be a
    mapping or an iterable of (name, tensor), including complete fused weights.
    The caller must freeze training mutations for the duration of the export.
    """
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"export destination already exists: {output}")
    raw = config.to_dict() if hasattr(config, "to_dict") else config
    data = canonical_bytes(raw)
    if len(data) > 1 << 20:
        raise ContractError("draft config is too large")
    config = msgspec.json.decode(data, type=dict)
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ContractError("export requires an explicit Hugging Face model_type")
    if (
        config.get("dtype") is not None
        and config.get("torch_dtype") is not None
        and config["dtype"] != config["torch_dtype"]
    ):
        raise ContractError("draft dtype aliases disagree")
    try:
        config = AutoConfig.for_model(
            model_type,
            **{key: value for key, value in config.items() if key != "model_type"},
        ).to_dict()
    except (TypeError, ValueError) as error:
        raise ContractError(f"unsupported draft config: {error}") from error
    data = canonical_bytes(config)
    if len(data) > 1 << 20:
        raise ContractError("resolved draft config is too large")
    contract, dtype, shapes, fused = _layout(config)
    prepared = _cpu_weights(weights, dtype, shapes, fused)
    if (acceptance_report is None) != (
        contract.validation.acceptance_report_sha256 is None
    ):
        raise ContractError(
            "acceptance report and its contract digest must be supplied together"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.export-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "checkpoint"
        staging.mkdir()
        validation = staging / "validation"
        validation.mkdir()
        fixture = validation / "inputs.safetensors"
        shutil.copyfile(golden_fixture, fixture)
        if _digest_file(fixture) != contract.validation.golden_fixture_sha256:
            raise ContractError("golden fixture differs from the declared contract")
        with safe_open(fixture, framework="pt", device="cpu") as source:
            if not source.keys():
                raise ContractError("golden fixture is empty")
        if acceptance_report is not None:
            destination = validation / "acceptance.json"
            shutil.copyfile(acceptance_report, destination)
            if (
                _digest_file(destination)
                != contract.validation.acceptance_report_sha256
            ):
                raise ContractError(
                    "acceptance report differs from the declared contract"
                )
        save_file(prepared, staging / "model.safetensors", metadata={"format": "pt"})
        (staging / "config.json").write_bytes(data)
        artifacts = {
            str(path.relative_to(staging)): _digest_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        receipt = {
            "status": "exported",
            "contract_sha256": contract.fingerprint,
            "artifact_sha256": artifacts,
            "parameters": len(prepared),
            "tensor_bytes": sum(
                value.numel() * value.element_size() for value in prepared.values()
            ),
            "dtype": str(dtype).removeprefix("torch."),
            "requires_fixed_input_parity": True,
        }
        (staging / "export.json").write_bytes(canonical_bytes(receipt))
        for path in staging.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"export destination already exists: {output}")
        staging.rename(output)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--golden-fixture", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--acceptance-report")
    args = parser.parse_args(argv)
    try:
        with Path(args.config).open("rb") as stream:
            data = stream.read((1 << 20) + 1)
        if len(data) > 1 << 20:
            raise ContractError("draft config is too large")
        receipt = export_target_kv_checkpoint(
            msgspec.json.decode(data, type=dict),
            load_file(args.weights, device="cpu"),
            golden_fixture=args.golden_fixture,
            output_dir=args.output_dir,
            acceptance_report=args.acceptance_report,
        )
    except (
        ContractError,
        OSError,
        ValueError,
        msgspec.DecodeError,
        SafetensorError,
    ) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
