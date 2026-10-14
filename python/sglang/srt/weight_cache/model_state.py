# SPDX-License-Identifier: Apache-2.0
"""Experimental post-load state transport for DeepSeek V4.1.

Tensor payloads use the existing IPC transport. The manifest contains only
references and primitive metadata; no tensors, callables, or CUDA resources.
Host Engram uses daemon-owned memfds or CUDA HOST_NUMA VMM allocations. Mutable request state is not cached.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os

import torch

STATE_ENTRY = "__sglang_model_state__"
TENSOR_PREFIX = "__sglang_model_tensor__/"
CONTRACT = "deepseek_v41_post_load_v3"
# Post-load flags consumed by the model, rather than purely constructor flags.
MODULE_FLAGS = (
    "ape_converted",
    "_mxfp4_backend",
    "_mega_moe_weights_built",
    "block_fp8_mxfp8_ready",
    "intermediate_size_per_partition",
    "hidden_pad",
    "intermediate_pad",
    "pad_id",
    "image_token_id",
    "k_scale_float",
    "v_scale_float",
    "use_padded_loading",
)
PARAM_FLAGS = ("format_ue8m0",)


def is_dsv41(model_config) -> bool:
    cfg = getattr(model_config, "hf_text_config", model_config.hf_config)
    return getattr(cfg, "model_type", None) in ("deepseek_v41", "deepseek_v4.1")


def model_state_contract(model_config, extra_config) -> str:
    if not is_dsv41(model_config):
        return ""
    if isinstance(extra_config, str):
        extra_config = json.loads(extra_config)
    if not isinstance(extra_config or {}, dict):
        raise ValueError("model-loader-extra-config must be a JSON object")
    if (extra_config or {}).get("dsv41_weight_cache") is not True:
        raise ValueError(
            "DeepSeek V4.1 IPC recovery is experimental; explicitly set "
            "model-loader-extra-config must contain dsv41_weight_cache=true."
        )
    return CONTRACT


def host_table_backend(extra_config):
    if isinstance(extra_config, str):
        extra_config = json.loads(extra_config)
    backend = (extra_config or {}).get("dsv41_host_table_backend", "memfd")
    if backend not in ("memfd", "cuda_vmm"):
        raise ValueError("dsv41_host_table_backend must be memfd or cuda_vmm")
    return backend


def layout_fingerprint(model_config, server_args) -> str:
    """Stamp choices that change packed weights or derived model state."""
    fields = (
        "moe_runner_backend",
        "fp8_gemm_runner_backend",
        "fp4_gemm_runner_backend",
        "bf16_gemm_backend",
        "quantization",
        "enable_shared_experts_fusion",
        "enable_deterministic_inference",
        "flashinfer_mxfp4_moe_precision",
        "enable_symm_mem",
    )
    data = {
        "contract": CONTRACT,
        "host_backend": host_table_backend(
            getattr(server_args, "model_loader_extra_config", None)
        ),
        "config": model_config.hf_config.to_dict(),
        "flags": {k: getattr(server_args, k, None) for k in fields},
        "env": {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith(
                (
                    "SGLANG_OPT_",
                    "SGLANG_DSV4",
                    "SGLANG_ENABLE_DSV41_ENGRAM",
                    "SGLANG_DEEPGEMM_",
                    "SGLANG_MXFP",
                    "SGLANG_USE_AITER",
                )
            )
        },
    }
    libraries = {}
    for package in (
        "sglang-kernel",
        "flashinfer-python",
        "deep-gemm",
        "triton",
        "tvm-ffi",
        "tilelang",
    ):
        try:
            libraries[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            libraries[package] = ""
    data["libraries"] = libraries
    # Two checkouts may have the same library version but different weight
    # transforms. Include source content, not paths, in the identity.
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    paths = (
        "weight_cache/model_state.py",
        "models/deepseek_v4.py",
        "layers/engram.py",
        "weight_cache/transport.py",
        "weight_cache/host_vmm.py",
        "layers/attention/dsv4/compressor.py",
        "layers/quantization/fp8.py",
        "layers/quantization/fp8_utils.py",
        "layers/quantization/mxfp4_flashinfer_trtllm_moe.py",
        "layers/moe/mega_moe.py",
    )
    data["source"] = {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _primitive(value):
    if value is None or type(value) in (bool, int, float, str):
        return True
    return isinstance(value, (list, tuple)) and all(_primitive(v) for v in value)


def _tensor_tree(value):
    if isinstance(value, torch.Tensor):
        return True
    return (
        isinstance(value, (tuple, list))
        and bool(value)
        and all(_tensor_tree(v) or v is None for v in value)
    )


def _class_name(obj):
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def export_model_state(model, state_tensors):
    """Capture derived tensor references and metadata after weight transforms.

    This runs before any request/KV initialization. Module registries and
    quantization runner objects remain constructed in the client; they are
    never pickled. Derived tensor trees keep their tuple/list structure.
    """
    manifest = {
        "contract": CONTRACT,
        "schema": {},
        "objects": {},
        "attrs": [],
        "params": {},
        "extras": {},
    }
    for name, (tensor, is_param) in state_tensors.items():
        manifest["schema"][name] = [list(tensor.shape), str(tensor.dtype), is_param]
    tensor_ids = {}

    def encode(value):
        if isinstance(value, torch.Tensor):
            key = tensor_ids.get(id(value))
            if key is None:
                key = f"{TENSOR_PREFIX}{len(tensor_ids)}"
                tensor_ids[id(value)] = key
                state_tensors[key] = (value.detach(), False)
                manifest["extras"][key] = [list(value.shape), str(value.dtype)]
            return {"tensor": key}
        return {
            "tuple" if isinstance(value, tuple) else "list": [
                None if v is None else encode(v) for v in value
            ]
        }

    def capture(path, obj, quant=False):
        manifest["objects"][path] = _class_name(obj)
        for attr, value in vars(obj).items():
            if _tensor_tree(value):
                manifest["attrs"].append([path, attr, encode(value)])
            elif attr in MODULE_FLAGS or (quant and _primitive(value)):
                if not _primitive(value):
                    raise ValueError(f"Non-primitive post-load flag: {path}.{attr}")
                manifest["attrs"].append([path, attr, {"value": value}])

    for path, module in model.named_modules():
        capture(path, module)
        method = getattr(module, "quant_method", None)
        if method is not None:
            qpath = f"{path}.quant_method" if path else "quant_method"
            capture(qpath, method, quant=True)
            wrapped = getattr(method, "_fp8", None)
            if wrapped is not None:
                capture(qpath + "._fp8", wrapped, quant=True)
    for name, param in model.named_parameters(remove_duplicate=False):
        flags = {k: getattr(param, k) for k in PARAM_FLAGS if hasattr(param, k)}
        if flags:
            manifest["params"][name] = flags
    return manifest


def host_table_exports(model, state_tensors):
    """Describe registered Engram views without serializing CPU tensor payloads."""
    entries, fds = {}, []
    for path, module in model.named_modules():
        table = getattr(module, "host_table", None)
        if table is None:
            continue
        if table.fd is None:
            raise ValueError(
                "IPC host Engram requires daemon-owned shareable host storage"
            )
        descriptor = {
            "index": len(fds),
            "module": path,
            "layout": table.layout,
            "nbytes": table.nbytes,
        }
        if getattr(table, "backend", "memfd") == "cuda_vmm":
            descriptor.update(
                backend="cuda_vmm",
                allocation_nbytes=table._vmm.allocation_nbytes,
                numa_id=table._vmm.numa_id,
            )
        fds.append(table.fd)
        for attr in ("weight", "scale"):
            name = f"{path}.{attr}"
            tensor, is_param = state_tensors[name]
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError(f"Invalid host Engram tensor: {name}")
            entries[name] = {
                "host_table": descriptor,
                "host_offset": tensor.data_ptr() - table.bytes.data_ptr(),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "is_param": is_param,
            }
    return entries, fds


def _resolve(model, path):
    obj = model
    for part in path.split(".") if path else ():
        obj = getattr(obj, part)
    return obj


def validate_model_state(model, manifest):
    if manifest.get("contract") != CONTRACT:
        raise ValueError("Missing or incompatible V4.1 post-load state contract")
    for path, cls in manifest["objects"].items():
        if _class_name(_resolve(model, path)) != cls:
            raise ValueError(f"V4.1 IPC object class mismatch at {path}")


def restore_model_state(model, manifest, tensors):
    validate_model_state(model, manifest)

    def decode(value):
        if "tensor" in value:
            return tensors[value["tensor"]]
        if "value" in value:
            return value["value"]
        if "tuple" in value:
            return tuple(None if v is None else decode(v) for v in value["tuple"])
        if "list" in value:
            return [None if v is None else decode(v) for v in value["list"]]
        raise ValueError("Invalid V4.1 tensor descriptor")

    for path, attr, value in manifest["attrs"]:
        setattr(_resolve(model, path), attr, decode(value))
    for name, flags in manifest["params"].items():
        for flag, value in flags.items():
            if flag not in PARAM_FLAGS:
                raise ValueError(f"Unknown V4.1 parameter flag: {flag}")
            setattr(_resolve(model, name), flag, value)
    # Bind only process-local references. Never call post_load_weights(), which
    # would transform shared scales/APE a second time and corrupt all clients.
    layers = getattr(getattr(model, "model", None), "layers", ())
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        topk_config = getattr(getattr(mlp, "topk", None), "topk_config", None)
        gate = getattr(mlp, "gate", None)
        if topk_config is not None and gate is not None:
            # TopKConfig is a dataclass, not a registered module. Its cached
            # parameter reference still points to the constructor's meta bias.
            topk_config.correction_bias = gate.e_score_correction_bias
        if hasattr(layer, "_init_hyper_connections"):
            layer._init_hyper_connections()
        if hasattr(layer, "bind_next"):
            layer.bind_next(layers[i + 1] if i + 1 < len(layers) else None)
