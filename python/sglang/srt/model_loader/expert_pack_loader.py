# SPDX-License-Identifier: Apache-2.0
"""SSD expert-pack loader for deepseek-v4-flash and text-only kimi-k3.

Only these two language-model paths are currently supported. The multimodal
kimi-k3 model is outside the scope of this loader.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import torch
from torch import nn

from sglang.kernels.ops.moe.expert_pack_mxfp4 import prewarm_mxfp4_extension
from sglang.srt.layers.moe.expert_pack import (
    ExpertPackStore,
    KimiGGMLExpertPackStore,
)
from sglang.srt.layers.quantization.expert_pack import (
    ExpertPackConfig,
    _clamped_swiglu,
)
from sglang.srt.model_loader.deepseek4_gguf import (
    build_deepseek4_checkpoint_name_map,
    routed_expert_tensor,
)
from sglang.srt.model_loader.expert_pack_config import (
    KIMI_K3_MODEL_TYPE,
    validate_expert_pack_model_config,
)
from sglang.srt.model_loader.kimi_k3_gguf import kimi_k3_nonexpert_weights_iterator
from sglang.srt.model_loader.loader import (
    BaseModelLoader,
    _initialize_model,
    device_loading_context,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.runtime_context import get_exec, get_parallel

logger = logging.getLogger(__name__)


def _bf16_tensor(data: np.ndarray) -> torch.Tensor:
    raw = np.asarray(data)
    if raw.dtype != np.uint8 or raw.shape[-1] % 2:
        raise ValueError("GGUF BF16 payload does not have a byte-pair layout")
    values = raw.view(np.uint16).reshape(*raw.shape[:-1], raw.shape[-1] // 2)
    return torch.from_numpy(values.copy()).view(torch.bfloat16)


def _compressor_component(source_name: str) -> str | None:
    if "_compressor_kv.weight" in source_name:
        return "kv"
    if "_compressor_gate.weight" in source_name:
        return "gate"
    return None


def _fused_compressor_name(checkpoint_name: str) -> str:
    result = checkpoint_name.replace(".wkv.weight", ".wkv_gate.weight")
    result = result.replace(".wgate.weight", ".wkv_gate.weight")
    if result == checkpoint_name:
        raise ValueError(f"invalid compressor checkpoint name: {checkpoint_name}")
    return result


def deepseek4_nonexpert_weights_iterator(
    source_path: str | os.PathLike[str],
    num_layers: int,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Yield exact non-routed tensors without materializing routed experts."""

    import gguf

    reader = gguf.GGUFReader(str(source_path), mode="r")
    names = [tensor.name for tensor in reader.tensors]
    mapping = build_deepseek4_checkpoint_name_map(gguf, names, num_layers)
    tensors = {tensor.name: tensor for tensor in reader.tensors}

    # GGUF quant methods must know the type before the raw qweight arrives.
    for tensor in reader.tensors:
        if routed_expert_tensor(tensor.name) is not None:
            continue
        weight_type = tensor.tensor_type
        if weight_type.name == "Q8_0":
            component = _compressor_component(tensor.name)
            if component == "gate":
                continue
            checkpoint_name = (
                _fused_compressor_name(mapping[tensor.name])
                if component == "kv"
                else mapping[tensor.name]
            )
            if not checkpoint_name.endswith(".weight"):
                raise ValueError(
                    f"quantized tensor maps to a non-weight parameter: {tensor.name}"
                )
            yield (
                checkpoint_name.removesuffix("weight") + "qweight_type",
                torch.tensor(int(weight_type), dtype=torch.uint8),
            )

    for tensor in reader.tensors:
        if routed_expert_tensor(tensor.name) is not None:
            continue
        checkpoint_name = mapping[tensor.name]
        weight_type = tensor.tensor_type
        if weight_type.name == "Q8_0":
            component = _compressor_component(tensor.name)
            if component == "gate":
                continue
            if component == "kv":
                gate_name = tensor.name.replace("_compressor_kv", "_compressor_gate")
                gate = tensors.get(gate_name)
                if gate is None or gate.tensor_type != weight_type:
                    raise ValueError(
                        f"missing matching compressor gate tensor: {gate_name}"
                    )
                checkpoint_name = _fused_compressor_name(checkpoint_name)
                raw_weight = torch.cat(
                    (torch.tensor(tensor.data), torch.tensor(gate.data)), dim=0
                )
            else:
                raw_weight = torch.tensor(tensor.data)
            yield checkpoint_name.removesuffix("weight") + "qweight", raw_weight
        elif weight_type.name == "BF16":
            yield checkpoint_name, _bf16_tensor(tensor.data)
        elif weight_type.name in ("F32", "I32"):
            yield checkpoint_name, torch.tensor(tensor.data)
        else:
            raise ValueError(
                f"unsupported non-routed GGUF type {weight_type.name} for {tensor.name}"
            )


class ExpertPackModelLoader(BaseModelLoader):
    def __init__(self, load_config) -> None:
        super().__init__(load_config)
        config = dict(load_config.model_loader_extra_config or {})
        pack_path = config.get("pack_path") or os.getenv("SGLANG_EXPERT_PACK_PATH")
        if not pack_path:
            raise ValueError(
                "expert_pack load format requires pack_path or SGLANG_EXPERT_PACK_PATH"
            )
        self.config = config
        self.pack_path = Path(pack_path).resolve()
        self.manifest_path = (
            Path(config["manifest_path"]).resolve()
            if config.get("manifest_path")
            else None
        )
        self.source_path = (
            Path(config["source_path"]).resolve() if config.get("source_path") else None
        )

    def download_model(self, model_config) -> None:
        if not Path(model_config.model_path).is_dir():
            raise ValueError(
                "expert_pack model_path must be the verified tokenizer/config directory"
            )

    def load_model(self, *, model_config, device_config) -> nn.Module:
        hf_config = model_config.hf_config
        model_kind, model_errors = validate_expert_pack_model_config(hf_config)
        if model_errors:
            details = "\n".join(f"- {error}" for error in model_errors)
            raise ValueError(f"Invalid expert_pack model configuration:\n{details}")
        is_kimi = model_kind == KIMI_K3_MODEL_TYPE
        if is_kimi:
            if self.manifest_path is None or not self.manifest_path.is_file():
                raise FileNotFoundError("Kimi-K3 expert_pack requires manifest_path")

        parallel = get_parallel()
        exec_config = get_exec()
        if (
            parallel.tp_size != 1
            or parallel.moe_dp_size != 1
            or parallel.moe_ep_size != 1
            or not exec_config.graph.disable_cuda_graph
            or not exec_config.moe.disable_shared_experts_fusion
        ):
            raise RuntimeError(
                "expert_pack ServerArgs invariants were not applied before model load"
            )

        if is_kimi:
            stats_path = self.config.get("stats_path")
            store = KimiGGMLExpertPackStore(
                self.pack_path,
                manifest_path=self.manifest_path,
                expected_layers=93,
                expected_experts=896,
                expected_top_k=16,
                cache_vram_mib=int(self.config.get("cache_vram_mib", 4 * 1024)),
                cache_vram_reserve_mib=int(
                    self.config.get("cache_vram_reserve_mib", 2 * 1024)
                ),
                stage_slots=int(self.config.get("stage_slots", 16)),
                read_splits=int(self.config.get("read_splits", 1)),
                direct_io=bool(self.config.get("direct_io", True)),
                stats_flush_interval=int(self.config.get("stats_flush_interval", 0)),
                stats_path=stats_path,
            )
            weights = kimi_k3_nonexpert_weights_iterator(self.manifest_path)
        else:
            stats_path = self.config.get("stats_path") or os.getenv(
                "SGLANG_EXPERT_PACK_STATS_PATH"
            )
            required = (
                "source_path",
                "source_sha256",
                "model_identity_sha256",
                "config_sha256",
            )
            missing = [name for name in required if not self.config.get(name)]
            if missing:
                raise ValueError(
                    "expert_pack loader config is missing: "
                    + ", ".join(sorted(missing))
                )
            if self.source_path is None or not self.source_path.is_file():
                raise FileNotFoundError("DeepSeek source GGUF is missing")
            store = ExpertPackStore(
                self.pack_path,
                manifest_path=self.manifest_path,
                expected_layers=int(hf_config.num_hidden_layers),
                expected_experts=int(hf_config.n_routed_experts),
                expected_top_k=int(hf_config.num_experts_per_tok),
                expected_source_sha256=self.config["source_sha256"],
                expected_model_identity_sha256=self.config["model_identity_sha256"],
                expected_config_sha256=self.config["config_sha256"],
                cache_vram_mib=int(
                    self.config.get(
                        "cache_vram_mib",
                        os.getenv("SGLANG_EXPERT_CACHE_VRAM_MIB", 20 * 1024),
                    )
                ),
                cache_vram_reserve_mib=int(
                    self.config.get("cache_vram_reserve_mib", 3 * 1024)
                ),
                stage_slots=int(
                    self.config.get(
                        "stage_slots", os.getenv("SGLANG_EXPERT_STAGE_SLOTS", 8)
                    )
                ),
                read_splits=int(self.config.get("read_splits", 1)),
                direct_io=bool(self.config.get("direct_io", True)),
                stats_flush_interval=int(self.config.get("stats_flush_interval", 0)),
                stats_path=stats_path,
            )
            weights = deepseek4_nonexpert_weights_iterator(
                self.source_path, int(hf_config.num_hidden_layers)
            )
        quant_config = ExpertPackConfig(store)
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(model_config, self.load_config, quant_config)
            loaded_params = model.load_weights(weights)
            if is_kimi:
                if loaded_params is None:
                    raise RuntimeError(
                        "Kimi-K3 load_weights did not return its parameter coverage"
                    )
                expected_params = {name for name, _ in model.named_parameters()}
                missing_params = sorted(expected_params - set(loaded_params))
                if missing_params:
                    preview = ", ".join(missing_params[:16])
                    raise RuntimeError(
                        "Kimi-K3 GGUF did not initialize all model parameters: "
                        f"missing={len(missing_params)} [{preview}]"
                    )
                logger.info(
                    "Kimi-K3 parameter coverage complete: loaded=%d expected=%d",
                    len(set(loaded_params) & expected_params),
                    len(expected_params),
                )
            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)

        store.initialize_device_cache(target_device)
        if not is_kimi:
            prewarm_started = time.monotonic()
            prewarm_mxfp4_extension()
            activation_input = torch.zeros(
                (1, int(hf_config.moe_intermediate_size)),
                dtype=model_config.dtype,
                device=target_device,
            )
            _clamped_swiglu(activation_input, activation_input, hf_config.swiglu_limit)
            torch.cuda.synchronize(target_device)
            del activation_input
            torch.cuda.empty_cache()
            logger.info(
                "Expert-pack CUDA extension and clamped SwiGLU prewarmed in %.3fs",
                time.monotonic() - prewarm_started,
            )
        dense_bytes = sum(
            value.numel() * value.element_size()
            for value in list(model.parameters()) + list(model.buffers())
        )
        store.stats["dense_bytes"] = dense_bytes
        model.expert_pack_store = store
        logger.info(
            "Loaded verified DeepSeek expert-pack model: source_sha256=%s "
            "pack_sha256=%s dense_bytes=%d resident_experts=0",
            store.header.source_blob_sha256,
            store.pack_sha256,
            dense_bytes,
        )
        return model.eval()
