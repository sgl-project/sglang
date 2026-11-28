# SPDX-License-Identifier: Apache-2.0
"""Build a LoadSpec from the model config and the runtime context."""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING, Optional

from sglang.srt.distributed.utils import get_pp_layer_partition
from sglang.srt.layers import dp_attention
from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import use_cutedsl_moe_swap_ab
from sglang.srt.layers.moe.utils import MoeA2ABackend, resolve_moe_runner_backend
from sglang.srt.layers.quantization import get_quantization_config
from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
from sglang.srt.layers.quantization.fp8_utils import get_fp8_gemm_runner_backend
from sglang.srt.layers.quantization.unquant import get_bf16_gemm_backend
from sglang.srt.model_loader.loader import PreshardedModelLoader
from sglang.srt.model_loader.utils import get_model_architecture
from sglang.srt.runtime_context import get_exec, get_model, get_parallel

from .protocol import compute_env_stamp, hash_quant_config, resolve_cache_quantization
from .schema import LoadSpec

if TYPE_CHECKING:
    import torch
    from transformers import PretrainedConfig

    from sglang.srt.configs.model_config import ModelConfig


def _hf_config_hash(hf_config: PretrainedConfig) -> str:
    # Composite HF models can have their own locator on each nested config.
    def omit_local_paths(fields):
        fields.pop("_name_or_path", None)
        fields.pop("name_or_path", None)
        return fields

    fields = json.loads(
        hf_config.to_json_string(use_diff=False), object_hook=omit_local_paths
    )
    encoded = json.dumps(fields, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def path_checkpoint_id() -> str:
    """The cache daemon's checkpoint ID: the requested model path and revision.

    ModelConfig may swap in a base model (Kev adapters), so read the request.
    """
    path, revision = get_model().model_path, get_model().revision
    return f"{path}@{revision}" if revision else path


def _quant_files_hash(model_config: ModelConfig, quant_method: str) -> str:
    """Hash the quantization config files next to a local checkpoint.

    A ModelOpt checkpoint keeps its per-layer recipe in hf_quant_config.json, which
    hf_config.quantization_config does not hold. The names are the ones the loader
    looks for (ModelConfig and get_quant_config): that file, and files ending in one
    of the method's own config filenames. A file the loader then does not read still
    counts, which can refuse a pair and never matches one. A checkpoint that is not
    a local directory has none here.
    """
    folder = model_config.model_path
    if not os.path.isdir(folder):
        return ""
    suffixes = ()
    if quant_method:
        try:
            suffixes = tuple(
                get_quantization_config(quant_method).get_config_filenames()
            )
        except ValueError:  # not a registered method: the loader refuses it
            pass
    digests = {}
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not name.endswith(".json") or not os.path.isfile(path):
            continue
        if name == "hf_quant_config.json" or (suffixes and name.endswith(suffixes)):
            with open(path, "rb") as f:
                digests[name] = hashlib.sha256(f.read()).hexdigest()
    if not digests:
        return ""
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()


def _model_class(model_config: ModelConfig) -> str:
    """The class get_model_architecture builds the model from.

    It can import a model's remote code; the loader does the same when it builds
    the model.
    """
    if not getattr(model_config.hf_config, "architectures", None):
        return ""
    model_cls, _ = get_model_architecture(model_config)
    return f"{model_cls.__module__}.{model_cls.__qualname__}"


def build_load_spec(
    model_config: ModelConfig,
    *,
    checkpoint_id: str,
    device: torch.device,
    draft_model_idx: Optional[int] = None,
) -> LoadSpec:
    """Describe the weights this rank is about to load.

    Call it after the parallel groups and the GEMM backends are initialized, and
    before the model is built: construction and loading rewrite the HF and
    quantization configs.
    """
    quant_method, quant_config = resolve_cache_quantization(model_config)
    architectures = model_config.hf_config.architectures
    parallel = get_parallel()
    execution = get_exec()
    pp_partition = get_pp_layer_partition(parallel.pp_size)
    # From the published config: a dense model's process never sets the MoE flags.
    moe_runner = resolve_moe_runner_backend(execution.moe.moe_runner_backend)
    a2a_backend = MoeA2ABackend(execution.moe.moe_a2a_backend)
    swap_ab = None
    if moe_runner.is_flashinfer_cutedsl() and (
        a2a_backend.is_none() or a2a_backend.is_flashinfer()
    ):
        swap_ab = use_cutedsl_moe_swap_ab(device)

    return LoadSpec(
        # Model config.
        checkpoint_id=checkpoint_id,
        model_arch=architectures[0] if architectures else "",
        model_class=_model_class(model_config),
        hf_config_hash=_hf_config_hash(model_config.hf_config),
        dtype=str(model_config.dtype),
        role="draft" if model_config.is_draft_model else "target",
        draft_model_idx=draft_model_idx,
        # Quantization config.
        quant_method=quant_method,
        quant_config_hash=hash_quant_config(quant_config),
        quant_files_hash=_quant_files_hash(model_config, quant_method),
        # Parallel config: the layout.
        tp_size=parallel.tp_size,
        pp_size=parallel.pp_size,
        dp_size=parallel.dp_size,
        ep_size=parallel.moe_ep_size,
        # MoE-DP can alias a wider attention-CP communicator; compare its configured width.
        moe_dp_size=parallel.moe_dp_size,
        attn_dp_size=parallel.attn_dp_size,
        attn_cp_size=parallel.attn_cp_size,
        moe_cp_size=dp_attention.get_moe_cp_size(),
        moe_dense_tp_size=parallel.moe_dense_tp_size,
        enable_dp_lm_head=parallel.enable_dp_lm_head,
        pp_layer_partition=(
            None if pp_partition is None else ",".join(map(str, pp_partition))
        ),
        # Parallel config: this rank's shard.
        tp_rank=parallel.tp_rank,
        pp_rank=parallel.pp_rank,
        moe_dp_rank=parallel.moe_dp_rank,
        moe_ep_rank=parallel.moe_ep_rank,
        # Expert placement config.
        ep_num_redundant_experts=execution.moe.ep_num_redundant_experts,
        enable_eplb=execution.moe.enable_eplb,
        init_expert_location=PreshardedModelLoader._normalize_init_expert_location(
            execution.moe.init_expert_location
        ),
        # Kernel config.
        moe_a2a_backend=execution.moe.moe_a2a_backend,
        moe_runner_backend=moe_runner.value,
        fp4_gemm_runner_backend=get_fp4_gemm_runner_backend().value,
        fp8_gemm_runner_backend=get_fp8_gemm_runner_backend().value,
        bf16_gemm_backend=get_bf16_gemm_backend().value,
        flashinfer_cutedsl_swap_ab=swap_ab,
        enable_fp32_lm_head=execution.features.enable_fp32_lm_head,
        # Environment.
        **compute_env_stamp(),
    )
