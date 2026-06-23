from __future__ import annotations

import logging
import os
import socket
import threading
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.load_config import LoadFormat
from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    trigger_init_weights_send_group_for_remote_instance_request,
)
from sglang.srt.utils.network import NetworkAddress

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


def maybe_downgrade_dtype_for_legacy_gpu(
    *, server_args: ServerArgs, model_config: ModelConfig
) -> None:
    if torch.cuda.get_device_capability()[0] < 8:
        logger.info(
            "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
        )
        server_args.dtype = "float16"
        model_config.dtype = torch.float16
        if torch.cuda.get_device_capability()[1] < 5:
            raise RuntimeError("SGLang only supports sm75 and above.")


def maybe_trigger_remote_instance_nccl_send_group(
    *, server_args: ServerArgs, tp_rank: int
) -> None:
    if (
        server_args.load_format == LoadFormat.REMOTE_INSTANCE
        and server_args.remote_instance_weight_loader_backend
        == RemoteInstanceWeightLoaderBackend.NCCL
    ):
        if tp_rank == 0:
            instance_ip = NetworkAddress.resolve_host(socket.gethostname())
            t = threading.Thread(
                target=trigger_init_weights_send_group_for_remote_instance_request,
                args=(
                    server_args.remote_instance_weight_loader_seed_instance_ip,
                    server_args.remote_instance_weight_loader_seed_instance_service_port,
                    server_args.remote_instance_weight_loader_send_weights_group_ports,
                    instance_ip,
                ),
            )
            t.start()


def load_kv_cache_scales(*, model, server_args: ServerArgs) -> None:
    if server_args.kv_cache_dtype == "fp8_e4m3":
        if server_args.quantization_param_path is not None:
            if callable(getattr(model, "load_kv_cache_scales", None)):
                model.load_kv_cache_scales(server_args.quantization_param_path)
                logger.info(
                    "Loaded KV cache scaling factors from %s",
                    server_args.quantization_param_path,
                )
            else:
                raise RuntimeError(
                    "Using FP8 KV cache and scaling factors provided but "
                    "model %s does not support loading scaling factors.",
                    model.__class__,
                )
        else:
            logger.warning(
                "Using FP8 KV cache but no scaling factors "
                "provided. Defaulting to scaling factors of 1.0. "
                "This may lead to less accurate results!"
            )


def resolve_sliding_window_size(model, model_config: ModelConfig) -> Optional[int]:
    # Parse other args
    sliding_window_size = None
    if hasattr(model, "get_attention_sliding_window_size"):
        sliding_window_size = model.get_attention_sliding_window_size()
    elif model_config.is_hybrid_swa and model_config.sliding_window_size is not None:
        # sliding window field in model config may have different meaning for different kinds of models (e.g., dllm), here we only consider the sliding window in SWA model
        sliding_window_size = model_config.sliding_window_size
    elif model_config.attention_chunk_size is not None:
        sliding_window_size = model_config.attention_chunk_size
        logger.info(
            f"Setting sliding_window_size to be attention_chunk_size: {sliding_window_size}"
        )
    return sliding_window_size


def maybe_register_debug_tensor_dump_hook(
    *,
    model,
    server_args: ServerArgs,
    spec_algorithm: SpeculativeAlgorithm,
    is_draft_worker: bool,
    tp_size: int,
    tp_rank: int,
    pp_rank: int,
) -> None:
    if server_args.debug_tensor_dump_output_folder is not None:
        dump_folder = server_args.debug_tensor_dump_output_folder
        if spec_algorithm.is_eagle():
            role = "draft" if is_draft_worker else "target"
            dump_folder = os.path.join(dump_folder, role)
        register_forward_hook_for_model(
            model,
            dump_folder,
            server_args.debug_tensor_dump_layers,
            tp_size,
            tp_rank,
            pp_rank,
        )
