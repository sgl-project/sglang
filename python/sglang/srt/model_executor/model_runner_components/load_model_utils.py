from __future__ import annotations

import datetime
import logging
import os
import socket
import threading
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch
import torch.distributed as dist

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.model_loader.loader import get_model_loader
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    trigger_init_weights_send_group_for_remote_instance_request,
)
from sglang.srt.utils.common import is_npu
from sglang.srt.utils.network import NetworkAddress

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

_is_npu = is_npu()


UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # leave more time for post data processing


class LoadedModel(msgspec.Struct, frozen=True, kw_only=True):
    loader: Any
    model: Any
    remote_instance_weight_info: Optional[Any]
    startup_weight_load: Optional[Any] = None


def maybe_downgrade_dtype_for_legacy_gpu(
    *, server_args: ServerArgs, model_config: ModelConfig
) -> None:
    if torch.cuda.get_device_capability()[0] < 8:
        logger.info(
            "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
        )
        from sglang.srt.arg_groups.overrides import declare_load_time_override

        declare_load_time_override(
            "ModelRunner._sm80_dtype_fallback", {"dtype": "float16"}
        )
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


def report_online_quantization(*, model, server_args: ServerArgs) -> None:
    # TODO: Make sure all models have `quant_config` attribute, and all online quantization methods register which layers they actually quantize.
    quantized_layers = getattr(
        getattr(model, "quant_config", None), "quantized_layers", None
    )
    if (
        server_args.quantization is not None
        and isinstance(quantized_layers, tuple)
        and len(quantized_layers) == 2
    ):
        layer_types, quantized_layers_count = quantized_layers
        logger.info(
            f"Online {server_args.quantization} quantization: quantized {quantized_layers_count} layers of types: {layer_types}"
        )


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


def build_load_config(
    *,
    server_args: ServerArgs,
    tp_rank: int,
    remote_instance_weight_transporter_engine: Any,
    remote_instance_weight_transporter_session_id: str,
    draft_model_idx: Optional[int],
) -> LoadConfig:
    from sglang.srt.configs.modelopt_config import ModelOptConfig

    modelopt_config = ModelOptConfig(
        quant=server_args.modelopt_quant,
        checkpoint_restore_path=server_args.modelopt_checkpoint_restore_path,
        checkpoint_save_path=server_args.modelopt_checkpoint_save_path,
        export_path=server_args.modelopt_export_path,
        quantize_and_serve=server_args.quantize_and_serve,
    )

    return LoadConfig(
        load_format=server_args.load_format,
        download_dir=server_args.download_dir,
        model_loader_extra_config=server_args.model_loader_extra_config,
        tp_rank=tp_rank,
        remote_instance_weight_loader_seed_instance_ip=server_args.remote_instance_weight_loader_seed_instance_ip,
        remote_instance_weight_loader_seed_instance_service_port=server_args.remote_instance_weight_loader_seed_instance_service_port,
        remote_instance_weight_loader_send_weights_group_ports=server_args.remote_instance_weight_loader_send_weights_group_ports,
        remote_instance_weight_loader_backend=server_args.remote_instance_weight_loader_backend,
        remote_instance_weight_loader_transfer_engine=remote_instance_weight_transporter_engine,
        remote_instance_weight_loader_transfer_engine_session_id=remote_instance_weight_transporter_session_id,
        modelexpress_url=server_args.modelexpress_url,
        modelexpress_transport=server_args.modelexpress_transport,
        modelopt_config=modelopt_config,
        rl_quant_profile=server_args.rl_quant_profile,
        draft_model_idx=draft_model_idx,
    )


def load_model_with_memory_saver(
    *,
    server_args: ServerArgs,
    model_config: ModelConfig,
    load_config: LoadConfig,
    device: str,
    gpu_id: int,
    memory_saver_adapter: Any,
    is_draft_worker: bool,
) -> LoadedModel:
    # Remove monkey_patch when linear.py quant remove dependencies with vllm
    monkey_patch_vllm_parallel_state()

    enable_cpu_backup = server_args.enable_weights_cpu_backup or (
        is_draft_worker and server_args.enable_draft_weights_cpu_backup
    )
    remote_instance_weight_info = None
    startup_weight_load = None
    with memory_saver_adapter.region(
        GPU_MEMORY_TYPE_WEIGHTS,
        enable_cpu_backup=enable_cpu_backup,
    ):
        loader = get_model_loader(
            load_config=load_config,
            model_config=model_config,
        )
        device_config = DeviceConfig(device, gpu_id)
        if server_args.startup_weight_load_mode == "overlap":
            from sglang.srt.model_executor.model_runner_components.startup_weight_load import (
                StartupWeightLoadManager,
                StartupWeightLoadOptions,
            )

            startup_weight_load = StartupWeightLoadManager.create(
                loader=loader,
                model_config=model_config,
                load_config=load_config,
                device_config=device_config,
                options=StartupWeightLoadOptions.from_server_args(
                    server_args=server_args,
                    is_draft_worker=is_draft_worker,
                ),
            )
            model = startup_weight_load.prepare()
        else:
            model = loader.load_model(
                model_config=model_config,
                device_config=device_config,
            )
        if hasattr(loader, "remote_instance_transfer_engine_weight_info"):
            remote_instance_weight_info = (
                loader.remote_instance_transfer_engine_weight_info
            )
    # Cache needs to be cleared after loading model weights (in the loader.load_model function).
    # To avoid conflict with memory_saver_adapter.region, empty_cache operation is now moved here.
    if _is_npu:
        torch.npu.empty_cache()
    monkey_patch_vllm_parallel_state(reverse=True)

    return LoadedModel(
        loader=loader,
        model=model,
        remote_instance_weight_info=remote_instance_weight_info,
        startup_weight_load=startup_weight_load,
    )


def dist_barrier_after_load(
    *,
    elastic_ep_backend: Optional[str],
    tp_rank: int,
    is_ep_scale_joiner: bool = False,
) -> None:
    if elastic_ep_backend == "mooncake":
        # Mooncake does not support `monitored_barrier`
        if not is_ep_scale_joiner:
            dist.barrier(group=get_tp_group().cpu_group)
    else:
        # Handle the case where some ranks do not finish loading.
        try:
            dist.monitored_barrier(
                group=get_tp_group().cpu_group,
                timeout=datetime.timedelta(seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S),
                wait_all_ranks=True,
            )
        except RuntimeError:
            raise ValueError(
                f"TP rank {tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
            ) from None
