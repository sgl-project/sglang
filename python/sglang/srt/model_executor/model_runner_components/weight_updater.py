from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_exec, get_lora, get_model
from sglang.srt.utils import (
    MultiprocessingSerializer,
    dynamic_import,
    get_available_gpu_memory,
    init_custom_process_group,
)
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _unsupported_derived_weight_cache_error() -> Optional[str]:
    """Reject online weight updates that derived-weight caches cannot survive.

    The HPC-Ops bf16xfp32 GEMM caches the fp32 weight split; in-place loader
    writes are invisible to it, so an update would silently keep serving the
    old weights. The check is startup-determined and rank-uniform, so an
    update never proceeds on some workers while rejected on others.
    """
    from sglang.kernels.ops.attention.dsv4.gemm import hpc_bf16xfp32_gemm_enabled

    if hpc_bf16xfp32_gemm_enabled():
        return (
            "Online weight updates are not supported while the HPC-Ops "
            "bf16xfp32 GEMM optimization is enabled: the cached weight "
            "split would keep serving the old weights."
        )
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class WeightUpdater:
    tp_rank: int
    device: str
    gpu_id: int
    model_config: ModelConfig
    custom_weight_loaders: dict
    get_model: Callable[[], Any]
    update_model_fields: Callable[..., None]
    recapture_cuda_graph: Callable[[], None]
    get_model_runner: Callable[[], ModelRunner]
    _model_update_group: dict = field(default_factory=dict)

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        assert torch.distributed.is_initialized(), (
            "Default torch process group must be initialized"
        )
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            na = NetworkAddress(master_address, master_port)
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def destroy_weights_update_group(self, group_name):
        try:
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            logger.error(message)
            return False, message

    def _assert_weight_cache_inactive(self: WeightUpdater, op: str) -> None:
        """Reject weight mutations while the CUDA IPC weight cache is active:
        param.data is the daemon's master copy shared with every co-attached
        engine, so an in-place update would silently corrupt them all.
        """
        mode = get_model().weight_cache_mode
        if mode != "off":
            raise RuntimeError(
                f"[weight_cache] {op} is not supported while the weight cache is "
                f"active (--weight-cache-mode {mode}): model weights are shared "
                f"with the daemon via CUDA IPC, so mutating them in place would "
                f"corrupt the daemon's master copy and every co-attached engine. "
                f"Restart with --weight-cache-mode off to use this operation."
            )

    def update_weights_from_disk(
        self: WeightUpdater,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
        model_loader_extra_config: Optional[Union[str, dict]] = None,
        rebuild_model: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights from the disk.

        By default the weights are loaded in-place into the existing model. Some
        quantization paths (e.g. MXFP4 MoE on the flashinfer backend) replace the
        raw parameters with derived ones in ``process_weights_after_loading``,
        which makes a second in-place ``load_weights`` impossible. ``rebuild_model``
        instead drops the current model, constructs a fresh one and re-runs the
        full load + post-processing pipeline; CUDA graphs are recaptured since
        they hold pointers into the old weights.
        """
        self._assert_weight_cache_inactive("update_weights_from_disk")
        error = _unsupported_derived_weight_cache_error()
        if error is not None:
            return False, error

        if model_loader_extra_config is None:
            model_loader_extra_config = (
                self.get_model_runner().load_config.model_loader_extra_config
            )

        logger.info(
            f"Update engine weights online from disk begin. "
            f"load_format={load_format} "
            f"model_loader_extra_config={model_loader_extra_config} "
            f"rebuild_model={rebuild_model} "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id, empty_cache=False):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(
            load_format=load_format,
            model_loader_extra_config=model_loader_extra_config,
        )

        if rebuild_model:
            if weight_name_filter is not None:
                return False, "rebuild_model does not support weight_name_filter."
            return self._rebuild_model_from_disk(
                model_path=model_path,
                load_format=load_format,
                load_config=load_config,
            )

        # Only support DefaultModelLoader for now
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.get_model())
            )
            if weight_name_filter is not None:
                iter = (
                    (name, weight) for name, weight in iter if weight_name_filter(name)
                )

            return iter

        def model_load_weights(model, iter):
            loader.load_weights_and_postprocess(model, iter, target_device)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.get_model(), iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                model_load_weights(self.get_model(), iter)
                return False, message

        self.update_model_fields(
            model,
            model_path=model_path,
            load_format=load_format,
            load_config=load_config,
        )

        if recapture_cuda_graph and (
            self.device == "cuda"
            or self.device == "musa"
            or (
                current_platform.is_out_of_tree()
                and current_platform.support_cuda_graph()
            )
        ):
            self.recapture_cuda_graph()

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def _mem_stats(self: WeightUpdater) -> str:
        stats = f"avail mem={get_available_gpu_memory(self.device, self.gpu_id, empty_cache=False):.2f} GB"
        if self.device == "cuda":
            stats += (
                f", allocated={torch.cuda.memory_allocated(self.gpu_id) / 2**30:.2f} GB"
                f", reserved={torch.cuda.memory_reserved(self.gpu_id) / 2**30:.2f} GB"
            )
        return stats

    def _rebuild_model_reject_reason(
        self: WeightUpdater, runner: ModelRunner
    ) -> Optional[str]:
        """Startup-determined configurations whose post-load state is bound to
        the model instance or its weights, so dropping the model would leave a
        dangling or stale collaborator."""
        if runner.spec_algorithm is not None and runner.spec_algorithm.is_speculative():
            return (
                "rebuild_model is not supported with speculative decoding: the "
                "draft model shares parameters with the target model."
            )
        if get_lora().enable_lora:
            return (
                "rebuild_model is not supported with LoRA: the lora manager and "
                "its adapters are bound to the dropped model instance."
            )
        if get_exec().moe.enable_eplb or get_exec().moe.elastic_ep_backend is not None:
            return (
                "rebuild_model is not supported with EPLB / elastic EP: the "
                "expert-location mapping is state that lives in the weights "
                "being replaced."
            )
        return None

    def _rebuild_model_from_disk(
        self: WeightUpdater,
        *,
        model_path: str,
        load_format: str,
        load_config: LoadConfig,
    ) -> tuple[bool, str]:
        runner = self.get_model_runner()
        reason = self._rebuild_model_reject_reason(runner)
        if reason is not None:
            return False, reason

        # The caching allocator only reuses freed blocks on the stream they were
        # allocated on. The model was loaded on the default stream at startup,
        # while the scheduler serves requests from its own stream, so the drop
        # and the reload must both run on the default stream or the new model
        # is allocated from fresh segments on top of the cached old ones.
        device_module = torch.get_device_module(self.device)
        if self.device == "cpu":
            return self._rebuild_model_body(
                runner,
                model_path=model_path,
                load_format=load_format,
                load_config=load_config,
            )
        device_module.synchronize()
        with device_module.stream(device_module.default_stream()):
            result = self._rebuild_model_body(
                runner,
                model_path=model_path,
                load_format=load_format,
                load_config=load_config,
            )
        device_module.default_stream().synchronize()
        return result

    def _rebuild_model_body(
        self: WeightUpdater,
        runner: ModelRunner,
        *,
        model_path: str,
        load_format: str,
        load_config: LoadConfig,
    ) -> tuple[bool, str]:
        from sglang.srt.model_executor.model_runner_components.layer_setup import (
            adjust_hybrid_swa_layer_ids,
            resolve_layer_indices,
        )
        from sglang.srt.model_executor.model_runner_components.load_model_utils import (
            load_kv_cache_scales,
            load_model_with_memory_saver,
            resolve_sliding_window_size,
        )
        from sglang.srt.model_executor.model_runner_components.moe_ep_setup import (
            prepare_moe_topk,
        )
        from sglang.srt.model_executor.runner_utils import (
            set_global_graph_memory_pool,
        )
        from sglang.srt.utils.offloader import get_offloader

        logger.info(f"rebuild_model: before drop. {self._mem_stats()}")

        # A still-pending overlapped startup load must commit before the drop:
        # its prefetch targets the weights about to be freed.
        if runner.startup_weight_load is not None:
            runner.finalize_startup_weight_load()

        # The KV pool keeps its startup sizing, so the rebuilt model must span
        # the same layer range; verified against layer_info after the load.
        old_layer_span = (
            runner.layer_info.start_layer,
            runner.layer_info.end_layer,
        )

        # Captured graphs replay kernels bound to the old weight pointers and
        # must not outlive the model they were captured against. Attention
        # backends and the eager runner hold references into the model and are
        # recreated below.
        runner.decode_cuda_graph_runner = None
        runner.prefill_cuda_graph_runner = None
        runner.eager_runner = None
        runner.attn_backend = None
        runner.decode_attn_backend = None
        runner.decode_attn_backend_group = None
        runner.graph_shared_output = None
        # The shared graph pool's use_count drops to zero when the last old
        # graph is destroyed; re-capture into the same id would trip the
        # allocator's use_count assert, so the next capture mints a fresh pool.
        set_global_graph_memory_pool(None)
        old_model = runner.model
        runner.model = None
        # Graph capture caches submodule lists (attention_layers, moe_layers,
        # ...) on the runner; anything pointing into the old model keeps its
        # weights alive.
        old_module_ids = {id(m) for m in old_model.modules()}
        for name, value in list(vars(runner).items()):
            if isinstance(value, torch.nn.Module) and id(value) in old_module_ids:
                setattr(runner, name, None)
            elif isinstance(value, (list, tuple)) and any(
                id(v) in old_module_ids for v in value
            ):
                setattr(runner, name, type(value)())
        runner.kv_cache_configurator.model = None
        # Release the device storage explicitly so a stray reference to the
        # old model (bound methods, helper objects) cannot keep its weights
        # resident while the new model is being allocated.
        for module in old_model.modules():
            for p in module._parameters.values():
                if p is not None:
                    p.data = p.data.new_empty(0)
            for k, b in module._buffers.items():
                if b is not None:
                    module._buffers[k] = b.new_empty(0)
            for k, v in list(vars(module).items()):
                if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                    setattr(module, k, v.new_empty(0))
        del old_model
        gc.collect()
        if self.device != "cpu":
            torch.get_device_module(self.device).empty_cache()
        logger.info(f"rebuild_model: dropped old model. {self._mem_stats()}")

        try:
            loaded = load_model_with_memory_saver(
                model_config=self.model_config,
                load_config=load_config,
                device=self.device,
                gpu_id=self.gpu_id,
                memory_saver_adapter=runner.memory_saver_adapter,
                is_draft_worker=runner.is_draft_worker,
            )
        except Exception as e:
            # The old model is gone; there is nothing to roll back to.
            logger.exception("rebuild_model: failed to load the new model")
            return False, f"Failed to rebuild model: {e}."

        runner.loader = loaded.loader
        runner.startup_weight_load = loaded.startup_weight_load
        transporter = runner.remote_instance_weight_transporter
        if loaded.remote_instance_weight_info is not None:
            transporter.weight_info = loaded.remote_instance_weight_info
        elif transporter.weight_info is not None:
            # The registered regions point into the dropped model; force the
            # transporter to re-register against the new weights below.
            transporter.weight_info = None
        runner.kv_cache_configurator.model = loaded.model
        self.update_model_fields(
            loaded.model,
            model_path=model_path,
            load_format=load_format,
            load_config=load_config,
        )
        runner.sliding_window_size = resolve_sliding_window_size(
            loaded.model, self.model_config
        )
        runner.prefill_aware_swa = (
            hasattr(loaded.model, "is_prefill_aware_swa")
            and loaded.model.is_prefill_aware_swa()
        )
        runner.dtype = self.model_config.dtype
        logger.info(f"rebuild_model: new model loaded. {self._mem_stats()}")

        if runner.startup_weight_load is not None:
            runner.finalize_startup_weight_load()

        # Mirror the model-dependent steps of ModelRunner.initialize() /
        # load_model(): everything that was derived from the old model object
        # has to be rebuilt against the new one. Managers holding
        # get_model=lambdas (e.g. expert backup client) follow the swap on
        # their own; pool-bound components (kv_index_translator, canary) are
        # untouched.
        if not runner.is_draft_worker:
            get_offloader().post_init()
        runner.maybe_precompile_model_kernels_after_loading()
        load_kv_cache_scales(
            model=loaded.model, kv_cache_dtype=get_model().kv_cache_dtype
        )
        prepare_moe_topk(
            model=loaded.model,
            model_config=runner.model_config,
            moe_ep_size=runner.ps.moe_ep_size,
            moe_ep_rank=runner.ps.moe_ep_rank,
        )
        runner.maybe_init_dwdp()
        transporter.maybe_register_and_publish_weight_info()
        runner.layer_info = resolve_layer_indices(
            model=loaded.model,
            model_config=runner.model_config,
            is_draft_worker=runner.is_draft_worker,
            spec_algorithm=runner.spec_algorithm,
        )
        if (
            runner.layer_info.start_layer,
            runner.layer_info.end_layer,
        ) != old_layer_span:
            return False, (
                "rebuild_model produced a different layer span "
                f"({runner.layer_info.start_layer}..{runner.layer_info.end_layer} "
                f"vs {old_layer_span[0]}..{old_layer_span[1]}): the KV pool was "
                "sized for the original model. Restart the server to serve "
                "this checkpoint."
            )
        adjust_hybrid_swa_layer_ids(
            model_config=runner.model_config,
            start_layer=runner.layer_info.start_layer,
            end_layer=runner.layer_info.end_layer,
            is_hybrid_swa=runner.is_hybrid_swa,
        )
        runner.maybe_apply_post_load_model_transforms()
        runner.maybe_enable_batch_invariant_mode()
        runner.configure_kv_cache_dtype()
        runner.init_routed_experts_capturer()
        runner.init_indexer_capturer()
        runner.init_attention_backends()
        runner.init_cuda_graphs()

        logger.info("Update weights end.")
        return True, "Succeeded to rebuild model weights."

    def update_weights_from_distributed(
        self: WeightUpdater,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
    ):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """
        self._assert_weight_cache_inactive("update_weights_from_distributed")
        error = _unsupported_derived_weight_cache_error()
        if error is not None:
            return False, error

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        if load_format == "flattened_bucket":
            return self._update_bucketed_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.get_model().load_weights(weights)
            return True, "Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def _update_bucketed_weights_from_distributed(
        self: WeightUpdater, names, dtypes, shapes, group_name
    ):
        try:
            named_tensors = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                named_tensors.append(
                    (
                        name,
                        torch.empty(shape, dtype=target_dtype, device=self.device),
                    )
                )
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor()
            torch.distributed.broadcast(
                flattened_tensor,
                src=0,
                group=self._model_update_group[group_name],
            )
            reconstructed_tensors = bucket.reconstruct_tensors()
            self.get_model().load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self: WeightUpdater,
        named_tensors: List[Tuple[str, Union[torch.Tensor, LocalSerializedTensor]]],
        load_format: Optional[str] = None,
    ):
        error = _unsupported_derived_weight_cache_error()
        if error is not None:
            return False, error

        monkey_patch_torch_reductions()
        self._assert_weight_cache_inactive("update_weights_from_tensor")
        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        device_module = torch.get_device_module(self.device)
        infered_device = device_module.current_device()

        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.get_model(), named_tensors)
        elif load_format in self.custom_weight_loaders:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.get_model(), named_tensors)
        elif load_format is None:
            self.get_model().load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def _update_weights_from_flattened_bucket(
        self: WeightUpdater,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates"""
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        self.get_model().load_weights(reconstructed_tensors)

        return True, "Success"

    def update_weights_from_ipc(self: WeightUpdater, recv_req):
        """Update weights from IPC for checkpoint-engine integration."""
        self._assert_weight_cache_inactive("update_weights_from_ipc")
        error = _unsupported_derived_weight_cache_error()
        if error is not None:
            return False, error

        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self.get_model_runner())
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device):
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
