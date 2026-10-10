from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import (
    DefaultModelLoader,
    get_model_loader,
    post_load_weights,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_model, get_parallel
from sglang.srt.utils import (
    MultiprocessingSerializer,
    dynamic_import,
    get_available_gpu_memory,
    init_custom_process_group,
)
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.external_receiver import (
    WeightUpdateReceiverContext,
    build_weight_update_receiver,
)
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _unsupported_derived_weight_cache_error(
    model: Optional[torch.nn.Module] = None,
) -> Optional[str]:
    """Reject online weight updates that derived-weight caches cannot survive.

    The HPC-Ops bf16xfp32 GEMM caches the fp32 weight split; in-place loader
    writes are invisible to it, so an update would silently keep serving the
    old weights. The check is startup-determined and rank-uniform, so an
    update never proceeds on some workers while rejected on others.
    """
    if model is not None and any(
        getattr(module, "_hc_attn_tf32_parts", None) is not None
        or getattr(module, "_hc_ffn_tf32_parts", None) is not None
        for module in model.modules()
    ):
        return (
            "Online weight updates are not supported while compensated mHC "
            "weight splits are active: captured CUDA graphs retain these derived "
            "weights. Restart with SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 to use "
            "online weight updates."
        )

    if model is not None:
        # Model-owned caches can publish the same rank-uniform constraint
        # without importing individual model implementations in the updater.
        for module in model.modules():
            reason = getattr(module, "_derived_weight_cache_error", None)
            if reason is not None:
                return reason
    from sglang.kernels.ops.gemm.bf16_fp32 import hpc_bf16xfp32_gemm_enabled

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
    # Import paths a request may name as receiver (--weight-update-receivers).
    weight_update_receivers: Optional[List[str]] = None
    _external_receivers: dict = field(default_factory=dict)

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
        receiver=None,
        receiver_init_payload=None,
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

        if receiver is None and receiver_init_payload is not None:
            message = "Failed to initialize custom process group: receiver_init_payload requires receiver."
            logger.error(message)
            return False, message
        if receiver is not None:
            try:
                if group_name in self._external_receivers or (
                    group_name in self._model_update_group
                ):
                    raise ValueError(f"group {group_name!r} already exists")
                self._external_receivers[group_name] = build_weight_update_receiver(
                    receiver,
                    self.weight_update_receivers,
                    WeightUpdateReceiverContext(
                        model=self.get_model(),
                        device=torch.device(self.device),
                        tp_rank=self.tp_rank,
                        tp_size=get_parallel().tp_size,
                        group_name=group_name,
                        master_address=master_address,
                        master_port=master_port,
                        world_size=world_size,
                        rank_offset=rank_offset,
                        init_payload=receiver_init_payload,
                    ),
                )
                logger.info(
                    f"init external weight-update receiver: receiver={receiver}, "
                    f"group_name={group_name}, rank_offset={rank_offset}, "
                    f"rank={rank}, world_size={world_size}"
                )
                return True, "Succeeded to initialize external weight-update receiver."
            except Exception as e:
                message = f"Failed to initialize external weight-update receiver: {e}."
                logger.error(message)
                return False, message

        # A name owned by an external receiver cannot be reused for a torch
        # process group: group_name routes both updates and destroy, so the
        # shadowed group would leak behind the receiver.
        if group_name in self._external_receivers:
            message = (
                f"Failed to initialize custom process group: group {group_name!r} "
                "already exists as an external weight-update receiver."
            )
            logger.error(message)
            return False, message
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
        is_receiver = group_name in self._external_receivers
        try:
            if is_receiver:
                # Forget the receiver first so a failing destroy() is not retried.
                self._external_receivers.pop(group_name).destroy()
                return True, "Succeeded to destroy external weight-update receiver."
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            kind = (
                "external weight-update receiver"
                if is_receiver
                else "custom process group"
            )
            message = f"Failed to destroy {kind}: {e}."
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
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        self._assert_weight_cache_inactive("update_weights_from_disk")
        error = _unsupported_derived_weight_cache_error(self.get_model())
        if error is not None:
            return False, error

        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id, empty_cache=False):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

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

    def receive_weights_from_distributed(
        self: WeightUpdater,
        *,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
        receiver_payload: Optional[dict] = None,
    ):
        """Receive one broadcast without loading it; only the target runner joined the group."""
        external = self._external_receivers.get(group_name)
        if external is not None:
            if names or dtypes or shapes:
                raise ValueError(
                    f"Group {group_name!r} is an external receiver: tensor "
                    "names/dtypes/shapes must be empty; per-round data goes in "
                    "receiver_payload"
                )
            if load_format is not None:
                raise ValueError(
                    f"Group {group_name!r} is an external receiver: load_format "
                    "is not supported on receiver rounds; the receiver writes "
                    "the model itself, so a load format would be silently ignored"
                )
            # A receiver round is an in-place weight mutation, so it takes the
            # same guards as the load paths; on raise the scheduler reports
            # failure and records no weight version.
            self._assert_weight_cache_inactive("update_weights_from_distributed")
            error = _unsupported_derived_weight_cache_error(self.get_model())
            if error is not None:
                raise RuntimeError(error)
            external.receive(receiver_payload)
            return []
        if receiver_payload is not None:
            raise ValueError(
                f"Group {group_name!r} has no external receiver to take receiver_payload"
            )
        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        if load_format == "flattened_bucket":
            return self._receive_bucketed_weights_from_distributed(
                names=names, dtypes=dtypes, shapes=shapes, group_name=group_name
            )

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
        return weights

    def _receive_bucketed_weights_from_distributed(
        self: WeightUpdater, *, names, dtypes, shapes, group_name
    ):
        bucket = FlattenedTensorBucket.empty(names, dtypes, shapes, self.device)
        flattened_tensor = bucket.get_flattened_tensor()
        torch.distributed.broadcast(
            flattened_tensor,
            src=0,
            group=self._model_update_group[group_name],
        )
        return bucket.reconstruct_tensors()

    def begin_weight_update(self: WeightUpdater) -> None:
        DefaultModelLoader.restore_weights_before_loading(
            self.get_model(), torch.device(self.device)
        )

    def end_weight_update(self: WeightUpdater, *, run_post_load: bool) -> None:
        if run_post_load:
            post_load_weights(self.get_model())
        DefaultModelLoader.postprocess_weights(
            self.get_model(), torch.device(self.device)
        )

    def load_weights_from_distributed(
        self: WeightUpdater, named_tensors: List[Tuple[str, torch.Tensor]]
    ) -> Tuple[bool, str]:
        self._assert_weight_cache_inactive("update_weights_from_distributed")
        error = _unsupported_derived_weight_cache_error(self.get_model())
        if error is not None:
            return False, error
        try:
            self.get_model().load_weights(named_tensors)
        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg
        return True, "Succeeded to update parameter online."

    def update_weights_from_tensor(
        self: WeightUpdater,
        named_tensors: List[Tuple[str, Union[torch.Tensor, LocalSerializedTensor]]],
        load_format: Optional[str] = None,
    ):
        error = _unsupported_derived_weight_cache_error(self.get_model())
        if error is not None:
            return False, error

        monkey_patch_torch_reductions()
        self._assert_weight_cache_inactive("update_weights_from_tensor")
        # We need to get device after patch otherwise the device would be wrong
        device_module = torch.get_device_module(self.device)
        infered_device = device_module.current_device()

        # Two input shapes reach this point. The per-tensor formats hand over a
        # list of (name, tensor) pairs, where a tensor may still be a
        # LocalSerializedTensor carrying one CUDA IPC handle per TP rank: open
        # this rank's handle and move the result to this device before loading.
        # A flattened bucket is instead a dict holding one big device tensor and
        # the metadata that says how to slice it; its loader below does that.
        if load_format != "flattened_bucket":
            named_tensors = [
                (
                    name,
                    _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device),
                )
                for name, tensor in named_tensors
            ]

        if load_format == "flattened_bucket":
            self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )
        elif load_format == "direct":
            _model_load_weights_direct(self.get_model(), named_tensors)
        elif load_format in self.custom_weight_loaders:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.get_model(), named_tensors)
        elif load_format is None:
            self.get_model().load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        # Tensors deserialized from CUDA IPC handles alias storage owned by the
        # sender, who may free or reuse it as soon as this call returns. The
        # loads above only enqueue device-to-device copies, so wait for them
        # before handing control back.
        device_module.synchronize()
        return True, "Success"

    def _update_weights_from_flattened_bucket(
        self: WeightUpdater,
        flattened_tensor_bucket_dict,
    ) -> None:
        """Load a flattened bucket, raising if reconstruction or loading fails."""
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

    def update_weights_from_ipc(self: WeightUpdater, recv_req):
        """Update weights from IPC for checkpoint-engine integration."""
        self._assert_weight_cache_inactive("update_weights_from_ipc")
        error = _unsupported_derived_weight_cache_error(self.get_model())
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
