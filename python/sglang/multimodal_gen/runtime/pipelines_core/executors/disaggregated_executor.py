"""
Pipeline executor for disaggregated execution with async communication support.
"""

import traceback
from collections import deque
from enum import Enum, auto
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.distributed import Work

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.dist_utils import get_disagg_communicator
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
    Timer,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DenoisingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class StageDisaggregationRole(Enum):
    NONE_DENOISE = auto()
    DENOISE = auto()
    COMMON = auto()


def get_stage_disagg_role(stage: PipelineStage):
    if isinstance(stage, DenoisingStage) or isinstance(stage, TimestepPreparationStage):
        return StageDisaggregationRole.DENOISE
    else:
        return StageDisaggregationRole.NONE_DENOISE


class PendingTransfer:
    """Tracks an in-flight async communication."""

    def __init__(self, batch_id: str, works: List[Optional[Work]], tensors: dict):
        self.batch_id = batch_id
        self.works = works  # List of Work handles
        self.tensors = tensors  # Keep references to prevent GC


class DisaggregatedExecutor(PipelineExecutor):
    """
    Executor that handles the disaggregated pipeline flow with async communication:
    Non-DiT (Encode) -> DiT (Denoise) -> Non-DiT (Decode)

    Supports async communication to overlap computation and communication.
    """

    def __init__(self, server_args: ServerArgs):
        super().__init__(server_args)
        self.comm = get_disagg_communicator()
        assert (
            self.comm is not None
        ), "DisaggregatedExecutor requires an initialized DisaggCommunicator"

        # Async communication queues
        self.pending_sends = deque()  # Queue of PendingTransfer
        self.pending_recvs = deque()  # Queue of PendingTransfer

        # Max in-flight transfers to prevent memory explosion
        self.max_pending_transfers = 2

        logger.info(
            f"DisaggregatedExecutor initialized on rank {dist.get_rank()} "
            f"with role {self.comm.role}, max_pending={self.max_pending_transfers}"
        )

    def execute(
        self,
        stages: List["PipelineStage"],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        """
        Executes stages with async communication support.

        Flow:
        1. Non-DiT: Encode -> Async Send to DiT
        2. DiT: Async Recv -> Denoise -> Async Send to Non-DiT
        3. Non-DiT: Async Recv -> Decode -> Send final result to DiT master
        4. DiT Master: Recv final result -> Return to client
        """

        with self.profile_execution(batch):
            # Split stages by role
            denoise_start_idx = -1
            denoise_end_idx = -1

            for i, stage in enumerate(stages):
                if get_stage_disagg_role(stage) == StageDisaggregationRole.DENOISE:
                    if denoise_start_idx == -1:
                        denoise_start_idx = i
                    denoise_end_idx = i

            if denoise_start_idx == -1:
                logger.warning("No denoising stage found! Running locally.")
                return self._run_local(stages, batch)

            pre_denoise_stages = stages[:denoise_start_idx]
            denoise_stages = stages[denoise_start_idx : denoise_end_idx + 1]
            post_denoise_stages = stages[denoise_end_idx + 1 :]

            # --- PHASE 1: Encoding (Non-DiT) ---
            if self.comm.is_non_dit_rank():

                # Run encoding stages
                for stage in pre_denoise_stages:
                    with Timer(stage.__class__.__name__):
                        batch = stage(batch, server_args)

                # Async send to DiT (non-blocking)
                works = self._async_send_batch_to_dit(batch)

                # Track pending send
                if works:
                    pending = PendingTransfer(
                        batch_id=batch.request_id or "batch",
                        works=works,
                        tensors={},  # Tensors are kept alive by batch object
                    )
                    self.pending_sends.append(pending)

                # Flow control: if queue is full, wait for oldest send
                if len(self.pending_sends) >= self.max_pending_transfers:
                    oldest = self.pending_sends.popleft()
                    self.comm.wait_all_works(oldest.works)
            # --- PHASE 2: Denoising (DiT) ---
            if self.comm.is_dit_rank():

                # Async receive from Non-DiT
                batch, recv_works = self._async_recv_batch_from_non_dit(batch)

                # Wait for receive to complete before denoising
                if recv_works:
                    self.comm.wait_all_works(recv_works)

                # Broadcast to all DiT workers (if SP/TP)
                self._broadcast_batch_in_dit_group(batch)

                try:
                    # Run denoising stages
                    for stage in denoise_stages:
                        with Timer(stage.__class__.__name__):
                            batch = stage(batch, server_args)
                except Exception as e:
                    print(traceback.format_exc())

                # Only DiT master sends result back
                if dist.get_rank() == self.comm.dit_master_rank:
                    works = self._async_send_batch_to_non_dit(batch)
                    if works:
                        # Wait for send to complete (simple approach for now)
                        self.comm.wait_all_works(works)
                else:
                    # DiT workers are done
                    return OutputBatch()

            # --- PHASE 3: Decoding (Non-DiT) ---
            if self.comm.is_non_dit_rank():
                # Wait for any pending sends from Phase 1
                while self.pending_sends:
                    pending = self.pending_sends.popleft()
                    self.comm.wait_all_works(pending.works)

                # Async receive from DiT
                batch, recv_works = self._async_recv_batch_from_dit(batch)

                # Wait for receive
                if recv_works:
                    self.comm.wait_all_works(recv_works)
                # Run decoding stages
                for stage in post_denoise_stages:
                    with Timer(stage.__class__.__name__):
                        batch = stage(batch, server_args)

                # Convert to OutputBatch
                output_batch = OutputBatch(
                    output=batch.output,
                    trajectory_timesteps=batch.trajectory_timesteps,
                    trajectory_latents=batch.trajectory_latents,
                    trajectory_decoded=getattr(batch, "trajectory_decoded", None),
                    timings=batch.timings,
                    error=None,
                )

                # Send final result to DiT master
                self._send_final_result_to_dit_master(output_batch)

            # --- PHASE 4: DiT Master receives final result ---
            if dist.get_rank() == self.comm.dit_master_rank:
                output_batch = OutputBatch()
                output_batch = self._recv_final_result_from_non_dit(output_batch)
                return output_batch

            # Non-master ranks return empty batch
            return OutputBatch()

    def _run_local(self, stages, batch):
        """Fallback: run all stages locally."""
        for stage in stages:
            batch = stage(batch, self.server_args)
        return batch

    def _broadcast_batch_in_dit_group(self, batch: Req):
        """Broadcast batch data within DiT group after master receives from Non-DiT."""
        # This ensures all DiT workers have the input for SP/TP parallel processing
        from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj

        # Step 1: Broadcast metadata info so all workers know what tensors to expect
        if dist.get_rank() == self.comm.dit_master_rank:
            metadata = getattr(batch, "_recv_metadata", {})
            tensor_infos = getattr(batch, "_recv_tensor_infos", {})
            list_tensor_infos = getattr(batch, "_recv_list_tensor_infos", {})
        else:
            metadata = None
            tensor_infos = None
            list_tensor_infos = None

        # Broadcast metadata using PyObj (works across all ranks)
        dit_group = self.comm.get_my_group()
        if dit_group is None or dist.get_world_size(group=dit_group) == 1:
            # Single rank in group, no need to broadcast
            return

        # Broadcast metadata using CUDA tensors (NCCL backend requires CUDA)
        world_rank = dist.get_rank()
        backend = dist.get_backend(group=dit_group)
        use_cpu = backend != "nccl"  # Only use CPU if not NCCL

        logger.debug(
            f"[DiT Rank {dist.get_rank()}] Broadcasting metadata with backend={backend}, use_cpu={use_cpu}"
        )

        metadata_pkg = broadcast_pyobj(
            (metadata, tensor_infos, list_tensor_infos),
            rank=world_rank,  # current rank (global)
            dist_group=dit_group,  # dit group
            src=self.comm.dit_master_rank,  # source rank (global)
            force_cpu_device=use_cpu,  # Use CUDA for NCCL backend
        )

        if dist.get_rank() != self.comm.dit_master_rank:
            # Non-master workers: unpack metadata and create tensors
            metadata, tensor_infos, list_tensor_infos = metadata_pkg

            # Update batch metadata
            for k, v in metadata.items():
                setattr(batch, k, v)

            # Create empty tensors that will be filled by broadcast
            device = get_local_torch_device()

            for name, (shape, dtype) in tensor_infos.items():
                tensor = torch.empty(shape, dtype=dtype, device=device)
                if not tensor.is_cuda and dist.get_backend() == "nccl":
                    # Force to CUDA if NCCL backend
                    tensor = tensor.cuda()
                setattr(batch, name, tensor)

            # Create lists of empty tensors
            for name, list_info in list_tensor_infos.items():
                list_length, tensor_indices, shapes_dtypes = list_info

                actual_tensors = []
                for idx, (shape, dtype) in enumerate(shapes_dtypes):
                    tensor = torch.empty(shape, dtype=dtype, device=device)
                    if not tensor.is_cuda and dist.get_backend() == "nccl":
                        tensor = tensor.cuda()
                    actual_tensors.append(tensor)

                # Reconstruct list with None placeholders
                result_list = [None] * list_length
                for idx, tensor in zip(tensor_indices, actual_tensors):
                    result_list[idx] = tensor

                setattr(batch, name, result_list)

        # Step 2: Broadcast all tensors
        # Now all workers have the tensor objects, broadcast the data

        if hasattr(batch, "__dict__"):
            for k, v in batch.__dict__.items():
                if k.startswith("_recv_"):
                    continue  # Skip internal metadata

                if isinstance(v, torch.Tensor):
                    # Ensure tensor is on CUDA before broadcast (NCCL requires CUDA)
                    if not v.is_cuda:
                        v = v.cuda()
                        setattr(batch, k, v)

                    # Double check before broadcast
                    if not v.is_cuda:
                        raise RuntimeError(
                            f"[Rank {dist.get_rank()}] Tensor {k} is still on CPU after cuda() call: {v.device}"
                        )

                    # Broadcast single tensor
                    try:
                        self.comm.broadcast_in_group(v)
                    except Exception as e:
                        logger.error(
                            f"[Rank {dist.get_rank()}] Failed to broadcast tensor {k}: device={v.device}, error={e}"
                        )
                        raise

                elif isinstance(v, list) and v and len(v) > 0:
                    # Broadcast list of tensors
                    for idx, item in enumerate(v):
                        if isinstance(item, torch.Tensor):
                            # Ensure tensor is on CUDA
                            if not item.is_cuda:
                                item = item.cuda()
                                v[idx] = item

                            # Double check
                            if not item.is_cuda:
                                raise RuntimeError(
                                    f"[Rank {dist.get_rank()}] List tensor {k}[{idx}] is still on CPU after cuda() call"
                                )

                            try:
                                self.comm.broadcast_in_group(item)
                            except Exception as e:
                                logger.error(
                                    f"[Rank {dist.get_rank()}] Failed to broadcast list tensor {k}[{idx}]: device={item.device}, error={e}"
                                )
                                raise

    # --- Async Communication Methods ---

    def _async_send_batch_to_dit(self, batch: Req) -> List[Optional[Work]]:
        """
        Async send batch from Non-DiT to DiT.
        Uses batched P2P operations to avoid serialization.
        Returns list of Work handles.
        """
        import io
        import pickle

        tensors_to_send = {}
        tensor_infos = {}
        list_tensor_infos = {}

        # Extract tensors from batch
        if hasattr(batch, "__dict__"):
            for k, v in batch.__dict__.items():
                if isinstance(v, torch.Tensor):
                    tensors_to_send[k] = v
                    tensor_infos[k] = (v.shape, v.dtype)
                elif isinstance(v, list) and v and len(v) > 0:
                    tensor_indices = []
                    actual_tensors = []
                    shapes_dtypes = []

                    for idx, t in enumerate(v):
                        if isinstance(t, torch.Tensor):
                            tensor_indices.append(idx)
                            actual_tensors.append(t)
                            shapes_dtypes.append((t.shape, t.dtype))

                    if actual_tensors:
                        tensors_to_send[k] = actual_tensors
                        list_tensor_infos[k] = (len(v), tensor_indices, shapes_dtypes)

        # Metadata
        metadata = {
            k: v
            for k, v in batch.__dict__.items()
            if k not in tensor_infos and k not in list_tensor_infos
        }

        # Serialize metadata
        buffer = io.BytesIO()
        pickle.dump((metadata, tensor_infos, list_tensor_infos), buffer)
        meta_bytes = torch.tensor(
            bytearray(buffer.getvalue()), dtype=torch.uint8, device="cpu"
        )

        if dist.get_backend() == "nccl":
            meta_bytes = meta_bytes.cuda()

        # Prepare metadata size tensor
        size_tensor = torch.tensor(
            [meta_bytes.numel()], dtype=torch.long, device=meta_bytes.device
        )

        # Collect all tensors to send in a batch
        all_tensors = [size_tensor, meta_bytes]

        # Add single tensors
        for name in tensor_infos.keys():
            tensor = tensors_to_send[name]
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            all_tensors.append(tensor)

        # Add list tensors
        for name in list_tensor_infos.keys():
            tensor_list = tensors_to_send[name]
            for tensor in tensor_list:
                if not tensor.is_cuda:
                    tensor = tensor.cuda()
                all_tensors.append(tensor)

        # Batch send all tensors at once
        works = self.comm.batch_isend_to_dit(all_tensors)

        return works

    def _async_recv_batch_from_non_dit(
        self, batch: Req
    ) -> tuple[Req, List[Optional[Work]]]:
        """
        Async receive batch from Non-DiT at DiT group.
        Uses batched P2P operations to avoid serialization.
        Master receives, then data is broadcasted in _broadcast_batch_in_dit_group.
        Returns (batch, list of Work handles).
        """
        import pickle

        works = []

        # Only DiT master receives metadata and data
        # Other workers will get data via broadcast
        if dist.get_rank() == self.comm.dit_master_rank:
            # Step 1: Receive size and metadata in a batch
            meta_shapes_dtypes = [
                (torch.Size([1]), torch.long),  # size tensor
            ]
            meta_tensors, meta_works = self.comm.batch_irecv_from_non_dit(
                meta_shapes_dtypes
            )

            # Wait for size to arrive
            if meta_works:
                for work in meta_works:
                    work.wait()

            size_tensor = meta_tensors[0]
            size = size_tensor.item()

            # Step 2: Receive metadata bytes
            metadata_shapes_dtypes = [(torch.Size([size]), torch.uint8)]
            metadata_tensors, metadata_works = self.comm.batch_irecv_from_non_dit(
                metadata_shapes_dtypes
            )

            # Wait for metadata to arrive
            if metadata_works:
                for work in metadata_works:
                    work.wait()

            meta_bytes = metadata_tensors[0]
            meta_data_cpu = meta_bytes.cpu().numpy().tobytes()
            metadata, tensor_infos, list_tensor_infos = pickle.loads(meta_data_cpu)

            # Store metadata for broadcasting
            batch._recv_metadata = metadata
            batch._recv_tensor_infos = tensor_infos
            batch._recv_list_tensor_infos = list_tensor_infos

            # Update batch metadata
            for k, v in metadata.items():
                setattr(batch, k, v)

            # Step 3: Prepare to receive all data tensors in a batch
            all_shapes_dtypes = []
            tensor_names = []

            # Collect single tensors
            for name, (shape, dtype) in tensor_infos.items():
                all_shapes_dtypes.append((shape, dtype))
                tensor_names.append(("single", name))

            # Collect list tensors
            for name, list_info in list_tensor_infos.items():
                list_length, tensor_indices, shapes_dtypes = list_info
                for idx, (shape, dtype) in enumerate(shapes_dtypes):
                    all_shapes_dtypes.append((shape, dtype))
                    tensor_names.append(
                        ("list", name, idx, list_length, tensor_indices)
                    )

            # Batch receive all data tensors
            if all_shapes_dtypes:
                all_tensors, data_works = self.comm.batch_irecv_from_non_dit(
                    all_shapes_dtypes
                )
                works.extend(data_works)

                # Assign tensors back to batch
                tensor_idx = 0
                list_tensors = {}  # Temporary storage for list tensors

                for info in tensor_names:
                    tensor = all_tensors[tensor_idx]

                    if info[0] == "single":
                        setattr(batch, info[1], tensor)
                    else:  # 'list'
                        name, idx, list_length, tensor_indices = (
                            info[1],
                            info[2],
                            info[3],
                            info[4],
                        )
                        if name not in list_tensors:
                            list_tensors[name] = {
                                "length": list_length,
                                "indices": tensor_indices,
                                "tensors": [],
                            }
                        list_tensors[name]["tensors"].append(tensor)

                    tensor_idx += 1

                # Reconstruct lists
                for name, list_data in list_tensors.items():
                    result_list = [None] * list_data["length"]
                    for idx, tensor in zip(list_data["indices"], list_data["tensors"]):
                        result_list[idx] = tensor
                    setattr(batch, name, result_list)

        # Non-master workers will get data via broadcast_batch_in_dit_group

        return batch, works

    def _async_send_batch_to_non_dit(self, batch: Req) -> List[Optional[Work]]:
        """
        Async send batch from DiT to Non-DiT.
        Uses batched P2P operations to avoid serialization.
        """
        import io
        import pickle

        tensors_to_send = {}
        tensor_infos = {}
        list_tensor_infos = {}

        if hasattr(batch, "__dict__"):
            for k, v in batch.__dict__.items():
                if isinstance(v, torch.Tensor):
                    tensors_to_send[k] = v
                    tensor_infos[k] = (v.shape, v.dtype)
                elif isinstance(v, list) and v and len(v) > 0:
                    tensor_indices = []
                    actual_tensors = []
                    shapes_dtypes = []

                    for idx, t in enumerate(v):
                        if isinstance(t, torch.Tensor):
                            tensor_indices.append(idx)
                            actual_tensors.append(t)
                            shapes_dtypes.append((t.shape, t.dtype))

                    if actual_tensors:
                        tensors_to_send[k] = actual_tensors
                        list_tensor_infos[k] = (len(v), tensor_indices, shapes_dtypes)

        metadata = {
            k: v
            for k, v in batch.__dict__.items()
            if k not in tensor_infos and k not in list_tensor_infos
        }

        buffer = io.BytesIO()
        pickle.dump((metadata, tensor_infos, list_tensor_infos), buffer)
        meta_bytes = torch.tensor(
            bytearray(buffer.getvalue()), dtype=torch.uint8, device="cpu"
        )

        if dist.get_backend() == "nccl":
            meta_bytes = meta_bytes.cuda()

        size_tensor = torch.tensor(
            [meta_bytes.numel()], dtype=torch.long, device=meta_bytes.device
        )

        # Collect all tensors to send in a batch
        all_tensors = [size_tensor, meta_bytes]

        for name in tensor_infos.keys():
            tensor = tensors_to_send[name]
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            all_tensors.append(tensor)

        for name in list_tensor_infos.keys():
            tensor_list = tensors_to_send[name]
            for tensor in tensor_list:
                if not tensor.is_cuda:
                    tensor = tensor.cuda()
                all_tensors.append(tensor)

        # Batch send all tensors at once
        works = self.comm.batch_isend_to_non_dit(all_tensors)

        return works

    def _async_recv_batch_from_dit(
        self, batch: Req
    ) -> tuple[Req, List[Optional[Work]]]:
        """
        Async receive batch from DiT at Non-DiT group.
        Uses batched P2P operations to avoid serialization.
        """
        import pickle

        works = []

        # Step 1: Receive size tensor
        size_shapes_dtypes = [(torch.Size([1]), torch.long)]
        size_tensors, size_works = self.comm.batch_irecv_from_dit(size_shapes_dtypes)

        # Wait for size
        if size_works:
            for work in size_works:
                work.wait()

        size_tensor = size_tensors[0]
        size = size_tensor.item()

        # Step 2: Receive metadata
        meta_shapes_dtypes = [(torch.Size([size]), torch.uint8)]
        meta_tensors, meta_works = self.comm.batch_irecv_from_dit(meta_shapes_dtypes)

        # Wait for metadata
        if meta_works:
            for work in meta_works:
                work.wait()

        meta_bytes = meta_tensors[0]
        meta_data_cpu = meta_bytes.cpu().numpy().tobytes()
        metadata, tensor_infos, list_tensor_infos = pickle.loads(meta_data_cpu)

        for k, v in metadata.items():
            setattr(batch, k, v)

        # Step 3: Prepare to receive all data tensors in a batch
        all_shapes_dtypes = []
        tensor_names = []

        # Collect single tensors
        for name, (shape, dtype) in tensor_infos.items():
            all_shapes_dtypes.append((shape, dtype))
            tensor_names.append(("single", name))

        # Collect list tensors
        for name, list_info in list_tensor_infos.items():
            list_length, tensor_indices, shapes_dtypes = list_info
            for idx, (shape, dtype) in enumerate(shapes_dtypes):
                all_shapes_dtypes.append((shape, dtype))
                tensor_names.append(("list", name, idx, list_length, tensor_indices))

        # Batch receive all data tensors
        if all_shapes_dtypes:
            all_tensors, data_works = self.comm.batch_irecv_from_dit(all_shapes_dtypes)
            works.extend(data_works)

            # Assign tensors back to batch
            tensor_idx = 0
            list_tensors = {}

            for info in tensor_names:
                tensor = all_tensors[tensor_idx]

                if info[0] == "single":
                    setattr(batch, info[1], tensor)
                else:  # 'list'
                    name, idx, list_length, tensor_indices = (
                        info[1],
                        info[2],
                        info[3],
                        info[4],
                    )
                    if name not in list_tensors:
                        list_tensors[name] = {
                            "length": list_length,
                            "indices": tensor_indices,
                            "tensors": [],
                        }
                    list_tensors[name]["tensors"].append(tensor)

                tensor_idx += 1

            # Reconstruct lists
            for name, list_data in list_tensors.items():
                result_list = [None] * list_data["length"]
                for idx, tensor in zip(list_data["indices"], list_data["tensors"]):
                    result_list[idx] = tensor
                setattr(batch, name, result_list)

        return batch, works

    # --- Methods for Final Result Transfer ---
    # Uses batched P2P to avoid serialization warnings

    def _send_final_result_to_dit_master(self, output_batch: OutputBatch):
        """Send OutputBatch from non-dit to dit master using batched P2P."""
        import io
        import pickle

        # Prepare metadata (non-tensor fields)
        has_output = output_batch.output is not None

        metadata = {
            "timings": output_batch.timings,
            "error": output_batch.error,
            "has_output": has_output,
        }

        if has_output:
            metadata["output_shape"] = tuple(output_batch.output.shape)
            metadata["output_dtype"] = str(output_batch.output.dtype)

        # Serialize metadata
        buffer = io.BytesIO()
        pickle.dump(metadata, buffer)
        meta_bytes = torch.tensor(
            bytearray(buffer.getvalue()), dtype=torch.uint8, device="cpu"
        )

        if dist.get_backend() == "nccl":
            meta_bytes = meta_bytes.cuda()

        size_tensor = torch.tensor(
            [meta_bytes.numel()], dtype=torch.long, device=meta_bytes.device
        )

        # Collect all tensors to send in a batch
        all_tensors = [size_tensor, meta_bytes]

        if has_output:
            output_tensor = output_batch.output
            if not output_tensor.is_cuda:
                output_tensor = output_tensor.cuda()
            all_tensors.append(output_tensor)

        # Batch send all tensors at once
        works = self.comm.batch_isend_to_dit(all_tensors)

        # Wait for completion (this is final result, no need to overlap)
        for work in works:
            if work:
                work.wait()

    def _recv_final_result_from_non_dit(self, output_batch: OutputBatch):
        """Receive OutputBatch from non-dit at dit master using batched P2P."""
        import pickle

        # Step 1: Receive size tensor
        size_shapes_dtypes = [(torch.Size([1]), torch.long)]
        size_tensors, size_works = self.comm.batch_irecv_from_non_dit(
            size_shapes_dtypes
        )

        # Wait for size
        for work in size_works:
            if work:
                work.wait()

        size_tensor = size_tensors[0]
        size = size_tensor.item()

        # Step 2: Receive metadata
        meta_shapes_dtypes = [(torch.Size([size]), torch.uint8)]
        meta_tensors, meta_works = self.comm.batch_irecv_from_non_dit(
            meta_shapes_dtypes
        )

        # Wait for metadata
        for work in meta_works:
            if work:
                work.wait()

        meta_bytes = meta_tensors[0]
        meta_data_cpu = meta_bytes.cpu().numpy().tobytes()
        metadata = pickle.loads(meta_data_cpu)
        # Update output_batch with metadata
        output_batch.timings = metadata["timings"]
        output_batch.error = metadata["error"]

        # Step 3: Receive output tensor if it exists
        if metadata["has_output"]:
            shape = metadata["output_shape"]
            dtype_str = metadata["output_dtype"]

            # Parse dtype
            if "float32" in dtype_str:
                dtype = torch.float32
            elif "float16" in dtype_str:
                dtype = torch.float16
            else:
                dtype = torch.float32  # default

            # Batch receive output tensor
            output_shapes_dtypes = [(torch.Size(shape), dtype)]
            output_tensors, output_works = self.comm.batch_irecv_from_non_dit(
                output_shapes_dtypes
            )

            # Wait for output
            for work in output_works:
                if work:
                    work.wait()

            output_tensor = output_tensors[0]
            output_batch.output = output_tensor.to(get_local_torch_device())
        else:
            output_batch.output = None

        return output_batch
