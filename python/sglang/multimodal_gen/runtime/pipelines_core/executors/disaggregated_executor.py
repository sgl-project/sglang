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
                logger.debug(f"[Non-DiT Rank {dist.get_rank()}] Phase 1: Encoding")

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
                    logger.debug(
                        f"[Non-DiT] Waited for send completion: {oldest.batch_id}"
                    )

            # --- PHASE 2: Denoising (DiT) ---
            if self.comm.is_dit_rank():
                logger.debug(f"[DiT Rank {dist.get_rank()}] Phase 2: Denoising")

                # Async receive from Non-DiT
                batch, recv_works = self._async_recv_batch_from_non_dit(batch)
                logger.debug(
                    f"[DiT Rank {dist.get_rank()}] Phase 2: _async_recv_batch_from_non_dit"
                )

                # Wait for receive to complete before denoising
                if recv_works:
                    self.comm.wait_all_works(recv_works)
                    logger.debug(f"[DiT] Received batch from Non-DiT")

                logger.debug(f"[DiT Rank {dist.get_rank()}] Phase 2: wait_all_works")

                # Broadcast to all DiT workers (if SP/TP)
                self._broadcast_batch_in_dit_group(batch)

                logger.debug(
                    f"[DiT Rank {dist.get_rank()}] Phase 2: _broadcast_batch_in_dit_group"
                )

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
                        logger.debug(f"[DiT Master] Sent batch to Non-DiT")
                else:
                    # DiT workers are done
                    return OutputBatch()

            # --- PHASE 3: Decoding (Non-DiT) ---
            if self.comm.is_non_dit_rank():
                logger.debug(f"[Non-DiT Rank {dist.get_rank()}] Phase 3: Decoding")

                # Wait for any pending sends from Phase 1
                while self.pending_sends:
                    pending = self.pending_sends.popleft()
                    self.comm.wait_all_works(pending.works)

                # Async receive from DiT
                batch, recv_works = self._async_recv_batch_from_dit(batch)

                # Wait for receive
                if recv_works:
                    self.comm.wait_all_works(recv_works)
                    logger.debug(f"[Non-DiT] Received batch from DiT")

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
        print(f"_broadcast_batch_in_dit_group...")
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
            logger.debug(
                f"[DiT Worker {dist.get_rank()}] Creating tensors on device: {device}"
            )

            for name, (shape, dtype) in tensor_infos.items():
                tensor = torch.empty(shape, dtype=dtype, device=device)
                if not tensor.is_cuda and dist.get_backend() == "nccl":
                    # Force to CUDA if NCCL backend
                    tensor = tensor.cuda()
                setattr(batch, name, tensor)
                logger.debug(
                    f"[DiT Worker {dist.get_rank()}] Created tensor {name}: device={tensor.device}"
                )

            # Create lists of empty tensors
            for name, list_info in list_tensor_infos.items():
                list_length, tensor_indices, shapes_dtypes = list_info

                actual_tensors = []
                for idx, (shape, dtype) in enumerate(shapes_dtypes):
                    tensor = torch.empty(shape, dtype=dtype, device=device)
                    if not tensor.is_cuda and dist.get_backend() == "nccl":
                        tensor = tensor.cuda()
                    actual_tensors.append(tensor)
                    logger.debug(
                        f"[DiT Worker {dist.get_rank()}] Created list tensor {name}[{idx}]: device={tensor.device}"
                    )

                # Reconstruct list with None placeholders
                result_list = [None] * list_length
                for idx, tensor in zip(tensor_indices, actual_tensors):
                    result_list[idx] = tensor

                setattr(batch, name, result_list)

        # Step 2: Broadcast all tensors
        # Now all workers have the tensor objects, broadcast the data
        logger.debug(f"[DiT Rank {dist.get_rank()}] Step 2: Broadcasting tensors...")

        if hasattr(batch, "__dict__"):
            for k, v in batch.__dict__.items():
                if k.startswith("_recv_"):
                    continue  # Skip internal metadata

                if isinstance(v, torch.Tensor):
                    # Ensure tensor is on CUDA before broadcast (NCCL requires CUDA)
                    logger.debug(
                        f"[DiT Rank {dist.get_rank()}] Broadcasting tensor {k}: device={v.device}, shape={v.shape}"
                    )
                    if not v.is_cuda:
                        logger.warning(
                            f"[DiT Rank {dist.get_rank()}] Tensor {k} is on CPU, moving to CUDA"
                        )
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
                            logger.debug(
                                f"[DiT Rank {dist.get_rank()}] Broadcasting list tensor {k}[{idx}]: device={item.device}"
                            )
                            # Ensure tensor is on CUDA
                            if not item.is_cuda:
                                logger.warning(
                                    f"[DiT Rank {dist.get_rank()}] List tensor {k}[{idx}] is on CPU, moving to CUDA"
                                )
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

        logger.debug(f"[DiT Rank {dist.get_rank()}] Broadcasted batch within DiT group")

    # --- Async Communication Methods ---

    def _async_send_batch_to_dit(self, batch: Req) -> List[Optional[Work]]:
        """
        Async send batch from Non-DiT to DiT.
        Returns list of Work handles.
        """
        import io
        import pickle

        works = []

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

        # Send metadata size
        size_tensor = torch.tensor(
            [meta_bytes.numel()], dtype=torch.long, device=meta_bytes.device
        )
        work = self.comm.isend_to_dit(size_tensor)
        if work:
            works.append(work)

        # Send metadata
        work = self.comm.isend_to_dit(meta_bytes)
        if work:
            works.append(work)

        # Send single tensors
        for name in tensor_infos.keys():
            tensor = tensors_to_send[name]
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            work = self.comm.isend_to_dit(tensor)
            if work:
                works.append(work)

        # Send list of tensors
        for name in list_tensor_infos.keys():
            tensor_list = tensors_to_send[name]
            for tensor in tensor_list:
                if not tensor.is_cuda:
                    tensor = tensor.cuda()
                work = self.comm.isend_to_dit(tensor)
                if work:
                    works.append(work)

        logger.debug(
            f"[Async Send to DiT] Initiated {len(works)} async sends "
            f"({len(tensor_infos)} single tensors, {len(list_tensor_infos)} tensor lists)"
        )

        return works

    def _async_recv_batch_from_non_dit(
        self, batch: Req
    ) -> tuple[Req, List[Optional[Work]]]:
        """
        Async receive batch from Non-DiT at DiT group.
        Master receives, then data is broadcasted in _broadcast_batch_in_dit_group.
        Returns (batch, list of Work handles).
        """
        import pickle

        works = []

        # Only DiT master receives metadata and data
        # Other workers will get data via broadcast
        if dist.get_rank() == self.comm.dit_master_rank:
            # Recv metadata size
            size_tensor, work = self.comm.irecv_from_non_dit(
                torch.Size([1]), torch.long
            )
            if work:
                works.append(work)
                work.wait()  # Must wait to get size

            size = size_tensor.item()

            # Recv metadata
            meta_bytes, work = self.comm.irecv_from_non_dit(
                torch.Size([size]), torch.uint8
            )
            if work:
                works.append(work)
                work.wait()  # Must wait to parse metadata

            meta_data_cpu = meta_bytes.cpu().numpy().tobytes()
            metadata, tensor_infos, list_tensor_infos = pickle.loads(meta_data_cpu)

            # Store metadata for broadcasting
            batch._recv_metadata = metadata
            batch._recv_tensor_infos = tensor_infos
            batch._recv_list_tensor_infos = list_tensor_infos

            # Update batch metadata
            for k, v in metadata.items():
                setattr(batch, k, v)

            # Recv single tensors (async)
            for name, (shape, dtype) in tensor_infos.items():
                tensor, work = self.comm.irecv_from_non_dit(shape, dtype)
                setattr(batch, name, tensor)
                if work:
                    works.append(work)

            # Recv list of tensors (async)
            for name, list_info in list_tensor_infos.items():
                list_length, tensor_indices, shapes_dtypes = list_info

                actual_tensors = []
                for shape, dtype in shapes_dtypes:
                    tensor, work = self.comm.irecv_from_non_dit(shape, dtype)
                    actual_tensors.append(tensor)
                    if work:
                        works.append(work)

                # Reconstruct list
                result_list = [None] * list_length
                for idx, tensor in zip(tensor_indices, actual_tensors):
                    result_list[idx] = tensor

                setattr(batch, name, result_list)

            logger.debug(
                f"[DiT Master] Initiated {len(works)} async recvs "
                f"({len(tensor_infos)} single tensors, {len(list_tensor_infos)} tensor lists)"
            )

        # Non-master workers will get data via broadcast_batch_in_dit_group

        return batch, works

    def _async_send_batch_to_non_dit(self, batch: Req) -> List[Optional[Work]]:
        """Async send batch from DiT to Non-DiT."""
        import io
        import pickle

        works = []

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
        work = self.comm.isend_to_non_dit(size_tensor)
        if work:
            works.append(work)

        work = self.comm.isend_to_non_dit(meta_bytes)
        if work:
            works.append(work)

        for name in tensor_infos.keys():
            tensor = tensors_to_send[name]
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            work = self.comm.isend_to_non_dit(tensor)
            if work:
                works.append(work)

        for name in list_tensor_infos.keys():
            tensor_list = tensors_to_send[name]
            for tensor in tensor_list:
                if not tensor.is_cuda:
                    tensor = tensor.cuda()
                work = self.comm.isend_to_non_dit(tensor)
                if work:
                    works.append(work)

        logger.debug(f"[Async Send to Non-DiT] Initiated {len(works)} async sends")

        return works

    def _async_recv_batch_from_dit(
        self, batch: Req
    ) -> tuple[Req, List[Optional[Work]]]:
        """Async receive batch from DiT at Non-DiT group."""
        import pickle

        works = []

        size_tensor, work = self.comm.irecv_from_dit(torch.Size([1]), torch.long)
        if work:
            works.append(work)
            work.wait()

        size = size_tensor.item()

        meta_bytes, work = self.comm.irecv_from_dit(torch.Size([size]), torch.uint8)
        if work:
            works.append(work)
            work.wait()

        meta_data_cpu = meta_bytes.cpu().numpy().tobytes()
        metadata, tensor_infos, list_tensor_infos = pickle.loads(meta_data_cpu)

        for k, v in metadata.items():
            setattr(batch, k, v)

        for name, (shape, dtype) in tensor_infos.items():
            tensor, work = self.comm.irecv_from_dit(shape, dtype)
            setattr(batch, name, tensor)
            if work:
                works.append(work)

        for name, list_info in list_tensor_infos.items():
            list_length, tensor_indices, shapes_dtypes = list_info

            actual_tensors = []
            for shape, dtype in shapes_dtypes:
                tensor, work = self.comm.irecv_from_dit(shape, dtype)
                actual_tensors.append(tensor)
                if work:
                    works.append(work)

            result_list = [None] * list_length
            for idx, tensor in zip(tensor_indices, actual_tensors):
                result_list[idx] = tensor

            setattr(batch, name, result_list)

        logger.debug(f"[Async Recv from DiT] Initiated {len(works)} async recvs")

        return batch, works

    # --- Sync Methods for Final Result Transfer ---
    # Final result transfer uses sync communication (no need to overlap)

    def _send_final_result_to_dit_master(self, output_batch: OutputBatch):
        """Send OutputBatch from non-dit to dit master."""
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

        # Serialize and send metadata
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
        self.comm.send_to_dit(size_tensor)
        self.comm.send_to_dit(meta_bytes)

        # Send output tensor if it exists
        if has_output:
            output_tensor = output_batch.output
            if not output_tensor.is_cuda:
                output_tensor = output_tensor.cuda()
            self.comm.send_to_dit(output_tensor)

    def _recv_final_result_from_non_dit(self, output_batch: OutputBatch):
        """Receive OutputBatch from non-dit at dit master."""
        import pickle

        # Recv metadata
        size_tensor = self.comm.recv_from_non_dit(torch.Size([1]), torch.long)
        size = size_tensor.item()
        print(f"  Metadata size: {size}")

        print(f"  Receiving metadata...")
        meta_bytes = self.comm.recv_from_non_dit(torch.Size([size]), torch.uint8)
        meta_data_cpu = meta_bytes.cpu().numpy().tobytes()
        metadata = pickle.loads(meta_data_cpu)
        print(
            f"  Metadata received: has_output={metadata.get('has_output')}, error={metadata.get('error')}"
        )

        # Update output_batch with metadata
        output_batch.timings = metadata["timings"]
        output_batch.error = metadata["error"]

        # Recv output tensor if it exists
        if metadata["has_output"]:
            shape = metadata["output_shape"]
            dtype_str = metadata["output_dtype"]
            print(f"  Output shape: {shape}, dtype: {dtype_str}")

            # Parse dtype
            if "float32" in dtype_str:
                dtype = torch.float32
            elif "float16" in dtype_str:
                dtype = torch.float16
            else:
                dtype = torch.float32  # default

            print(f"  Receiving output tensor...")
            output_tensor = self.comm.recv_from_non_dit(torch.Size(shape), dtype)
            output_batch.output = output_tensor.to(get_local_torch_device())
            print(f"  Output tensor received with shape {shape}")
        else:
            output_batch.output = None
            print("  No output tensor")

        print(
            f"[Dit Master Rank {dist.get_rank()}] _recv_final_result_from_non_dit completed"
        )
        return output_batch
