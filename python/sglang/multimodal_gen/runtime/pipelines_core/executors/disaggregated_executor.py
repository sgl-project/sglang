"""
Pipeline executor for disaggregated execution.
"""

from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.dist_utils import get_disagg_communicator
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
    Timer,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage

logger = init_logger(__name__)


class DisaggregatedExecutor(PipelineExecutor):
    """
    Executor that handles the disaggregated pipeline flow:
    Non-DiT (Encode) -> DiT (Denoise) -> Non-DiT (Decode)
    """

    def __init__(self, server_args: ServerArgs):
        super().__init__(server_args)
        self.comm = get_disagg_communicator()
        assert (
            self.comm is not None
        ), "DisaggregatedExecutor requires an initialized DisaggCommunicator"
        print(
            f"DisaggregatedExecutor initialized on rank {dist.get_rank()} with role {self.comm.role}"
        )

    def execute(
        self,
        stages: List["PipelineStage"],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        """
        Executes the stages using a ping-pong mechanism between Non-DiT and DiT groups.

        Assumptions:
        1. Stages are ordered: [Encoding..., Denoising..., Decoding...]
        2. Denoising stages are identified by involving 'Transformer' or being compute-heavy.
        3. Encoding/Decoding stages run on Non-DiT.
        4. Denoising stages run on DiT.
        """

        with self.profile_execution(batch):
            # Identify stage types
            # In a real implementation, stages should self-identify their role/group requirements.
            # For now, we use a simple heuristic based on stage names or types.

            # Simple State Machine:
            # 0: Pre-Denoise (Run on Non-DiT)
            # 1: Denoise (Run on DiT)
            # 2: Post-Denoise (Run on Non-DiT)

            # Since we don't have explicit stage tagging yet, we split by index
            # or check stage class names.
            # Typically: TextEncoding -> LatentPreparation -> Denoising -> Decoding

            # Let's iterate and check where "Denoising" starts and ends.
            denoise_start_idx = -1
            denoise_end_idx = -1

            for i, stage in enumerate(stages):
                if isinstance(stage, DenoisingStage):
                    if denoise_start_idx == -1:
                        denoise_start_idx = i
                    denoise_end_idx = i

            if denoise_start_idx == -1:
                logger.warning(
                    "No denoising stage found! Running all on Non-DiT (fallback)."
                )
                # Fallback: Run everything on Non-DiT? Or maybe this is an image encoder only pipeline?
                # For safety in this proof-of-concept, we assume standard diffusion.
                return self._run_local(stages, batch)

            pre_denoise_stages = stages[:denoise_start_idx]
            denoise_stages = stages[denoise_start_idx : denoise_end_idx + 1]
            post_denoise_stages = stages[denoise_end_idx + 1 :]

            # --- PHASE 1: Pre-Denoise (Encoding) ---
            print(
                f"[Rank {dist.get_rank()}] Entering Phase 1. is_non_dit={self.comm.is_non_dit_rank()}, is_dit={self.comm.is_dit_rank()}"
            )
            if self.comm.is_non_dit_rank():
                print(
                    f"[Non-DiT Rank {dist.get_rank()}] Phase 1: Running Pre-Denoise stages..."
                )
                for stage in pre_denoise_stages:
                    print(
                        f"[Non-DiT Rank {dist.get_rank()}] Running stage: {stage.__class__.__name__}"
                    )
                    with Timer(stage.__class__.__name__):
                        batch = stage(batch, server_args)

                print(
                    f"[Non-DiT Rank {dist.get_rank()}] Phase 1 Complete. Sending data to DiT..."
                )
                print(
                    f"[Non-DiT Rank {dist.get_rank()}] self={self}, type={type(self)}"
                )
                print(
                    f"[Non-DiT Rank {dist.get_rank()}] hasattr _send_batch_to_dit: {hasattr(self, '_send_batch_to_dit')}"
                )
                print(f"[Non-DiT Rank {dist.get_rank()}] batch type: {type(batch)}")
                print(
                    f"[Non-DiT Rank {dist.get_rank()}] About to call _send_batch_to_dit..."
                )
                import sys

                sys.stdout.flush()  # Force flush

                # Get method reference first
                send_method = self._send_batch_to_dit
                print(
                    f"[Non-DiT Rank {dist.get_rank()}] Got method reference: {send_method}"
                )
                sys.stdout.flush()

                try:
                    send_method(batch)
                    print(
                        f"[Non-DiT Rank {dist.get_rank()}] _send_batch_to_dit returned successfully"
                    )
                    sys.stdout.flush()
                except Exception as e:
                    print(
                        f"[Non-DiT Rank {dist.get_rank()}] ERROR in _send_batch_to_dit: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    sys.stdout.flush()
                    raise

            elif self.comm.is_dit_rank():
                # DiT waits for data
                print(
                    f"[Dit Rank {dist.get_rank()}] Phase 1: Waiting for data from Non-DiT..."
                )
                print(
                    f"[Dit Rank {dist.get_rank()}] self.comm.non_dit_master_rank={self.comm.non_dit_master_rank}"
                )
                print(
                    f"[Dit Rank {dist.get_rank()}] self.comm.dit_master_rank={self.comm.dit_master_rank}"
                )
                import sys

                sys.stdout.flush()

                try:
                    batch = self._recv_batch_from_non_dit(batch)
                    print(
                        f"[Dit Rank {dist.get_rank()}] Received data from Non-DiT successfully"
                    )
                    sys.stdout.flush()
                except Exception as e:
                    print(
                        f"[Dit Rank {dist.get_rank()}] ERROR in _recv_batch_from_non_dit: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    sys.stdout.flush()
                    raise

            # --- PHASE 2: Denoise (DiT) ---
            if self.comm.is_dit_rank():
                print("Phase 2: Running Denoise stages on DiT...")
                for stage in denoise_stages:
                    with Timer(stage.__class__.__name__):
                        batch = stage(batch, server_args)

                # After denoising, latents are gathered (via gather_latents_for_sp)
                # All dit ranks have the complete result, but only dit master sends to non-dit
                if dist.get_rank() == self.comm.dit_master_rank:
                    print(
                        "Phase 2 Complete. DiT Master sending data back to Non-DiT..."
                    )
                    self._send_batch_to_non_dit(batch)
                else:
                    print(
                        f"Phase 2 Complete. DiT Worker (rank {dist.get_rank()}) done for this request."
                    )
                    # DiT workers are done. Non-dit will handle decoding.
                    # Return early to avoid unnecessary waiting
                    return OutputBatch()

            if self.comm.is_non_dit_rank():
                try:
                    # Non-DiT waits for result from DiT Master
                    print("Phase 2: Waiting for result from DiT Master...")
                    batch = self._recv_batch_from_dit(batch)

                    # --- PHASE 3: Post-Denoise (Decoding) ---
                    print("Phase 3: Running Post-Denoise stages on Non-DiT...")
                    for stage in post_denoise_stages:
                        with Timer(stage.__class__.__name__):
                            batch = stage(batch, server_args)

                    # Debug: check output status
                    print(
                        f"Phase 3 Complete. Output status: {batch.output is not None}, type: {type(batch.output) if batch.output is not None else 'None'}"
                    )
                    if batch.output is not None and isinstance(
                        batch.output, torch.Tensor
                    ):
                        print(
                            f"  Output shape: {batch.output.shape}, dtype: {batch.output.dtype}"
                        )

                    # Convert Req to OutputBatch
                    output_batch = OutputBatch(
                        output=batch.output,
                        trajectory_timesteps=batch.trajectory_timesteps,
                        trajectory_latents=batch.trajectory_latents,
                        trajectory_decoded=getattr(batch, "trajectory_decoded", None),
                        timings=batch.timings,
                        error=None,
                    )

                    # Non-dit has the final output, send it back to dit master for client response
                    print("Sending final result back to DiT Master...")
                    self._send_final_result_to_dit_master(output_batch)
                    print("Final result sent successfully.")
                except Exception as e:
                    logger.error(
                        f"[Non-DiT Rank {dist.get_rank()}] Error in Phase 2/3: {e}",
                        exc_info=True,
                    )
                    # Still need to send error info to dit master to avoid deadlock
                    # For now, just re-raise
                    raise

            # Dit master receives final result from non-dit
            if dist.get_rank() == self.comm.dit_master_rank:
                try:
                    print("Receiving final result from Non-DiT...")
                    output_batch = OutputBatch()
                    output_batch = self._recv_final_result_from_non_dit(output_batch)
                    print(
                        f"Final result received. Output status: {output_batch.output is not None}"
                    )
                    return output_batch
                except Exception as e:
                    logger.error(
                        f"[Dit Master] Error receiving final result: {e}", exc_info=True
                    )
                    raise

            # Dit workers return empty batch (they don't handle client responses)
            return OutputBatch()

    def _run_local(self, stages, batch):
        for stage in stages:
            batch = stage(batch, self.server_args)
        return batch

    # --- Helper methods for Data Transfer ---
    # These would ideally be in a Serializer/Deserializer class

    def _send_batch_to_dit(self, batch: Req):
        print(f"[_send_batch_to_dit ENTRY] Function entered, rank={dist.get_rank()}")
        import sys

        sys.stdout.flush()

        import io
        import pickle

        tensors_to_send = {}
        tensor_infos = {}  # name -> (shape, dtype)
        list_tensor_infos = {}  # name -> list of (shape, dtype)

        print(f"[_send_batch_to_dit] Checking batch attributes...")
        sys.stdout.flush()

        # Identify tensors (both single and lists)
        if hasattr(batch, "__dict__"):
            for k, v in batch.__dict__.items():
                if isinstance(v, torch.Tensor):
                    tensors_to_send[k] = v
                    tensor_infos[k] = (v.shape, v.dtype)
                elif isinstance(v, list) and v and len(v) > 0:
                    # Check if it's a list of tensors
                    # Build list with None placeholders and extract tensors separately
                    tensor_indices = []
                    actual_tensors = []
                    shapes_dtypes = []
                    
                    for idx, t in enumerate(v):
                        if isinstance(t, torch.Tensor):
                            tensor_indices.append(idx)
                            actual_tensors.append(t)
                            shapes_dtypes.append((t.shape, t.dtype))
                    
                    if actual_tensors:  # Only if there's at least one tensor
                        # Store: (list_length, tensor_indices, shapes_dtypes)
                        tensors_to_send[k] = actual_tensors
                        list_tensor_infos[k] = (len(v), tensor_indices, shapes_dtypes)

        # Debug: Print all fields and their types
        print(f"[DEBUG _send_batch_to_dit] All batch fields:")
        for k, v in batch.__dict__.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor, shape={v.shape}")
            elif isinstance(v, list):
                if len(v) > 0:
                    print(f"  {k}: List[{len(v)}], first={type(v[0])}")
                else:
                    print(f"  {k}: List[empty]")
            elif isinstance(v, tuple):
                print(f"  {k}: Tuple[{len(v)}], types={[type(x) for x in v]}")
            else:
                print(f"  {k}: {type(v)}, value_preview={str(v)[:100]}")

        # Metadata: everything that's not in tensors_to_send
        metadata = {
            k: v
            for k, v in batch.__dict__.items()
            if k not in tensor_infos and k not in list_tensor_infos
        }

        # Serialize metadata + tensor info
        buffer = io.BytesIO()
        pickle.dump((metadata, tensor_infos, list_tensor_infos), buffer)
        meta_bytes = torch.tensor(
            bytearray(buffer.getvalue()), dtype=torch.uint8, device="cpu"
        )

        if dist.get_backend() == "nccl":
            meta_bytes = meta_bytes.cuda()

        # 1. Send Metadata Size & Payload
        size_tensor = torch.tensor(
            [meta_bytes.numel()], dtype=torch.long, device=meta_bytes.device
        )
        print(f"[_send_batch_to_dit] About to send size_tensor: {size_tensor.item()}")
        import sys

        sys.stdout.flush()
        self.comm.send_to_dit(size_tensor)
        print(f"[_send_batch_to_dit] size_tensor sent, about to send meta_bytes")
        sys.stdout.flush()
        self.comm.send_to_dit(meta_bytes)
        print(f"[_send_batch_to_dit] meta_bytes sent")
        sys.stdout.flush()

        # 2. Send Single Tensors
        for name in tensor_infos.keys():
            tensor = tensors_to_send[name]
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            self.comm.send_to_dit(tensor)

        # 3. Send List of Tensors
        for name in list_tensor_infos.keys():
            tensor_list = tensors_to_send[name]
            for tensor in tensor_list:
                if not tensor.is_cuda:
                    tensor = tensor.cuda()
                self.comm.send_to_dit(tensor)

        print(
            f"_send_batch_to_dit: sent {len(tensor_infos)} single tensors and {len(list_tensor_infos)} tensor lists"
        )

    def _recv_batch_from_non_dit(self, batch: Req):
        import pickle

        print(
            f"[_recv_batch_from_non_dit Rank {dist.get_rank()}] Entered, waiting for size_tensor..."
        )
        import sys

        sys.stdout.flush()

        # 1. Recv Metadata Size
        size_tensor = self.comm.recv_from_non_dit(torch.Size([1]), torch.long)
        size = size_tensor.item()
        print(
            f"[_recv_batch_from_non_dit Rank {dist.get_rank()}] Received size: {size}"
        )
        sys.stdout.flush()

        # 2. Recv Metadata
        meta_bytes = self.comm.recv_from_non_dit(torch.Size([size]), torch.uint8)
        meta_data_cpu = meta_bytes.cpu().numpy().tobytes()
        metadata, tensor_infos, list_tensor_infos = pickle.loads(meta_data_cpu)

        # Update batch metadata
        for k, v in metadata.items():
            setattr(batch, k, v)

        # 3. Recv Single Tensors
        for name, (shape, dtype) in tensor_infos.items():
            tensor = self.comm.recv_from_non_dit(shape, dtype)
            tensor = tensor.to(get_local_torch_device())
            setattr(batch, name, tensor)

        # 4. Recv List of Tensors
        for name, list_info in list_tensor_infos.items():
            list_length, tensor_indices, shapes_dtypes = list_info
            
            # Receive actual tensors
            actual_tensors = []
            for shape, dtype in shapes_dtypes:
                tensor = self.comm.recv_from_non_dit(shape, dtype)
                tensor = tensor.to(get_local_torch_device())
                actual_tensors.append(tensor)
            
            # Reconstruct list with None placeholders
            result_list = [None] * list_length
            for idx, tensor in zip(tensor_indices, actual_tensors):
                result_list[idx] = tensor
            
            setattr(batch, name, result_list)

        print(
            f"_recv_batch_from_non_dit: received {len(tensor_infos)} single tensors and {len(list_tensor_infos)} tensor lists"
        )
        return batch

    def _send_batch_to_non_dit(self, batch: Req):
        # Send all tensors back to Non-DiT (primarily latents after denoising)
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
                    # Check if it's a list of tensors (may contain None)
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
        self.comm.send_to_non_dit(size_tensor)
        self.comm.send_to_non_dit(meta_bytes)

        # Send single tensors
        for name in tensor_infos.keys():
            tensor = tensors_to_send[name]
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            self.comm.send_to_non_dit(tensor)

        # Send list of tensors
        for name in list_tensor_infos.keys():
            tensor_list = tensors_to_send[name]
            for tensor in tensor_list:
                if not tensor.is_cuda:
                    tensor = tensor.cuda()
                self.comm.send_to_non_dit(tensor)

        print(
            f"_send_batch_to_non_dit: sent {len(tensor_infos)} single tensors and {len(list_tensor_infos)} tensor lists"
        )

    def _recv_batch_from_dit(self, batch: Req):
        import pickle

        size_tensor = self.comm.recv_from_dit(torch.Size([1]), torch.long)
        size = size_tensor.item()

        meta_bytes = self.comm.recv_from_dit(torch.Size([size]), torch.uint8)
        meta_data_cpu = meta_bytes.cpu().numpy().tobytes()
        metadata, tensor_infos, list_tensor_infos = pickle.loads(meta_data_cpu)

        for k, v in metadata.items():
            setattr(batch, k, v)

        # Recv single tensors
        for name, (shape, dtype) in tensor_infos.items():
            tensor = self.comm.recv_from_dit(shape, dtype)
            tensor = tensor.to(get_local_torch_device())
            setattr(batch, name, tensor)

        # Recv list of tensors
        for name, list_info in list_tensor_infos.items():
            list_length, tensor_indices, shapes_dtypes = list_info
            
            # Receive actual tensors
            actual_tensors = []
            for shape, dtype in shapes_dtypes:
                tensor = self.comm.recv_from_dit(shape, dtype)
                tensor = tensor.to(get_local_torch_device())
                actual_tensors.append(tensor)
            
            # Reconstruct list with None placeholders
            result_list = [None] * list_length
            for idx, tensor in zip(tensor_indices, actual_tensors):
                result_list[idx] = tensor
            
            setattr(batch, name, result_list)

        print(
            f"_recv_batch_from_dit: received {len(tensor_infos)} single tensors and {len(list_tensor_infos)} tensor lists"
        )
        return batch

    def _send_final_result_to_dit_master(self, output_batch: OutputBatch):
        """Send OutputBatch from non-dit to dit master."""
        import io
        import pickle

        print(
            f"[Non-DiT Rank {dist.get_rank()}] _send_final_result_to_dit_master called"
        )
        print(
            f"  non_dit_master_rank: {self.comm.non_dit_master_rank}, dit_master_rank: {self.comm.dit_master_rank}"
        )

        # Prepare metadata (non-tensor fields)
        has_output = output_batch.output is not None
        has_error = output_batch.error is not None

        print(f"  has_output: {has_output}, has_error: {has_error}")
        if has_output:
            print(
                f"  output shape: {output_batch.output.shape}, dtype: {output_batch.output.dtype}"
            )

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

        print(f"  Sending metadata size: {meta_bytes.numel()}")
        size_tensor = torch.tensor(
            [meta_bytes.numel()], dtype=torch.long, device=meta_bytes.device
        )
        self.comm.send_to_dit(size_tensor)
        self.comm.send_to_dit(meta_bytes)
        print("  Metadata sent")

        # Send output tensor if it exists
        if has_output:
            output_tensor = output_batch.output
            if not output_tensor.is_cuda:
                output_tensor = output_tensor.cuda()
            self.comm.send_to_dit(output_tensor)
            print(f"  Output tensor sent with shape {output_tensor.shape}")
        else:
            print("  No output tensor to send")

        print(
            f"[Non-DiT Rank {dist.get_rank()}] _send_final_result_to_dit_master completed"
        )

    def _recv_final_result_from_non_dit(self, output_batch: OutputBatch):
        """Receive OutputBatch from non-dit at dit master."""
        import pickle

        print(
            f"[Dit Master Rank {dist.get_rank()}] _recv_final_result_from_non_dit called"
        )
        print(f"  Waiting for metadata size...")

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
