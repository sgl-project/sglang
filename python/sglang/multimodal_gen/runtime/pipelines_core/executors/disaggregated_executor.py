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
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
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
    ) -> Req:
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
            if self.comm.is_non_dit_rank():
                print("Phase 1: Running Pre-Denoise stages on Non-DiT...")
                for stage in pre_denoise_stages:
                    with Timer(stage.__class__.__name__):
                        batch = stage(batch, server_args)

                # Prepare data to send to DiT
                # What needs to be sent? Usually 'prompt_embeds', 'latents' (if img2img), 'timesteps'
                # For now, we assume 'batch' object contains everything needed in tensors.
                # But 'batch' is a python object. We need to serialize it or send specific tensors.

                # Simplified: Send specific critical tensors.
                # In a robust system, we'd serialize the Batch/Req object (metadata) + Tensors (P2P).

                # PING: Send to DiT
                # We need to know WHAT tensors to send.
                # This is model specific.
                # For this generic implementation, we might need a contract.
                # Let's assume the batch object holds a dictionary of tensors we can introspect?

                # HACK for PoC: Assume Wan/Flux specific keys or iterate batch attributes.
                # Better: Pickle the whole batch metadata (small), and send big tensors separately?
                # Or use torch.save to buffer?

                # Let's try to send the key tensors:
                # 1. prompt_embeds / encoder_hidden_states
                # 2. pooled_prompt_embeds
                # 3. latents (initial noise)

                # For this Step 4, let's implement the CONTROL FLOW first.
                # The actual tensor transfer details might need refinement based on 'Req' structure.

                print("Phase 1 Complete. Sending data to DiT...")
                # self.comm.send_to_dit(batch_tensors)
                # Placeholder: We need a method to extract/inject tensors from Req
                self._send_batch_to_dit(batch)

            elif self.comm.is_dit_rank():
                # DiT waits for data
                print("Phase 1: Waiting for data from Non-DiT...")
                batch = self._recv_batch_from_non_dit(
                    batch
                )  # Update batch with received data

            # --- PHASE 2: Denoise (DiT) ---
            if self.comm.is_dit_rank():
                print("Phase 2: Running Denoise stages on DiT...")
                for stage in denoise_stages:
                    with Timer(stage.__class__.__name__):
                        batch = stage(batch, server_args)

                print("Phase 2 Complete. Sending data back to Non-DiT...")
                # PONG: Send back to Non-DiT
                self._send_batch_to_non_dit(batch)

            elif self.comm.is_non_dit_rank():
                # Non-DiT waits for result
                print("Phase 2: Waiting for result from DiT...")
                batch = self._recv_batch_from_dit(batch)

            # --- PHASE 3: Post-Denoise (Decoding) ---
            if self.comm.is_non_dit_rank():
                print("Phase 3: Running Post-Denoise stages on Non-DiT...")
                for stage in post_denoise_stages:
                    with Timer(stage.__class__.__name__):
                        batch = stage(batch, server_args)

            # DiT is done for this request.

        return batch

    def _run_local(self, stages, batch):
        for stage in stages:
            batch = stage(batch, self.server_args)
        return batch

    # --- Helper methods for Data Transfer ---
    # These would ideally be in a Serializer/Deserializer class

    def _send_batch_to_dit(self, batch: Req):
        import io
        import pickle

        tensors_to_send = {}
        tensor_infos = {}  # name -> (shape, dtype)

        # Identify tensors
        if hasattr(batch, "__dict__"):
            for k, v in batch.__dict__.items():
                if isinstance(v, torch.Tensor):
                    tensors_to_send[k] = v
                    tensor_infos[k] = (v.shape, v.dtype)

        metadata = {
            k: v for k, v in batch.__dict__.items() if not isinstance(v, torch.Tensor)
        }

        # Serialize metadata + tensor info
        buffer = io.BytesIO()
        pickle.dump((metadata, tensor_infos), buffer)
        meta_bytes = torch.tensor(
            bytearray(buffer.getvalue()), dtype=torch.uint8, device="cpu"
        )

        if dist.get_backend() == "nccl":
            meta_bytes = meta_bytes.cuda()

        # 1. Send Metadata Size & Payload
        size_tensor = torch.tensor(
            [meta_bytes.numel()], dtype=torch.long, device=meta_bytes.device
        )
        self.comm.send_to_dit(size_tensor)
        self.comm.send_to_dit(meta_bytes)

        # 2. Send Tensors
        for name, tensor in tensors_to_send.items():
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            self.comm.send_to_dit(tensor)

        print(f"_send_batch_to_dit: {batch}")

    def _recv_batch_from_non_dit(self, batch: Req):
        import pickle

        # 1. Recv Metadata Size
        size_tensor = self.comm.recv_from_non_dit(torch.Size([1]), torch.long)
        size = size_tensor.item()

        # 2. Recv Metadata
        meta_bytes = self.comm.recv_from_non_dit(torch.Size([size]), torch.uint8)
        meta_data_cpu = meta_bytes.cpu().numpy().tobytes()
        metadata, tensor_infos = pickle.loads(meta_data_cpu)

        # Update batch metadata
        for k, v in metadata.items():
            setattr(batch, k, v)

        # 3. Recv Tensors
        for name, (shape, dtype) in tensor_infos.items():
            tensor = self.comm.recv_from_non_dit(shape, dtype)
            # TODO: USE CUDA-IPC?
            tensor = tensor.to(get_local_torch_device())
            setattr(batch, name, tensor)

        if isinstance(batch.prompt_embeds, list):
            batch.prompt_embeds = [
                prompt_embeds.to(get_local_torch_device())
                for prompt_embeds in batch.prompt_embeds
            ]

        print(f"_recv_batch_from_non_dit: {batch}")
        return batch

    def _send_batch_to_non_dit(self, batch: Req):
        # We assume only changed tensors need to be sent back?
        # Or just send everything again?
        # For simplicity/correctness, send everything that is a tensor.
        # But efficiently, we might only want 'latents'.
        # Let's send all tensors found in batch to be safe.

        import io
        import pickle

        tensors_to_send = {}
        tensor_infos = {}

        if hasattr(batch, "__dict__"):
            for k, v in batch.__dict__.items():
                if isinstance(v, torch.Tensor):
                    tensors_to_send[k] = v
                    tensor_infos[k] = (v.shape, v.dtype)

        # No need to send metadata back? Maybe updated metadata (e.g. counters)?
        metadata = {
            k: v for k, v in batch.__dict__.items() if not isinstance(v, torch.Tensor)
        }

        buffer = io.BytesIO()
        pickle.dump((metadata, tensor_infos), buffer)
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

        for name, tensor in tensors_to_send.items():
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            self.comm.send_to_non_dit(tensor)

        print(f"_send_batch_to_non_dit: {batch}")

    def _recv_batch_from_dit(self, batch: Req):
        import pickle

        size_tensor = self.comm.recv_from_dit(torch.Size([1]), torch.long)
        size = size_tensor.item()

        meta_bytes = self.comm.recv_from_dit(torch.Size([size]), torch.uint8)
        meta_data_cpu = meta_bytes.cpu().numpy().tobytes()
        metadata, tensor_infos = pickle.loads(meta_data_cpu)

        for k, v in metadata.items():
            setattr(batch, k, v)

        for name, (shape, dtype) in tensor_infos.items():
            tensor = self.comm.recv_from_dit(shape, dtype)
            # TODO: USE CUDA-IPC?
            tensor = tensor.to(get_local_torch_device())
            setattr(batch, name, tensor)

        print(f"_recv_batch_from_dit: {batch}")
        return batch
