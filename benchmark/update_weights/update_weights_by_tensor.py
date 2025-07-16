# Copyright (c) 2025 RedAccel Authors. All Rights Reserved.

import argparse
import asyncio
import os
import time
from dataclasses import dataclass
from itertools import islice
from typing import Iterable, Iterator, List, Tuple

import torch
import torch.distributed as dist
import tqdm
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument(
    "--profile_dir", type=str, default="/home/jobuser/profiling_results/"
)
parser.add_argument(
    "--bucket_bytes",
    type=int,
    default=0,
    help="Bucket size in bytes. 0 means per-tensor",
)
parser.add_argument(
    "--benchmark", action="store_true", help="Run benchmark with different bucket sizes"
)
parser.add_argument(
    "--use_flattened_buckets",
    action="store_true",
    help="Use flattened tensor buckets for better memory efficiency",
)
args = parser.parse_args()


@dataclass
class TensorMetadata:
    """Metadata for a tensor in a flattened bucket"""

    name: str
    shape: torch.Size
    dtype: torch.dtype
    start_idx: int
    end_idx: int
    numel: int


class TensorBucket:
    """
    A bucket that flattens multiple tensors into a single tensor for efficient processing
    while preserving all metadata needed for reconstruction.
    """

    def __init__(self, named_tensors: List[Tuple[str, torch.Tensor]]):
        """
        Initialize a tensor bucket from a list of named tensors.

        Args:
            named_tensors: List of (name, tensor) tuples
        """
        self.metadata: List[TensorMetadata] = []
        self.flattened_tensor: torch.Tensor = None
        self.device = None
        self.total_elements = 0

        if not named_tensors:
            raise ValueError("Cannot create empty tensor bucket")

        # Collect metadata and flatten tensors
        flattened_tensors = []
        current_idx = 0

        for name, tensor in named_tensors:
            # Store device info from first tensor
            if self.device is None:
                self.device = tensor.device

            # Ensure all tensors are on the same device
            if tensor.device != self.device:
                tensor = tensor.to(self.device)

            flattened = tensor.flatten()
            flattened_tensors.append(flattened)

            # Store metadata
            metadata = TensorMetadata(
                name=name,
                shape=tensor.shape,
                dtype=tensor.dtype,
                start_idx=current_idx,
                end_idx=current_idx + flattened.numel(),
                numel=flattened.numel(),
            )
            self.metadata.append(metadata)
            current_idx += flattened.numel()

        # Concatenate all flattened tensors
        self.flattened_tensor = torch.cat(flattened_tensors, dim=0)
        self.total_elements = self.flattened_tensor.numel()

        # logger.debug(f"Created tensor bucket with {len(named_tensors)} tensors, "
        #             f"total elements: {self.total_elements:,}, "
        #             f"memory: {self.flattened_tensor.element_size() * self.total_elements / 1024**2:.2f} MB")

    def get_flattened_tensor(self) -> torch.Tensor:
        """Get the flattened tensor containing all bucket tensors"""
        return self.flattened_tensor

    def get_metadata(self) -> List[TensorMetadata]:
        """Get metadata for all tensors in the bucket"""
        return self.metadata

    def reconstruct_tensors(
        self, flattened_tensor: torch.Tensor = None
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Reconstruct original tensors from flattened tensor.

        Args:
            flattened_tensor: Optional flattened tensor to reconstruct from.
                            If None, uses the bucket's own flattened tensor.

        Returns:
            List of (name, tensor) tuples with original shapes
        """
        if flattened_tensor is None:
            flattened_tensor = self.flattened_tensor

        if flattened_tensor.numel() != self.total_elements:
            raise ValueError(
                f"Flattened tensor has {flattened_tensor.numel()} elements, "
                f"expected {self.total_elements}"
            )

        reconstructed = []
        for meta in self.metadata:
            # Extract the slice for this tensor
            tensor_slice = flattened_tensor[meta.start_idx : meta.end_idx]

            # Reshape to original shape
            tensor = tensor_slice.reshape(meta.shape)

            # Ensure correct dtype
            if tensor.dtype != meta.dtype:
                tensor = tensor.to(meta.dtype)

            reconstructed.append((meta.name, tensor))

        return reconstructed

    def size_bytes(self) -> int:
        """Get the total size of the bucket in bytes"""
        return self.flattened_tensor.element_size() * self.total_elements

    def __len__(self) -> int:
        """Number of tensors in the bucket"""
        return len(self.metadata)

    def __repr__(self) -> str:
        return (
            f"TensorBucket(tensors={len(self.metadata)}, "
            f"elements={self.total_elements:,}, "
            f"size={self.size_bytes() / 1024**2:.2f}MB)"
        )


def get_named_tensor_buckets(
    iterable: Iterator[tuple[str, torch.Tensor]], bucket_bytes: int
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """
    Group tensors into buckets based on a specified size in megabytes.
    Args:
        iterable: An iterator of tuples containing tensor names and tensors.
        bucket_bytes: The maximum size of each bucket in bytes.
    Yields:
        Lists of tuples, where each tuple contains a tensor name and its corresponding tensor.
    Example:
        >>> tensors = [('tensor1', torch.randn(1000, 1000)), ('tensor2', torch.randn(2000, 2000))]
        >>> for bucket in get_named_tensor_buckets(tensors, bucket_size_mb=10):
        ...     print(bucket)
        [('tensor1', tensor(...)), ('tensor2', tensor(...))]
    """
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    current_bucket = []
    current_size = 0
    for name, tensor in iterable:
        tensor_size = tensor.element_size() * tensor.numel()
        if current_size + tensor_size > bucket_bytes:
            if current_bucket:
                yield current_bucket
            current_bucket = [(name, tensor)]
            current_size = tensor_size
        else:
            current_bucket.append((name, tensor))
            current_size += tensor_size

    if current_bucket:
        yield current_bucket


def get_flattened_tensor_buckets(
    iterable: Iterator[tuple[str, torch.Tensor]], bucket_bytes: int
) -> Iterator[TensorBucket]:
    """
    Group tensors into flattened buckets based on a specified size in bytes.

    Args:
        iterable: An iterator of tuples containing tensor names and tensors.
        bucket_bytes: The maximum size of each bucket in bytes.

    Yields:
        TensorBucket objects containing flattened tensors with metadata.

    Example:
        >>> tensors = [('tensor1', torch.randn(1000, 1000)), ('tensor2', torch.randn(2000, 2000))]
        >>> for bucket in get_flattened_tensor_buckets(tensors, bucket_bytes=10*1024*1024):
        ...     print(bucket)
        TensorBucket(tensors=2, elements=5000000, size=20.00MB)
    """
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    current_bucket = []
    current_size = 0

    for name, tensor in iterable:
        tensor_size = tensor.element_size() * tensor.numel()
        if current_size + tensor_size > bucket_bytes:
            if current_bucket:
                yield TensorBucket(current_bucket)
            current_bucket = [(name, tensor)]
            current_size = tensor_size
        else:
            current_bucket.append((name, tensor))
            current_size += tensor_size

    if current_bucket:
        yield TensorBucket(current_bucket)


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def patch_profiler():
    from sglang.srt.model_executor.model_runner import ModelRunner

    ori_update = ModelRunner.update_weights_from_tensor

    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            skip_first=True, wait=10, warmup=1, active=4, repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile_dir),
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    )
    prof.start()

    logger.info(f"Profiling will be saved to {args.profile_dir}")

    def update_with_profiler(self, named_tensors, load_format=None):
        # Handle flattened bucket format before profiling
        # Call the original method for other formats
        res = ori_update(self, named_tensors, load_format)
        prof.step()
        return res

    ModelRunner.update_weights_from_tensor = update_with_profiler


patch_profiler()


def per_tensor_generator(model) -> Iterable[Tuple[str, torch.Tensor]]:
    """Generate tensors one by one, moving to GPU only when needed"""
    for name, tensor in model.named_parameters():
        yield name, tensor.cuda()


def analyze_bucket_distribution(model, bucket_bytes, rank, use_flattened=False):
    """Analyze how tensors are distributed into buckets (CPU-only analysis)"""
    if bucket_bytes == 0:
        return

    if rank == 0:
        bucket_type = "flattened" if use_flattened else "regular"
        logger.info(
            f"\nBUCKET DISTRIBUTION ANALYSIS ({bucket_type} buckets, bucket_size={bucket_bytes/1024**2:.1f}MB):"
        )

        # Use CPU-only tensor info for analysis to avoid GPU OOM
        def cpu_tensor_generator():
            for name, tensor in model.named_parameters():
                # Don't move to GPU, just analyze metadata
                yield name, tensor.cpu()

        named_tensors = list(cpu_tensor_generator())

        if use_flattened:
            buckets = list(
                get_flattened_tensor_buckets(iter(named_tensors), bucket_bytes)
            )
            bucket_sizes = [bucket.size_bytes() for bucket in buckets]
        else:
            buckets = list(get_named_tensor_buckets(iter(named_tensors), bucket_bytes))
            bucket_sizes = [
                sum(tensor.element_size() * tensor.numel() for _, tensor in bucket)
                for bucket in buckets
            ]

        logger.info(f"Total tensors: {len(named_tensors)}")
        logger.info(f"Total buckets: {len(buckets)}")

        avg_bucket_size = sum(bucket_sizes) / len(bucket_sizes) if bucket_sizes else 0
        max_bucket_size = max(bucket_sizes) if bucket_sizes else 0
        min_bucket_size = min(bucket_sizes) if bucket_sizes else 0

        logger.info(f"Average bucket size: {avg_bucket_size/1024**2:.1f} MB")
        logger.info(f"Max bucket size: {max_bucket_size/1024**2:.1f} MB")
        logger.info(f"Min bucket size: {min_bucket_size/1024**2:.1f} MB")

        # Show first few buckets as examples
        logger.info("\nFirst 3 buckets:")
        for i, bucket in enumerate(buckets[:3]):
            if use_flattened:
                bucket_size_bytes = bucket.size_bytes()
                logger.info(
                    f"  Bucket {i+1}: {len(bucket)} tensors, {bucket_size_bytes/1024**2:.1f} MB (flattened: {bucket.total_elements:,} elements)"
                )
                for meta in bucket.get_metadata()[:2]:  # Show first 2 tensors in bucket
                    tensor_size = (
                        meta.numel * torch.tensor([], dtype=meta.dtype).element_size()
                    )
                    logger.info(
                        f"    - {meta.name}: {meta.shape} ({tensor_size/1024**2:.1f} MB)"
                    )
                if len(bucket) > 2:
                    logger.info(f"    - ... and {len(bucket)-2} more tensors")
            else:
                bucket_size_bytes = sum(
                    tensor.element_size() * tensor.numel() for _, tensor in bucket
                )
                logger.info(
                    f"  Bucket {i+1}: {len(bucket)} tensors, {bucket_size_bytes/1024**2:.1f} MB"
                )
                for name, tensor in bucket[:2]:  # Show first 2 tensors in bucket
                    tensor_size = tensor.element_size() * tensor.numel()
                    logger.info(
                        f"    - {name}: {tensor.shape} ({tensor_size/1024**2:.1f} MB)"
                    )
                if len(bucket) > 2:
                    logger.info(f"    - ... and {len(bucket)-2} more tensors")
        logger.info("")


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"
    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]

    if rank == 0:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        inference_engine = AsyncEngine(
            model_path=args.model,
            dtype="bfloat16",
            mem_fraction_static=0.4,
            enable_memory_saver=True,
            tp_size=world_size,
            load_format="dummy",
            dist_init_addr=None,
            trust_remote_code=True,
            mm_attention_backend="fa3",
            attention_backend="fa3",
            disable_cuda_graph=True,
        )
    dist.barrier()

    # Load model on CPU with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Enable low CPU memory usage
        torch_dtype=torch.float16,  # Use half precision to save memory
    )
    tensor_num = len(model.state_dict())

    async def update_weights(named_tensors):
        initial_memory = torch.cuda.memory_allocated(rank) / 1024**3
        torch.cuda.reset_peak_memory_stats(rank)

        logger.info(
            f"Rank {rank} start to update weights with {tensor_num} tensors (per-tensor mode)."
        )
        logger.info(f"Rank {rank} initial memory: {initial_memory:.2f} GB")

        for name, tensor in tqdm.tqdm(named_tensors, total=tensor_num):
            serialized_tensor = MultiprocessingSerializer.serialize(tensor.detach())

            if rank == 0:
                gathered_serialized_tensors = [None for _ in range(world_size)]
            else:
                gathered_serialized_tensors = None

            dist.gather_object(
                obj=serialized_tensor,
                object_gather_list=gathered_serialized_tensors,
                dst=0,
            )

            if rank == 0:
                await inference_engine.update_weights_from_tensor(
                    named_tensors=[
                        (
                            name,
                            LocalSerializedTensor(values=gathered_serialized_tensors),
                        )
                    ],
                    load_format=None,
                    flush_cache=False,
                )

        peak_memory = torch.cuda.max_memory_allocated(rank) / 1024**3
        final_memory = torch.cuda.memory_allocated(rank) / 1024**3

        logger.info(
            f"Rank {rank} peak memory: {peak_memory:.2f} GB (increase: {peak_memory - initial_memory:.2f} GB)"
        )
        logger.info(f"Rank {rank} final memory: {final_memory:.2f} GB")
        dist.barrier()

    async def update_weights_by_bucket(named_tensors, bucket_bytes):
        initial_memory = torch.cuda.memory_allocated(rank) / 1024**3
        torch.cuda.reset_peak_memory_stats(rank)

        logger.info(
            f"Rank {rank} start to update weights with {tensor_num=} tensors and bucket_bytes={bucket_bytes} ({bucket_bytes/1024**2:.1f}MB)"
        )
        logger.info(f"Rank {rank} initial memory: {initial_memory:.2f} GB")

        pbar = tqdm.tqdm(total=tensor_num)
        total_buckets = 0
        max_bucket_memory = 0

        for bucket in get_named_tensor_buckets(named_tensors, bucket_bytes):
            # Monitor memory before processing bucket
            pre_bucket_memory = torch.cuda.memory_allocated(rank) / 1024**3

            serialized_tensor = [
                (name, MultiprocessingSerializer.serialize(tensor.detach()))
                for name, tensor in bucket
            ]

            # Monitor memory after serialization
            post_serialize_memory = torch.cuda.memory_allocated(rank) / 1024**3
            bucket_memory_increase = post_serialize_memory - pre_bucket_memory
            max_bucket_memory = max(max_bucket_memory, bucket_memory_increase)

            if rank == 0:
                gathered_serialized_tensors = [None for _ in range(world_size)]
            else:
                gathered_serialized_tensors = None

            dist.gather_object(
                obj=serialized_tensor,
                object_gather_list=gathered_serialized_tensors,
                dst=0,
            )

            if rank == 0:
                await inference_engine.update_weights_from_tensor(
                    named_tensors=[
                        (i[0][0], LocalSerializedTensor(values=[j[1] for j in i]))
                        for i in zip(*gathered_serialized_tensors)
                    ],
                    load_format=None,
                    flush_cache=False,
                )
            pbar.update(len(bucket))
            total_buckets += 1

        peak_memory = torch.cuda.max_memory_allocated(rank) / 1024**3
        final_memory = torch.cuda.memory_allocated(rank) / 1024**3

        logger.info(f"Rank {rank} processed {total_buckets} buckets")
        logger.info(
            f"Rank {rank} max single bucket memory increase: {max_bucket_memory:.2f} GB"
        )
        logger.info(
            f"Rank {rank} peak memory: {peak_memory:.2f} GB (increase: {peak_memory - initial_memory:.2f} GB)"
        )
        logger.info(f"Rank {rank} final memory: {final_memory:.2f} GB")
        dist.barrier()

    async def update_weights_by_flattened_bucket(named_tensors, bucket_bytes):
        initial_memory = torch.cuda.memory_allocated(rank) / 1024**3
        torch.cuda.reset_peak_memory_stats(rank)

        logger.info(
            f"Rank {rank} start to update weights with {tensor_num=} tensors using flattened buckets, bucket_bytes={bucket_bytes} ({bucket_bytes/1024**2:.1f}MB)"
        )
        logger.info(f"Rank {rank} initial memory: {initial_memory:.2f} GB")

        pbar = tqdm.tqdm(total=tensor_num)
        total_buckets = 0
        max_bucket_memory = 0
        total_flattened_elements = 0

        for flattened_bucket in get_flattened_tensor_buckets(
            named_tensors, bucket_bytes
        ):
            # Monitor memory before processing bucket
            pre_bucket_memory = torch.cuda.memory_allocated(rank) / 1024**3

            # Get the flattened tensor and metadata
            flattened_tensor = flattened_bucket.get_flattened_tensor()
            metadata = flattened_bucket.get_metadata()

            total_flattened_elements += flattened_tensor.numel()

            # Serialize the flattened tensor and metadata
            serialized_flattened = MultiprocessingSerializer.serialize(
                flattened_tensor.detach()
            )  # ipc already
            serialized_data = {
                "flattened_tensor": serialized_flattened,
                "metadata": metadata,
            }

            # Monitor memory after serialization
            post_serialize_memory = torch.cuda.memory_allocated(rank) / 1024**3
            bucket_memory_increase = post_serialize_memory - pre_bucket_memory
            max_bucket_memory = max(max_bucket_memory, bucket_memory_increase)

            if rank == 0:
                gathered_serialized_data = [None for _ in range(world_size)]
            else:
                gathered_serialized_data = None

            dist.gather_object(
                obj=serialized_data,
                object_gather_list=gathered_serialized_data,
                dst=0,
            )

            if rank == 0:
                # Send flattened tensor + metadata to inference engine
                # Let the inference engine handle unpacking

                # Create a special entry for the flattened bucket
                bucket_name = f"__flattened_bucket_{total_buckets}"

                # Prepare the flattened tensor data with metadata
                flattened_data = {
                    "serialized_tensors": [
                        rank_data["flattened_tensor"]
                        for rank_data in gathered_serialized_data
                    ],
                    "metadata": [
                        rank_data["metadata"] for rank_data in gathered_serialized_data
                    ],
                    "bucket_id": total_buckets,
                }

                # Serialize the flattened data
                serialized_flattened_data = MultiprocessingSerializer.serialize(
                    flattened_data
                )

                # Create LocalSerializedTensor with entries for all ranks
                # Each rank gets the same serialized data since rank 0 handles all processing
                serialized_values = [
                    serialized_flattened_data for _ in range(world_size)
                ]

                # Use the existing update_weights_from_tensor method with a special format
                await inference_engine.update_weights_from_tensor(
                    named_tensors=[
                        (bucket_name, LocalSerializedTensor(values=serialized_values))
                    ],
                    load_format="flattened_bucket",
                    flush_cache=False,
                )

            pbar.update(len(flattened_bucket))
            total_buckets += 1

        peak_memory = torch.cuda.max_memory_allocated(rank) / 1024**3
        final_memory = torch.cuda.memory_allocated(rank) / 1024**3

        logger.info(f"Rank {rank} processed {total_buckets} flattened buckets")
        logger.info(
            f"Rank {rank} total flattened elements: {total_flattened_elements:,}"
        )
        logger.info(
            f"Rank {rank} max single bucket memory increase: {max_bucket_memory:.2f} GB"
        )
        logger.info(
            f"Rank {rank} peak memory: {peak_memory:.2f} GB (increase: {peak_memory - initial_memory:.2f} GB)"
        )
        logger.info(f"Rank {rank} final memory: {final_memory:.2f} GB")
        dist.barrier()

    async def benchmark_different_bucket_sizes():
        """Benchmark different bucket sizes with memory monitoring"""
        bucket_sizes = [
            (0, "per-tensor"),
            (512 * 1024 * 1024, "512MB"),  # 512MB
            # (1024 * 1024 * 1024, "1GB"),  # 1GB
            # (2 * 1024 * 1024 * 1024, "2GB"),  # 2GB - comment out large sizes to avoid OOM
        ]

        # Add flattened bucket variants
        flattened_bucket_sizes = [
            (128 * 1024 * 1024, "128MB-flat"),  # 128MB flattened
            (256 * 1024 * 1024, "512MB-flat"),  # 512MB flattened
            (512 * 1024 * 1024, "512MB-flat"),  # 512MB flattened
            (1024 * 1024 * 1024, "1GB-flat"),  # 1GB flattened
            (2 * 1024 * 1024 * 1024, "2GB-flat"),  # 2GB flattened
            (4 * 1024 * 1024 * 1024, "4GB-flat"),  # 4GB flattened
            (8 * 1024 * 1024 * 1024, "8GB-flat"),  # 8GB flattened
            (16 * 1024 * 1024 * 1024, "16GB-flat"),  # 16GB flattened
        ]
        bucket_sizes.extend(flattened_bucket_sizes)

        results = {}

        for bucket_bytes, size_name in bucket_sizes:
            if rank == 0:
                logger.info(f"\n{'='*50}")
                logger.info(f"Benchmarking {size_name} bucket size")
                logger.info(f"{'='*50}")

                # Check GPU memory before test
                memory_free = torch.cuda.memory_reserved(
                    rank
                ) - torch.cuda.memory_allocated(rank)
                logger.info(
                    f"Available GPU memory before test: {memory_free/1024**3:.2f} GB"
                )

            # Analyze bucket distribution first
            use_flattened = size_name.endswith("-flat")
            # analyze_bucket_distribution(model, bucket_bytes, rank, use_flattened)

            # Reset memory stats before each test
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(rank)

            # Record initial memory
            initial_memory = torch.cuda.memory_allocated(rank) / 1024**3

            start_time = time.time()

            named_tensors = per_tensor_generator(model)
            if bucket_bytes == 0:
                await update_weights(named_tensors)
            else:
                # Check if this is a flattened bucket variant
                if size_name.endswith("-flat"):
                    await update_weights_by_flattened_bucket(
                        named_tensors, bucket_bytes
                    )
                else:
                    await update_weights_by_bucket(named_tensors, bucket_bytes)

            end_time = time.time()
            elapsed = end_time - start_time

            # Record memory stats
            final_memory = torch.cuda.memory_allocated(rank) / 1024**3
            peak_memory = torch.cuda.max_memory_allocated(rank) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(rank) / 1024**3

            results[size_name] = {
                "time": elapsed,
                "initial_memory_gb": initial_memory,
                "final_memory_gb": final_memory,
                "peak_memory_gb": peak_memory,
                "reserved_memory_gb": memory_reserved,
                "memory_increase_gb": peak_memory - initial_memory,
            }

            if rank == 0:
                logger.info(f"{size_name} completed in {elapsed:.2f} seconds")
                logger.info(f"  Initial memory: {initial_memory:.2f} GB")
                logger.info(f"  Peak memory: {peak_memory:.2f} GB")
                logger.info(f"  Final memory: {final_memory:.2f} GB")
                logger.info(f"  Memory increase: {peak_memory - initial_memory:.2f} GB")
                logger.info(f"  Reserved memory: {memory_reserved:.2f} GB")

        if rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info("BENCHMARK RESULTS")
            logger.info(f"{'='*60}")
            logger.info(
                f"{'Method':<12} {'Time(s)':<8} {'Peak(GB)':<9} {'Increase(GB)':<12} {'Reserved(GB)':<12}"
            )
            logger.info(f"{'-'*60}")
            for size_name, stats in results.items():
                logger.info(
                    f"{size_name:<12} {stats['time']:<8.2f} {stats['peak_memory_gb']:<9.2f} "
                    f"{stats['memory_increase_gb']:<12.2f} {stats['reserved_memory_gb']:<12.2f}"
                )
            logger.info(f"{'='*60}")

            # Memory efficiency analysis
            logger.info("\nMEMORY EFFICIENCY ANALYSIS:")
            baseline_time = results["per-tensor"]["time"]
            baseline_peak = results["per-tensor"]["peak_memory_gb"]

            for size_name, stats in results.items():
                if size_name != "per-tensor":
                    time_speedup = baseline_time / stats["time"]
                    memory_overhead = stats["peak_memory_gb"] - baseline_peak
                    logger.info(
                        f"{size_name}: {time_speedup:.1f}x faster, "
                        f"{memory_overhead:+.2f}GB memory overhead"
                    )
            logger.info(f"{'='*60}")

    loop = asyncio.get_event_loop()

    if args.benchmark:
        loop.run_until_complete(benchmark_different_bucket_sizes())
    else:
        # Analyze bucket distribution for single run
        # analyze_bucket_distribution(model, args.bucket_bytes, rank, args.use_flattened_buckets)

        named_tensors = per_tensor_generator(model)
        if args.bucket_bytes == 0:
            loop.run_until_complete(update_weights(named_tensors))
        else:
            if args.use_flattened_buckets:
                loop.run_until_complete(
                    update_weights_by_flattened_bucket(named_tensors, args.bucket_bytes)
                )
            else:
                loop.run_until_complete(
                    update_weights_by_bucket(named_tensors, args.bucket_bytes)
                )

    if rank == 0:
        inference_engine.shutdown()


if __name__ == "__main__":
    main()
