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
from transformers import AutoModelForCausalLM
from verl.workers.rollout.sglang_rollout.sglang_rollout import AsyncEngine

from sglang.srt.model_executor.model_runner import LocalSerializedTensor, FlattenedTensorBucket, FlattenedTensorMetadata
from sglang.srt.utils import MultiprocessingSerializer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--profile_dir", type=str, default="/tmp/verl_sglang_profile/")
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
) -> Iterator[FlattenedTensorBucket]:
    """
    Group tensors into flattened buckets based on a specified size in bytes.

    Args:
        iterable: An iterator of tuples containing tensor names and tensors.
        bucket_bytes: The maximum size of each bucket in bytes.

    Yields:
        FlattenedTensorBucket objects containing flattened tensors with metadata.

    Example:
        >>> tensors = [('tensor1', torch.randn(1000, 1000)), ('tensor2', torch.randn(2000, 2000))]
        >>> for bucket in get_flattened_tensor_buckets(tensors, bucket_bytes=10*1024*1024):
        ...     print(bucket)
        FlattenedTensorBucket(num_tensors=2, num_elements=5000000, size=20.00MB)
    """
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    current_bucket = []
    current_size = 0

    for name, tensor in iterable:
        tensor_size = tensor.element_size() * tensor.numel()
        if current_size + tensor_size > bucket_bytes:
            if current_bucket:
                yield FlattenedTensorBucket(named_tensors=current_bucket)
            current_bucket = [(name, tensor)]
            current_size = tensor_size
        else:
            current_bucket.append((name, tensor))
            current_size += tensor_size

    if current_bucket:
        yield FlattenedTensorBucket(named_tensors=current_bucket)


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
        # Call the original method for other formats
        res = ori_update(self, named_tensors, load_format)
        prof.step()
        return res

    ModelRunner.update_weights_from_tensor = update_with_profiler


if args.profile_dir:
    patch_profiler()


def per_tensor_generator(model) -> Iterable[Tuple[str, torch.Tensor]]:
    """Generate tensors one by one, moving to GPU only when needed"""
    for name, tensor in model.named_parameters():
        yield name, tensor.cuda()


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
            # post_serialize_memory = torch.cuda.memory_allocated(rank) / 1024**3
            # bucket_memory_increase = post_serialize_memory - pre_bucket_memory
            # max_bucket_memory = max(max_bucket_memory, bucket_memory_increase)

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

        # logger.info(f"Rank {rank} processed {total_buckets} flattened buckets")
        # logger.info(
        #     f"Rank {rank} total flattened elements: {total_flattened_elements:,}"
        # )
        # # logger.info(
        # #     f"Rank {rank} max single bucket memory increase: {max_bucket_memory:.2f} GB"
        # # )
        # logger.info(
        #     f"Rank {rank} peak memory: {peak_memory:.2f} GB (increase: {peak_memory - initial_memory:.2f} GB)"
        # )
        # logger.info(f"Rank {rank} final memory: {final_memory:.2f} GB")
        dist.barrier()

    async def benchmark_different_bucket_sizes():
        """Benchmark different bucket sizes with memory monitoring"""
        bucket_sizes = [
            # (0, "per-tensor"),
            # (512 * 1024 * 1024, "512MB"),  # 512MB
            #(1024 * 1024 * 1024, "1GB"),  # 1GB
        ]

        # Add flattened bucket variants
        flattened_bucket_sizes = [
            # (512 * 1024 * 1024, "512MB-flat"),  # 512MB flattened
            (1024 * 1024 * 1024, "1GB-flat"),  # 1GB flattened
            #(2 * 1024 * 1024 * 1024, "2GB-flat"),  # 2GB flattened
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

            # Reset memory stats before each test
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(rank)

            # Record initial memory
            initial_memory = torch.cuda.memory_allocated(rank) / 1024**3

            start_time = time.time()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile_dir),
                profile_memory=True,
                with_stack=True,
                record_shapes=True,
            ):
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
