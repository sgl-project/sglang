"""
Benchmark the latency of running a single static batch without a server.

This script does not launch a server and uses the low-level APIs.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

# Usage (latency test)
## with dummy weights:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --load-format dummy
## sweep through multiple data points and store (append) the results in a jsonl file:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --run-name test_run
## run with profiling:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --profile
## run with profiling to custom directory:
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 --input-len 256 --profile
## run with CUDA profiler (nsys):
nsys profile --force-overwrite=true -o bench_one_batch python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 --input-len 256 --profile --profile-activities CUDA_PROFILER
# Usage (correctness test):
python -m sglang.bench_one_batch --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --correctness-test

# Usage (multimodal/vision model testing):
## Latency test with images:
python -m sglang.bench_one_batch --model-path meta-llama/Llama-4-Scout-17B-Vision-Instruct --image-test --image-resolution 1080p --image-count 2 --batch-size 1 --input-len 512 --output-len 128
## Correctness test with images:
python -m sglang.bench_one_batch --model-path meta-llama/Llama-4-Scout-17B-Vision-Instruct --correctness-test --image-test --image-urls https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png --verify-image-understanding

## Reference output (of the correctness test above, can be gpu dependent):
input_ids=[[1, 450, 7483, 310, 3444, 338], [1, 450, 7483, 310, 278, 3303, 13187, 290, 338], [1, 20628, 338, 263, 6575, 1460, 2462, 322, 306, 763]]

prefill logits (first half): tensor([[-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [ -9.1875, -10.2500,   2.7129,  ...,  -4.3359,  -4.0664,  -4.1328]],
       device='cuda:0')

prefill logits (final): tensor([[-8.3125, -7.1172,  3.3457,  ..., -4.9570, -4.1328, -3.4141],
        [-8.9141, -9.0156,  4.1445,  ..., -4.9922, -4.4961, -4.0781],
        [-9.6328, -9.0547,  4.0195,  ..., -5.3047, -4.7148, -4.4570]],
       device='cuda:0')

========== Prompt 0 ==========
<s> The capital of France is Paris.
The capital of the United States is Washington, D.C.


========== Prompt 1 ==========
<s> The capital of the United Kindom is London.
The capital of the United Kingdom is London.
The capital of the

========== Prompt 2 ==========
<s> Today is a sunny day and I like to go for a walk in the park.
I'm going to the park
"""

import argparse
import copy
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import time
from io import BytesIO
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import requests
import torch
import torch.distributed as dist
from PIL import Image

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalInputs,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.scheduler_dp_attn_mixin import prepare_mlp_sync_batch_raw
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    is_cuda_alike,
    is_xpu,
    kill_process_tree,
    maybe_reindex_device_id,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

try:
    from sglang.srt.mem_cache.swa_memory_pool import (
        SWATokenToKVPoolAllocator as _SWAAlloc,
    )
except ImportError:
    _SWAAlloc = None


def _fix_swa_full_pool_if_empty(model_runner, rank_print=print):
    """For hybrid-SWA models whose full-attention allocator was built with size 0,
    redirect it to share the SWA allocator so bench correctness tests can allocate tokens.

    The mapping tensor is referenced by three objects via weakref.proxy:
      1. alloc.full_to_swa_index_mapping  (strong ref — owner)
      2. kvcache.full_to_swa_index_mapping (weakref.proxy registered at alloc init)
      3. xpu_backend.full_to_swa_index_mapping (copy of the weakref.proxy from kvcache)

    We must keep the new identity tensor alive AND update all three holders so no
    stale weakref.proxy remains pointing at the dead original tensor.
    """
    if _SWAAlloc is None:
        return
    alloc = getattr(model_runner, "token_to_kv_pool_allocator", None)
    if not isinstance(alloc, _SWAAlloc):
        return
    if alloc.full_attn_allocator.available_size() > 0:
        return

    rank_print(
        "[bench_one_batch] full-attention KV pool is empty (all-SWA model config). "
        "Aliasing full allocator to SWA allocator for correctness test."
    )

    # Redirect full allocator → SWA allocator.
    alloc.full_attn_allocator = alloc.swa_attn_allocator

    # Build an identity mapping: full-index → same SWA index.
    # Size must cover all SWA indices + page_size pad + sentinel at [-1].
    swa_size = alloc.swa_attn_allocator.size
    mapping_len = swa_size + alloc.page_size + 1
    identity = torch.arange(mapping_len, dtype=torch.int64, device=alloc.device)
    identity[-1] = -1  # sentinel: -1 maps to -1

    # Update strong ref on the allocator (this is what keeps the tensor alive).
    alloc.full_to_swa_index_mapping = identity

    # Update kvcache — it stores whatever was passed to register_mapping(); replace it
    # with the new strong-ref tensor (not a weakref.proxy) so it stays alive.
    kvcache = getattr(alloc, "_kvcache", None)
    if kvcache is not None and hasattr(kvcache, "full_to_swa_index_mapping"):
        kvcache.full_to_swa_index_mapping = identity

    # Update the attention backend — it captured kvcache.full_to_swa_index_mapping
    # at __init__ time; replace its copy with the live tensor.
    attn_backend = getattr(model_runner, "attn_backend", None)
    if attn_backend is not None and hasattr(attn_backend, "full_to_swa_index_mapping"):
        attn_backend.full_to_swa_index_mapping = identity


profile_activities = [torch.profiler.ProfilerActivity.CPU] + [
    profiler_activity
    for available, profiler_activity in [
        (is_cuda_alike(), torch.profiler.ProfilerActivity.CUDA),
        (is_xpu(), torch.profiler.ProfilerActivity.XPU),
    ]
    if available
]


def start_profile(profile_activities, profile_record_shapes=False, rank_print=print):
    """
    Abstracted function to start profiling based on profile_activities.
    Returns profiler object (or None).
    """
    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStart()
            rank_print("CUDA Profiler started (nsys will begin capturing)")
        except Exception as e:
            rank_print(f"Failed to start CUDA profiler: {e}")
        return None
    else:
        activities = []
        if "CPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "GPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if activities:
            profiler = torch.profiler.profile(
                activities=activities,
                with_stack=True,
                record_shapes=profile_record_shapes,
            )
            profiler.start()
            return profiler
        return None


def stop_profile(
    profiler,
    profile_activities,
    rank_print=print,
    save_trace=False,
    trace_filename=None,
    stage=None,
):
    """
    Abstracted function to stop profiling based on profile_activities.
    Optionally saves trace results and prints completion messages.
    """
    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStop()
            rank_print("CUDA Profiler stopped (nsys should dump traces)")
        except Exception as e:
            rank_print(f"Failed to stop CUDA profiler: {e}")
    elif profiler is not None:
        profiler.stop()

    if save_trace:
        if profiler is not None:
            if trace_filename:
                _save_profile_trace_results(profiler, trace_filename)
                stage_desc = f"for {stage}" if stage else ""
                rank_print(
                    f"torch profiler chrome trace {stage_desc} saved to {trace_filename}"
                )
        if "CUDA_PROFILER" in profile_activities:
            rank_print(f"CUDA profiler trace for {stage} completed")


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    prompt_filename: str = ""
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    log_decode_step: int = 0
    profile: bool = False
    profile_record_shapes: bool = False
    profile_activities: Tuple[str] = ("CPU", "GPU")
    profile_stage: str = "all"
    profile_filename_prefix: str = "profile"
    # Multimodal support
    image_test: bool = False
    image_urls: Tuple[str] = ()
    image_count: int = 1
    image_resolution: str = "1080p"
    image_content: str = "random"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--prompt-filename", type=str, default=BenchArgs.prompt_filename
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=BenchArgs.log_decode_step,
            help="Log decode latency by step, default is set to zero to disable.",
        )
        parser.add_argument("--profile", action="store_true", help="Enable profiling.")
        parser.add_argument(
            "--profile-record-shapes",
            action="store_true",
            help="Record tensor shapes in profiling results.",
        )
        parser.add_argument(
            "--profile-activities",
            type=str,
            nargs="+",
            default=["CPU", "GPU"],
            choices=["CPU", "GPU", "CUDA_PROFILER"],
            help="Profiler activities: CPU, GPU, CUDA_PROFILER. If CPU/GPU, use torch profiler. If CUDA_PROFILER, use CUDA profiler.",
        )
        parser.add_argument(
            "--profile-stage",
            type=str,
            default=BenchArgs.profile_stage,
            choices=["all", "prefill", "decode"],
            help="Which stage to profile: all, prefill, or decode only.",
        )
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=BenchArgs.profile_filename_prefix,
            help="Prefix of the profiling file names. The full profiling result file(s) be "
            '"[profile_filename_prefix]_batch[batch_size]_input[input_len]_output[output_len].trace.json.gz"',
        )
        parser.add_argument(
            "--image-test",
            action="store_true",
            help="Enable multimodal (vision) support for the benchmark.",
        )
        parser.add_argument(
            "--image-urls",
            type=str,
            nargs="+",
            default=[],
            help="List of image URLs to use for multimodal testing. Can be local file paths or HTTP URLs.",
        )
        parser.add_argument(
            "--image-count",
            "--num-images",
            dest="image_count",
            type=int,
            default=BenchArgs.image_count,
            help="Number of images per request for generated synthetic image inputs.",
        )
        parser.add_argument(
            "--image-resolution",
            type=str,
            default=BenchArgs.image_resolution,
            help=(
                "Resolution for generated synthetic images. "
                "Supports presets 4k/1080p/720p/360p or custom 'heightxwidth' (e.g., 1080x1920)."
            ),
        )
        parser.add_argument(
            "--image-content",
            type=str,
            default=BenchArgs.image_content,
            choices=["random", "blank"],
            help="Content for generated synthetic images. Supports random and blank.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def load_model(server_args, port_args, gpu_id, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )

    # Initialize multimodal embedding cache for vision models
    if model_config.is_multimodal:
        from sglang.srt.managers.mm_utils import init_mm_embedding_cache

        # Initialize with a reasonable cache size (100MB by default)
        cache_size_mb = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "100"))
        init_mm_embedding_cache(max_size=cache_size_mb * 1024 * 1024)
        rank_print(
            f"Initialized multimodal embedding cache with size {cache_size_mb}MB"
        )

    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


def load_images_from_urls(image_urls, rank_print):
    """Load images from URLs or local file paths."""
    images = []
    for url in image_urls:
        try:
            if url.startswith("http://") or url.startswith("https://"):
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                images.append(Image.open(BytesIO(response.content)))
                rank_print(f"Loaded image from URL: {url}")
            else:
                # Assume it's a local file path
                images.append(Image.open(url))
                rank_print(f"Loaded image from file: {url}")
        except Exception as e:
            rank_print(f"Failed to load image from {url}: {e}")
            raise
    return images


def parse_image_resolution(image_resolution: str) -> Tuple[int, int]:
    """Parse image resolution into (width, height).

    Supports presets '4k', '1080p', '720p', '360p' and custom
    'heightxwidth' format (e.g., '1080x1920' means height=1080, width=1920).
    """
    resolution_to_size = {
        "4k": (3840, 2160),
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "360p": (640, 360),
    }

    resolution = image_resolution.strip().lower()
    if resolution in resolution_to_size:
        return resolution_to_size[resolution]

    if "x" in resolution:
        parts = resolution.split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            height = int(parts[0])
            width = int(parts[1])
            if height > 0 and width > 0:
                return (width, height)

    raise ValueError(
        f"Unsupported image resolution: {image_resolution}. "
        "Choose from 4k, 1080p, 720p, 360p, or provide custom 'heightxwidth' (e.g., 1080x1920)."
    )


def resolve_image_size(bench_args: BenchArgs) -> Tuple[int, int]:
    """Resolve synthetic image size from --image-resolution."""
    return parse_image_resolution(bench_args.image_resolution)


def generate_random_images(
    num_images, width, height, rank_print, image_content="random"
):
    """Generate random synthetic images for latency testing.

    Args:
        num_images: Number of images to generate
        width: Image width in pixels
        height: Image height in pixels
        rank_print: Print function to use

    Returns:
        List of PIL Image objects with random RGB data
    """
    images = []
    for i in range(num_images):
        if image_content == "blank":
            image_data = np.full((height, width, 3), 255, dtype=np.uint8)
        else:
            image_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(image_data, mode="RGB")
        images.append(image)

    rank_print(
        f"Generated {num_images} {image_content} image(s) with size {width}x{height}"
    )
    return images


def process_images_to_multimodal_inputs(images, model_runner, tokenizer, input_ids):
    """
    Process PIL images into MultimodalInputs format by using the model's image processor.

    This function loads the model's processor and converts images into the format
    expected by the model's forward pass.

    Args:
        images: List of PIL Image objects
        model_runner: The ModelRunner instance
        tokenizer: The tokenizer instance
        input_ids: The token IDs that contain image placeholder tokens

    Returns:
        MultimodalInputs object
    """
    if not images:
        return None

    try:
        from transformers import AutoProcessor

        # Load the processor for the model
        processor = AutoProcessor.from_pretrained(
            model_runner.model_config.model_path, trust_remote_code=True
        )

        # Create a dummy text with image placeholders for processor
        # Llama-4 uses <|image|> as the placeholder token
        image_token_str = "<|image|>"
        dummy_text = image_token_str * len(images)

        # Process images - this creates the tensors needed by the vision encoder
        processor_output = processor(
            text=[dummy_text], images=images, return_tensors="pt"
        )

        # Move tensors to the model's device
        device = model_runner.device
        for key in processor_output:
            if isinstance(processor_output[key], torch.Tensor):
                processor_output[key] = processor_output[key].to(device)

        # Create MultimodalDataItem for the processor output
        # For Llama-4 and similar models, pixel_values contains the processed images
        if "pixel_values" not in processor_output:
            print(
                f"WARNING: pixel_values not found in processor output. Keys: {processor_output.keys()}"
            )
            return None

        # Get the image token ID
        image_token_id = None
        try:
            image_token_ids = tokenizer.encode(
                image_token_str, add_special_tokens=False
            )
            if len(image_token_ids) > 0:
                image_token_id = image_token_ids[0]
        except Exception:
            # Fallback to vocabulary lookup
            if hasattr(tokenizer, "vocab") and image_token_str in tokenizer.vocab:
                image_token_id = tokenizer.vocab[image_token_str]
            elif hasattr(tokenizer, "get_vocab"):
                vocab = tokenizer.get_vocab()
                if image_token_str in vocab:
                    image_token_id = vocab[image_token_str]

        if image_token_id is None:
            print(f"WARNING: Could not find image token ID for '{image_token_str}'")
            return None

        # Find image token positions in input_ids
        image_token_positions = [
            idx for idx, token_id in enumerate(input_ids) if token_id == image_token_id
        ]

        if not image_token_positions:
            print(f"WARNING: Image token {image_token_id} not found in input_ids")
            print(f"  Input IDs (first 20): {input_ids[:20]}")
            return None

        # For Llama-4, we need to create offsets for where images appear
        # Each image token position gets its own offset
        offsets = [(pos, pos) for pos in image_token_positions]

        # Create model-specific data dict with all processor outputs except pixel_values
        model_specific_data = {}
        for key, value in processor_output.items():
            if key != "pixel_values" and isinstance(value, torch.Tensor):
                model_specific_data[key] = value

        # Create a single MultimodalDataItem with all the images
        mm_item = MultimodalDataItem(
            modality=Modality.MULTI_IMAGES if len(images) > 1 else Modality.IMAGE,
            format=MultimodalInputFormat.PROCESSOR_OUTPUT,
            feature=processor_output["pixel_values"],
            offsets=offsets,
            model_specific_data=model_specific_data,
        )
        mm_item.set_pad_value()

        # Create MultimodalInputs object
        mm_inputs = MultimodalInputs(mm_items=[mm_item])
        mm_inputs.im_token_id = image_token_id

        print(
            f"✓ Created MultimodalInputs with {len(images)} image(s), offsets={offsets}, im_token_id={image_token_id}"
        )

        return mm_inputs

    except Exception as e:
        print(f"Error processing images: {e}")
        print("=" * 80)
        print("WARNING: Image processing failed")
        print("=" * 80)
        print()
        print("For reliable multimodal benchmarking, consider using:")
        print("1. examples/multimodal_bench_wrapper.py (uses Engine API)")
        print("2. test/srt/test_vlm_input_format.py for working examples")
        print()
        import traceback

        traceback.print_exc()
        print("=" * 80)
        return None


def prepare_inputs_for_correctness_test(
    bench_args, tokenizer, custom_prompts, images=None, model_runner=None
):
    prompts = (
        custom_prompts
        if custom_prompts
        else [
            "The capital of France is",
            "The capital of the United Kindom is",
            "Today is a sunny day and I like",
        ]
    )

    # For multimodal, use image description prompt with image tokens
    if bench_args.image_test and images:
        # For multimodal models, we need to include image placeholder tokens in the prompt
        image_token_str = "<|image|>"
        image_placeholders = " ".join([image_token_str] * len(images))
        prompts = [f"{image_placeholders} Describe the images in detail."]

    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=tmp_input_ids,
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))

        # Add multimodal inputs if images are provided
        if bench_args.image_test and images and model_runner:
            req.multimodal_inputs = process_images_to_multimodal_inputs(
                images, model_runner, tokenizer, req.origin_input_ids
            )

        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req: Req = reqs[i]
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
            i, : bench_args.cut_len
        ].to(req.prefix_indices.dtype)
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        # Release the req_pool_idx slot so the second extend() treats every
        # request as a fresh allocation.  The actual KV-cache data is already
        # captured in prefix_indices (indices into token_to_kv_pool), so
        # freeing the row in req_to_token_pool does not lose any KV state.
        if req.req_pool_idx is not None:
            model_runner.req_to_token_pool.free(req)
            req.kv_committed_len = 0
    return reqs


def prepare_synthetic_inputs_for_latency_test(
    batch_size,
    input_len,
    custom_inputs=None,
    images=None,
    model_runner=None,
    tokenizer=None,
):

    # For multimodal tests, we need to create input sequences with image placeholder tokens
    image_token_positions = []
    if images and model_runner and tokenizer:
        # Get the image placeholder token for this model
        # Llama-4 uses <|image|> token
        image_token = "<|image|>"

        # Create a simple prompt with image placeholders
        prompt_template = f"{image_token} Describe this image in detail."

        # Encode the prompt to get the token IDs
        encoded = tokenizer.encode(prompt_template)

        # Find where the image token appears in the encoded sequence
        # Encode just the image token to find its ID
        image_token_ids = tokenizer.encode(image_token, add_special_tokens=False)

        print(f"image_token_str='{image_token}', image_token_ids={image_token_ids}")
        print(f"Full prompt: '{prompt_template}'")
        print(f"Encoded prompt: {encoded[:20]}... (showing first 20)")

        if len(image_token_ids) > 0:
            image_token_id = image_token_ids[0]
            # Find all positions of the image token
            positions = [
                idx
                for idx, token_id in enumerate(encoded)
                if token_id == image_token_id
            ]
            print(f"Found image_token_id {image_token_id} at positions: {positions}")
            if positions:
                # For simplicity, use the first image token position
                # In the real case, there might be multiple images
                image_token_positions = [(positions[0], positions[0])]

        if not image_token_positions:
            print(f"WARNING: Could not find image token in encoded sequence!")
            print(f"  This may cause issues with multimodal processing")

        # If the encoded prompt is shorter than input_len, pad with random tokens
        if len(encoded) < input_len:
            # Pad with random tokens (excluding special tokens)
            padding_tokens = np.random.randint(
                100, 10000, input_len - len(encoded), dtype=np.int32
            ).tolist()
            encoded = encoded + padding_tokens
        elif len(encoded) > input_len:
            # Truncate if too long
            encoded = encoded[:input_len]

        # Create the same input for all requests in the batch
        input_ids = [encoded] * batch_size
    else:
        # Non-multimodal: use random tokens or custom inputs
        input_ids = (
            custom_inputs
            if custom_inputs
            else np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
        )

    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    # For multimodal batch processing, create multimodal inputs once and share across all requests
    # This ensures consistent pad_value usage and avoids duplication
    shared_multimodal_inputs = None
    if images and model_runner and tokenizer:
        # Create multimodal inputs once using the first request's input_ids as reference
        shared_multimodal_inputs = process_images_to_multimodal_inputs(
            images, model_runner, tokenizer, list(input_ids[0])
        )
        if shared_multimodal_inputs:
            print(
                f"✓ Created shared multimodal inputs for batch of {len(input_ids)} requests"
            )

            # Important: Call pad_input_ids to replace image tokens with pad_values
            # This is required for the mask to find image tokens during forward pass
            if hasattr(model_runner.model, "pad_input_ids"):
                # Update the input_ids to use pad_values instead of image token IDs
                padded_input_ids = model_runner.model.pad_input_ids(
                    list(input_ids[0]), shared_multimodal_inputs
                )
                # Update all input_ids in the batch with the padded version
                input_ids = [padded_input_ids] * len(input_ids)
                print(f"✓ Applied pad_input_ids: {len(input_ids[0])} tokens")

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))

        # Share the same multimodal inputs across all requests in the batch
        if shared_multimodal_inputs:
            req.multimodal_inputs = shared_multimodal_inputs

        reqs.append(req)

    return reqs


class TreeCacheNamespace(SimpleNamespace):
    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()

    def evict(self, params):
        return None

    def pretty_print(self):
        return None

    def evictable_size(self):
        return 0

    def full_evictable_size(self):
        return 0

    def swa_evictable_size(self):
        return 0

    def full_lru_list_evictable_size(self):
        return 0

    def swa_lru_list_evictable_size(self):
        return 0


@torch.no_grad
def extend(reqs, model_runner):
    # Create dummy tree_cache for benchmarks (no prefix caching, just allocation)
    dummy_tree_cache = TreeCacheNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )

    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output = model_runner.forward(forward_batch).logits_output
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def decode(input_token_ids, batch, model_runner):
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output = model_runner.forward(forward_batch).logits_output
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits


def _maybe_prepare_mlp_sync_batch(batch: ScheduleBatch, model_runner):
    if require_mlp_sync(model_runner.server_args):
        prepare_mlp_sync_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=1,
            tp_group=model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
            offload_tags=set(),
        )


def _read_prompts_from_file(prompt_file, rank_print):
    """Read custom prompts from the file specified by `--prompt-filename`."""
    if not prompt_file:
        return []
    if not os.path.exists(prompt_file):
        rank_print(
            f"Custom prompt file {prompt_file} not found. Using default inputs..."
        )
        return []
    with open(prompt_file, "r") as pf:
        return pf.readlines()


def _get_torch_profiler_output_dir():
    return os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp")


def _create_torch_profiler_filename(
    profile_filename_prefix, batch_size, input_len, output_len, stage
):
    output_dir = _get_torch_profiler_output_dir()
    filename = f"{profile_filename_prefix}_batch{batch_size}_input{input_len}_output{output_len}_{stage}.trace.json.gz"
    return os.path.join(output_dir, filename)


def _save_profile_trace_results(profiler, filename):
    parent_dir = os.path.dirname(os.path.abspath(filename))
    os.makedirs(parent_dir, exist_ok=True)
    profiler.export_chrome_trace(filename)
    print(
        profiler.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total"
        )
    )


def correctness_test(
    server_args,
    port_args,
    bench_args,
    gpu_id,
    tp_rank,
):
    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)

    # For hybrid-SWA models whose full-attention KV pool ended up with size 0
    # (e.g. all-SWA or reduced model configs), alias the full allocator to the SWA
    # allocator so correctness-test token allocations can succeed.
    _fix_swa_full_pool_if_empty(model_runner, rank_print)

    # Load or generate images if multimodal is enabled
    images = None
    if bench_args.image_test:
        if bench_args.image_urls:
            # Load real images from URLs/files for correctness test
            rank_print(
                f"Loading {len(bench_args.image_urls)} images for multimodal test..."
            )
            images = load_images_from_urls(bench_args.image_urls, rank_print)
            rank_print(f"Loaded {len(images)} images successfully")
        else:
            width, height = resolve_image_size(bench_args)
            rank_print(
                f"Generating {bench_args.image_count} synthetic image(s) at {width}x{height}"
            )
            images = generate_random_images(
                bench_args.image_count,
                width,
                height,
                rank_print,
                image_content=bench_args.image_content,
            )

    # Prepare inputs
    custom_prompts = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
    input_ids, reqs = prepare_inputs_for_correctness_test(
        bench_args, tokenizer, custom_prompts, images=images, model_runner=model_runner
    )
    rank_print(f"\n{input_ids=}\n")

    if bench_args.cut_len > 0:
        # Prefill
        next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
        rank_print(f"prefill logits (first half): {next_token_logits} \n")

    # Prepare extend inputs
    reqs = prepare_extend_inputs_for_correctness_test(
        bench_args, input_ids, reqs, model_runner
    )

    # Extend (prefill w/ KV cache)
    next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
    rank_print(f"prefill logits (final): {next_token_logits} \n")

    # Decode
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    for _ in range(bench_args.output_len[0] - 1):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])

    # Print output texts
    for i in range(len(reqs)):
        rank_print(f"========== Prompt {i} ===========")
        output_text = tokenizer.decode(output_ids[i])
        rank_print(output_text, "\n")


def synchronize(device):
    torch.get_device_module(device).synchronize()


def latency_test_run_once(
    run_name,
    model_runner,
    rank_print,
    reqs,
    batch_size,
    input_len,
    output_len,
    device,
    log_decode_step,
    profile,
    profile_record_shapes,
    profile_activities,
    profile_filename_prefix,
    profile_stage,
    tp_rank,
):
    max_batch_size = model_runner.max_total_num_tokens // (input_len + output_len)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping ({batch_size}, {input_len}, {output_len}) due to max batch size limit"
        )
        return

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    tot_latency = 0

    profiler = None
    enable_profile_prefill = profile and profile_stage in ["all", "prefill"]
    if enable_profile_prefill:
        profiler = start_profile(
            profile_activities,
            profile_record_shapes=profile_record_shapes,
            rank_print=rank_print,
        )

    synchronize(device)
    tic = time.perf_counter()
    next_token_ids, _, batch = extend(reqs, model_runner)
    synchronize(device)
    prefill_latency = time.perf_counter() - tic

    if enable_profile_prefill:
        trace_filename = _create_torch_profiler_filename(
            profile_filename_prefix, batch_size, input_len, output_len, "prefill"
        )
        stop_profile(
            profiler,
            profile_activities,
            rank_print=rank_print,
            save_trace=True,
            trace_filename=trace_filename,
            stage="prefill",
        )

    tot_latency += prefill_latency
    throughput = input_len * batch_size / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput

    decode_latencies = []
    profile_step_of_interest = output_len // 2
    enable_profile_decode = profile and profile_stage in ["all", "decode"]
    for i in range(output_len - 1):
        synchronize(device)
        profiler = None
        if enable_profile_decode and i == profile_step_of_interest:
            profiler = start_profile(
                profile_activities,
                profile_record_shapes=profile_record_shapes,
                rank_print=rank_print,
            )

        tic = time.perf_counter()
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        synchronize(device)
        latency = time.perf_counter() - tic

        if enable_profile_decode and i == profile_step_of_interest:
            trace_filename = _create_torch_profiler_filename(
                profile_filename_prefix, batch_size, input_len, output_len, "decode"
            )
            stop_profile(
                profiler,
                profile_activities,
                rank_print=rank_print,
                save_trace=True,
                trace_filename=trace_filename,
                stage="decode",
            )

        tot_latency += latency
        throughput = batch_size / latency
        decode_latencies.append(latency)
        if i < 5 or (log_decode_step > 0 and i % log_decode_step == 0):
            rank_print(
                f"Decode {i}. Batch size: {batch_size}, latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
            )

    # Record decode timing from 2nd output
    if output_len > 1:
        med_decode_latency = np.median(decode_latencies)
        med_decode_throughput = batch_size / med_decode_latency
        rank_print(
            f"Decode.  median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    throughput = (input_len + output_len) * batch_size / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput
    return measurement_results


def latency_test(
    server_args,
    port_args,
    bench_args,
    gpu_id,
    tp_rank,
):
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)

    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, tp_rank
        )

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)

    # For hybrid-SWA models whose full-attention KV pool ended up with size 0
    # (e.g. all-SWA or reduced model configs), alias the full allocator to the SWA
    # allocator so latency-test token allocations can succeed.
    _fix_swa_full_pool_if_empty(model_runner, rank_print)

    # Load or generate images if multimodal is enabled
    images = None
    if bench_args.image_test:
        if bench_args.image_urls:
            # Load real images from URLs/files
            rank_print(
                f"Loading {len(bench_args.image_urls)} images for multimodal test..."
            )
            images = load_images_from_urls(bench_args.image_urls, rank_print)
            rank_print(f"Loaded {len(images)} images successfully")
        else:
            width, height = resolve_image_size(bench_args)
            rank_print(
                f"Generating {bench_args.image_count} synthetic image(s) at {width}x{height} for latency test..."
            )
            images = generate_random_images(
                bench_args.image_count,
                width,
                height,
                rank_print,
                image_content=bench_args.image_content,
            )

    # Prepare inputs for warm up
    reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.batch_size[0],
        bench_args.input_len[0],
        images=images,
        model_runner=model_runner,
        tokenizer=tokenizer,
    )

    # Warm up
    rank_print("Warmup ...")
    latency_test_run_once(
        bench_args.run_name,
        model_runner,
        rank_print,
        reqs,
        bench_args.batch_size[0],
        bench_args.input_len[0],
        min(32, bench_args.output_len[0]),  # shorter decoding to speed up the warmup
        server_args.device,
        log_decode_step=0,
        profile=False,
        profile_record_shapes=False,
        profile_activities=("CPU", "GPU"),
        profile_filename_prefix="",
        profile_stage="all",
        tp_rank=tp_rank,
    )

    rank_print("Benchmark ...")

    custom_inputs = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
    custom_inputs = [tokenizer.encode(p.strip()) for p in custom_inputs]
    custom_input_len = len(custom_inputs)

    # Run the sweep
    result_list = []
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        bs_aligned_inputs = []
        if custom_inputs:
            if custom_input_len == bs:
                bs_aligned_inputs = custom_inputs
            elif custom_input_len > bs:
                rank_print(
                    f"Custom input size ({custom_input_len}) is larger than batch_size ({bs}). "
                    f"Using the first {bs} prompts."
                )
                bs_aligned_inputs = copy.deepcopy(custom_inputs[:bs])
            else:
                rank_print(
                    f"Custom input size ({custom_input_len}) is smaller than batch_size ({bs}). "
                    f"Pad to the desired batch_size with the last prompt."
                )
                bs_aligned_inputs = copy.deepcopy(custom_inputs)
                bs_aligned_inputs.extend(
                    [bs_aligned_inputs[-1]] * (bs - custom_input_len)
                )

        reqs = prepare_synthetic_inputs_for_latency_test(
            bs,
            il,
            bs_aligned_inputs,
            images=images,
            model_runner=model_runner,
            tokenizer=tokenizer,
        )
        ret = latency_test_run_once(
            bench_args.run_name,
            model_runner,
            rank_print,
            reqs,
            bs,
            il,
            ol,
            server_args.device,
            bench_args.log_decode_step,
            bench_args.profile if tp_rank == 0 else None,
            bench_args.profile_record_shapes if tp_rank == 0 else None,
            bench_args.profile_activities,
            bench_args.profile_filename_prefix,
            bench_args.profile_stage,
            tp_rank,
        )
        if ret is not None:
            result_list.append(ret)

    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")

    if server_args.tp_size > 1:
        destroy_distributed_environment()


def main(server_args, bench_args):
    server_args.cuda_graph_max_bs = max(bench_args.batch_size)

    # bench_one_batch --image-test implies multimodal execution.
    # Keep explicit user choice intact if --enable-multimodal is provided.
    if bench_args.image_test and server_args.enable_multimodal is None:
        server_args.enable_multimodal = True

    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            with maybe_reindex_device_id(tp_rank) as gpu_id:
                proc = multiprocessing.Process(
                    target=work_func,
                    args=(
                        server_args,
                        port_args,
                        bench_args,
                        gpu_id,
                        tp_rank,
                    ),
                )
                proc.start()
                workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
