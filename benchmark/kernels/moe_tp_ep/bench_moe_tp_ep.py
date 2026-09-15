"""Compare the real Qwen3 MoE block at equal global token counts on Ascend.

TP receives replicated inputs; EP receives disjoint token shards. Timing covers
router, experts and reduction/combine, but excludes input-layout conversion and
output gathering. This is an eager module benchmark, not serving latency.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from workload import make_input


def build_block(model_path, ep_size, backend, deepep_mode, random_init=False):
    """Construct the MoE block with the global state the model code expects."""
    import torch_npu  # noqa: F401
    from transformers import AutoConfig

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import (
        get_moe_expert_parallel_rank,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.eplb.expert_location import (
        compute_initial_expert_location_metadata,
        set_global_expert_location_metadata,
    )
    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if (backend == "none" and ep_size != 1) or (
        backend == "deepep" and (ep_size != world or world < 2)
    ):
        raise ValueError("Use EP=1 for none, or EP=WORLD_SIZE>=2 for deepep")
    torch.npu.set_device(local_rank)

    server_args = ServerArgs(
        model_path=model_path,
        tp_size=world,
        ep_size=ep_size,
        moe_a2a_backend=backend,
        deepep_mode=deepep_mode,
        device="npu",
        attention_backend="ascend",
        trust_remote_code=True,
    )
    # The MoE layer reads global server args during construction.
    set_global_server_args_for_scheduler(server_args)
    # initialize_moe_config() dropped its argument once the MoE flags moved into
    # the published configuration; accept either form so the benchmark runs
    # against both current main and slightly older checkouts.
    if inspect.signature(initialize_moe_config).parameters:
        initialize_moe_config(server_args)
    else:
        initialize_moe_config()

    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        backend="hccl",
        moe_a2a_backend=backend,
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world,
        expert_model_parallel_size=server_args.ep_size,
    )

    # The DeepEP path looks up global expert location metadata, which ModelRunner
    # normally installs. Build the trivial layout here for the same purpose.
    model_config = ModelConfig.from_server_args(server_args)
    # Same story here: server_args was removed from the signature, so dispatch on
    # what the installed version actually accepts.
    meta_params = inspect.signature(compute_initial_expert_location_metadata).parameters
    if "server_args" in meta_params:
        metadata = compute_initial_expert_location_metadata(
            server_args, model_config, get_moe_expert_parallel_rank()
        )
    else:
        metadata = compute_initial_expert_location_metadata(
            model_config, get_moe_expert_parallel_rank()
        )
    set_global_expert_location_metadata(metadata)

    from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with torch.device("npu"):
        block = Qwen3MoeSparseMoeBlock(
            layer_id=0, config=config, quant_config=None, prefix="bench"
        )
    block = block.to(torch.bfloat16).eval()

    if random_init:
        raise ValueError("Load checkpoint weights for a matched TP/EP comparison")
    return block, config, server_args


def finalize_weights(block):
    """Run each module's post-load weight processing.

    This must happen *after* weights are loaded. It rewrites MoE parameters into
    the NPU layout, and running it first would leave ``weight_loader`` sharding
    along the wrong dimension.
    """
    for module in block.modules():
        qm = getattr(module, "quant_method", None)
        if qm is not None and hasattr(qm, "process_weights_after_loading"):
            qm.process_weights_after_loading(module)


def load_real_weights(block, config, model_path, layer_id=0, skew_experts=0):
    """Load one layer's real MoE weights through SGLang's own weight loader.

    Both configurations start from the same checkpoint tensors and let
    ``FusedMoE.weight_loader`` shard them, so pure TP (which splits each expert's
    intermediate dimension) and TP+EP (which splits the expert set) end up holding
    the same logical weights.

    ``skew_experts`` scales up the router rows of the first N experts to bias the
    routing distribution. Note this skews the load but does not by itself create
    zero-token experts.
    """
    from safetensors import safe_open

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    with open(
        os.path.join(model_path, "model.safetensors.index.json"), encoding="utf-8"
    ) as f:
        weight_map = json.load(f)["weight_map"]
    handles = {}
    weight_hash = hashlib.sha256()

    def read(name):
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(model_path, shard), framework="pt")
        tensor = handles[shard].get_tensor(name)
        weight_hash.update(name.encode())
        weight_hash.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        return tensor

    params = dict(block.named_parameters())
    prefix = f"model.layers.{layer_id}.mlp"

    gate_w = read(f"{prefix}.gate.weight").float()
    if skew_experts > 0:
        gate_w[:skew_experts] *= 50.0
        gate_w[skew_experts:] *= 0.01
    with torch.no_grad():
        params["gate.weight"].copy_(gate_w.to(params["gate.weight"].dtype))

    mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=config.num_experts,
    )
    for param_name, weight_name, expert_id, shard_id in mapping:
        proj = weight_name.split(".")[-2]
        target = f"experts.{expert_id}.{proj}.weight".replace(weight_name, param_name)
        if target not in params:
            continue
        params[target].weight_loader(
            params[target],
            read(f"{prefix}.experts.{expert_id}.{proj}.weight"),
            target,
            shard_id=shard_id,
            expert_id=expert_id,
        )
    block.benchmark_weight_sha256 = weight_hash.hexdigest()


def make_forward_batch(num_tokens, is_extend=True):
    """Minimal stand-in for the few ForwardBatch fields the MoE path reads."""
    from sglang.srt.layers.dp_attention import set_is_extend_in_batch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    # DeepEP picks normal vs low-latency from this phase flag, which the real
    # runtime sets before every forward.
    set_is_extend_in_batch(is_extend)

    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND if is_extend else ForwardMode.DECODE,
        num_token_non_padded=torch.tensor(num_tokens, dtype=torch.int32, device="npu"),
    )


@torch.inference_mode()
def time_forward(block, hidden, fb, warmup, iters):
    for _ in range(warmup):
        block(hidden, fb)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        dist.barrier()
        torch.npu.synchronize()
        wall_start = time.perf_counter()
        start.record()
        block(hidden, fb)
        end.record()
        torch.npu.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000
        samples.append([start.elapsed_time(end), wall_ms])
    local = torch.tensor(samples, dtype=torch.float32, device=hidden.device)
    ranks = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(ranks, local)
    per_rank = torch.stack(ranks).cpu()
    return per_rank.max(dim=0).values.tolist(), per_rank.tolist()


def environment_metadata():
    import sglang

    source = Path(sglang.__file__).resolve()
    root = source.parents[2]

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    packages = {}
    for name in ("torch", "torch-npu", "deep-ep", "sglang"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    script_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    source_diff = git("diff", "HEAD")
    return dict(
        packages=packages,
        sglang_commit=git("rev-parse", "HEAD")
        or os.environ.get("BENCHMARK_SGLANG_COMMIT"),
        sglang_diff_sha256=(
            hashlib.sha256(source_diff.encode()).hexdigest()
            if source_diff is not None
            else None
        ),
        scripts_sha256=script_hashes,
        environment={
            key: value
            for key, value in os.environ.items()
            if key.startswith(("DEEPEP_", "HCCL_", "SGLANG_DEEPEP_"))
            or key in ("ASCEND_CUSTOM_OPP_PATH", "DEEP_USE_MODE")
        },
        execution_mode="torch.inference_mode; eager",
        image_digest=os.environ.get("BENCHMARK_IMAGE_DIGEST"),
    )


@torch.inference_mode()
def profile_forward(block, hidden, fb, iters, output_dir):
    import torch_npu

    torch.npu.synchronize()
    dist.barrier()
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
    ) as prof:
        for _ in range(iters):
            block(hidden, fb)
            torch.npu.synchronize()
            prof.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--backend", default="none", choices=["none", "deepep"])
    parser.add_argument("--deepep-mode", default="auto")
    parser.add_argument(
        "--phase",
        default="prefill",
        choices=["prefill", "decode"],
        help="prefill exercises the DeepEP normal path, decode the low-latency path",
    )
    parser.add_argument("--global-tokens", default="128,512,2048,4096,8192")
    parser.add_argument("--input-seed", type=int, default=20260922)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeat-id", type=int, default=0)
    parser.add_argument("--profile-dir")
    parser.add_argument("--profile-iters", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    counts = [int(n) for n in args.global_tokens.split(",")]
    if min(counts) < 1 or args.iters < 1 or args.warmup < 0:
        parser.error(
            "token counts and iters must be positive; warmup must be nonnegative"
        )
    if args.profile_dir and len(counts) != 1:
        parser.error("profile one global token count per process")

    block, config, sa = build_block(
        args.model_path, args.ep_size, args.backend, args.deepep_mode
    )
    load_real_weights(block, config, args.model_path)
    finalize_weights(block)
    result = dict(
        schema_version=2,
        backend=args.backend,
        tp_size=sa.tp_size,
        ep_size=sa.ep_size,
        phase=args.phase,
        real_weights=True,
        repeat_id=args.repeat_id,
        input_seed=args.input_seed,
        warmup=args.warmup,
        iters=args.iters,
        num_experts=config.num_experts,
        topk=config.num_experts_per_tok,
        hidden=config.hidden_size,
        weights_sha256=block.benchmark_weight_sha256,
        metadata=environment_metadata(),
        results=[],
    )
    for n in counts:
        hidden, per_rank_counts, input_hash = make_input(
            n, config.hidden_size, args.backend, args.input_seed, torch.device("npu")
        )
        fb = make_forward_batch(hidden.shape[0], is_extend=args.phase == "prefill")
        samples, rank_samples = time_forward(block, hidden, fb, args.warmup, args.iters)
        device_ms = [s[0] for s in samples]
        median = statistics.median(device_ms)
        result["results"].append(
            dict(
                global_tokens=n,
                tokens_per_rank=per_rank_counts,
                input_sha256=input_hash,
                median_ms=median,
                p95_ms=sorted(device_ms)[
                    min(len(device_ms) - 1, int(len(device_ms) * 0.95))
                ],
                wall_median_ms=statistics.median(s[1] for s in samples),
                global_tokens_per_s=n * 1000 / median,
                samples_max_rank_ms=samples,
                samples_per_rank_ms=rank_samples,
            )
        )
        if dist.get_rank() == 0:
            print(
                f"global_tokens={n} local_tokens={per_rank_counts} median_ms={median:.4f}",
                flush=True,
            )
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        if args.profile_dir:
            profile_forward(block, hidden, fb, args.profile_iters, args.profile_dir)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
