"""Synthetic full-model Qwen4-Exp GDN runtime benchmark.

This intentionally uses a tiny random Qwen4-Exp config so it can exercise the
native SGLang model, RadixLinearAttention, GDNAttnBackend, and HybridReqToTokenPool
without downloading model weights.
"""

from __future__ import annotations

import argparse
import os
import socket
import time
import types
from dataclasses import dataclass

import torch

from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.utils import resolve_linear_attn_backends
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position_torch,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpTextSparseMoeBlock,
)
from sglang.srt.runtime_context import get_context, get_parallel


@dataclass(frozen=True)
class RuntimeBundle:
    model: Qwen4ExpForConditionalGeneration
    backend: HybridLinearAttnBackend
    req_pool: HybridReqToTokenPool


@dataclass(frozen=True)
class DistributedEnv:
    rank: int
    world_size: int
    local_rank: int


def _parse_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _single_rank_init_method() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _distributed_env() -> DistributedEnv:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank if world_size > 1 else 0)))
    return DistributedEnv(rank=rank, world_size=world_size, local_rank=local_rank)


def _select_device(requested_device: str, dist_env: DistributedEnv) -> str:
    if not requested_device.startswith("cuda"):
        return requested_device
    if not torch.cuda.is_available():
        return requested_device
    if ":" in requested_device:
        torch.cuda.set_device(torch.device(requested_device))
        return requested_device
    device_index = dist_env.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_index)
    return f"cuda:{device_index}"


def _ensure_distributed(device: str, dist_env: DistributedEnv, tp_size: int) -> None:
    if not device.startswith("cuda"):
        return
    if tp_size > 1 and dist_env.world_size != tp_size:
        raise ValueError(
            f"tp_size={tp_size} requires WORLD_SIZE={tp_size}; "
            f"got WORLD_SIZE={dist_env.world_size}. Use torchrun for TP screens."
        )

    from sglang.srt.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    if not torch.distributed.is_initialized():
        init_method = (
            "env://" if dist_env.world_size > 1 else _single_rank_init_method()
        )
        init_distributed_environment(
            world_size=dist_env.world_size,
            rank=dist_env.rank,
            local_rank=dist_env.local_rank,
            distributed_init_method=init_method,
            backend="nccl",
        )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=tp_size,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        backend="nccl",
    )


def _destroy_single_rank_distributed(device: str) -> None:
    if not device.startswith("cuda") or not torch.distributed.is_initialized():
        return

    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()


def _layer_types(num_layers: int, full_attention_interval: int) -> list[str]:
    if full_attention_interval <= 0:
        return ["linear_attention"] * num_layers
    return [
        (
            "qwen_sparse_attention"
            if (idx + 1) % full_attention_interval == 0
            else "linear_attention"
        )
        for idx in range(num_layers)
    ]


def _build_config(args: argparse.Namespace) -> Qwen4ExpConfig:
    layer_types = _layer_types(args.num_layers, args.full_attention_interval)
    ple_layer_ids = []
    if args.enable_ple:
        ple_layer_ids = [
            layer_idx + 1
            for layer_idx, layer_type in enumerate(layer_types)
            if layer_type == "linear_attention"
        ][: args.num_ple_layers]
    return Qwen4ExpConfig(
        text_config={
            "vocab_size": args.vocab_size,
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_layers,
            "num_attention_heads": args.num_attention_heads,
            "num_key_value_heads": args.num_key_value_heads,
            "head_dim": args.head_dim,
            "linear_key_head_dim": args.linear_head_dim,
            "linear_value_head_dim": args.linear_head_dim,
            "linear_num_key_heads": args.linear_num_key_heads,
            "linear_num_value_heads": args.linear_num_value_heads,
            "linear_conv_kernel_dim": args.linear_conv_kernel_dim,
            "moe_intermediate_size": args.moe_intermediate_size,
            "shared_expert_intermediate_size": args.shared_expert_intermediate_size,
            "num_experts": args.num_experts,
            "num_experts_per_tok": args.num_experts_per_tok,
            "hc_count": args.hc_count,
            "hc_lowrank": args.hc_lowrank,
            "layer_types": layer_types,
            "max_position_embeddings": max(args.seq_len * 2, 256),
            "eos_token_id": 2,
            "dtype": args.dtype,
            "mamba_ssm_dtype": args.mamba_ssm_dtype,
            "ple_layer_ids": ple_layer_ids,
            "ple_embed_dim": args.ple_embed_dim or args.hidden_size,
            "ple_conv_kernel_size": args.ple_conv_kernel_size,
            "ngram_size": args.ngram_size,
            "heads_per_ngram": args.heads_per_ngram,
            "ngram_vocab_size_base": args.ngram_vocab_size_base,
            "make_ngram_vocab_size_divisible_by": args.make_ngram_vocab_size_divisible_by,
            "split_ngram_parts": args.split_ngram_parts,
        },
        vision_config={
            "model_type": "qwen4_exp",
            "hidden_size": args.hidden_size,
            "out_hidden_size": args.hidden_size,
        },
    )


class _UnusedFullAttentionBackend:
    needs_cpu_seq_lens = False
    supports_ragged_verify_graph = True
    max_context_len = 0

    def __init__(self, req_pool: HybridReqToTokenPool, dtype: torch.dtype):
        self.req_to_token_pool = req_pool
        self.token_to_kv_pool = None
        self.data_type = dtype

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        del forward_batch

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ) -> None:
        del forward_batch, in_capture

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        del forward_batch

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 1

    def get_cpu_graph_seq_len_fill_value(self) -> int:
        return 1

    def forward_extend(self, *args, **kwargs):
        raise RuntimeError("Qwen4-Exp benchmark full-attention path is model-local.")

    def forward_decode(self, *args, **kwargs):
        raise RuntimeError("Qwen4-Exp benchmark full-attention path is model-local.")


def _build_runtime(
    config: Qwen4ExpConfig, dtype: torch.dtype, device: str, max_reqs: int
) -> RuntimeBundle:
    model = (
        Qwen4ExpForConditionalGeneration(config).to(device=device, dtype=dtype).eval()
    )
    req_pool = HybridReqToTokenPool(
        size=max_reqs,
        mamba_size=max_reqs,
        mamba_spec_state_size=max_reqs,
        max_context_len=config.text_config.max_position_embeddings,
        device=device,
        enable_memory_saver=False,
        cache_params=config.text_config.mamba2_cache_params,
        mamba_layer_ids=config.text_config.linear_layer_ids,
        enable_mamba_extra_buffer=False,
    )
    req_pool.req_index_to_mamba_index_mapping[:max_reqs] = torch.arange(
        1, max_reqs + 1, device=device, dtype=torch.int32
    )
    runner = types.SimpleNamespace(
        device=torch.device(device),
        is_draft_worker=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool=None,
        server_args=types.SimpleNamespace(enable_unified_memory=False),
        linear_attn_backends=resolve_linear_attn_backends(),
    )
    backend = HybridLinearAttnBackend(
        _UnusedFullAttentionBackend(req_pool, dtype),
        GDNAttnBackend(runner),
        config.text_config.full_attention_layer_ids,
    )
    return RuntimeBundle(model=model, backend=backend, req_pool=req_pool)


def _make_forward_batch(
    config: Qwen4ExpConfig,
    batch_size: int,
    seq_len: int,
    input_ids: torch.Tensor,
    device: str,
) -> ForwardBatch:
    total_tokens = batch_size * seq_len
    seq_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    positions, extend_start_loc = compute_position_torch(
        torch.zeros_like(seq_lens), seq_lens
    )
    return ForwardBatch(
        batch_size=batch_size,
        input_ids=input_ids,
        req_pool_indices=torch.arange(batch_size, device=device, dtype=torch.int64),
        seq_lens=seq_lens,
        out_cache_loc=torch.arange(total_tokens, device=device, dtype=torch.int64),
        seq_lens_sum=total_tokens,
        forward_mode=ForwardMode.EXTEND,
        seq_lens_cpu=torch.full((batch_size,), seq_len, dtype=torch.int32),
        positions=positions,
        extend_seq_lens=seq_lens,
        extend_prefix_lens=torch.zeros_like(seq_lens),
        extend_start_loc=extend_start_loc,
        extend_prefix_lens_cpu=[0] * batch_size,
        extend_seq_lens_cpu=[seq_len] * batch_size,
        return_logprob=False,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def _clear_runtime_state(bundle: RuntimeBundle) -> None:
    mamba_cache = bundle.req_pool.mamba_pool.mamba_cache
    for conv_state in mamba_cache.conv:
        conv_state.zero_()
    mamba_cache.temporal.zero_()


def _set_reference_moe(bundle: RuntimeBundle, enabled: bool) -> None:
    for module in bundle.model.modules():
        if isinstance(module, Qwen4ExpTextSparseMoeBlock):
            module.force_reference_moe = enabled


def _set_sequence_ple(bundle: RuntimeBundle, enabled: bool) -> None:
    bundle.model.model.force_sequence_ple = enabled


@torch.inference_mode()
def _run_reference(bundle: RuntimeBundle, forward_batch: ForwardBatch):
    return bundle.model(
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch,
    )


@torch.inference_mode()
def _run_radix(bundle: RuntimeBundle, forward_batch: ForwardBatch):
    bundle.backend.init_forward_metadata(forward_batch)
    with forward_context(ForwardContext(attn_backend=bundle.backend)):
        return bundle.model(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
        )


def _measure(
    label: str, fn, warmups: int, iters: int, device: str
) -> tuple[float, object]:
    last = None
    for _ in range(warmups):
        last = fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        last = fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ms = elapsed / iters * 1000
    print(f"{label}_runtime iters={iters} total_s={elapsed:.6f} ms_per_iter={ms:.3f}")
    return ms, last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--mamba-ssm-dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--vocab-size", type=int, default=257)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--full-attention-interval", type=int, default=0)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--num-key-value-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--linear-head-dim", type=int, default=16)
    parser.add_argument("--linear-num-key-heads", type=int, default=2)
    parser.add_argument("--linear-num-value-heads", type=int, default=4)
    parser.add_argument("--linear-conv-kernel-dim", type=int, default=4)
    parser.add_argument("--moe-intermediate-size", type=int, default=32)
    parser.add_argument("--shared-expert-intermediate-size", type=int, default=32)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-experts-per-tok", type=int, default=1)
    parser.add_argument("--hc-count", type=int, default=2)
    parser.add_argument("--hc-lowrank", type=int, default=16)
    parser.add_argument("--enable-ple", action="store_true")
    parser.add_argument("--num-ple-layers", type=int, default=1)
    parser.add_argument("--ple-embed-dim", type=int, default=None)
    parser.add_argument("--ple-conv-kernel-size", type=int, default=4)
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--heads-per-ngram", type=int, default=8)
    parser.add_argument("--ngram-vocab-size-base", type=int, default=1024)
    parser.add_argument("--make-ngram-vocab-size-divisible-by", type=int, default=128)
    parser.add_argument("--split-ngram-parts", type=int, default=8)
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Tensor parallel size. Defaults to WORLD_SIZE under torchrun, else 1.",
    )
    args = parser.parse_args()

    dist_env = _distributed_env()
    args.device = _select_device(args.device, dist_env)
    tp_size = args.tp_size or dist_env.world_size
    dtype = _parse_dtype(args.dtype)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    print("benchmark=qwen4_exp_runtime")
    print("visible", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print(
        "distributed",
        f"rank={dist_env.rank}",
        f"world_size={dist_env.world_size}",
        f"local_rank={dist_env.local_rank}",
        f"tp_size={tp_size}",
        f"device={args.device}",
    )
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    if torch.cuda.is_available():
        print("cuda_device_count", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            print(
                "gpu",
                idx,
                torch.cuda.get_device_name(idx),
                torch.cuda.get_device_capability(idx),
            )
    _ensure_distributed(args.device, dist_env, tp_size)
    if torch.distributed.is_initialized():
        print(
            "parallel",
            f"world_rank={get_parallel().world_rank}",
            f"tp_size={get_parallel().tp_size}",
            f"attn_tp_size={get_parallel().attn_tp_size}",
            f"tp_rank={get_parallel().tp_rank}",
        )

    config = _build_config(args)
    print(
        "ple",
        f"enabled={bool(config.text_config.ple_layer_ids)}",
        f"layers={config.text_config.ple_layer_ids}",
        f"split_ngram_parts={config.text_config.split_ngram_parts}",
    )
    input_ids = torch.randint(
        0,
        config.text_config.vocab_size,
        (args.batch_size * args.seq_len,),
        device=args.device,
    )

    parallel_override = (
        get_parallel().override(attn_dp_size=1)
        if torch.distributed.is_initialized()
        else get_parallel().override(tp_size=1, attn_tp_size=1, attn_dp_size=1)
    )
    with parallel_override, get_context().override_server_args(
        enable_dp_lm_head=False, enable_fp32_lm_head=False
    ):
        bundle = _build_runtime(config, dtype, args.device, max(args.batch_size, 2))

        def make_batch() -> ForwardBatch:
            return _make_forward_batch(
                config, args.batch_size, args.seq_len, input_ids, args.device
            )

        _set_reference_moe(bundle, True)
        _set_sequence_ple(bundle, True)
        full_reference = _run_reference(bundle, make_batch())
        _clear_runtime_state(bundle)
        _set_sequence_ple(bundle, False)
        python_moe = _run_radix(bundle, make_batch())
        _clear_runtime_state(bundle)
        _set_reference_moe(bundle, False)
        fused_moe = _run_radix(bundle, make_batch())
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        full_max_abs = (
            (
                full_reference.next_token_logits.float()
                - fused_moe.next_token_logits.float()
            )
            .abs()
            .max()
        )
        full_denom = (
            full_reference.next_token_logits.float().abs().max().clamp_min(1e-6)
        )
        full_rel = full_max_abs / full_denom
        moe_max_abs = (
            (python_moe.next_token_logits.float() - fused_moe.next_token_logits.float())
            .abs()
            .max()
        )
        moe_denom = python_moe.next_token_logits.float().abs().max().clamp_min(1e-6)
        moe_rel = moe_max_abs / moe_denom
        print(
            "correctness_full_reference_vs_fused "
            f"logits_shape={tuple(fused_moe.next_token_logits.shape)} "
            f"max_abs={float(full_max_abs):.8f} rel_to_ref_max={float(full_rel):.8f}"
        )
        print(
            "correctness_python_moe_vs_fused "
            f"logits_shape={tuple(fused_moe.next_token_logits.shape)} "
            f"max_abs={float(moe_max_abs):.8f} rel_to_ref_max={float(moe_rel):.8f}"
        )

        def full_reference_fn():
            _set_reference_moe(bundle, True)
            _set_sequence_ple(bundle, True)
            return _run_reference(bundle, make_batch())

        def radix_python_moe_fn():
            _set_reference_moe(bundle, True)
            _set_sequence_ple(bundle, False)
            _clear_runtime_state(bundle)
            return _run_radix(bundle, make_batch())

        def radix_fused_moe_fn():
            _set_reference_moe(bundle, False)
            _set_sequence_ple(bundle, False)
            _clear_runtime_state(bundle)
            return _run_radix(bundle, make_batch())

        full_ref_ms, _ = _measure(
            "full_reference_loop",
            full_reference_fn,
            args.warmups,
            args.iters,
            args.device,
        )
        python_moe_ms, _ = _measure(
            "radix_python_moe",
            radix_python_moe_fn,
            args.warmups,
            args.iters,
            args.device,
        )
        fused_moe_ms, _ = _measure(
            "radix_fused_moe", radix_fused_moe_fn, args.warmups, args.iters, args.device
        )
        print(
            "speedup "
            f"full_reference_loop_ms={full_ref_ms:.3f} "
            f"radix_python_moe_ms={python_moe_ms:.3f} "
            f"radix_fused_moe_ms={fused_moe_ms:.3f} "
            f"moe_speedup={python_moe_ms / fused_moe_ms:.3f}x "
            f"full_reference_speedup={full_ref_ms / fused_moe_ms:.3f}x"
        )
        if args.device.startswith("cuda"):
            print(f"peak_mem_mb {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}")
    _destroy_single_rank_distributed(args.device)


if __name__ == "__main__":
    main()
