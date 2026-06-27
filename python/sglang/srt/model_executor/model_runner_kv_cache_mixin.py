from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.model_config import (
    get_dsa_index_head_dim,
    is_deepseek_dsa,
    is_deepseek_v4,
)
from sglang.srt.distributed.parallel_state import get_moe_ep_group, get_world_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.common import get_req_to_token_extra_context_len
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    NoOpMHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.platforms import current_platform
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    is_float4_e2m1fn_x2,
    is_hip,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig


# the ratio of mamba cache pool size to max_running_requests
MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY = 1
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP = 1

logger = logging.getLogger(__name__)


def _get_dsv4_compress_state_dtypes() -> tuple[torch.dtype, torch.dtype]:
    dtype_name = envs.SGLANG_DSV4_COMPRESS_STATE_DTYPE.get().strip().lower()
    if dtype_name in ("float32", "fp32"):
        return torch.float32, torch.float32
    if dtype_name in ("bfloat16", "bf16"):
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            raise ValueError(
                "SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16 is not supported when "
                "SGLANG_OPT_USE_ONLINE_COMPRESS=1; online c128 state must stay float32."
            )
        return torch.bfloat16, torch.bfloat16
    raise ValueError(
        "Unsupported SGLANG_DSV4_COMPRESS_STATE_DTYPE="
        f"{dtype_name!r}. Expected one of: float32, fp32, bfloat16, bf16."
    )


_is_npu = is_npu()
_is_hip = is_hip()


class ModelRunnerKVCacheMixin:
    def _profile_available_bytes(self: ModelRunner, pre_model_load_memory: int) -> int:
        # KV pool budget = currently-free GPU memory minus the non-static runtime
        # slack (pre_model_load_memory * (1 - mem_fraction_static)). Whatever is
        # already resident (model weights, etc.) is thus charged against it.
        available_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )

        rest_memory = available_gpu_memory - pre_model_load_memory * (
            1 - self.mem_fraction_static
        )
        if self.mambaish_config is not None:
            rest_memory = self.handle_max_mamba_cache(rest_memory)

        # Loaded weights (target + draft) can exceed the static budget
        if rest_memory <= 0:
            minimum_mem_fraction_static = (
                1 - available_gpu_memory / pre_model_load_memory
            )
            suggested_mem_fraction_static = (
                math.ceil(minimum_mem_fraction_static * 1000) / 1000
            )
            raise ValueError(
                f"Loaded weights leave no GPU memory for the KV cache under "
                f"--mem-fraction-static={self.mem_fraction_static}. "
                f"Raise --mem-fraction-static above "
                f"{suggested_mem_fraction_static:.3f} "
                f"(minimum viable = 1 - available/pre = "
                f"{minimum_mem_fraction_static:.4f}). If using speculative "
                f"decoding, draft weights are now counted."
            )

        return int(rest_memory * (1 << 30))  # return in bytes

    def handle_max_mamba_cache(self: ModelRunner, total_rest_memory):
        config = self.mambaish_config
        server_args = self.server_args
        assert config is not None

        has_spec_dec = not self.spec_algorithm.is_none()
        if has_spec_dec:
            assert server_args.speculative_num_draft_tokens is not None
            assert server_args.max_running_requests is not None

        if server_args.max_mamba_cache_size is not None:
            # Use explicitly set max_mamba_cache_size
            server_args.max_mamba_cache_size = server_args.max_mamba_cache_size // (
                server_args.dp_size if server_args.enable_dp_attention else 1
            )
            # Reserve intermediate memory based on capped max_num_reqs
            if has_spec_dec:
                ratio = self._calculate_mamba_ratio()
                capped_reqs = min(
                    server_args.max_running_requests
                    // (self.dp_size if server_args.enable_dp_attention else 1),
                    server_args.max_mamba_cache_size // ratio,
                )
                intermediate_size = (
                    config.mamba2_cache_params.mamba_cache_per_req
                    * capped_reqs
                    * server_args.speculative_num_draft_tokens
                )
                total_rest_memory = total_rest_memory - (intermediate_size / (1 << 30))
        elif (
            server_args.disable_radix_cache
            and server_args.max_running_requests is not None
        ):
            # Use explicitly set max_running_requests when radix cache is disabled
            server_args.max_mamba_cache_size = server_args.max_running_requests // (
                server_args.dp_size if server_args.enable_dp_attention else 1
            )
            # Reserve intermediate memory based on capped max_num_reqs
            if has_spec_dec:
                intermediate_size = (
                    config.mamba2_cache_params.mamba_cache_per_req
                    * server_args.max_mamba_cache_size
                    * server_args.speculative_num_draft_tokens
                )
                total_rest_memory = total_rest_memory - (intermediate_size / (1 << 30))
        else:
            # Use ratio-based calculation to auto-fit available memory
            assert config.mamba2_cache_params.mamba_cache_per_req > 0
            per_req = config.mamba2_cache_params.mamba_cache_per_req

            # Solve jointly for max_mamba_cache_size accounting for intermediate memory.
            # The mamba budget (from the ratio split) must cover both:
            #   1. main mamba state: max_mamba_cache_size * per_req
            #   2. intermediate states: (max_mamba_cache_size / ratio) * D * per_req
            # So: max_mamba_cache_size * per_req * (1 + D/ratio) = mamba_budget_bytes
            mamba_budget = (
                total_rest_memory
                * server_args.mamba_full_memory_ratio
                / (1 + server_args.mamba_full_memory_ratio)
            )
            mamba_budget_bytes = mamba_budget * (1 << 30)

            if has_spec_dec:
                ratio = self._calculate_mamba_ratio()
                D = server_args.speculative_num_draft_tokens
                # Joint solve: main_state + intermediate = mamba_budget
                server_args.max_mamba_cache_size = int(
                    mamba_budget_bytes // (per_req * (1 + D / ratio))
                )
                # Intermediate memory is included in mamba_budget, subtract it
                # so the return value only has main_state subtracted from total
                capped_reqs = min(
                    server_args.max_running_requests
                    // (self.dp_size if server_args.enable_dp_attention else 1),
                    server_args.max_mamba_cache_size // ratio,
                )
                intermediate_size = per_req * capped_reqs * D
                total_rest_memory = total_rest_memory - (intermediate_size / (1 << 30))
            else:
                server_args.max_mamba_cache_size = int(mamba_budget_bytes // per_req)

        # Validate: max_mamba_cache_size must be positive after memory allocation.
        # A non-positive value means GPU memory is insufficient for the requested
        # configuration. Fail fast with actionable advice instead of silently
        # producing garbled output at runtime.
        if server_args.max_mamba_cache_size <= 0:
            raise RuntimeError(
                f"Not enough GPU memory for hybrid (mamba/linear-attention) state cache. "
                f"Computed max_mamba_cache_size={server_args.max_mamba_cache_size} "
                f"(total_rest_memory={total_rest_memory:.2f} GB, "
                f"mamba_cache_per_req={config.mamba2_cache_params.mamba_cache_per_req / (1 << 20):.2f} MB). "
                f"Try: (1) reduce --max-running-requests, "
                f"(2) increase --mem-fraction-static, "
                f"(3) reduce --speculative-num-draft-tokens, or "
                f"(4) use GPUs with more memory."
            )

        mamba_state_memory = (
            server_args.max_mamba_cache_size
            * config.mamba2_cache_params.mamba_cache_per_req
            / (1 << 30)
        )
        return total_rest_memory - mamba_state_memory

    def calculate_mla_kv_cache_dim(self: ModelRunner) -> int:
        is_dsa_model = is_deepseek_dsa(self.model_config.hf_config)
        kv_cache_dtype = self.kv_cache_dtype
        kv_lora_rank = self.model_config.kv_lora_rank
        qk_rope_head_dim = self.model_config.qk_rope_head_dim
        kv_cache_dim = kv_lora_rank + qk_rope_head_dim  # default mla kv cache dim

        # For non-DSA models, MLA kv cache dim is simply kv_lora_rank + qk_rope_head_dim
        if not is_dsa_model:
            return kv_cache_dim

        # TRTLLM backend does not override kv_cache_dim for MLA kv cache
        # Assuming dsa prefill and decode backends are the same when using trtllm MLA backend,
        # since it is not compatible for trtllm and other mla attn backend due to the different
        # kv cache layout.
        if (
            self.server_args.dsa_prefill_backend == "trtllm"
            or self.server_args.dsa_decode_backend == "trtllm"
        ):
            return kv_cache_dim

        # On HIP, TileLang and AITER DSA kernels consume the raw MLA KV layout:
        # nope(512 fp8) + rope(64 fp8), without extra per-block scales.
        if _is_hip and (
            self.server_args.dsa_prefill_backend in ("tilelang", "aiter")
            or self.server_args.dsa_decode_backend in ("tilelang", "aiter")
        ):
            return kv_cache_dim

        quant_block_size = DSATokenToKVPool.quant_block_size
        rope_storage_dtype = DSATokenToKVPool.rope_storage_dtype
        # Calculate override_kv_cache_dim for FP8 storage in backends that use scaled KV layout
        # (excluding TRTLLM and HIP raw-layout kernels).
        # kv_lora_rank + scale storage (kv_lora_rank // quant_block_size * 4 bytes) + rope dimension storage
        # Note: rope dimension is stored in original dtype (bf16), not quantized to fp8
        if kv_cache_dtype == torch.float8_e4m3fn:
            assert (
                kv_lora_rank % quant_block_size == 0
            ), f"kv_lora_rank {kv_lora_rank} must be multiple of quant_block_size {quant_block_size}"

            return (
                kv_lora_rank
                + kv_lora_rank // quant_block_size * 4
                + qk_rope_head_dim * rope_storage_dtype.itemsize
            )

        return kv_cache_dim

    def _calculate_mamba_ratio(self: ModelRunner) -> int:
        if self.server_args.disable_radix_cache:
            return 1

        additional_ratio = 0
        if self.server_args.enable_mamba_extra_buffer():
            # ping-pong buffer size is 2 when overlap schedule is on, 1 otherwise.
            # Lazy mode saves 1 slot (2 → 1) for overlap; non-overlap already uses 1.
            if not self.server_args.disable_overlap_schedule:
                if self.server_args.enable_mamba_extra_buffer_lazy():
                    additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY
                else:
                    additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP
            else:
                assert (
                    not self.server_args.enable_mamba_extra_buffer_lazy()
                ), "Lazy extra buffer requires overlap schedule (--disable-overlap-schedule is incompatible)"
                additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP

        return MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO + additional_ratio

    def _validate_prefill_only_disable_kv_cache_pool_family(
        self: ModelRunner,
        is_dsa_model: bool,
        is_dsv4_model: bool,
        current_platform,
    ):
        if not self.server_args.prefill_only_disable_kv_cache or self.is_draft_worker:
            return

        unsupported_pool_family = None
        if is_dsv4_model:
            unsupported_pool_family = "DeepSeekV4TokenToKVPool"
        elif current_platform.is_out_of_tree() and not self.mambaish_config:
            unsupported_pool_family = "out-of-tree platform KV pool"
        elif (
            self.server_args.attention_backend == "ascend" and not self.mambaish_config
        ):
            unsupported_pool_family = "NPU/Ascend KV pool"
        elif self.use_mla_backend and is_dsa_model:
            unsupported_pool_family = "DSA/MLA KV pool"
        elif self.use_mla_backend and not self.mambaish_config:
            unsupported_pool_family = "MLA KV pool"
        elif self.is_hybrid_swa:
            unsupported_pool_family = "SWA KV pool"
        elif self.mambaish_config:
            unsupported_pool_family = "hybrid linear/Mamba KV pool"
        elif is_float4_e2m1fn_x2(self.kv_cache_dtype):
            unsupported_pool_family = "FP4 MHA KV pool"

        if unsupported_pool_family is not None:
            raise RuntimeError(
                "--prefill-only-disable-kv-cache is not supported for "
                f"{unsupported_pool_family}. Supported configurations today: plain MHA "
                "models on CUDA with the FA (fa3/fa4) prefill backend, --is-embedding, "
                "--chunked-prefill-size=-1, --disable-radix-cache, no context-parallel "
                "attention, no HiSparse, and --kv-cache-dtype != fp4_e2m1."
            )

    def _init_pools(self: ModelRunner):
        """Initialize the memory pools."""
        max_num_reqs = self.max_running_requests

        # Initialize req_to_token_pool
        if self.req_to_token_pool is None:
            max_spec_draft_tokens = self.server_args.max_speculative_num_draft_tokens
            extra_max_context_len = get_req_to_token_extra_context_len(self.server_args)

            if self.server_args.disaggregation_mode == "decode":
                from sglang.srt.disaggregation.decode import (
                    DecodeReqToTokenPool,
                    HybridMambaDecodeReqToTokenPool,
                )

                # Extra slots for pre-allocated requests
                pre_alloc_size = self.server_args.disaggregation_decode_extra_slots
                if config := self.mambaish_config:
                    self.req_to_token_pool = HybridMambaDecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        cache_params=config.mamba2_cache_params,
                        mamba_layer_ids=(
                            [
                                i
                                for i in config.mamba2_cache_params.layers
                                if self.start_layer <= i < self.end_layer
                            ]
                        ),
                        speculative_num_draft_tokens=max_spec_draft_tokens,
                        speculative_eagle_topk=self.server_args.speculative_eagle_topk,
                        enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
                        pre_alloc_size=pre_alloc_size,
                        enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
                        mamba_size=self.server_args.max_mamba_cache_size,
                        start_layer=self.start_layer,
                    )
                else:
                    self.req_to_token_pool = DecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        pre_alloc_size=pre_alloc_size,
                    )
            elif config := self.mambaish_config:
                self.req_to_token_pool = HybridReqToTokenPool(
                    size=max_num_reqs,
                    mamba_size=self.server_args.max_mamba_cache_size,
                    mamba_spec_state_size=max_num_reqs,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    cache_params=config.mamba2_cache_params,
                    mamba_layer_ids=(
                        [
                            i
                            for i in config.mamba2_cache_params.layers
                            if self.start_layer <= i < self.end_layer
                        ]
                    ),
                    enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
                    enable_mamba_extra_buffer_lazy=self.server_args.enable_mamba_extra_buffer_lazy(),
                    speculative_num_draft_tokens=max_spec_draft_tokens,
                    speculative_eagle_topk=self.server_args.speculative_eagle_topk,
                    enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
                    start_layer=self.start_layer,
                    enable_linear_replayssm=self.server_args.enable_linear_replayssm,
                    linear_replayssm_cache_len=self.server_args.linear_replayssm_cache_len,
                )
            else:
                # DSV4 on NPU needs an extended ReqToTokenPool holding per-req
                # swa/c4/c128/c{4,128}_state tables; others stay on the stock one.
                req_to_token_pool_cls = ReqToTokenPool
                if _is_npu and is_deepseek_v4(self.model_config.hf_config):
                    from sglang.srt.hardware_backend.npu.dsv4.dsv4_req_to_token_pool import (
                        DSV4NPUReqToTokenPool,
                    )

                    req_to_token_pool_cls = DSV4NPUReqToTokenPool

                self.req_to_token_pool = req_to_token_pool_cls(
                    size=max_num_reqs,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                )
        else:
            # Draft worker shares req_to_token_pool with the target worker.
            assert self.is_draft_worker

        # Initialize token_to_kv_pool
        is_dsa_model = is_deepseek_dsa(self.model_config.hf_config)
        is_dsv4_model = is_deepseek_v4(self.model_config.hf_config)

        self._validate_prefill_only_disable_kv_cache_pool_family(
            is_dsa_model, is_dsv4_model, current_platform
        )

        if is_dsv4_model:
            swa_page_size = self.page_size
            if not _is_npu:
                assert swa_page_size == 256, "In paged swa mode, page_size must be 256."

            if self.is_draft_worker:
                from sglang.srt.models.deepseek_v4_nextn import (
                    COMPRESS_RATIO_NEXTN_LAYER,
                )

                compression_ratios = [
                    COMPRESS_RATIO_NEXTN_LAYER
                ] * self.num_effective_layers
            else:
                compression_ratios = self.model_config.compress_ratios

            # NPU + DSV4 → paged-state subclass: the fused compressor kernel
            # needs cache_mode=1 (paged); Atlas A3 rejects cache_mode=2 (ring),
            # so the CUDA ring-buffer state path can't be shared. CUDA keeps
            # DeepSeekV4TokenToKVPool unchanged; NPU recomputes state sizes below.
            if _is_npu:
                from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
                    DSV4NPUTokenToKVPool,
                    npu_state_pool_size,
                )

                pool_cls = DSV4NPUTokenToKVPool
                # Recompute state pool sizes for the NPU paged formula (CUDA's
                # ring sizes are dropped here). Tail-only allocation keeps the
                # per-req-budget formula sufficient at any prefill length: long
                # prompts allocate only ``tail+128`` (c4) / ``tail`` (c128)
                # slots (tail = seq_len % 128), and decode is drained by
                # sliding eviction in ``ScheduleBatch._evict_swa``.
                c4_state_pool_size = npu_state_pool_size(
                    ratio=4,
                    page_size=self.page_size,
                    max_num_reqs=self.max_running_requests,
                )
                c128_state_pool_size = npu_state_pool_size(
                    ratio=128,
                    page_size=self.page_size,
                    max_num_reqs=self.max_running_requests,
                )
            else:
                pool_cls = DeepSeekV4TokenToKVPool
                c4_state_pool_size = self.c4_state_pool_size
                c128_state_pool_size = self.c128_state_pool_size

            self.token_to_kv_pool = pool_cls(
                max_num_reqs=self.max_running_requests,
                # SWA ring is indexed by req_pool_idx; PD decode inflates req_to_token
                # past max_running_requests (pre-alloc), so size to the real capacity.
                num_req_slots=self.req_to_token_pool.req_to_token.shape[0],
                swa_size=self.swa_max_total_num_tokens,
                c4_size=self.c4_max_total_num_tokens,
                c128_size=self.c128_max_total_num_tokens,
                c4_state_pool_size=c4_state_pool_size,
                c128_state_pool_size=c128_state_pool_size,
                page_size=self.page_size,
                swa_page_size=swa_page_size,
                sliding_window=self.model_config.window_size,
                dtype=self.kv_cache_dtype,
                c4_state_dtype=self.c4_state_dtype,
                c128_state_dtype=self.c128_state_dtype,
                qk_nope_head_dim=self.model_config.qk_nope_head_dim,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                indexer_head_dim=self.model_config.index_head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                compression_ratios=compression_ratios,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                enable_hisparse=self.enable_hisparse,
                online_mtp_max_draft_tokens=(
                    self.server_args.max_speculative_num_draft_tokens or 0
                ),
            )
        elif current_platform.is_out_of_tree() and not self.mambaish_config:
            if self.use_mla_backend and is_dsa_model:
                PoolCls = current_platform.get_dsa_kv_pool_cls()
                self.token_to_kv_pool = PoolCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    kv_cache_dim=self.calculate_mla_kv_cache_dim(),
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
                )
            elif self.use_mla_backend:
                PoolCls = current_platform.get_mla_kv_pool_cls()
                self.token_to_kv_pool = PoolCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=(
                        self.model_config.index_head_dim if is_dsa_model else None
                    ),
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                PoolCls = current_platform.get_mha_kv_pool_cls()
                self.token_to_kv_pool = PoolCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif (
            self.server_args.attention_backend == "ascend" and not self.mambaish_config
        ):
            if self.is_hybrid_swa:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                kwargs = {}
                if self.is_hybrid_swa_compress:
                    kwargs = {
                        "swa_head_num": max(
                            1,
                            self.model_config.hf_text_config.swa_num_key_value_heads
                            // get_attention_tp_size(),
                        ),
                        "swa_head_dim": self.model_config.swa_head_dim,
                        "swa_v_head_dim": self.model_config.swa_v_head_dim,
                        "v_head_dim": self.model_config.v_head_dim,
                    }
                self.token_to_kv_pool = SWAKVPool(
                    size=self.full_max_total_num_tokens,
                    size_swa=self.swa_max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    enable_kvcache_transpose=False,
                    device=self.device,
                    token_to_kv_pool_class=NPUMHATokenToKVPool,
                    **kwargs,
                )
            elif self.use_mla_backend:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                self.token_to_kv_pool = NPUMLATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=(
                        self.model_config.index_head_dim if is_dsa_model else None
                    ),
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                self.token_to_kv_pool = NPUMHATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif self.use_mla_backend and is_dsa_model:
            PoolCls = (
                HiSparseDSATokenToKVPool if self.enable_hisparse else DSATokenToKVPool
            )
            pool_kwargs = {}
            if self.enable_hisparse:
                from sglang.srt.mem_cache.sparsity import parse_hisparse_config

                pool_kwargs["host_to_device_ratio"] = parse_hisparse_config(
                    self.server_args
                ).host_to_device_ratio
            self.token_to_kv_pool = PoolCls(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                kv_cache_dim=self.calculate_mla_kv_cache_dim(),
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
                **pool_kwargs,
            )
        elif self.use_mla_backend and not self.mambaish_config:
            assert not is_dsa_model
            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                self.token_to_kv_pool = MLATokenToKVPoolFP4(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                self.token_to_kv_pool = MLATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        else:
            if self.is_hybrid_swa:
                kwargs = {}
                if self.is_hybrid_swa_compress:
                    kwargs = {
                        "swa_head_num": max(
                            1,
                            self.model_config.hf_text_config.swa_num_key_value_heads
                            // get_attention_tp_size(),
                        ),
                        "swa_head_dim": self.model_config.swa_head_dim,
                        "swa_v_head_dim": self.model_config.swa_v_head_dim,
                        "v_head_dim": self.model_config.v_head_dim,
                    }
                self.token_to_kv_pool = SWAKVPool(
                    size=self.full_max_total_num_tokens,
                    size_swa=self.swa_max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    enable_kvcache_transpose=False,
                    device=self.device,
                    enable_kv_cache_copy=(
                        self.server_args.speculative_algorithm is not None
                    ),
                    **kwargs,
                )
            elif config := self.mambaish_config:
                extra_args = {}
                if self.use_mla_backend:
                    extra_args = {
                        "kv_lora_rank": self.model_config.kv_lora_rank,
                        "qk_rope_head_dim": self.model_config.qk_rope_head_dim,
                    }
                self.token_to_kv_pool = HybridLinearKVPool(
                    page_size=self.page_size,
                    size=self.max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    # if draft worker, we only need 1 attention layer's kv pool
                    full_attention_layer_ids=(
                        [0]
                        if self.is_draft_worker
                        else [
                            i
                            for i in config.full_attention_layer_ids
                            if self.start_layer <= i < self.end_layer
                        ]
                    ),
                    enable_kvcache_transpose=False,
                    device=self.device,
                    mamba_pool=self.req_to_token_pool.mamba_pool,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    enable_kv_cache_copy=(
                        self.server_args.speculative_algorithm is not None
                    ),
                    use_mla=self.use_mla_backend,
                    start_layer=self.start_layer,
                    **extra_args,
                )
            else:
                if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                    self.token_to_kv_pool = MHATokenToKVPoolFP4(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self.model_config.get_num_kv_heads(
                            get_attention_tp_size()
                        ),
                        head_dim=self.model_config.head_dim,
                        v_head_dim=self.model_config.v_head_dim,
                        layer_num=self.num_effective_layers,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        enable_alt_stream=not self.server_args.enable_pdmux,
                        enable_kv_cache_copy=(
                            self.server_args.speculative_algorithm is not None
                        ),
                    )
                else:
                    pool_cls = (
                        NoOpMHATokenToKVPool
                        if self.server_args.prefill_only_disable_kv_cache
                        else MHATokenToKVPool
                    )
                    self.token_to_kv_pool = pool_cls(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self.model_config.get_num_kv_heads(
                            get_attention_tp_size()
                        ),
                        head_dim=self.model_config.head_dim,
                        v_head_dim=self.model_config.v_head_dim,
                        layer_num=self.num_effective_layers,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        enable_alt_stream=not self.server_args.enable_pdmux,
                        enable_kv_cache_copy=(
                            self.server_args.speculative_algorithm is not None
                        ),
                    )

        # Initialize token_to_kv_pool_allocator
        need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
        if self.token_to_kv_pool_allocator is None:
            if current_platform.is_out_of_tree():
                AllocatorCls = current_platform.get_paged_allocator_cls()
                self.token_to_kv_pool_allocator = AllocatorCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=need_sort,
                )
            elif _is_npu and (
                self.server_args.attention_backend == "ascend"
                or is_dsv4_model
                or self.hybrid_gdn_config is not None
            ):
                if self.is_hybrid_swa:
                    # DSV4 on NPU: SWA allocator subclass that also drives the
                    # c4/c128 allocators, producing a DSV4OutCacheLoc per alloc.
                    if is_dsv4_model:
                        from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
                            DSV4NPUTokenToKVPoolAllocator,
                        )

                        swa_allocator_cls = DSV4NPUTokenToKVPoolAllocator
                    else:
                        swa_allocator_cls = SWATokenToKVPoolAllocator
                    self.token_to_kv_pool_allocator = swa_allocator_cls(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    from sglang.srt.hardware_backend.npu.allocator_npu import (
                        NPUPagedTokenToKVPoolAllocator,
                    )

                    self.token_to_kv_pool_allocator = NPUPagedTokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
            else:
                if self.is_hybrid_swa:
                    self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    if self.enable_hisparse:
                        from sglang.srt.mem_cache.sparsity import (
                            parse_hisparse_config,
                        )

                        hisparse_cfg = parse_hisparse_config(self.server_args)
                        self.token_to_kv_pool_allocator = (
                            HiSparseTokenToKVPoolAllocator(
                                self.max_total_num_tokens,
                                page_size=self.page_size,
                                dtype=self.kv_cache_dtype,
                                device=self.device,
                                kvcache=self.token_to_kv_pool,
                                need_sort=need_sort,
                                host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
                            )
                        )
                    elif self.page_size == 1 and self.dcp_size == 1:
                        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                            self.max_total_num_tokens,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )
                    else:
                        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                            self.max_total_num_tokens * self.dcp_size,
                            page_size=self.page_size * self.dcp_size,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )

            if self.enable_hisparse and is_dsv4_model:
                assert self.is_hybrid_swa, "DeepSeek V4 HiSparse requires SWA mode."
                self.token_to_kv_pool_allocator = (
                    DeepSeekV4HiSparseTokenToKVPoolAllocator(
                        self.token_to_kv_pool_allocator
                    )
                )

            # DSV4-NPU: wire allocator back-ref into req_to_token_pool so its
            # free(req) can release c4/c128 pool pages alongside the slot.
            if hasattr(self.req_to_token_pool, "register_dsv4_allocator"):
                self.req_to_token_pool.register_dsv4_allocator(
                    self.token_to_kv_pool_allocator
                )

        else:
            assert self.is_draft_worker
            if self.is_hybrid_swa:
                swa_allocator = getattr(
                    self.token_to_kv_pool_allocator,
                    "logical_attn_allocator",
                    self.token_to_kv_pool_allocator,
                )
                assert swa_allocator.__class__ == SWATokenToKVPoolAllocator
                self.token_to_kv_pool.full_to_swa_index_mapping = (
                    swa_allocator.full_to_swa_index_mapping
                )

        # Defensive check: the explicit validation above should reject known
        # unsupported pool families before allocation. Keep this guard here so
        # future pool-selection refactors fail at boot instead of on first use.
        if (
            self.server_args.prefill_only_disable_kv_cache
            and not self.is_draft_worker
            and not isinstance(self.token_to_kv_pool, NoOpMHATokenToKVPool)
        ):
            raise RuntimeError(
                "--prefill-only-disable-kv-cache expected NoOpMHATokenToKVPool but the "
                f"runtime pool is {type(self.token_to_kv_pool).__name__}. This pool "
                "family is not yet supported by --prefill-only-disable-kv-cache. "
                "Supported configurations today: plain MHA models on CUDA with the FA "
                "(fa3/fa4) prefill backend, --is-embedding, --chunked-prefill-size=-1, "
                "--disable-radix-cache, no context-parallel attention, no HiSparse, "
                "and --kv-cache-dtype != fp4_e2m1."
            )

    def _apply_token_constraints(self: ModelRunner, token_capacity: int) -> int:
        """Apply external constraints to token capacity: user cap, PP sync.

        Page alignment is handled by the configurator, not here.
        If constraints change the value, the configurator re-runs and re-aligns.
        """
        user_limit = self.server_args.max_total_tokens

        # Apply user-specified upper bound
        if user_limit is not None:
            if user_limit > token_capacity:
                logging.warning(
                    f"max_total_tokens={user_limit} is larger than the profiled value "
                    f"{token_capacity}. Use the profiled value instead."
                )
            token_capacity = min(token_capacity, user_limit)

        # Sync across PP ranks (each may have different layer counts)
        if self.pp_size > 1:
            tensor = torch.tensor(token_capacity, dtype=torch.int64)
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            token_capacity = tensor.item()

        return token_capacity

    def _resolve_max_num_reqs(self: ModelRunner, token_capacity: int) -> int:
        """Compute max concurrent requests (per dp worker) from the finalized
        token capacity."""
        # Estimate pool size (used as upper bound when user specifies max_running_requests)
        estimated = int(token_capacity / self.model_config.context_len * 512)
        estimated = max(min(estimated, 4096), 2048)

        max_num_reqs = self.server_args.max_running_requests
        if max_num_reqs is not None:
            requested_per_worker = max_num_reqs // self.dp_size
            max_num_reqs = min(requested_per_worker, token_capacity // 2)
        else:
            requested_per_worker = None
            max_num_reqs = min(estimated, token_capacity // 2)

        if self.mambaish_config is not None:
            ratio = self._calculate_mamba_ratio()
            max_num_reqs = min(
                max_num_reqs, self.server_args.max_mamba_cache_size // ratio
            )

            if max_num_reqs <= 0:
                raise RuntimeError(
                    f"Hybrid (mamba/linear-attention) state cache is too small to serve "
                    f"any requests. max_mamba_cache_size={self.server_args.max_mamba_cache_size}, "
                    f"mamba_ratio={ratio}, resulting max_num_reqs={max_num_reqs}. "
                    f"Try: (1) reduce --max-running-requests, "
                    f"(2) increase --mem-fraction-static, or "
                    f"(3) use GPUs with more memory."
                )
        if requested_per_worker is not None and max_num_reqs < requested_per_worker:
            logger.warning(
                "max_running_requests was reduced from the requested %d to %d "
                "(per dp worker) due to the available KV cache capacity.",
                requested_per_worker,
                max_num_reqs,
            )

        return self._clamp_deepep_low_latency_concurrency(max_num_reqs)

    def _is_deepep_low_latency(self: ModelRunner) -> bool:
        from sglang.srt.layers.moe.utils import is_deepep_dispatch_backend

        return (
            is_deepep_dispatch_backend(self.server_args.moe_a2a_backend)
            and self.server_args.deepep_mode != "normal"
        )

    def _clamp_deepep_low_latency_concurrency(
        self: ModelRunner, max_num_reqs: int
    ) -> int:
        """Cap per-rank decode concurrency for the DeepEP low_latency path.

        num_max (the dispatch cap, sized from this concurrency) must hold every
        rank's decode batch or DeepEP asserts, yet the all-to-all buffer is
        collective so all EP ranks must size it identically. Clamp to the
        FINISHED_SUM_TAG bound, then take the EP-group minimum: the result is
        uniform and <= every rank's own concurrency, so the buffer fits and no
        rank overruns it. The cap also lets the post-KV auto-tune read the
        already-synced request-pool size without its own reduction.
        """
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DEEPEP_LOW_LATENCY_MAX_DISPATCH_TOKENS,
        )

        if not self._is_deepep_low_latency():
            return max_num_reqs

        # The per-rank dispatch is concurrency * num_tokens_per_bs (spec/MTP verify
        # packs num_tokens_per_bs tokens per request); bound that product by
        # FINISHED_SUM_TAG, so the concurrency ceiling shrinks by the spec multiplier.
        tokens_per_req = self.server_args.speculative_num_draft_tokens or 1
        capped = min(
            max_num_reqs, DEEPEP_LOW_LATENCY_MAX_DISPATCH_TOKENS // tokens_per_req
        )
        ep_group = get_moe_ep_group()
        if ep_group.world_size > 1:
            tensor = torch.tensor(capped, dtype=torch.int64)
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MIN, group=ep_group.cpu_group
            )
            capped = int(tensor.item())

        if capped < max_num_reqs:
            logger.warning(
                "DeepEP low_latency capped per-rank decode concurrency to %d "
                "(from %d) to keep the all-to-all dispatch buffer within the "
                "FINISHED_SUM_TAG bound and uniform across the EP group.",
                capped,
                max_num_reqs,
            )
        return capped

    def _apply_memory_pool_config(self: ModelRunner, config: MemoryPoolConfig):
        """Apply a resolved MemoryPoolConfig and initialize pools."""
        self.max_total_num_tokens = config.max_total_num_tokens
        self.max_running_requests = config.max_running_requests
        if self.is_hybrid_swa:
            self.full_max_total_num_tokens = config.full_max_total_num_tokens
            self.swa_max_total_num_tokens = config.swa_max_total_num_tokens

        # DSV4 compressed-attention pool sizes. Draft worker reuses target's
        # full/swa sizes but does NOT own c4/c128/state pools (those live on
        # the target rank only); zero them out regardless of what config holds.
        if self.is_draft_worker:
            self.c4_max_total_num_tokens = 0
            self.c128_max_total_num_tokens = 0
            self.c4_state_pool_size = 0
            self.c128_state_pool_size = 0
        else:
            self.c4_max_total_num_tokens = config.c4_max_total_num_tokens
            self.c128_max_total_num_tokens = config.c128_max_total_num_tokens
            self.c4_state_pool_size = config.c4_state_pool_size
            self.c128_state_pool_size = config.c128_state_pool_size

        # Draft worker does not own the compression-state pools, but keep the
        # dtype attributes initialized so _init_pools can share one code path.
        if is_deepseek_v4(self.model_config.hf_config):
            self.c4_state_dtype, self.c128_state_dtype = (
                _get_dsv4_compress_state_dtypes()
            )

        self._init_pools()

    def _resolve_memory_pool_config(
        self: ModelRunner, pre_model_load_memory: int
    ) -> MemoryPoolConfig:
        """Profile GPU memory and resolve all pool parameters into a config."""
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )

        available_bytes = self._profile_available_bytes(pre_model_load_memory)
        page_size = self.server_args.page_size

        configurator = create_memory_pool_configurator(self)
        config = configurator.calculate_pool_sizes(available_bytes, page_size)

        # Apply external constraints (user cap, page alignment, PP sync)
        constrained = self._apply_token_constraints(config.max_total_num_tokens)
        if constrained != config.max_total_num_tokens:
            config = configurator.calculate_pool_sizes_from_max_tokens(
                constrained, page_size
            )

        config.max_running_requests = self._resolve_max_num_reqs(
            config.max_total_num_tokens
        )
        config.mem_fraction_static = self.server_args.mem_fraction_static
        return config

    def init_memory_pool(self: ModelRunner, pre_model_load_memory: int):
        if not self.spec_algorithm.is_none() and self.is_draft_worker:
            assert (
                self.memory_pool_config is not None
            ), "Draft worker requires memory_pool_config"
        else:
            self.memory_pool_config = self._resolve_memory_pool_config(
                pre_model_load_memory
            )

        self._apply_memory_pool_config(self.memory_pool_config)

        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )
