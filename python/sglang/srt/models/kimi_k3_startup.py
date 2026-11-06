"""CPU kernel compilation overlapped with Kimi checkpoint loading."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import torch

from sglang.srt.environ import envs


@dataclass(frozen=True)
class KdaPrecompileConfig:
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    safe_gate: bool


def _decode_builders(kda_config, text_config):
    # These are the first-forward factories used by the TP8 K3.1 fast path.
    # Build modules only: no collectives, model forwards, or tuning runs here.
    from sglang.srt.runtime_context import get_model, get_parallel, get_schedule

    parallel = get_parallel()
    if (
        kda_config is None
        or kda_config.num_heads != 12
        or kda_config.head_dim != 128
        or kda_config.dtype != torch.bfloat16
        or parallel.tp_size != 8
    ):
        return []

    from sglang.kernels.jit.utils import is_arch_support_pdl
    from sglang.kernels.ops.activation.kimi_k3_situ_and_mul import (
        _jit_situ_and_mul_module,
    )
    from sglang.kernels.ops.attention.clamp_position import _jit_clamp_position_module
    from sglang.kernels.ops.attention.concat_mla import _jit_concat_mla_absorb_q_module
    from sglang.kernels.ops.attention.dsv4.flashmla_sched_meta import (
        _jit_flashmla_sched_meta_module,
    )
    from sglang.kernels.ops.attention.dsv4.topk import _jit_topk_v2_module
    from sglang.kernels.ops.attention.kda_fused_decode import (
        _jit_kda_fused_decode_module,
    )
    from sglang.kernels.ops.elementwise.add3 import _jit_add3_module
    from sglang.kernels.ops.elementwise.mla_output_gate import (
        _jit_mla_output_gate_module,
    )
    from sglang.kernels.ops.gemm.dsv3_fused_a_gemm import dsv3_fused_a_gemm_module
    from sglang.kernels.ops.gemm.tiny_gemm import _get_module, _mma_min_m
    from sglang.kernels.ops.moe.dsv4 import _jit_mask_topk_module
    from sglang.kernels.ops.moe.moe_route_quant_fused import available
    from sglang.kernels.ops.moe.moe_route_radix import _jit_route_radix_module
    from sglang.kernels.ops.quantization.per_token_group_quant import (
        _jit_module as quant_module,
    )

    builders = [
        _jit_topk_v2_module,
        partial(_jit_topk_v2_module, deterministic=True),
        _jit_kda_fused_decode_module,
        partial(_jit_situ_and_mul_module, torch.bfloat16, torch.bfloat16),
        partial(_jit_situ_and_mul_module, torch.float32, torch.bfloat16),
        available,  # Preserve the routing kernel's optional-build fallback.
        _jit_mask_topk_module,
        _jit_add3_module,
        _jit_concat_mla_absorb_q_module,
        _jit_flashmla_sched_meta_module,
        _jit_mla_output_gate_module,
        _jit_route_radix_module,
        partial(_jit_clamp_position_module, torch.int64),
        partial(
            quant_module,
            torch.float32,
            torch.float8_e4m3fn,
            32,
            True,
            True,
            True,
            False,
            False,
            is_arch_support_pdl(),
        ),
    ]
    for n, k in ((1536, 128), (144, 7168), (160, 7168)):
        # Use the current dispatcher schedules on both sides of its crossover.
        threshold = _mma_min_m(n, k, torch.bfloat16)
        for use_mma in {1 >= threshold, 16 >= threshold}:
            builders.append(
                partial(_get_module, n, k, torch.bfloat16, torch.bfloat16, use_mma)
            )
    from sglang.kernels.ops.kvcache.dsa_fused_store_index_cache import (
        can_use_dsa_fused_store,
    )

    builders.append(
        partial(
            can_use_dsa_fused_store,
            torch.bfloat16,
            torch.int64,
            get_schedule().page_size,
        )
    )
    if parallel.dcp_size == 8 and getattr(text_config, "index_topk", None) == 2048:
        from sglang.kernels.ops.attention.dsa.dcp_topk_merge import (
            _jit_dcp_topk_merge_module,
        )
        from sglang.srt.layers.attention.dsa.dsa_dcp_sharded import _DCP_MERGE_CAND_K

        builders.append(partial(_jit_dcp_topk_merge_module, 8, 2048, _DCP_MERGE_CAND_K))
    if (
        getattr(text_config, "q_lora_rank", None) == 1536
        and getattr(text_config, "kv_lora_rank", None) == 512
    ):
        builders.append(
            partial(dsv3_fused_a_gemm_module, 7168, 2112, is_arch_support_pdl())
        )
    if (
        get_model().kv_cache_dtype == "fp8_e4m3"
        and getattr(text_config, "kv_lora_rank", None) == 512
        and getattr(text_config, "qk_rope_head_dim", None) == 64
    ):
        from sglang.kernels.ops.kvcache.set_mla_kv_buffer import (
            can_use_set_mla_kv_buffer,
        )

        # 512 FP8 values + four FP32 scales, and 64 BF16 rope values.
        builders.append(partial(can_use_set_mla_kv_buffer, 528, 128))
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.layers.communication import k3_ar_fusion

    comm = parallel.tp_group.ca_comm
    if isinstance(comm, CustomAllReduceV2) and not comm.disabled:
        from sglang.kernels.ops.communication.all_reduce import get_all_reduce_module

        builders.append(
            partial(get_all_reduce_module, torch.bfloat16, parallel.tp_size)
        )
    if k3_ar_fusion.enabled():
        from sglang.kernels.ops.communication.all_reduce_residual import _jit_module
        from sglang.kernels.ops.communication.gemm_ag import (
            _jit_module as gemm_ag_module,
        )

        builders.append(partial(_jit_module, parallel.tp_size))
        builders.append(gemm_ag_module)
    return builders


@contextmanager
def precompile_kimi_kernels(
    hidden_size, num_bank_rows, device, kda_config=None, text_config=None
):
    from sglang.srt.layers.attn_residual import _use_fast

    if (
        not envs.SGLANG_ENABLE_KIMI_STARTUP_PRECOMPILE.get()
        or device.type != "cuda"
        or not 1 <= num_bank_rows <= 8
        or not _use_fast(hidden_size)
    ):
        yield
        return

    from sglang.kernels.arch import current_arch
    from sglang.kernels.jit.utils import parallel_compile
    from sglang.kernels.ops.attention.attn_res import (
        _TMA_BEST_CONFIG,
        _tuned_fused_tma_module,
        _tuning,
    )

    # Initialize the architecture before starting worker threads. All builders
    # use this same device and target; they never launch a model forward.
    target = current_arch().cuda
    if target is None or target.capability != torch.cuda.get_device_capability(device):
        yield
        return
    configs = sorted(
        {
            config
            for nvb in range(1, num_bank_rows + 1)
            for config in (_tuning(nvb, 0), _TMA_BEST_CONFIG[nvb])
        }
    )

    def build(config):
        with torch.cuda.device(device):
            return _tuned_fused_tma_module(*config)

    builders = [partial(build, config) for config in configs]
    if kda_config is not None:
        from sglang.kernels.ops.attention.fla.chunk_intra import (
            precompile_kda_small_grid,
        )
        from sglang.kernels.ops.attention.fla.kda import precompile_kda_output

        # The output autotuner has 36 compile candidates. Start it early so
        # its CPU compilation is hidden by the checkpoint I/O and H2D work.
        builders.insert(
            0,
            partial(
                precompile_kda_output,
                num_heads=kda_config.num_heads,
                head_dim=kda_config.head_dim,
                dtype=kda_config.dtype,
                device=device,
            ),
        )

        builders.append(
            partial(
                precompile_kda_small_grid,
                num_heads=kda_config.num_heads,
                head_dim=kda_config.head_dim,
                dtype=kda_config.dtype,
                safe_gate=kda_config.safe_gate,
                device=device,
            )
        )

    def on_device(builder):
        with torch.cuda.device(device):
            return builder()

    builders.extend(
        partial(on_device, build) for build in _decode_builders(kda_config, text_config)
    )

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="kimi-precompile"
    ) as pool:
        future = pool.submit(parallel_compile, builders, 6)
        try:
            yield
        finally:
            # Surface compilation failures and join all jobs before inference
            # can call the same module factories or capture CUDA graphs.
            future.result()
