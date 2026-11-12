import math
from functools import partial
from importlib.metadata import version
from itertools import groupby
from types import SimpleNamespace
from typing import NamedTuple

import torch
from torch import nn

from sglang.kernels.ops.inkling2 import experts_sm90, marlin_sm90
from sglang.kernels.ops.inkling2.comm import (
    ROUTED_EXCHANGE_MIN_ROWS,
    all_reduce,
    multicast_all_reduce,
    multicast_all_reduce_input,
    oneshot_all_reduce,
    oneshot_all_reduce_input,
    prefill_exchange,
    prepare_prefill_exchange,
    prepare_routed_exchange,
    routed_exchange,
)
from sglang.kernels.ops.inkling2.linear import (
    linear,
    linear_k_group_partials,
    uses_k_groups,
)
from sglang.kernels.ops.inkling2.moe import (
    routed_partial,
    routed_rows,
    routed_swiglu,
    swiglu,
    unweighted_post_reorder,
)
from sglang.kernels.ops.inkling2.mxfp8 import (
    cutlass_weight_scales,
)
from sglang.kernels.ops.inkling2.mxfp8 import expert_matmul as mxfp8_matmul
from sglang.kernels.ops.inkling2.mxfp8 import (
    prefill_expert_matmul as mxfp8_prefill_matmul,
)
from sglang.kernels.ops.inkling2.mxfp8 import (
    prefill_swiglu_expert_matmul as mxfp8_prefill_swiglu_matmul,
)
from sglang.kernels.ops.inkling2.mxfp8 import (
    routed_expert_matmul as mxfp8_routed_matmul,
)
from sglang.kernels.ops.inkling2.mxfp8 import (
    routed_swiglu_expert_matmul as mxfp8_routed_swiglu_matmul,
)
from sglang.kernels.ops.inkling2.nvfp4 import expert_matmul as nvfp4_matmul
from sglang.kernels.ops.inkling2.nvfp4 import (
    routed_expert_matmul,
    routed_swiglu_expert_matmul,
    swizzle_weight_scales,
)
from sglang.kernels.ops.inkling2.route import (
    ROUTE_MAX_IDS,
    ROUTE_MAX_ROWS,
    route_metadata,
    route_topk,
)
from sglang.kernels.ops.moe.inkling_moe import (
    compute_expert_block_metadata,
    compute_grouped_gemm_metadata,
    get_src2dst,
    grouped_gemm_triton,
    post_reorder,
    pre_reorder,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig
from sglang.srt.layers.utils.common import set_weight_attrs
from sglang.srt.models.inkling_common.moe import renorm_topk_logits_scaled
from sglang.srt.runtime_context import (
    cutedsl_moe_max_num_tokens,
    get_parallel,
    get_platform,
)

BF16_COMBINE_MIN_TOKENS = 512


class RoutedOutput(NamedTuple):
    """Expert rows in routing order, all-reduced; combined by the consumer.
    top_k 0: rows are already combined per token (src2dst None)."""

    rows: torch.Tensor
    src2dst: torch.Tensor | None
    top_k: int


class Inkling2Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
            if config.use_gate_bias
            else None
        )
        self.register_buffer(
            "_one", torch.ones((), dtype=torch.float32), persistent=False
        )

    def forward(self, x):
        if uses_k_groups(x, self.weight):
            partials = linear_k_group_partials(x.contiguous(), self.weight)
        else:
            partials = linear(x, self.weight, out_dtype=torch.float32)[None]
        if self.topk == 8:
            _, ids, weights = route_topk(
                partials, self.bias, self.topk, self.route_scale, self._one
            )
            return weights, ids
        logits, ids = route_topk(partials, self.bias, self.topk)
        weights, _ = renorm_topk_logits_scaled(
            logits, ids, 0, self.route_scale, self._one
        )
        return weights, ids


class Inkling2Experts(nn.Module):
    def __init__(self, config, num_experts, checkpoint_prefix, quant_config):
        super().__init__()
        parallel = get_parallel()
        self.tp_rank, self.tp_size = parallel.moe_tp_rank, parallel.moe_tp_size
        self.ep_rank = parallel.moe_ep_rank
        self.ep_size = parallel.moe_ep_size
        self.num_global_experts = num_experts
        if num_experts % self.ep_size:
            raise ValueError("The expert count must be divisible by expert parallelism")
        num_experts //= self.ep_size
        if config.intermediate_size % self.tp_size:
            raise ValueError(
                "Expert intermediate size must be divisible by tensor parallelism"
            )
        intermediate = config.intermediate_size // self.tp_size
        self.quantize_w13 = quant_config is not None and quant_config.quantizes_input(
            checkpoint_prefix + "w13_weight"
        )
        self.quantize_w2 = quant_config is not None and quant_config.quantizes_input(
            checkpoint_prefix + "w2_weight"
        )
        self._nvfp4_scale_shapes: dict[str, tuple[int, ...]] = {}
        for name, shape, dim, quantized in (
            (
                "w13",
                (num_experts, 2 * intermediate, config.hidden_size),
                1,
                self.quantize_w13,
            ),
            (
                "w2",
                (num_experts, config.hidden_size, intermediate),
                2,
                self.quantize_w2,
            ),
        ):
            loader = partial(self._load_shard, dim=dim)
            nvfp4 = (
                quant_config is not None
                and checkpoint_prefix + name + "_weight" in quant_config.nvfp4_weights
            )
            if nvfp4 and self.tp_size != 1:
                raise ValueError(
                    "NVFP4 experts require expert parallelism so activation scales "
                    "cover each expert's full input width"
                )
            block_size = 16 if nvfp4 else 32
            if quantized and shape[-1] % block_size:
                raise ValueError(
                    f"Quantized tensor-parallel weight width must be divisible by {block_size}"
                )
            weight_shape = (*shape[:-1], shape[-1] // 2) if nvfp4 else shape
            weight_dtype = (
                torch.uint8
                if nvfp4
                else torch.float8_e4m3fn
                if quantized
                else torch.get_default_dtype()
            )
            weight = nn.Parameter(
                self._empty_weight(weight_shape, weight_dtype), requires_grad=False
            )
            self.register_parameter(name + "_weight", weight)
            set_weight_attrs(weight, {"weight_loader": loader})
            if quantized:
                scale = nn.Parameter(
                    torch.empty(
                        *shape[:-1],
                        shape[-1] // block_size,
                        dtype=torch.float8_e4m3fn if nvfp4 else torch.uint8,
                    ),
                    requires_grad=False,
                )
                self.register_parameter(name + "_scale", scale)
                if nvfp4:
                    # Replaced by the swizzled GEMM layout after loading.
                    self._nvfp4_scale_shapes[name] = tuple(scale.shape)
                    loader = partial(
                        self._load_shard, dim=dim, checkpoint_shape=tuple(scale.shape)
                    )
                set_weight_attrs(scale, {"weight_loader": loader})
            else:
                self.register_buffer(name + "_scale", None, persistent=False)
            if nvfp4:
                for suffix, size in (
                    ("amax", num_experts),
                    ("outer_scale_denominator", 1),
                ):
                    param = nn.Parameter(
                        torch.empty(size, dtype=torch.float32), requires_grad=False
                    )
                    self.register_parameter(name + "_" + suffix, param)
                    set_weight_attrs(
                        param,
                        {
                            "weight_loader": partial(
                                self._load_shard, dim=None, expert_axis=suffix == "amax"
                            )
                        },
                    )
        self.gate_cap, self.up_cap = config.mlp_gate_softcap, config.mlp_up_softcap
        self.sm90 = get_platform().is_sm90
        self.output_sizes = {"w13": 2 * intermediate, "w2": config.hidden_size}
        self._marlin = {}
        self.max_routed_ids = min(
            ROUTE_MAX_ROWS * config.num_experts_per_tok, ROUTE_MAX_IDS
        )
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.mega = None
        aligned = all(
            w.shape[1] % 128 == 0 and w.shape[2] % 64 == 0
            for w in (self.w13_weight, self.w2_weight)
        )
        dtype = self.w13_weight.dtype
        self._routed_kernels = (
            dtype
            if aligned
            and dtype == self.w2_weight.dtype
            and dtype in (torch.uint8, torch.float8_e4m3fn)
            else None
        )

    def _empty_weight(self, shape, dtype):
        if (
            torch.get_default_device().type == "cuda"
            and get_moe_a2a_backend().is_megamoe()
            and self.ep_size > 1
        ):
            weight = domain_paged_empty(shape, dtype)
            if weight is not None:
                return weight
        return torch.empty(shape, dtype=dtype)

    def _load_shard(self, param, loaded, dim, expert_axis=True, checkpoint_shape=None):
        # Reload in place: captured CUDA graphs hold the swizzled buffer address.
        swizzled = (
            checkpoint_shape is not None and tuple(param.shape) != checkpoint_shape
        )
        target = param.data.new_empty(checkpoint_shape) if swizzled else param.data
        expected = list(target.shape)
        if expert_axis:
            expected[0] *= self.ep_size
        if dim is not None:
            expected[dim] *= self.tp_size
        if tuple(loaded.shape) != tuple(expected):
            raise ValueError(
                f"Expert tensor shape mismatch: {loaded.shape} != {expected}"
            )
        if loaded.dtype != param.dtype:
            raise ValueError(
                f"Expert tensor dtype mismatch: {loaded.dtype} != {param.dtype}"
            )
        if expert_axis:
            loaded = loaded.chunk(self.ep_size, dim=0)[self.ep_rank]
        target.copy_(
            loaded if dim is None else loaded.chunk(self.tp_size, dim=dim)[self.tp_rank]
        )
        if swizzled:
            param.data.copy_(swizzle_weight_scales(target))

    def _mega_covers(self, low, high):
        return self.mega is not None and (
            low > high
            or self.mega.min_tokens <= low
            and high <= self.mega.max_tokens * self.ep_size
        )

    def prepare_routed_exchange(self):
        if self.ep_size == 1 or self.sm90:
            return
        group, hidden = get_parallel().moe_ep_group, self.w2_weight.shape[1]
        routed_tokens = self.max_routed_ids // self.top_k
        min_routed_tokens = -(-ROUTED_EXCHANGE_MIN_ROWS // self.top_k)
        if not (
            envs.SGLANG_OPT_INKLING2_EP_PARTIAL_COMBINE.get()
            or self._mega_covers(min_routed_tokens, routed_tokens)
        ):
            prepare_routed_exchange(group, self.max_routed_ids, hidden)
        prefill_tokens = min(cutedsl_moe_max_num_tokens(), PREFILL_EXCHANGE_MAX_TOKENS)
        if not self._mega_covers(routed_tokens + 1, prefill_tokens):
            prepare_prefill_exchange(group, prefill_tokens, self.top_k, hidden)

    def prepare_mxfp8_cutlass_scales(self):
        """CUTLASS-layout copies of the MXFP8 weight scales for the prefill GEMMs,
        rewritten in place on reload (captured CUDA graphs hold the addresses)."""
        if not (
            self.ep_size > 1
            and not self.sm90
            and self.w13_weight.dtype == self.w2_weight.dtype == torch.float8_e4m3fn
        ):
            return
        for name in ("w13", "w2"):
            current = getattr(self, name + "_cutlass_scale", None)
            setattr(
                self,
                name + "_cutlass_scale",
                cutlass_weight_scales(getattr(self, name + "_scale"), current),
            )

    def prepare_mega_moe(self):
        """DeepGEMM MegaMoE over the loaded expert tensors in place (`inkling2`
        layout); runs after the NVFP4 scale swizzle and the MXFP8 CUTLASS scales."""
        if not (
            get_moe_a2a_backend().is_megamoe()
            and self.ep_size > 1
            and not self.sm90
            and self._routed_kernels is not None
        ):
            return
        if (self.gate_cap, self.up_cap) != (4.0, 25.0):
            raise ValueError("MegaMoE SiTU needs MLP softcaps (4, 25)")
        if version("sgl-deep-gemm") != MEGA_MOE_DEEP_GEMM_VERSION:
            raise RuntimeError(
                f"Inkling2 MegaMoE needs sgl-deep-gemm {MEGA_MOE_DEEP_GEMM_VERSION} "
                "(docker/inkling2/apply_deep_gemm_patches.sh)"
            )
        import deep_gemm

        from sglang.srt.layers.moe.mega_moe import (
            _configure_mega_moe_deep_gemm_num_sms,
        )

        max_tokens = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()

        nvfp4 = self.w13_weight.dtype == torch.uint8
        kwargs = {"recipe": (1, 1, 16 if nvfp4 else 32), "weight_layout": "inkling2"}
        if nvfp4:
            l1 = (self.w13_weight.view(torch.int8), self.w13_scale.view(torch.int32))
            l2 = (self.w2_weight.view(torch.int8), self.w2_scale.view(torch.int32))
            # L2 activations take a static per-expert scale sized for the SiTU
            # bound |act| <= 4 * 25 * route_scale.
            act_scale = 6.0 * 448.0 / (4.0 * 25.0 * self.route_scale)
            l1_alpha = self.w13_amax / self.w13_outer_scale_denominator
            kwargs.update(
                use_x_scales=True,
                l1_alphas=torch.stack([l1_alpha, l1_alpha], dim=1).contiguous(),
                l2_alphas=(
                    self.w2_amax / self.w2_outer_scale_denominator / act_scale
                ).contiguous(),
                l2_act_scales=torch.full_like(self.w2_amax, act_scale),
            )
        else:
            l1 = (self.w13_weight, self.w13_cutlass_scale.view(torch.int32))
            l2 = (self.w2_weight, self.w2_cutlass_scale.view(torch.int32))
        mma_type = "nvfp4xnvfp4" if nvfp4 else "fp8xfp8"
        self.mega = SimpleNamespace(
            deep_gemm=deep_gemm,
            num_sms=_configure_mega_moe_deep_gemm_num_sms,
            nvfp4=nvfp4,
            mma_type=mma_type,
            l1=l1,
            l2=l2,
            kwargs=kwargs,
            min_tokens=envs.SGLANG_OPT_INKLING2_MEGA_MOE_MIN_TOKENS.get(),
            max_tokens=max_tokens,
            buffer=_mega_moe_buffer(
                get_parallel().moe_ep_group.device_group,
                self.num_global_experts,
                max_tokens,
                self.top_k,
                self.w2_weight.shape[1],
                self.w13_weight.shape[1] // 2,
                mma_type,
            ),
        )

    def _forward_mega(self, x, weights, ids, combine):
        """Each EP rank feeds its slice of the (replicated) tokens to MegaMoE, which
        writes the combined rows of every slice to every rank."""
        mega, deep_gemm = self.mega, self.mega.deep_gemm
        tokens = x.shape[0]
        per_rank = -(-tokens // self.ep_size)
        start = min(self.ep_rank * per_rank, tokens)
        local = min(tokens - start, per_rank)
        buf = mega.buffer
        deep_gemm.mega_moe_pre_dispatch(
            x[start : start + local],
            ids[start : start + local].int(),
            weights[start : start + local].float(),
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=local,
            group_size=16 if mega.nvfp4 else 32,
            mma_type=mega.mma_type,
            buf_x_scales=buf.x_scales if mega.nvfp4 else None,
        )
        if local < per_rank:
            buf.topk_idx[local:per_rank].fill_(-1)
        y = buf.gathered_y[: per_rank * self.ep_size]
        with mega.num_sms(deep_gemm):
            deep_gemm.fp8_fp4_mega_moe(
                y,
                mega.l1,
                mega.l2,
                buf,
                activation="situ",
                fast_math=True,
                broadcast_y=True,
                **mega.kwargs,
            )
        out = y[:tokens]
        return out if combine else RoutedOutput(out, None, 0)

    def swizzle_nvfp4_scales(self):
        """Convert freshly loaded NVFP4 block scales to the GEMM layout once (on Hopper,
        repack NVFP4 and MXFP8 experts for Marlin)."""
        if self.sm90:
            for name in ("w13", "w2"):
                weight = getattr(self, name + "_weight")
                scale = getattr(self, name + "_scale")
                if not weight.is_cuda or not weight.numel():
                    continue
                if name in self._nvfp4_scale_shapes:
                    self._marlin[name] = marlin_sm90.repack(weight.data, scale.data)
                elif weight.dtype == torch.float8_e4m3fn:
                    self._marlin[name] = marlin_sm90.repack_mxfp8(
                        weight.data, scale.data
                    )
                else:
                    continue
                weight.data = weight.data.new_empty((weight.shape[0], 0, 0))
                scale.data = scale.data.new_empty((scale.shape[0], 0, 0))
            if self._marlin:
                self._identity = torch.arange(
                    1 << 17, dtype=torch.int32, device=self.w13_weight.device
                )
            return
        for name, checkpoint_shape in self._nvfp4_scale_shapes.items():
            scale = getattr(self, name + "_scale")
            if scale.is_cuda and tuple(scale.shape) == checkpoint_shape:
                if torch.cuda.get_device_capability(scale.device) not in (
                    (10, 0),
                    (10, 3),
                    (10, 7),
                ):
                    raise ValueError(
                        "NVFP4 expert computation requires an SM100, SM103 or SM107 GPU"
                    )
                scale.data = swizzle_weight_scales(scale.data)

    def _quantized_matmul(self, name, x, metadata=None):
        weight, scale = getattr(self, name + "_weight"), getattr(self, name + "_scale")
        if weight.dtype == torch.uint8:
            return nvfp4_matmul(
                x,
                weight,
                scale,
                getattr(self, name + "_amax"),
                getattr(self, name + "_outer_scale_denominator"),
                metadata=metadata,
            )
        return mxfp8_matmul(x, weight, scale, metadata=metadata)

    def forward(self, x, weights, ids, combine=True):
        if x.shape[0] == 0:
            return x
        if self._marlin:
            return self._forward_source(x, weights, ids, combine)
        if (
            self.mega is not None
            and self.mega.min_tokens
            <= x.shape[0]
            <= self.mega.max_tokens * self.ep_size
        ):
            return self._forward_mega(x, weights, ids, combine)
        if self.ep_size > 1 and ids.numel() <= self.max_routed_ids:
            return self._forward_routed(x, weights, ids, combine)
        exchange = (
            prefill_exchange(get_parallel().moe_ep_group, ids.numel(), x.shape[1])
            if self.ep_size > 1 and x.dtype == torch.bfloat16
            else None
        )
        if exchange is not None and self._routed_kernels is not None:
            return self._forward_prefill(x, weights, ids, exchange)
        sorted_ids, permutation = torch.sort(ids.to(torch.int16).flatten(), stable=True)
        src2dst = get_src2dst(permutation)
        counts, offsets, block_offsets, schedule = compute_grouped_gemm_metadata(
            sorted_ids,
            self.num_global_experts,
            block_size_m=128,
        )
        if self.ep_size > 1:
            start = self.ep_rank * self.w13_weight.shape[0]
            end = start + self.w13_weight.shape[0]
            counts, offsets = counts[start:end], offsets[start : end + 1]
            block_offsets, schedule = compute_expert_block_metadata(
                counts, ids.numel(), block_size_m=128
            )
        inputs = pre_reorder(x, src2dst, ids.shape[1])
        metadata = (counts, offsets, schedule)
        gate_up = (
            self._quantized_matmul("w13", inputs, metadata)
            if self.quantize_w13
            else grouped_gemm_triton(
                inputs,
                self.w13_weight,
                self.w13_weight.shape[0],
                counts,
                offsets,
                block_offsets,
                schedule,
                block_size_m=128,
            )
        )
        if self.ep_size > 1:
            rows = torch.arange(gate_up.shape[0], device=x.device)
            outside = (rows < offsets[0]) | (rows >= offsets[-1])
            gate_up.masked_fill_(outside[:, None], 0)
        gammas = pre_reorder(weights.reshape(-1, 1), src2dst, 1).flatten()
        activated = swiglu(gate_up, gammas, self.gate_cap, self.up_cap)
        down = (
            self._quantized_matmul("w2", activated, metadata)
            if self.quantize_w2
            else grouped_gemm_triton(
                activated,
                self.w2_weight,
                self.w2_weight.shape[0],
                counts,
                offsets,
                block_offsets,
                schedule,
                block_size_m=128,
            )
        )
        if self.ep_size > 1:
            down.masked_fill_(outside[:, None], 0)
            down = get_parallel().moe_ep_group.all_reduce(down)
        return post_reorder(down, src2dst, torch.ones_like(weights))

    def _nvfp4_routed(self, x, dst2src, top_k, gammas, counts, offsets, schedule):
        accumulator, x_amax = routed_expert_matmul(
            x.contiguous(),
            dst2src,
            top_k,
            self.w13_weight,
            self.w13_scale,
            counts,
            offsets,
            schedule,
        )
        accumulator, x_amax = routed_swiglu_expert_matmul(
            accumulator,
            x_amax,
            self.w13_amax,
            self.w13_outer_scale_denominator,
            gammas,
            self.gate_cap,
            self.up_cap,
            self.w2_weight,
            self.w2_scale,
            counts,
            offsets,
            x.dtype,
            schedule,
        )
        return accumulator, (x_amax, self.w2_amax, self.w2_outer_scale_denominator)

    def _forward_prefill(self, x, weights, ids, exchange):
        """The grouped path for this rank's expert rows only: token rows are
        gathered into the quantized GEMM inputs, and the exchange combines rows
        across ranks instead of an all-reduce of every routed row."""
        num_local = self.w13_weight.shape[0]
        top_k = ids.shape[1]
        sorted_ids, dst2src = torch.sort(ids.to(torch.int16).flatten(), stable=True)
        dst2src = dst2src.int()
        src2dst = get_src2dst(dst2src)
        counts, offsets, _, _ = compute_grouped_gemm_metadata(
            sorted_ids, self.num_global_experts, block_size_m=128
        )
        start, end = self.ep_rank * num_local, (self.ep_rank + 1) * num_local
        counts, offsets = counts[start:end], offsets[start : end + 1]
        gammas = weights.flatten()[dst2src]
        if self._routed_kernels == torch.uint8:
            _, schedule = compute_expert_block_metadata(
                counts, ids.numel(), block_size_m=128
            )
            accumulator, nvfp4_scales = self._nvfp4_routed(
                x, dst2src, top_k, gammas, counts, offsets, schedule
            )
        else:
            gate_up = mxfp8_prefill_matmul(
                x,
                dst2src,
                top_k,
                self.w13_weight,
                self.w13_cutlass_scale,
                counts,
                offsets,
            )
            accumulator = mxfp8_prefill_swiglu_matmul(
                gate_up,
                gammas,
                self.gate_cap,
                self.up_cap,
                self.w2_weight,
                self.w2_cutlass_scale,
                counts,
                offsets,
            )
            nvfp4_scales = None
        routed_rows(
            accumulator,
            offsets,
            nvfp4_scales=nvfp4_scales,
            exchange=exchange,
            dst2src=dst2src,
        )
        return exchange.combine(src2dst, top_k)

    def _forward_source(self, x, weights, ids, combine):
        """SM90 NVFP4 experts on rows in source (token, top-k) order: Marlin gathers
        token rows itself, so tokens are quantized once and never reordered."""
        tokens, top_k = ids.shape
        num_local = self.w13_weight.shape[0]
        nvfp4 = self._marlin["w13"][2] is not None
        start = self.ep_rank * num_local
        flat = ids.flatten()
        n = flat.numel()
        if n <= ROUTE_MAX_IDS:
            _, dst2src, _, counts, offsets, *_ = route_metadata(
                flat, weights.flatten(), start, num_local, self.num_global_experts
            )
        else:
            sorted_ids, permutation = torch.sort(flat.to(torch.int16), stable=True)
            counts, offsets, _, _ = compute_grouped_gemm_metadata(
                sorted_ids, self.num_global_experts, block_size_m=128
            )
            counts = counts[start : start + num_local]
            offsets = offsets[start : start + num_local + 1]
            dst2src = permutation.to(torch.int32)
        if marlin_sm90.uses_gemv(self._marlin["w13"], tokens, num_local):
            matmul = partial(
                marlin_sm90.gemv,
                metadata=(counts, offsets, dst2src),
                tokens=tokens,
            )
        else:
            routing = marlin_sm90.route(counts, offsets, n, tokens, dst2src)
            matmul = partial(marlin_sm90.gemm, routing=routing)
        caps = (self.gate_cap, self.up_cap)
        if nvfp4:
            xq, x_amax = experts_sm90.fake_quantize(x.contiguous())
        else:
            xq = experts_sm90.fake_quantize_mxfp8(x.contiguous())
        acc = matmul(xq, self._marlin["w13"], self.output_sizes["w13"], top_k=top_k)
        if nvfp4:
            hidden, hidden_amax = marlin_sm90.swiglu_fake_quantize(
                acc,
                flat,
                weights.flatten(),
                (x_amax, self.w13_amax, self.w13_outer_scale_denominator),
                start,
                num_local,
                top_k,
                caps,
            )
            scales = (hidden_amax, self.w2_amax, self.w2_outer_scale_denominator)
        else:
            hidden = marlin_sm90.swiglu_fake_quantize_mxfp8(
                acc, flat, weights.flatten(), start, num_local, caps
            )
            scales = (None, None, None)
        down = matmul(hidden, self._marlin["w2"], self.output_sizes["w2"], top_k=1)
        group = get_parallel().moe_ep_group
        out = (
            oneshot_all_reduce_input(
                group, tokens, down.shape[1], torch.bfloat16, tag="moe"
            )
            if tokens > BF16_COMBINE_MIN_TOKENS
            else None
        )
        sums = marlin_sm90.source_partial(
            down,
            flat,
            scales,
            start,
            num_local,
            top_k,
            out=(
                out
                if out is not None
                else multicast_all_reduce_input(
                    group, (tokens, down.shape[1]), torch.float32
                )
            ),
        )
        if out is not None:
            combined = oneshot_all_reduce(sums, group, acc_f32=True, tag="moe")
        else:
            combined = multicast_all_reduce(sums, group)
        if not combine:
            return RoutedOutput(combined, self._identity[:tokens], 1)
        return combined.to(x.dtype)

    def _routed_matmul(self, name, x, metadata, block_offsets):
        weight = getattr(self, name + "_weight")
        if weight.dtype == torch.uint8:
            accumulator, x_amax = nvfp4_matmul(
                x,
                weight,
                getattr(self, name + "_scale"),
                getattr(self, name + "_amax"),
                getattr(self, name + "_outer_scale_denominator"),
                metadata=metadata,
                raw=True,
            )
            scales = (
                x_amax,
                getattr(self, name + "_amax"),
                getattr(self, name + "_outer_scale_denominator"),
            )
            return accumulator, scales
        if weight.dtype == torch.float8_e4m3fn:
            return mxfp8_matmul(
                x, weight, getattr(self, name + "_scale"), metadata=metadata
            ), None
        counts, offsets, schedule = metadata
        return grouped_gemm_triton(
            x,
            weight,
            weight.shape[0],
            counts,
            offsets,
            block_offsets,
            schedule,
            block_size_m=128,
        ), None

    def _forward_routed(self, x, weights, ids, combine):
        num_local = self.w13_weight.shape[0]
        (
            src2dst,
            dst2src,
            gammas,
            counts,
            offsets,
            block_offsets,
            schedule,
            sorted_ids,
        ) = route_metadata(
            ids.flatten(),
            weights.flatten(),
            self.ep_rank * num_local,
            num_local,
            self.num_global_experts,
        )
        exchange = (
            routed_exchange(get_parallel().moe_ep_group, ids.numel(), x.shape[1])
            if x.dtype == torch.bfloat16
            else None
        )
        route = (src2dst, sorted_ids, num_local, exchange)
        if self._routed_kernels == torch.uint8:
            accumulator, scales = self._nvfp4_routed(
                x, dst2src, ids.shape[1], gammas, counts, offsets, schedule
            )
            return self._routed_output(
                accumulator,
                offsets,
                src2dst,
                ids.shape[1],
                x.dtype,
                scales,
                combine,
                route,
            )
        metadata = (counts, offsets, schedule)
        if self._routed_kernels == torch.float8_e4m3fn:
            gate_up = mxfp8_routed_matmul(
                x, dst2src, ids.shape[1], self.w13_weight, self.w13_scale, metadata
            )
            down = mxfp8_routed_swiglu_matmul(
                gate_up,
                gammas,
                self.gate_cap,
                self.up_cap,
                self.w2_weight,
                self.w2_scale,
                metadata,
                ids.shape[0],
            )
            return self._routed_output(
                down, offsets, src2dst, ids.shape[1], x.dtype, None, combine, route
            )
        inputs = pre_reorder(x, src2dst, ids.shape[1])
        gate_up, scales = self._routed_matmul("w13", inputs, metadata, block_offsets)
        activated = routed_swiglu(
            gate_up,
            gammas,
            offsets,
            self.gate_cap,
            self.up_cap,
            out_dtype=x.dtype,
            nvfp4_scales=scales,
        )
        down, scales = self._routed_matmul("w2", activated, metadata, block_offsets)
        return self._routed_output(
            down, offsets, src2dst, ids.shape[1], x.dtype, scales, combine, route
        )

    def _routed_output(
        self, down, offsets, src2dst, top_k, out_dtype, nvfp4_scales, combine, route
    ):
        group = get_parallel().moe_ep_group
        _, sorted_ids, num_local, exchange = route
        if (
            self.ep_size > 1
            and envs.SGLANG_OPT_INKLING2_EP_PARTIAL_COMBINE.get()
            and src2dst.numel() >= PARTIAL_COMBINE_MIN_TOKENS * top_k
        ):
            partial_sums = routed_partial(
                down, offsets, src2dst, top_k, nvfp4_scales=nvfp4_scales
            )
            out = group.all_reduce(partial_sums).to(out_dtype)
            return out if combine else RoutedOutput(out, None, 0)
        if exchange is not None:
            routed_rows(down, offsets, nvfp4_scales=nvfp4_scales, exchange=exchange)
            return exchange.combine(src2dst, sorted_ids, num_local, top_k)
        down = routed_rows(
            down, offsets, out_dtype=out_dtype, nvfp4_scales=nvfp4_scales
        )
        down = group.all_reduce(down)
        if not combine:
            return RoutedOutput(down, src2dst, top_k)
        return unweighted_post_reorder(down, src2dst, top_k)


PREFILL_EXCHANGE_MAX_TOKENS = 32768
PARTIAL_COMBINE_MIN_TOKENS = 16
MEGA_MOE_DEEP_GEMM_VERSION = "0.2.6+inkling2.0f96f72.1"

_MEGA_MOE_BUFFERS = {}
_MEGA_MOE_MMA_TYPES = ("nvfp4xnvfp4", "fp8xfp8")


def _mega_moe_buffer(group, experts, max_tokens, top_k, hidden, intermediate, mma):
    """One symmetric allocation, sized for the larger MMA type, backs the buffers
    of both the NVFP4 and the MXFP8 layers (layers run in sequence)."""
    import deep_gemm
    from deep_gemm.mega import SymmBuffer

    key = (id(group), experts, max_tokens, top_k, hidden, intermediate)
    if (key, mma) not in _MEGA_MOE_BUFFERS:
        args = (group, experts, max_tokens, top_k, hidden, intermediate)
        if key not in _MEGA_MOE_BUFFERS:
            tokens = -(-max_tokens // deep_gemm._C.get_token_alignment_for_mega_moe())
            tokens *= deep_gemm._C.get_token_alignment_for_mega_moe()
            largest = max(
                _MEGA_MOE_MMA_TYPES,
                key=lambda m: deep_gemm._C.get_symm_buffer_size_for_mega_moe(
                    group.size(),
                    experts,
                    tokens,
                    top_k,
                    hidden,
                    intermediate,
                    m,
                    "situ",
                    0,
                ),
            )
            _MEGA_MOE_BUFFERS[key] = SymmBuffer(
                *args, mma_type=largest, activation="situ", gather_output=True
            )
        _MEGA_MOE_BUFFERS[key, mma] = SymmBuffer(
            *args,
            mma_type=mma,
            activation="situ",
            base=_MEGA_MOE_BUFFERS[key],
            gather_output=True,
        )
    return _MEGA_MOE_BUFFERS[key, mma]


class _DomainPages:
    """Owns the reservation behind a `domain_paged_empty` tensor."""

    def __init__(self, reservation, num_bytes):
        self.reservation = reservation
        self.__cuda_array_interface__ = {
            "shape": (num_bytes,),
            "typestr": "|u1",
            "data": (reservation.base, False),
            "version": 3,
        }

    def __del__(self):
        torch.cuda.synchronize()
        self.reservation.close()


def domain_paged_empty(shape, dtype):
    """Uninitialized contiguous `(E, N, K)` expert weights on the current GPU whose
    pages each sit in the locality domain holding most of their bytes, row slice `d`
    of every expert counting for domain `d` (the N slice MegaMoE gives that domain's
    SMs). Returns None on GPUs with one locality domain."""
    from cuda.bindings import driver as cu

    from sglang.srt.utils.cuda_vmm_utils import (
        VmmReservation,
        check_drv,
        get_allocation_granularity,
        make_device_allocation_prop,
    )

    attr = getattr(
        cu.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT", None
    )
    if attr is None:
        return None
    device = torch.cuda.current_device()
    err, domains = cu.cuDeviceGetAttribute(attr, device)
    if err != cu.CUresult.CUDA_SUCCESS or domains < 2 or shape[1] % domains:
        return None
    page = get_allocation_granularity(
        make_device_allocation_prop(device, handle_types=None),
        cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
    )
    expert = math.prod(shape[1:]) * dtype.itemsize
    total = expert * shape[0]
    slice_bytes = expert // domains

    def owned(end, domain):
        whole, rest = divmod(min(end, total), expert)
        return whole * slice_bytes + min(
            max(rest - domain * slice_bytes, 0), slice_bytes
        )

    page_domains = [
        max(range(domains), key=lambda d: owned(p + page, d) - owned(p, d))
        for p in range(0, total, page)
    ]
    props = []
    for domain in range(domains):
        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = (
            cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
        )
        prop.location.localized.deviceId = device
        prop.location.localized.localityDomainId = domain
        props.append(prop)
    reservation = VmmReservation(
        len(page_domains) * page, props[0], device, alignment=page
    )
    try:
        offset = 0
        for domain, run in groupby(page_domains):
            pages = len(list(run))
            # The driver rounds locality-domain allocations up to a power of two pages.
            while pages:
                size = (1 << (pages.bit_length() - 1)) * page
                handle = check_drv(
                    cu.cuMemCreate(size, props[domain], 0), "cuMemCreate(locality)"
                )
                try:
                    reservation.map_existing(offset, size, handle)
                finally:
                    check_drv(cu.cuMemRelease(handle), "cuMemRelease(locality)")
                offset += size
                pages -= size // page
    except BaseException:
        reservation.close()
        raise
    out = torch.as_tensor(_DomainPages(reservation, total), device=f"cuda:{device}")
    return out.view(dtype).view(shape)


class Inkling2MoE(nn.Module):
    def __init__(
        self, config, layer_id, prefix, quant_config: Inkling2QuantConfig | None
    ):
        super().__init__()
        parallel = get_parallel()
        if parallel.moe_dp_size != 1 or (
            parallel.moe_ep_size > 1 and parallel.moe_tp_size > 1
        ):
            raise ValueError("Inkling2 experts support tensor or expert parallelism")
        self.gate = Inkling2Gate(config)
        checkpoint_prefix = "model." + prefix + ".experts."
        self.experts = Inkling2Experts(
            config, config.n_routed_experts, checkpoint_prefix, quant_config
        )

    def forward(self, x, forward_batch=None, combine=True):
        """combine=False may return a RoutedOutput instead of the combined output."""
        if x.shape[0] == 0:
            return x
        weights, ids = self.gate(x)
        if not combine and get_parallel().moe_tp_size == 1:
            return self.experts(x, weights, ids, combine=False)
        out = self.experts(x, weights, ids)
        if get_parallel().moe_tp_size > 1:
            out = all_reduce(out, get_parallel().moe_tp_group)
        return out
