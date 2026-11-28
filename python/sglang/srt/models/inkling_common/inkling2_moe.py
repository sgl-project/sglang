import math
from functools import partial
from itertools import groupby
from types import SimpleNamespace
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.activation.inkling2_swiglu_quant_lowm_sm120 import (
    nvfp4_swiglu_quant_bf16_lowm,
)
from sglang.kernels.ops.communication.inkling2_comm import (
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
from sglang.kernels.ops.gemm import inkling2_grouped_sm120 as grouped_sm120
from sglang.kernels.ops.gemm import inkling2_marlin_sm90 as marlin_sm90
from sglang.kernels.ops.gemm.inkling2_linear import (
    linear,
    linear_k_group_partials,
    uses_k_groups,
)
from sglang.kernels.ops.gemm.inkling2_moe_gemm_lowm_sm120 import (
    mxfp8_expert_matmul_lowm,
    nvfp4_expert_matmul_lowm_scaled,
)
from sglang.kernels.ops.gemm.inkling2_mxfp8 import (
    routed_expert_matmul as mxfp8_routed_matmul,
)
from sglang.kernels.ops.gemm.inkling2_mxfp8 import (
    routed_swiglu_expert_matmul as mxfp8_routed_swiglu_matmul,
)
from sglang.kernels.ops.gemm.inkling2_nvfp4 import (
    routed_expert_matmul,
    routed_swiglu_expert_matmul,
)
from sglang.kernels.ops.moe import inkling2_experts_sm90 as experts_sm90
from sglang.kernels.ops.moe import inkling2_experts_sm120 as experts_sm120
from sglang.kernels.ops.moe import inkling2_moe_lowm_sm120 as moe_lowm_sm120
from sglang.kernels.ops.moe.inkling2_moe import (
    routed_partial,
    routed_rows,
    unweighted_post_reorder,
)
from sglang.kernels.ops.moe.inkling2_route import (
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
from sglang.kernels.ops.quantization.inkling2_mxfp8_scales_sm120 import (
    mxfp8_weight_scales,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.srt.layers.utils.common import set_weight_attrs
from sglang.srt.models.inkling_common.moe import renorm_topk_logits_scaled
from sglang.srt.runtime_context import (
    cutedsl_moe_max_num_tokens,
    get_parallel,
    get_platform,
)

BF16_COMBINE_MIN_TOKENS = 512


def _capped_swiglu(gate_up, gammas, gate_cap, up_cap, offsets=None):
    """swiglu of interleaved (gate, up) columns, each capped as cap * tanh(x / cap),
    with the sigmoid of the uncapped gate; rows outside [offsets[0], offsets[-1])
    are zero. Only BF16 experts take this path."""
    gate, up = gate_up.float().unflatten(-1, (-1, 2)).unbind(-1)
    sigmoid = torch.sigmoid(gate)
    if gate_cap > 0:
        gate = gate_cap * torch.tanh(gate / gate_cap)
    if up_cap > 0:
        up = up_cap * torch.tanh(up / up_cap)
    out = gate * sigmoid * up * gammas.float()[:, None]
    if offsets is not None:
        rows = torch.arange(out.shape[0], device=out.device)
        out[(rows < offsets[0]) | (rows >= offsets[-1])] = 0
    return out.to(gate_up.dtype)


def _has_forward_hooks(*modules):
    return any(m._forward_hooks or m._forward_pre_hooks for m in modules)


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

    def partials(self, x):
        """FP32 router split-K partials, [groups, tokens, experts]."""
        if uses_k_groups(x, self.weight):
            return linear_k_group_partials(x.contiguous(), self.weight)
        return linear(x, self.weight, out_dtype=torch.float32)[None]

    def forward(self, x):
        partials = self.partials(x)
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
        platform = get_platform()
        # SM120 has no NVLink multicast.
        self.multicast_combine = not platform.is_sm120
        self.output_sizes = {"w13": 2 * intermediate, "w2": config.hidden_size}
        self._marlin = {}
        self.max_routed_ids = min(
            ROUTE_MAX_ROWS * config.num_experts_per_tok, ROUTE_MAX_IDS
        )
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.mega = None
        self._mega_padded = None
        self.w13_cutlass_scale = None
        self.w2_cutlass_scale = None
        self.w13_cute_scale = None
        self.w2_cute_scale = None
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
        quantized = {self.w13_weight.dtype, self.w2_weight.dtype} - {
            torch.get_default_dtype()
        }
        self._lacks_routed_kernels = bool(quantized) and self._routed_kernels is None
        # SM120 NVFP4 and MXFP8 experts use the SM120 block-scaled grouped GEMMs,
        # whose activation scale layouts assume local rows start at row 0 and, for
        # MXFP8, pack four 32-wide scale blocks per word.
        self.native_sm120 = (
            platform.is_sm120
            and self.ep_size == 1
            and self._routed_kernels is not None
            and experts_sm120.can_quantize(
                widths=(config.hidden_size, intermediate), experts=num_experts
            )
            and (
                dtype == torch.uint8
                or all(w.shape[2] % 128 == 0 for w in (self.w13_weight, self.w2_weight))
            )
        )
        # SM90 lacks block-scaled MMA, and the SM100 kernels need tcgen05: other
        # SM90 and SM120 experts run through Marlin.
        self.use_marlin = platform.is_sm90 or (
            platform.is_sm120 and not self.native_sm120
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
            param.data.copy_(swizzle_blockscale(target).flatten(1))

    def _mega_covers(self, low, high):
        return self.mega is not None and (low > high or self.mega.min_tokens <= low)

    def prepare_routed_exchange(self):
        if self.ep_size == 1 or self.use_marlin:
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
        """CUTLASS-layout copies of the MXFP8 weight scales for MegaMoE,
        rewritten in place on reload (captured CUDA graphs hold the addresses)."""
        if not (
            get_moe_a2a_backend().is_megamoe()
            and self.ep_size > 1
            and not self.use_marlin
            and self.w13_weight.dtype == self.w2_weight.dtype == torch.float8_e4m3fn
        ):
            return
        for name in ("w13", "w2"):
            swizzled = swizzle_blockscale(
                getattr(self, name + "_scale").view(torch.uint8)
            )
            current = getattr(self, name + "_cutlass_scale")
            if current is None:
                setattr(self, name + "_cutlass_scale", swizzled)
            else:
                current.copy_(swizzled)

    def prepare_mega_moe(self):
        """DeepGEMM MegaMoE over the loaded expert tensors in place
        (`gate_up_interleaved` layout); runs after the NVFP4 scale swizzle and the
        MXFP8 CUTLASS scales."""
        if not (
            get_moe_a2a_backend().is_megamoe()
            and self.ep_size > 1
            and not self.use_marlin
            and self._routed_kernels is not None
        ):
            return
        if (self.gate_cap, self.up_cap) != (4.0, 25.0):
            raise ValueError("MegaMoE SiTU needs MLP softcaps (4, 25)")
        import deep_gemm

        from sglang.srt.layers.moe.mega_moe import (
            _configure_mega_moe_deep_gemm_num_sms,
        )

        max_tokens = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()

        nvfp4 = self.w13_weight.dtype == torch.uint8
        kwargs = {
            "recipe": (1, 1, 16 if nvfp4 else 32),
            "weight_layout": "gate_up_interleaved",
        }
        if nvfp4:
            w13, w13_sf, w2, w2_sf = self._mega_nvfp4_weights()
            l1 = (w13.view(torch.int8), w13_sf.view(torch.int32))
            l2 = (w2.view(torch.int8), w2_sf.view(torch.int32))
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
            local_input=envs.SGLANG_OPT_INKLING2_MEGA_MOE_LOCAL_INPUT.get(),
            buffer=_mega_moe_buffer(
                get_parallel().moe_ep_group.device_group,
                self.num_global_experts,
                max_tokens,
                self.top_k,
                self.w2_weight.shape[1],
                l1[0].shape[1] // 2,
                mma_type,
            ),
        )

    def mega_token_slice(self, tokens):
        """(start, count) of the tokens this rank dispatches to MegaMoE."""
        per_rank = -(-tokens // self.ep_size)
        start = min(self.ep_rank * per_rank, tokens)
        return start, min(tokens - start, per_rank)

    def mega_local_input_rows(self, tokens):
        """mega_token_slice(tokens) if the MoE input is needed for those rows only."""
        if (
            self._marlin
            or self._lacks_routed_kernels
            or self.mega is None
            or not self.mega.local_input
            or not self.mega.min_tokens <= tokens <= self.mega.max_tokens * self.ep_size
        ):
            return None
        start, count = self.mega_token_slice(tokens)
        return (start, count) if count >= MEGA_MOE_LOCAL_INPUT_MIN_ROWS else None

    def _mega_nvfp4_weights(self):
        """NVFP4 MegaMoE tiles the L2 K (intermediate) dimension in blocks of 256;
        narrower multiples run on copies zero-padded to the next multiple (zero
        weights, zero scales). Rewritten in place on reload."""
        tensors = (self.w13_weight, self.w13_scale, self.w2_weight, self.w2_scale)
        experts, hidden, intermediate = self.w2_weight.shape
        intermediate *= 2
        pad = -intermediate % 256
        if not pad:
            return tensors
        w13, w13_sf, w2, w2_sf = (t.view(torch.uint8) for t in tensors)
        # Swizzled scales are [E][N / 128][K / 64][512 B]: w13 gains N tiles, w2 K atoms.
        padded = (
            F.pad(w13, (0, 0, 0, 2 * pad)),
            F.pad(
                w13_sf.view(experts, 2 * intermediate // 128, -1),
                (0, 0, 0, 2 * pad // 128),
            ),
            F.pad(w2, (0, pad // 2)),
            F.pad(
                w2_sf.view(experts, hidden // 128, intermediate // 64, -1),
                (0, 0, 0, pad // 64),
            ),
        )
        if self._mega_padded is None:
            with torch.device(w13.device):
                self._mega_padded = tuple(
                    torch.empty_like(t)
                    if i % 2
                    else self._empty_weight(t.shape, t.dtype)
                    for i, t in enumerate(padded)
                )
        for dst, src in zip(self._mega_padded, padded):
            dst.copy_(src)
        w13, w13_sf, w2, w2_sf = self._mega_padded
        return w13, w13_sf.view(experts, -1), w2, w2_sf.view(experts, -1)

    def _forward_mega(self, x, weights, ids, combine, tokens=None):
        """Each EP rank feeds its slice of the (replicated) tokens to MegaMoE, which
        writes the combined rows of every slice to every rank. A forward larger than
        the MegaMoE buffer runs in buffer-sized pieces. With `tokens`, the inputs
        hold only this rank's slice; such a forward always fits the buffer."""
        mega, deep_gemm = self.mega, self.mega.deep_gemm
        sliced = tokens is not None
        tokens = tokens if sliced else x.shape[0]
        cap = mega.max_tokens * self.ep_size
        if tokens > cap:
            assert not sliced
            out = torch.empty_like(x)
            for s in range(0, tokens, cap):
                rows = slice(s, s + cap)
                out[rows] = self._forward_mega(x[rows], weights[rows], ids[rows], True)
            return out if combine else RoutedOutput(out, None, 0)
        per_rank = -(-tokens // self.ep_size)
        start, local = self.mega_token_slice(tokens)
        if not sliced:
            rows = slice(start, start + local)
            x, ids, weights = x[rows], ids[rows], weights[rows]
        buf = mega.buffer
        deep_gemm.mega_moe_pre_dispatch(
            x,
            ids.int(),
            weights.float(),
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
        """Convert freshly loaded NVFP4 block scales to the GEMM layout once (on Hopper
        and SM120 without the native path, repack NVFP4 and MXFP8 experts for
        Marlin)."""
        if self.native_sm120:
            self._prepare_sm120_scales()
            return
        if self.use_marlin:
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
                scale.data = swizzle_blockscale(scale.data).flatten(1)

    def _prepare_sm120_scales(self):
        """SM120 GEMM layouts of the loaded weight scales: NVFP4 scales swizzled once
        (_load_shard re-swizzles reloads), MXFP8 scales packed for the CuTe GEMM and
        rewritten in place on reload (captured CUDA graphs hold the addresses)."""
        if self._routed_kernels == torch.uint8:
            for scale, checkpoint_shape in (
                (self.w13_scale, self._nvfp4_scale_shapes["w13"]),
                (self.w2_scale, self._nvfp4_scale_shapes["w2"]),
            ):
                if scale.is_cuda and tuple(scale.shape) == checkpoint_shape:
                    scale.data = swizzle_blockscale(scale.data).flatten(1)
        elif self.w13_scale.is_cuda and self.w2_scale.is_cuda:
            self.w13_cute_scale = mxfp8_weight_scales(
                self.w13_scale, out=self.w13_cute_scale
            )
            self.w2_cute_scale = mxfp8_weight_scales(
                self.w2_scale, out=self.w2_cute_scale
            )

    def forward(self, x, weights, ids, combine=True):
        if x.shape[0] == 0:
            return x
        if self.native_sm120:
            return self._forward_sm120(x, weights, ids, combine)
        if self._marlin:
            return self._forward_source(x, weights, ids, combine)
        if self._lacks_routed_kernels:
            raise ValueError(
                "Inkling2 quantized experts run only on the routed kernels, which need "
                "one NVFP4 or MXFP8 precision with 128 x 64 aligned weights; got "
                f"ep_size={self.ep_size}, "
                f"w13 {tuple(self.w13_weight.shape)} {self.w13_weight.dtype}, "
                f"w2 {tuple(self.w2_weight.shape)} {self.w2_weight.dtype}"
            )
        if self.mega is not None and self.mega.min_tokens <= x.shape[0]:
            return self._forward_mega(x, weights, ids, combine)
        local = self.ep_size == 1 and self._routed_kernels is not None
        if (self.ep_size > 1 or local) and ids.numel() <= self.max_routed_ids:
            return self._forward_routed(x, weights, ids, combine)
        if local:
            return self._forward_prefill(x, weights, ids, None)
        exchange = (
            prefill_exchange(get_parallel().moe_ep_group, ids.numel(), x.shape[1])
            if self.ep_size > 1 and x.dtype == torch.bfloat16
            else None
        )
        if exchange is not None and self._routed_kernels is not None:
            return self._forward_prefill(x, weights, ids, exchange)
        if self._routed_kernels is not None:
            raise ValueError(
                f"Inkling2 quantized experts got a {x.shape[0]}-token MoE forward, "
                "beyond the prefill exchange and MegaMoE capacity; lower "
                f"--chunked-prefill-size to at most {PREFILL_EXCHANGE_MAX_TOKENS}"
            )
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
        gate_up = self._bf16_matmul("w13", inputs, metadata, block_offsets)
        if self.ep_size > 1:
            rows = torch.arange(gate_up.shape[0], device=x.device)
            outside = (rows < offsets[0]) | (rows >= offsets[-1])
            gate_up.masked_fill_(outside[:, None], 0)
        gammas = pre_reorder(weights.reshape(-1, 1), src2dst, 1).flatten()
        activated = _capped_swiglu(gate_up, gammas, self.gate_cap, self.up_cap)
        down = self._bf16_matmul("w2", activated, metadata, block_offsets)
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
        across ranks instead of an all-reduce of every routed row (without an
        exchange, a single rank combines its rows in place)."""
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
        _, schedule = compute_expert_block_metadata(
            counts, ids.numel(), block_size_m=128
        )
        if self._routed_kernels == torch.uint8:
            accumulator, nvfp4_scales = self._nvfp4_routed(
                x, dst2src, top_k, gammas, counts, offsets, schedule
            )
        else:
            accumulator = self._mxfp8_routed(
                x, dst2src, top_k, gammas, (counts, offsets, schedule), x.shape[0]
            )
            nvfp4_scales = None
        if exchange is None:
            down = routed_rows(
                accumulator, offsets, out_dtype=x.dtype, nvfp4_scales=nvfp4_scales
            )
            return unweighted_post_reorder(down, src2dst, top_k)
        routed_rows(
            accumulator,
            offsets,
            nvfp4_scales=nvfp4_scales,
            exchange=exchange,
            dst2src=dst2src,
        )
        return exchange.combine(src2dst, top_k)

    def _forward_source(self, x, weights, ids, combine):
        """Marlin experts on rows in source (token, top-k) order: Marlin
        gathers token rows itself, so tokens are quantized once and never reordered."""
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
        if self.multicast_combine:
            bf16 = tokens > BF16_COMBINE_MIN_TOKENS
            out = oneshot_all_reduce_input(
                group,
                tokens,
                down.shape[1],
                torch.bfloat16 if bf16 else torch.float32,
                tag="moe" if bf16 else "moe_f32",
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
                combined = oneshot_all_reduce(
                    sums, group, acc_f32=bf16, tag="moe" if bf16 else "moe_f32"
                )
            else:
                combined = multicast_all_reduce(sums, group)
        else:
            combined = group.all_reduce(
                marlin_sm90.source_partial(down, flat, scales, start, num_local, top_k)
            )
        if not combine:
            return RoutedOutput(combined, self._identity[:tokens], 1)
        return combined.to(x.dtype)

    def _mxfp8_routed(self, x, dst2src, top_k, gammas, metadata, tokens):
        gate_up = mxfp8_routed_matmul(
            x, dst2src, top_k, self.w13_weight, self.w13_scale, metadata
        )
        return mxfp8_routed_swiglu_matmul(
            gate_up,
            gammas,
            self.gate_cap,
            self.up_cap,
            self.w2_weight,
            self.w2_scale,
            metadata,
            tokens,
        )

    def _forward_sm120_lowm(self, x, ids, src2dst, dst2src, gammas, offsets, combine):
        x_c = x.contiguous()
        if self._routed_kernels == torch.uint8:
            values, scales, amax = moe_lowm_sm120.nvfp4_quant_dedup_lowm(
                x_c, ids, src2dst, offsets
            )
            gate_up = nvfp4_expert_matmul_lowm_scaled(
                values,
                scales,
                self.w13_weight,
                self.w13_scale,
                offsets,
                amax,
                self.w13_amax,
                self.w13_outer_scale_denominator,
            )
            values, scales, amax = nvfp4_swiglu_quant_bf16_lowm(
                gate_up, gammas, offsets, self.gate_cap, self.up_cap
            )
            down = nvfp4_expert_matmul_lowm_scaled(
                values,
                scales,
                self.w2_weight,
                self.w2_scale,
                offsets,
                amax,
                self.w2_amax,
                self.w2_outer_scale_denominator,
            )
        elif not moe_lowm_sm120.can_use_lowm_mxfp8(x.shape[0]):
            down = self._sm120_mxfp8(x_c, dst2src, gammas, offsets, 8)
        else:
            values, scales = experts_sm120.mxfp8_routed_quantize(
                x=x_c, dst2src=dst2src, top_k=8, offsets=offsets
            )
            gate_up = mxfp8_expert_matmul_lowm(
                values, scales, self.w13_weight, self.w13_scale, offsets
            )
            values, scales = experts_sm120.mxfp8_routed_swiglu_quantize(
                acc=gate_up,
                gammas=gammas,
                offsets=offsets,
                gate_cap=self.gate_cap,
                up_cap=self.up_cap,
            )
            down = mxfp8_expert_matmul_lowm(
                values, scales, self.w2_weight, self.w2_scale, offsets
            )
        if not combine:
            return RoutedOutput(down, src2dst, 8)
        return moe_lowm_sm120.moe_combine_bf16(down, src2dst, 8)

    def _forward_sm120(self, x, weights, ids, combine):
        """SM120 NVFP4 or MXFP8 experts on rows in expert order (single EP rank):
        routed quantization into the SM120 grouped GEMMs, then the per-token sum of
        the (scaled) rows."""
        tokens, top_k = ids.shape
        if moe_lowm_sm120.can_use_lowm_moe(
            tokens,
            self.num_global_experts,
            top_k,
            x.shape[1],
            self.output_sizes["w13"] // 2,
            x.device,
        ):
            src2dst, dst2src, gammas, offsets = moe_lowm_sm120.route_metadata_lowm(
                ids, weights, self.num_global_experts
            )
            return self._forward_sm120_lowm(
                x, ids, src2dst, dst2src, gammas, offsets, combine
            )
        src2dst, dst2src, gammas, offsets = self._sm120_route(ids, weights)
        if self._routed_kernels == torch.uint8:
            down, scales = self._sm120_nvfp4(
                x=x.contiguous(),
                dst2src=dst2src,
                gammas=gammas,
                offsets=offsets,
                top_k=top_k,
                tokens=tokens,
            )
        else:
            down = self._sm120_mxfp8(
                x=x.contiguous(),
                dst2src=dst2src,
                gammas=gammas,
                offsets=offsets,
                top_k=top_k,
            )
            scales = None
        out = routed_partial(down, offsets, src2dst, top_k, nvfp4_scales=scales).to(
            x.dtype
        )
        return out if combine else RoutedOutput(out, None, 0)

    def _sm120_route(self, ids, weights):
        """(src2dst, dst2src, FP32 gammas in sorted order, expert offsets) of the
        routed rows of this single EP rank."""
        flat = ids.flatten()
        if flat.numel() <= ROUTE_MAX_IDS:
            src2dst, dst2src, gammas, _, offsets, *_ = route_metadata(
                flat,
                weights.flatten(),
                0,
                self.num_global_experts,
                self.num_global_experts,
            )
            return src2dst, dst2src, gammas, offsets
        sorted_ids, dst2src = torch.sort(flat.to(torch.int16), stable=True)
        dst2src = dst2src.int()
        _, offsets, _, _ = compute_grouped_gemm_metadata(
            sorted_ids, self.num_global_experts, block_size_m=128
        )
        return (
            get_src2dst(dst2src),
            dst2src,
            weights.flatten().float()[dst2src],
            offsets,
        )

    def _sm120_nvfp4(self, x, dst2src, gammas, offsets, top_k, tokens):
        values, scales, amax = experts_sm120.nvfp4_routed_quantize(
            x=x, dst2src=dst2src, top_k=top_k, offsets=offsets
        )
        gate_up = grouped_sm120.nvfp4_expert_matmul(
            values=values,
            scales=scales,
            weight=self.w13_weight,
            weight_scales=self.w13_scale,
            offsets=offsets,
            tokens=tokens,
        )
        values, scales, amax = experts_sm120.nvfp4_routed_swiglu_quantize(
            acc=gate_up,
            acc_amax=amax,
            w_amax=self.w13_amax,
            w_denominator=self.w13_outer_scale_denominator,
            gammas=gammas,
            offsets=offsets,
            gate_cap=self.gate_cap,
            up_cap=self.up_cap,
        )
        down = grouped_sm120.nvfp4_expert_matmul(
            values=values,
            scales=scales,
            weight=self.w2_weight,
            weight_scales=self.w2_scale,
            offsets=offsets,
            tokens=tokens,
        )
        return down, (amax, self.w2_amax, self.w2_outer_scale_denominator)

    def _sm120_mxfp8(self, x, dst2src, gammas, offsets, top_k):
        values, scales = experts_sm120.mxfp8_routed_quantize(
            x=x, dst2src=dst2src, top_k=top_k, offsets=offsets
        )
        gate_up = grouped_sm120.mxfp8_expert_matmul(
            values=values,
            scales=scales,
            weight=self.w13_weight,
            weight_scales=self.w13_cute_scale,
            offsets=offsets,
        )
        values, scales = experts_sm120.mxfp8_routed_swiglu_quantize(
            acc=gate_up,
            gammas=gammas,
            offsets=offsets,
            gate_cap=self.gate_cap,
            up_cap=self.up_cap,
        )
        return grouped_sm120.mxfp8_expert_matmul(
            values=values,
            scales=scales,
            weight=self.w2_weight,
            weight_scales=self.w2_cute_scale,
            offsets=offsets,
        )

    def _bf16_matmul(self, name, x, metadata, block_offsets):
        weight = getattr(self, name + "_weight")
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
        )

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
            down = self._mxfp8_routed(
                x, dst2src, ids.shape[1], gammas, metadata, ids.shape[0]
            )
        else:
            inputs = pre_reorder(x, src2dst, ids.shape[1])
            gate_up = self._bf16_matmul("w13", inputs, metadata, block_offsets)
            activated = _capped_swiglu(
                gate_up, gammas, self.gate_cap, self.up_cap, offsets
            )
            down = self._bf16_matmul("w2", activated, metadata, block_offsets)
        return self._routed_output(
            down, offsets, src2dst, ids.shape[1], x.dtype, None, combine, route
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
# Above 256 rows the router and RMSNorm pick the same kernel at any row count.
MEGA_MOE_LOCAL_INPUT_MIN_ROWS = 257
PARTIAL_COMBINE_MIN_TOKENS = 16

_MEGA_MOE_BUFFERS = {}
_MEGA_MOE_MMA_TYPES = ("nvfp4xnvfp4", "fp8xfp8")


def _mega_moe_buffer(group, experts, max_tokens, top_k, hidden, intermediate, mma):
    """One symmetric allocation, sized for the larger MMA type, backs the buffers
    of both the NVFP4 and the MXFP8 layers (layers run in sequence)."""
    import deep_gemm
    from deep_gemm.mega import SymmBuffer

    # NVFP4 layers run at the intermediate width padded to 256, MXFP8 layers unpadded.
    sized = -(-intermediate // 256) * 256
    key = (id(group), experts, max_tokens, top_k, hidden, sized)
    args = (group, experts, max_tokens, top_k, hidden, intermediate)
    if (key, intermediate, mma) not in _MEGA_MOE_BUFFERS:
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
                    sized,
                    m,
                    "situ",
                    0,
                ),
            )
            _MEGA_MOE_BUFFERS[key] = SymmBuffer(
                *args[:-1],
                sized,
                mma_type=largest,
                activation="situ",
                gather_output=True,
            )
        _MEGA_MOE_BUFFERS[key, intermediate, mma] = SymmBuffer(
            *args,
            mma_type=mma,
            activation="situ",
            base=_MEGA_MOE_BUFFERS[key],
            gather_output=True,
        )
    return _MEGA_MOE_BUFFERS[key, intermediate, mma]


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

    def local_input_rows(self, tokens, combine):
        """(start, count) of the MoE-input rows this rank needs, or None for all."""
        if not combine or get_parallel().moe_tp_size > 1:
            return None
        return self.experts.mega_local_input_rows(tokens)

    def forward(self, x, forward_batch=None, combine=True, total_tokens=None):
        """combine=False may return a RoutedOutput instead of the combined output.
        With total_tokens, x holds only local_input_rows(total_tokens, combine)."""
        if x.shape[0] == 0:
            return x
        if (
            self.experts.native_sm120
            and total_tokens is None
            and get_parallel().moe_tp_size == 1
            and x.shape[0] <= moe_lowm_sm120.LOWM_FUSED_ROUTE_MAX_TOKENS
            and moe_lowm_sm120.can_use_lowm_moe(
                x.shape[0],
                self.experts.num_global_experts,
                self.gate.topk,
                x.shape[1],
                self.experts.output_sizes["w13"] // 2,
                x.device,
            )
            # The fused route calls neither module: keep their --forward-hooks.
            and not _has_forward_hooks(self.gate, self.experts)
        ):
            ids, _, src2dst, dst2src, gammas, offsets = (
                moe_lowm_sm120.route_topk_meta_lowm(
                    self.gate.partials(x),
                    self.gate.bias,
                    self.gate.route_scale,
                    self.gate._one,
                )
            )
            return self.experts._forward_sm120_lowm(
                x, ids, src2dst, dst2src, gammas, offsets, combine=combine
            )
        weights, ids = self.gate(x)
        if total_tokens is not None:
            return self.experts._forward_mega(
                x, weights, ids, combine, tokens=total_tokens
            )
        if not combine and get_parallel().moe_tp_size == 1:
            return self.experts(x, weights, ids, combine=False)
        out = self.experts(x, weights, ids)
        if get_parallel().moe_tp_size > 1:
            out = all_reduce(out, get_parallel().moe_tp_group)
        return out
