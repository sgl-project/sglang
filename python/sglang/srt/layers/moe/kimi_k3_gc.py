"""Optional GC decode for Kimi K3 with replicated shared experts and static EP placement.

GC weights expert outputs during combine. Native Mega MoE applies the routing
weights before quantizing the W2 input, so their FP8 rounding can differ.
"""

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import is_tbo_enabled
from sglang.srt.runtime_context import get_exec, get_lora, get_parallel

logger = logging.getLogger(__name__)


class KimiK3GCDecode:
    def __init__(self, moe, workspace, zero_prefix, *, routed_scale):
        from gc_moe import fp8_fp4_gc_moe_k3, transform_weights_for_gc_moe_k3

        self.run = fp8_fp4_gc_moe_k3
        self.workspace = workspace
        self.zero_prefix = zero_prefix
        self.routed = moe.experts._gc_k3_weights
        self.shared = transform_weights_for_gc_moe_k3(
            moe.shared_experts.gate_up_proj.weight, moe.shared_experts.down_proj.weight
        )
        self.decode_only = moe.experts._gc_k3_decode_only
        if self.decode_only:
            # The worker cannot fall back to the dense shared MLP. Keep the
            # registered parameters and the adapter on the same packed storage.
            moe.shared_experts.gate_up_proj.weight.data = self.shared[0]
            moe.shared_experts.down_proj.weight.data = self.shared[1]
        self.front = moe._front_w.detach()
        self.bias = moe.gate.e_score_correction_bias.detach()
        self.norm = moe.routed_expert_norm.weight.detach()
        self.up = moe.routed_expert_up_proj.weight.detach()
        self.eps = moe.routed_expert_norm.variance_epsilon
        self.renormalize = moe.topk.topk_config.renormalize
        self.scale = routed_scale
        self.attn_dp_size = get_parallel().attn_dp_size

    def use_for(self, batch):
        # DP shards can be idle while their peers decode or extend. All EP ranks
        # must choose the same collective, including ranks with zero live rows.
        mode = batch.global_forward_mode
        if mode is None and (self.attn_dp_size == 1 or self.decode_only):
            mode = batch.forward_mode
        if self.decode_only:
            if mode is None or not mode.is_decode_or_idle():
                raise RuntimeError(f"Unexpected forward mode on GC-only decode worker: {mode}")
            return True
        return mode is not None and mode.is_decode()

    def __call__(self, x, prefix, valid):
        out = torch.empty_like(x)
        if prefix is None:
            # Native K3 rounds the projected latent to BF16 before adding shared.
            prefix = self.zero_prefix[: x.shape[0]]
        if valid is not None:
            # SGLang graph buffers hold the live row count in a one-element tensor.
            valid = valid.reshape(())
        self.run(
            out,
            *self.routed,
            self.workspace,
            hidden_states=x,
            front_weight=self.front,
            router_bias=self.bias,
            shared_l1_weights=self.shared[0],
            shared_l2_weights=self.shared[1],
            norm_weight=self.norm,
            up_proj_weight=self.up,
            prefix_sum=prefix,
            norm_eps=self.eps,
            routed_scaling_factor=self.scale,
            route_renormalize=self.renormalize,
            valid_token_count=valid,
        )
        return out


@torch.no_grad()
def prepare_gc_decode(model):
    from sglang.srt.models.kimi_k3 import KimiK3MoE

    layers = [
        m
        for m in model.modules()
        if isinstance(m, KimiK3MoE) and m.experts._prepare_gc_k3
    ]
    if not layers:
        return
    if (
        is_tbo_enabled()
        or get_exec().moe.init_expert_location != "trivial"
        or torch.cuda.get_device_capability()[0] != 10
    ):
        raise ValueError(
            "GC decode requires SM100, serialized forwards and contiguous EP placement"
        )
    if (
        get_lora().enable_lora
        or get_exec().features.enable_return_routed_experts
        or get_exec().moe.expert_distribution_recorder_mode is not None
    ):
        raise ValueError("GC decode requires static weights and no routed-expert capture")

    from gc_moe import get_symm_buffer_for_gc_moe_k3

    workspaces = {}
    capacity = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    for m in layers:
        if m._gc_decode is not None:
            continue
        if not (
            m._shared_experts_tp1
            and m.shared_experts is not None
            and m._ep_front_eligible
            and m._front_w.dtype == torch.bfloat16
            and m.routed_expert_norm is not None
            and hasattr(m.experts, "_gc_k3_weights")
        ):
            raise ValueError(
                "GC decode requires the BF16 EP front, replicated BF16 shared experts and MXFP4 routed weights"
            )
        routed_scale = m.routed_scaling_factor
        if routed_scale is None:
            routed_scale = 1.0
        if routed_scale != 1.0 and m.experts.should_fuse_routed_scaling_factor_in_topk:
            raise ValueError("GC decode applies routed scaling after routed combine")
        hs, si = m.shared_experts.down_proj.weight.shape
        key = (
            m.experts.num_experts,
            capacity,
            m._mega_top_k,
            m.moe_hidden_size,
            m._mega_intermediate_size,
            m.num_shared_experts,
            hs,
            si,
        )
        if key not in workspaces:
            ws = get_symm_buffer_for_gc_moe_k3(
                get_parallel().moe_ep_group.device_group,
                *key[:5],
                num_shared_experts=m.num_shared_experts,
                shared_hidden=hs,
                shared_intermediate_hidden=si,
            )
            zeros = m._front_w.new_zeros((capacity, hs))
            workspaces[key] = ws, zeros
        m._gc_decode = KimiK3GCDecode(m, *workspaces[key], routed_scale=routed_scale)
        m._derived_weight_cache_error = (
            "GC decode caches packed expert weights; restart to reload weights"
        )
    logger.info("Prepared Kimi K3 GC decode for %d layers", len(layers))
