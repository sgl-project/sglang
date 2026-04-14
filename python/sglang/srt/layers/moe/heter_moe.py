"""Heterogeneous-precision MoE layer.

Stores expert weights in multiple precisions (e.g., BF16 + INT4).
Classifies experts per-batch based on token load, runs separate
group-GEMMs per precision, sums outputs.

Integration approach (per reference.md):
  1. Load BF16 model normally via standard path (--model-path)
  2. Post-load: swap FusedMoE → HeterFusedMoE
     - BF16 expert weights transferred from the loaded FusedMoE
     - INT4 expert weights loaded from secondary GPTQ checkpoint
     - GPTQ weights repacked to Marlin format
  3. Forward: per-batch policy assigns experts to groups,
     each group dispatched through its own kernel

Dispatch approach (following TRT-LLM reference):
  - All groups keep full [num_experts, ...] weight tensors
  - Per-group: zero topk_weights for non-group experts
  - Kernel naturally skips experts with zero weights (no token-expert pairs)
  - No weight subsetting or expert ID remapping at runtime
  - Stable tensor shapes → CUDA graph compatible
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import outplace_fused_experts
from sglang.srt.layers.moe.heter_policy import (
    HeterDispatchPolicy,
    create_policy,
)

logger = logging.getLogger(__name__)


def _parse_heter_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        cfg = json.load(f)
    groups = cfg["groups"]
    total_ratio = sum(g["size_ratio"] for g in groups)
    assert abs(total_ratio - 1.0) < 1e-3, (
        f"size_ratios must sum to 1.0, got {total_ratio}"
    )
    return cfg


class HeterFusedMoE(nn.Module):
    """Multi-precision MoE layer with per-batch dynamic expert assignment.

    Created post-load from an already-loaded BF16 FusedMoE plus an INT4
    GPTQ checkpoint. Each precision group has its own full [num_experts, ...]
    weight tensors. At forward time, routing weights are zeroed for non-group
    experts so each kernel processes only its assigned experts.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        heter_config: Dict[str, Any],
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.dtype = dtype
        self.device = device or torch.device("cuda")

        self.group_cfgs = heter_config["groups"]
        self.num_groups = len(self.group_cfgs)
        self.group_ratios = [g["size_ratio"] for g in self.group_cfgs]

        policy_name = heter_config.get("policy", "expert_load")
        policy_kwargs = heter_config.get("policy_params", {})
        self.policy: HeterDispatchPolicy = create_policy(
            policy_name,
            num_experts=num_experts,
            group_size_ratios=self.group_ratios,
            device=self.device,
            **policy_kwargs,
        )

        self._int4_is_k_full = True
        self._int4_group_size = 128

        self._init_group_weights()

    def _init_group_weights(self) -> None:
        """Create weight containers for groups specified in the config.

        Only creates weights for groups defined at __init__ time (direct
        construction path, e.g. for testing). The from_fused_moe() path
        transfers weights from an existing FusedMoE instead.
        """
        E, H, I = self.num_experts, self.hidden_size, self.intermediate_size
        for idx, gcfg in enumerate(self.group_cfgs):
            num_bits = gcfg.get("num_bits", 16)
            if num_bits == 16:
                if not hasattr(self, "w13_weight"):
                    self._init_bf16_params(E, H, I)
            elif num_bits == 4:
                if not hasattr(self, "int4_w13_qweight"):
                    self._init_int4_params(E, H, I, gcfg)
            elif num_bits == 8:
                if not hasattr(self, "w13_weight"):
                    self._init_int8_params(E, H, I)

    def init_fake_weights(self, seed: int = 0) -> None:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        for name, param in self.named_parameters():
            if param.dtype in (torch.bfloat16, torch.float16, torch.float32):
                param.data.normal_(0, 0.02, generator=gen)
            elif param.dtype == torch.int8:
                param.data.copy_(
                    torch.randint(
                        -128, 127, param.shape, device=self.device, dtype=torch.int8
                    )
                )
            elif param.dtype == torch.int32:
                param.data.copy_(
                    torch.randint(
                        0, 2**31 - 1, param.shape, device=self.device, dtype=torch.int32
                    )
                )

    @classmethod
    def from_fused_moe(
        cls,
        fused_moe: nn.Module,
        heter_config: Dict[str, Any],
    ) -> "HeterFusedMoE":
        """Create HeterFusedMoE from an already-loaded BF16 FusedMoE.

        Transfers BF16 expert weights from the FusedMoE. INT4 weights
        are loaded separately via load_int4_weights().
        """
        E = fused_moe.w13_weight.shape[0]
        # w13_weight: [E, 2*I_part, H], w2_weight: [E, H, I_part]
        I_part = fused_moe.w2_weight.shape[2]
        H = fused_moe.w13_weight.shape[2]
        device = fused_moe.w13_weight.device
        dtype = fused_moe.w13_weight.dtype

        instance = cls(
            num_experts=E,
            hidden_size=H,
            intermediate_size=I_part,
            top_k=getattr(fused_moe, "top_k", 1),
            heter_config=heter_config,
            dtype=dtype,
            device=device,
        )

        instance.w13_weight = nn.Parameter(
            fused_moe.w13_weight.data, requires_grad=False
        )
        instance.w2_weight = nn.Parameter(fused_moe.w2_weight.data, requires_grad=False)

        return instance

    def _init_bf16_params(self, E: int, H: int, I: int) -> None:
        """Create BF16 weight containers."""
        self.register_parameter(
            "w13_weight",
            nn.Parameter(
                torch.empty(E, 2 * I, H, dtype=self.dtype, device=self.device),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "w2_weight",
            nn.Parameter(
                torch.empty(E, H, I, dtype=self.dtype, device=self.device),
                requires_grad=False,
            ),
        )

    def _init_int8_params(self, E: int, H: int, I: int) -> None:
        """Create INT8 weight and scale containers."""
        self.register_parameter(
            "w13_weight",
            nn.Parameter(
                torch.empty(E, 2 * I, H, dtype=torch.int8, device=self.device),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "w2_weight",
            nn.Parameter(
                torch.empty(E, H, I, dtype=torch.int8, device=self.device),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "w13_weight_scale",
            nn.Parameter(
                torch.ones(E, 2 * I, 1, dtype=torch.float32, device=self.device),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "w2_weight_scale",
            nn.Parameter(
                torch.ones(E, H, 1, dtype=torch.float32, device=self.device),
                requires_grad=False,
            ),
        )

    def _init_int4_params(self, E: int, H: int, I: int, gcfg: Dict) -> None:
        """Create GPTQ-format INT4 parameter containers.

        Shapes match GPTQMarlinMoEMethod.create_weights for compatibility
        with gptq_marlin_moe_repack.
        """
        group_size = gcfg.get("group_size", 128)
        pack_factor = 8  # 8 INT4 values packed in one INT32

        # qweight: packed INT4 in INT32
        # w13: fused gate+up, shape [E, H//pack_factor, 2*I]
        self.int4_w13_qweight = nn.Parameter(
            torch.zeros(
                E, H // pack_factor, 2 * I, dtype=torch.int32, device=self.device
            ),
            requires_grad=False,
        )
        # w2: down_proj, shape [E, I//pack_factor, H]
        self.int4_w2_qweight = nn.Parameter(
            torch.zeros(E, I // pack_factor, H, dtype=torch.int32, device=self.device),
            requires_grad=False,
        )

        # scales: [E, K//group_size, N]
        scales_size_w13 = H // group_size
        scales_size_w2 = I // group_size
        self.int4_w13_scales = nn.Parameter(
            torch.zeros(
                E, scales_size_w13, 2 * I, dtype=torch.float16, device=self.device
            ),
            requires_grad=False,
        )
        self.int4_w2_scales = nn.Parameter(
            torch.zeros(E, scales_size_w2, H, dtype=torch.float16, device=self.device),
            requires_grad=False,
        )

        # qzeros
        self.int4_w13_qzeros = nn.Parameter(
            torch.zeros(
                E,
                scales_size_w13,
                2 * I // pack_factor,
                dtype=torch.int32,
                device=self.device,
            ),
            requires_grad=False,
        )
        self.int4_w2_qzeros = nn.Parameter(
            torch.zeros(
                E,
                scales_size_w2,
                H // pack_factor,
                dtype=torch.int32,
                device=self.device,
            ),
            requires_grad=False,
        )

        # g_idx (will be emptied for non-desc_act models)
        self.int4_w13_g_idx = nn.Parameter(
            torch.zeros(E, H, dtype=torch.int32, device=self.device),
            requires_grad=False,
        )
        self.int4_w2_g_idx = nn.Parameter(
            torch.zeros(E, I, dtype=torch.int32, device=self.device),
            requires_grad=False,
        )
        self.int4_w13_g_idx_sort_indices = nn.Parameter(
            torch.zeros(E, H, dtype=torch.int32, device=self.device),
            requires_grad=False,
        )
        self.int4_w2_g_idx_sort_indices = nn.Parameter(
            torch.zeros(E, I, dtype=torch.int32, device=self.device),
            requires_grad=False,
        )

        self._int4_group_size = group_size
        self._int4_is_k_full = True  # Will be set based on desc_act

    def load_int4_weights(self, checkpoint_path: str, layer_id: int) -> None:
        """Load INT4 expert weights from a GPTQ checkpoint into our params.

        Reads per-expert GPTQ tensors (qweight, scales, qzeros, g_idx)
        and stacks them into [E, ...] format.
        """
        from safetensors import safe_open

        E = self.num_experts
        I = self.intermediate_size

        # Find safetensors files
        if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".safetensors"):
            st_files = [checkpoint_path]
        else:
            st_files = sorted(
                f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")
            )
            st_files = [os.path.join(checkpoint_path, f) for f in st_files]

        prefix = f"model.layers.{layer_id}.mlp.experts."

        for st_file in st_files:
            with safe_open(st_file, framework="pt", device=str(self.device)) as f:
                for key in f.keys():
                    if not key.startswith(prefix):
                        continue

                    # Parse: model.layers.{l}.mlp.experts.{e}.{proj}.{attr}
                    suffix = key[len(prefix) :]
                    # suffix like "0.gate_proj.qweight"
                    match = re.match(
                        r"(\d+)\.(gate_proj|up_proj|down_proj)\.(qweight|scales|qzeros|g_idx)",
                        suffix,
                    )
                    if not match:
                        continue

                    expert_id = int(match.group(1))
                    proj = match.group(2)
                    attr = match.group(3)

                    if expert_id >= E:
                        continue

                    tensor = f.get_tensor(key)

                    self._fill_int4_param(expert_id, proj, attr, tensor)

        logger.info(
            f"Loaded INT4 expert weights for layer {layer_id} from {checkpoint_path}"
        )

    def _fill_int4_param(
        self, expert_id: int, proj: str, attr: str, tensor: torch.Tensor
    ) -> None:
        """Fill a single expert's INT4 parameter into the fused [E, ...] tensor."""
        I = self.intermediate_size

        if proj == "down_proj":
            # down_proj → w2
            target = getattr(self, f"int4_w2_{attr}")
            target.data[expert_id].copy_(tensor)
        elif proj in ("gate_proj", "up_proj"):
            # gate_proj/up_proj → w13 (fused along last dim)
            target = getattr(self, f"int4_w13_{attr}")
            if attr == "g_idx":
                # g_idx is per-input-dim, same for gate and up
                # Just store it (gate_proj's is canonical)
                if proj == "gate_proj":
                    target.data[expert_id].copy_(tensor)
            else:
                # Fuse along last dimension: gate=[0:I], up=[I:2*I]
                # For qzeros, the last dim is divided by pack_factor
                last_dim = tensor.shape[-1]
                if proj == "gate_proj":
                    target.data[expert_id, :, :last_dim].copy_(tensor)
                else:  # up_proj
                    target.data[expert_id, :, last_dim : 2 * last_dim].copy_(tensor)

    def repack_int4_to_marlin(self) -> None:
        """Repack GPTQ INT4 weights to Marlin format for fused_marlin_moe.

        Must be called after load_int4_weights() and before forward().
        """
        from sglang.srt.layers.quantization.gptq import (
            gptq_marlin_moe_repack,
            marlin_moe_permute_scales,
        )

        E = self.num_experts
        H = self.hidden_size
        I = self.intermediate_size
        group_size = self._int4_group_size
        num_bits = 4
        pack_factor = 8
        device = self.device

        # No desc_act: reset g_idx to empty
        self.int4_w13_g_idx = nn.Parameter(
            torch.empty((E, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        self.int4_w2_g_idx = nn.Parameter(
            torch.empty((E, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        self.int4_w13_g_idx_sort_indices = nn.Parameter(
            torch.empty((E, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        self.int4_w2_g_idx_sort_indices = nn.Parameter(
            torch.empty((E, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        self._int4_is_k_full = True

        # Repack qweight: GPTQ → Marlin layout
        marlin_w13 = gptq_marlin_moe_repack(
            self.int4_w13_qweight.data,
            self.int4_w13_g_idx_sort_indices.data,
            self.int4_w13_qweight.shape[1] * pack_factor,  # size_k = H
            self.int4_w13_qweight.shape[2],  # size_n = 2*I
            num_bits,
        )
        self.int4_w13_qweight = nn.Parameter(marlin_w13, requires_grad=False)

        marlin_w2 = gptq_marlin_moe_repack(
            self.int4_w2_qweight.data,
            self.int4_w2_g_idx_sort_indices.data,
            self.int4_w2_qweight.shape[1] * pack_factor,  # size_k = I
            self.int4_w2_qweight.shape[2],  # size_n = H
            num_bits,
        )
        self.int4_w2_qweight = nn.Parameter(marlin_w2, requires_grad=False)

        # Permute scales for Marlin
        marlin_w13_scales = marlin_moe_permute_scales(
            s=self.int4_w13_scales.data,
            size_k=I,  # intermediate_size_per_partition
            size_n=self.int4_w13_scales.shape[2],
            group_size=group_size,
        )
        self.int4_w13_scales = nn.Parameter(marlin_w13_scales, requires_grad=False)

        marlin_w2_scales = marlin_moe_permute_scales(
            s=self.int4_w2_scales.data,
            size_k=I * group_size // (I // group_size)
            if group_size != -1
            else pack_factor,
            size_n=self.int4_w2_scales.shape[2],
            group_size=group_size,
        )
        self.int4_w2_scales = nn.Parameter(marlin_w2_scales, requires_grad=False)

        # Convert scales to model dtype (Marlin expects same dtype as hidden_states)
        if self.dtype == torch.bfloat16:
            self.int4_w13_scales.data = self.int4_w13_scales.data.to(torch.bfloat16)
            self.int4_w2_scales.data = self.int4_w2_scales.data.to(torch.bfloat16)

        logger.info(
            f"Repacked INT4 weights to Marlin format: "
            f"w13={list(self.int4_w13_qweight.shape)}, "
            f"w2={list(self.int4_w2_qweight.shape)}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output,
    ) -> torch.Tensor:
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
        router_logits = topk_output.router_logits

        # Sentinel = -1: Triton (BF16/INT8) fully skips expert ID -1.
        group_dispatches = self.policy.dispatch(topk_ids, topk_weights, sentinel=-1)
        # print(group_dispatches)

        output = None

        for group_idx, gcfg in enumerate(self.group_cfgs):
            group_ids, group_weights = group_dispatches[group_idx]

            num_bits = gcfg.get("num_bits", 16)

            if num_bits == 16:
                group_out = outplace_fused_experts(
                    hidden_states,
                    self.w13_weight,
                    self.w2_weight,
                    group_weights,
                    group_ids,
                )
            elif num_bits == 4:
                # expert_map enables is_ep=True inside fused_marlin_moe,
                # which makes the Marlin GEMM kernel skip expert_id=-1
                # blocks (marlin_template.h:380) and zeros cache3 for
                # unwritten slots.  The tensor itself is never forwarded
                # to the CUDA kernel — only its existence matters.
                _dummy_expert_map = torch.empty(
                    0, dtype=torch.int32, device=hidden_states.device
                )
                group_out = fused_marlin_moe(
                    hidden_states=hidden_states,
                    w1=self.int4_w13_qweight,
                    w2=self.int4_w2_qweight,
                    w1_scale=self.int4_w13_scales,
                    w2_scale=self.int4_w2_scales,
                    gating_output=router_logits,
                    topk_weights=group_weights,
                    topk_ids=group_ids,
                    expert_map=_dummy_expert_map,
                    num_bits=4,
                    is_k_full=self._int4_is_k_full,
                    inplace=False,
                )
            elif num_bits == 8:
                group_out = outplace_fused_experts(
                    hidden_states,
                    self.w13_weight,
                    self.w2_weight,
                    group_weights,
                    group_ids,
                    use_int8_w8a8=True,
                    per_channel_quant=True,
                    w1_scale=getattr(self, "w13_weight_scale", None),
                    w2_scale=getattr(self, "w2_weight_scale", None),
                )
            else:
                raise ValueError(f"Unsupported num_bits={num_bits}")

            if output is None:
                output = group_out
            else:
                output.add_(group_out)

        return output if output is not None else torch.zeros_like(hidden_states)


def apply_heter_precision(
    model: nn.Module,
    config_path: str,
    device: torch.device,
) -> None:
    """Post-load: swap FusedMoE layers with HeterFusedMoE.

    1. Parse heter_config.json
    2. For each MoE layer, create HeterFusedMoE from the loaded BF16 FusedMoE
    3. Load INT4 expert weights from secondary checkpoint
    4. Repack GPTQ → Marlin
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    heter_config = _parse_heter_config(config_path)

    # Find INT4 group's checkpoint path
    int4_checkpoint = None
    for gcfg in heter_config["groups"]:
        if gcfg.get("num_bits", 16) == 4:
            int4_checkpoint = gcfg["checkpoint"]
            break

    if int4_checkpoint is None:
        logger.warning("No INT4 group found in heter_config, skipping heter precision")
        return

    # Find all MoE layers and swap
    layers_module = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers_module = model.model.layers
    elif hasattr(model, "layers"):
        layers_module = model.layers

    if layers_module is None:
        raise ValueError("Cannot find model layers for heter precision application")

    num_swapped = 0
    for layer_id, layer in enumerate(layers_module):
        # Find the FusedMoE experts module
        fused_moe = None
        moe_parent = None
        moe_attr = None

        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            candidate = layer.mlp.experts
            if isinstance(candidate, FusedMoE):
                fused_moe = candidate
                moe_parent = layer.mlp
                moe_attr = "experts"

        if fused_moe is None:
            continue

        # Create HeterFusedMoE from the loaded BF16 FusedMoE
        heter_moe = HeterFusedMoE.from_fused_moe(fused_moe, heter_config)

        # Load INT4 expert weights
        heter_moe.load_int4_weights(int4_checkpoint, layer_id)

        # Repack GPTQ → Marlin
        heter_moe.repack_int4_to_marlin()

        # Swap
        setattr(moe_parent, moe_attr, heter_moe)
        num_swapped += 1

        if layer_id % 10 == 0:
            logger.info(f"Swapped layer {layer_id} FusedMoE → HeterFusedMoE")

    # Free memory from the old FusedMoE modules
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    logger.info(
        f"Applied heterogeneous precision to {num_swapped} MoE layers. "
        f"Groups: {[g['name'] for g in heter_config['groups']]}"
    )
