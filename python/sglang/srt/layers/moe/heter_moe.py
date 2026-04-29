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

Attention INT4 swap (optional):
  When ``heter_config["attention_num_bits"] == 4``, ``apply_heter_precision``
  also replaces every layer's ``self_attn.qkv_proj`` and ``self_attn.o_proj``
  with INT4 GPTQ-Marlin linears loaded from the same checkpoint as the INT4
  MoE group. Attention is fixed-precision (no per-batch dispatch / policy);
  the swap reuses the standard ``GPTQMarlinLinearMethod`` path used by
  ``--quantization gptq_marlin``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

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
    # ``size_ratio`` is only consumed by score-based policies (expert_load,
    # confidence, total_weight, random) to pick top-K experts per group.
    # ``expert_batch`` decides per-expert hot/cold at runtime from token
    # counts and ignores the ratios entirely -- so they're optional there.
    # NOTE: ``group_size`` on INT4 groups is the GPTQ quantization group
    # size (e.g. 128 scales per K), unrelated to the precision groups.
    policy = cfg.get("policy", "expert_load")
    # Policies that don't need static size_ratios (decide hot/cold at
    # runtime from token counts / curve lookup).
    if policy in ("expert_batch", "efficiency_promotion"):
        for g in groups:
            g.setdefault("size_ratio", 0.0)
    else:
        total_ratio = sum(g["size_ratio"] for g in groups)
        assert abs(total_ratio - 1.0) < 1e-3, (
            f"size_ratios must sum to 1.0, got {total_ratio}"
        )

    # ``bf16_promotion_threshold`` is a required top-level field. The runtime
    # promotes any expert with routed-token count >= threshold to BF16 in the
    # ABC dispatch loop, regardless of which scoring policy is in use. It is
    # NOT a policy-specific parameter -- do not nest it under ``policy_params``.
    if "bf16_promotion_threshold" not in cfg:
        # Old configs put the threshold under ``policy_params.threshold`` --
        # surface a clear migration error rather than silently defaulting.
        legacy = (cfg.get("policy_params") or {}).get("threshold")
        legacy_hint = (
            f" (legacy ``policy_params.threshold``={legacy} found -- "
            "move it to top-level ``bf16_promotion_threshold`` and remove "
            "from policy_params)"
            if legacy is not None else ""
        )
        raise ValueError(
            f"heter_config {config_path}: missing required top-level "
            f"``bf16_promotion_threshold``{legacy_hint}"
        )
    thr = cfg["bf16_promotion_threshold"]
    if not isinstance(thr, int) or isinstance(thr, bool) or thr < 0:
        raise ValueError(
            f"bf16_promotion_threshold must be a non-negative int, got {thr!r}"
        )
    # Load per-layer INT4-only expert lists if specified.
    # Format: {"layer_id": [expert_ids...], ...}
    # These experts have NO BF16 weights — always routed to INT4 kernel.
    int4_only_file = cfg.get("int4_only_experts_file")
    if int4_only_file is not None:
        with open(int4_only_file) as f:
            cfg["_int4_only_by_layer"] = json.load(f)
    else:
        cfg["_int4_only_by_layer"] = {}
    # Load per-layer BF16-only expert lists if specified.
    # Format: {"layer_id": [expert_ids...], ...}
    # These experts have NO INT4 weights — always routed to BF16 kernel.
    # If ALL experts in a layer are bf16-only, no INT4 weights are loaded
    # for that layer (saves VRAM).
    #
    # NOTE: This is NOT used in normal online serving. It exists as a VRAM
    # optimization for the per-layer INT4 sensitivity sweep
    # (expert_precision_assignment/sensitivity/per_moe_layer/), where only
    # one layer at a time is tested with INT4. Marking all other layers as
    # bf16-only avoids loading ~16 GB of unused INT4 weights.
    # Online serving uses {int4-only, heter(dual BF16+INT4)} experts only.
    bf16_only_file = cfg.get("bf16_only_experts_file")
    if bf16_only_file is not None:
        with open(bf16_only_file) as f:
            cfg["_bf16_only_by_layer"] = json.load(f)
    else:
        cfg["_bf16_only_by_layer"] = {}

    # Optional INT4 attention swap: replaces every layer's qkv_proj+o_proj
    # with GPTQ-Marlin INT4 linears loaded from the INT4 group's checkpoint.
    # Default 16 = no swap (back-compat). Only {16, 4} supported.
    attn_bits = cfg.get("attention_num_bits", 16)
    if not isinstance(attn_bits, int) or attn_bits not in (16, 4):
        raise ValueError(
            f"attention_num_bits must be 16 or 4, got {attn_bits!r}"
        )
    if attn_bits == 4:
        int4_groups_with_ckpt = [
            g for g in groups
            if g.get("num_bits", 16) == 4 and g.get("checkpoint")
        ]
        if not int4_groups_with_ckpt:
            raise ValueError(
                "attention_num_bits=4 requires an INT4 group with a "
                "'checkpoint' path in groups[] (the same checkpoint is "
                "reused for attention)."
            )
    cfg["attention_num_bits"] = attn_bits
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
        # EP support: num_experts is the LOCAL count when ep_size > 1.
        ep_size: int = 1,
        ep_rank: int = 0,
        num_global_experts: Optional[int] = None,
        reduce_results: bool = False,
        moe_tp_size: int = 1,
        # INT4-only expert support: list of expert IDs that are INT4-only
        int4_only_experts: Optional[list] = None,
        # BF16-only expert support: list of expert IDs that are BF16-only
        # (no INT4 weights loaded — always routed to BF16 kernel).
        # Only used by the per-layer sensitivity sweep for VRAM savings,
        # not in normal online serving.
        bf16_only_experts: Optional[list] = None,
        # Required when heter_config has "expert_importance_file" so we can
        # select this layer's per-expert importance slice.
        layer_id: Optional[int] = None,
    ):
        super().__init__()
        self.num_experts = num_experts  # local expert count
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.dtype = dtype
        self.device = device or torch.device("cuda")

        # EP state
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_global_experts = num_global_experts or num_experts
        self.reduce_results = reduce_results
        self.moe_tp_size = moe_tp_size
        self._local_expert_mapping: Optional[torch.Tensor] = None

        # Find group indices for INT4 and BF16
        self._int4_group_idx = next(
            (i for i, g in enumerate(heter_config["groups"])
             if g.get("num_bits", 16) == 4),
            0,
        )
        self._bf16_group_idx = next(
            (i for i, g in enumerate(heter_config["groups"])
             if g.get("num_bits", 16) == 16),
            1 - self._int4_group_idx,
        )

        # INT4-only expert support: build mask for dispatch remasking
        # and compact BF16 tensor (only dual-precision experts get BF16 weights)
        self._int4_only_experts = int4_only_experts or []
        int4_only_set = set(self._int4_only_experts)
        if self._int4_only_experts:
            mask = torch.zeros(num_experts, dtype=torch.bool, device=self.device)
            for eid in self._int4_only_experts:
                assert 0 <= eid < num_experts, f"Invalid expert ID {eid}"
                mask[eid] = True
            self._int4_only_mask = mask
            # BF16 compact tensor support: remap original expert IDs to [0, D)
            dual_experts = sorted(e for e in range(num_experts) if e not in int4_only_set)
            self._num_bf16_experts = len(dual_experts)
            remap = torch.full((num_experts,), -1, dtype=torch.long, device=self.device)
            for compact_idx, orig_idx in enumerate(dual_experts):
                remap[orig_idx] = compact_idx
            self._bf16_id_remap = remap
        else:
            self._int4_only_mask = None
            self._num_bf16_experts = num_experts
            self._bf16_id_remap = None

        # BF16-only expert support: these experts skip INT4 weight loading
        # and are always routed to the BF16 kernel.
        # Used only by the per-layer sensitivity sweep (VRAM optimization),
        # not in normal online serving.
        self._bf16_only_experts = bf16_only_experts or []
        self._bf16_only_set = set(self._bf16_only_experts)
        if self._bf16_only_experts:
            mask = torch.zeros(num_experts, dtype=torch.bool, device=self.device)
            for eid in self._bf16_only_experts:
                assert 0 <= eid < num_experts, f"Invalid expert ID {eid}"
                mask[eid] = True
            self._bf16_only_mask = mask
        else:
            self._bf16_only_mask = None
        # If ALL experts are bf16-only, skip INT4 param init entirely
        self._all_bf16_only = (len(self._bf16_only_set) == num_experts)

        self.group_cfgs = heter_config["groups"]
        self.num_groups = len(self.group_cfgs)
        self.group_ratios = [g["size_ratio"] for g in self.group_cfgs]

        policy_name = heter_config.get("policy", "expert_load")
        policy_kwargs = dict(heter_config.get("policy_params", {}))
        # Universal BF16 promotion threshold lives at the top level, not under
        # policy_params. Strip any stray legacy ``threshold`` from policy_params
        # so it isn't double-passed to the policy ctor.
        policy_kwargs.pop("threshold", None)
        bf16_promotion_threshold = int(heter_config["bf16_promotion_threshold"])

        importance_file = heter_config.get("expert_importance_file")
        self.expert_importance: Optional[torch.Tensor] = None
        if importance_file is not None:
            if layer_id is None:
                raise ValueError(
                    "heter_config.expert_importance_file is set but "
                    "layer_id was not passed to HeterFusedMoE"
                )
            with open(importance_file) as f:
                importance_by_layer = json.load(f)
            key = str(layer_id)
            if key not in importance_by_layer:
                raise KeyError(
                    f"expert_importance_file {importance_file} missing "
                    f"entry for layer {layer_id}"
                )
            row = importance_by_layer[key]
            if len(row) != num_experts:
                raise ValueError(
                    f"expert_importance layer {layer_id}: length "
                    f"{len(row)} != num_experts {num_experts}"
                )
            self.expert_importance = torch.tensor(
                row, dtype=torch.float32, device=self.device
            )
            policy_kwargs.setdefault("importance", self.expert_importance)

        self.policy: HeterDispatchPolicy = create_policy(
            policy_name,
            num_experts=num_experts,
            group_size_ratios=self.group_ratios,
            bf16_promotion_threshold=bf16_promotion_threshold,
            device=self.device,
            int4_only_mask=self._int4_only_mask,
            int4_group_idx=self._int4_group_idx,
            bf16_only_mask=self._bf16_only_mask,
            bf16_group_idx=self._bf16_group_idx,
            **policy_kwargs,
        )

        self._int4_is_k_full = True
        self._int4_group_size = 128

        # Sparse-active autotune: load the (n_active, m_per_expert)-keyed
        # tile dictionary produced by tune_bf16_sparse_sep.py. When set,
        # forward() wraps the BF16 path in override_split_config(...) to
        # pin per-direction tiles instead of the production E=128,N=768
        # JSON (which only has integer-M keys and saturates at M=4096).
        # See test/test_heter_moe/unittest/kernel_profile/results/x_star_curve.md.
        self._sparse_tile_bse_keys: Optional[List[int]] = None
        self._sparse_tile_by_bse: Optional[Dict[int, Tuple[Dict, Dict]]] = None
        if heter_config.get("pin_autotuned_tiles", True):
            sparse_path = heter_config.get(
                "sparse_tile_path",
                "test/test_heter_moe/unittest/kernel_profile/results/"
                "bf16_sparse_configs_sep.json",
            )
            self._init_sparse_tile_lookup(sparse_path)

        self._init_group_weights()

    def _init_sparse_tile_lookup(self, path: str) -> None:
        """Parse the sparse-active autotune JSON and pre-extract the row
        matching this layer's BF16 expert count (= self._num_bf16_experts).

        The JSON is keyed by ``"n{n}_bse{bse}"``. We only need the row at
        n = num_bf16_experts (constant per layer); per-call lookup then
        only does nearest-neighbor in the bse axis.
        """
        if not os.path.exists(path):
            # Silent fallback: if the kernel-profile artifact isn't present,
            # the BF16 path uses the production JSON via try_get_optimal_moe_config.
            return
        with open(path) as f:
            full = json.load(f)
        # Group cells by n_active
        by_n: Dict[int, Dict[int, Tuple[Dict, Dict]]] = {}
        for k, v in full.items():
            try:
                n_str, bse_str = k.split("_")
                n = int(n_str[1:])
                bse = int(bse_str[3:])
            except (ValueError, IndexError):
                continue
            if "up" not in v or "down" not in v:
                continue
            up_tile = {k_: v_ for k_, v_ in v["up"].items() if not k_.startswith("_")}
            down_tile = {k_: v_ for k_, v_ in v["down"].items() if not k_.startswith("_")}
            by_n.setdefault(n, {})[bse] = (up_tile, down_tile)
        if not by_n:
            return
        n_keys = sorted(by_n.keys())
        n_target = self._num_bf16_experts
        n_match = min(n_keys, key=lambda nn: abs(nn - n_target))
        self._sparse_tile_by_bse = by_n[n_match]
        self._sparse_tile_bse_keys = sorted(self._sparse_tile_by_bse.keys())

    def _lookup_sparse_tile(
        self, m_per_expert: int
    ) -> Optional[Tuple[Dict, Dict]]:
        """Nearest-bse lookup in the precomputed row. None if no autotune
        artifact is loaded.

        Manual argmin loop instead of ``min(..., key=lambda)`` — dynamo
        traces the forward path and rejects the ``key=`` kwarg form (see
        EfficiencyPromotionPolicy._lookup_x_runtime for context).
        """
        if not self._sparse_tile_by_bse:
            return None
        keys = self._sparse_tile_bse_keys
        best_idx = 0
        best_dist = abs(keys[0] - m_per_expert)
        for i in range(1, len(keys)):
            d = abs(keys[i] - m_per_expert)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return self._sparse_tile_by_bse[keys[best_idx]]

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
                if not hasattr(self, "w13_weight") and self._num_bf16_experts > 0:
                    self._init_bf16_params(E, H, I)
            elif num_bits == 4:
                if not hasattr(self, "int4_w13_qweight") and not self._all_bf16_only:
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
        int4_only_experts: Optional[list] = None,
        bf16_only_experts: Optional[list] = None,
        layer_id: Optional[int] = None,
    ) -> "HeterFusedMoE":
        """Create HeterFusedMoE from an already-loaded BF16 FusedMoE.

        Transfers BF16 expert weights from the FusedMoE. INT4 weights
        are loaded separately via load_int4_weights().
        Captures EP state so the layer can remap global expert IDs.
        """
        E = fused_moe.w13_weight.shape[0]  # num_local_experts with EP
        # w13_weight: [E, 2*I_part, H], w2_weight: [E, H, I_part]
        I_part = fused_moe.w2_weight.shape[2]
        H = fused_moe.w13_weight.shape[2]
        device = fused_moe.w13_weight.device
        dtype = fused_moe.w13_weight.dtype

        ep_size = getattr(fused_moe, "moe_ep_size", 1)
        ep_rank = getattr(fused_moe, "moe_ep_rank", 0)
        num_global_experts = getattr(fused_moe, "num_experts", E)
        reduce_results = getattr(fused_moe, "reduce_results", False)
        moe_tp_size = getattr(fused_moe, "moe_tp_size", 1)

        instance = cls(
            num_experts=E,
            hidden_size=H,
            intermediate_size=I_part,
            top_k=getattr(fused_moe, "top_k", 1),
            heter_config=heter_config,
            dtype=dtype,
            device=device,
            ep_size=ep_size,
            ep_rank=ep_rank,
            num_global_experts=num_global_experts,
            reduce_results=reduce_results,
            moe_tp_size=moe_tp_size,
            int4_only_experts=int4_only_experts,
            bf16_only_experts=bf16_only_experts,
            layer_id=layer_id,
        )

        # Copy BF16 weights: compact (dual-precision only) or full
        if int4_only_experts:
            int4_only_set = set(int4_only_experts)
            dual_experts = sorted(e for e in range(E) if e not in int4_only_set)
            if dual_experts:
                instance.w13_weight = nn.Parameter(
                    fused_moe.w13_weight.data[dual_experts].clone(), requires_grad=False
                )
                instance.w2_weight = nn.Parameter(
                    fused_moe.w2_weight.data[dual_experts].clone(), requires_grad=False
                )
            # else: all experts INT4-only, no BF16 weights needed
        else:
            instance.w13_weight = nn.Parameter(
                fused_moe.w13_weight.data, requires_grad=False
            )
            instance.w2_weight = nn.Parameter(
                fused_moe.w2_weight.data, requires_grad=False
            )

        return instance

    def _init_bf16_params(self, E: int, H: int, I: int) -> None:
        """Create BF16 weight containers.

        Uses compact size D (dual-precision experts only) when int4_only_experts
        is configured, otherwise full E.
        """
        D = self._num_bf16_experts
        self.register_parameter(
            "w13_weight",
            nn.Parameter(
                torch.empty(D, 2 * I, H, dtype=self.dtype, device=self.device),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "w2_weight",
            nn.Parameter(
                torch.empty(D, H, I, dtype=self.dtype, device=self.device),
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

        ``gcfg["group_size"]`` is the GPTQ quantization group size (one
        fp16 scale per ``group_size`` INT4 weights along K). Must match
        the checkpoint; unrelated to the precision-group concept above.
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
        and stacks them into [E, ...] format. With EP, only loads experts
        assigned to this rank (global ID → local ID mapping).
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

                    global_expert_id = int(match.group(1))
                    proj = match.group(2)
                    attr = match.group(3)

                    # EP: map global expert ID to local; skip non-local experts.
                    local_expert_id = self._map_global_to_local_expert_id(
                        global_expert_id
                    )
                    if local_expert_id < 0 or local_expert_id >= E:
                        continue

                    # Skip loading INT4 weights for bf16-only experts.
                    if local_expert_id in self._bf16_only_set:
                        continue

                    tensor = f.get_tensor(key)

                    self._fill_int4_param(local_expert_id, proj, attr, tensor)

        logger.info(
            f"Loaded INT4 expert weights for layer {layer_id} from {checkpoint_path}"
            f" (ep_rank={self.ep_rank}/{self.ep_size}, local_experts={E})"
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

    def _get_local_expert_mapping(self) -> Optional[torch.Tensor]:
        """Build mapping from global expert IDs to local IDs (lazy, cached).

        Returns None when ep_size == 1 (no remapping needed).
        Non-local experts map to -1.
        """
        if self.ep_size <= 1:
            return None
        if self._local_expert_mapping is None:
            mapping = torch.full(
                (self.num_global_experts,), -1,
                dtype=torch.int32, device=self.device,
            )
            start = self.ep_rank * self.num_experts
            end = start + self.num_experts
            mapping[start:end] = torch.arange(
                self.num_experts, dtype=torch.int32, device=self.device,
            )
            self._local_expert_mapping = mapping
        return self._local_expert_mapping

    def _map_global_to_local_expert_id(self, global_id: int) -> int:
        """Map a single global expert ID to local (for weight loading)."""
        if self.ep_size <= 1:
            return global_id
        start = self.ep_rank * self.num_experts
        if start <= global_id < start + self.num_experts:
            return global_id - start
        return -1

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output,
    ) -> torch.Tensor:
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
        router_logits = topk_output.router_logits

        # EP: remap global expert IDs → local; non-local experts → -1.
        mapping = self._get_local_expert_mapping()
        if mapping is not None:
            topk_ids = mapping[topk_ids]
            non_local = topk_ids < 0
            topk_weights = topk_weights.clone()
            topk_weights[non_local] = 0.0

        # Sentinel = -1: Triton (BF16/INT8) fully skips expert ID -1.
        group_dispatches = self.policy.dispatch(topk_ids, topk_weights, sentinel=-1)

        # Host-side: number of tokens in this batch (used by short-circuit
        # policies like batch_size). For other policies should_skip_group()
        # always returns False.
        num_tokens = topk_ids.shape[0]

        output = None

        for group_idx, gcfg in enumerate(self.group_cfgs):
            if self.policy.should_skip_group(group_idx, num_tokens):
                continue

            group_ids, group_weights = group_dispatches[group_idx]

            num_bits = gcfg.get("num_bits", 16)

            if num_bits == 16:
                # Skip BF16 kernel if all experts are INT4-only for this layer
                if self._num_bf16_experts == 0:
                    continue
                # Remap expert IDs to compact BF16 tensor indices
                if self._bf16_id_remap is not None:
                    safe_ids = group_ids.clamp(min=0)
                    remapped = self._bf16_id_remap[safe_ids]
                    group_ids = torch.where(group_ids >= 0, remapped, group_ids)
                # Pin autotuned tile if the kernel-profile artifact was loaded.
                # m_per_expert is computed from host-known shapes only — no
                # GPU→host sync (a .item() would break CUDA-graph capture
                # since the kernel call below runs inside the captured graph).
                # Approximation: under uniform routing each token's K slots
                # spread across self._num_bf16_experts, so per-expert load is
                # M_global * top_k / num_bf16_experts. Real Zipf concentrates
                # on a subset, but the autotune is hierarchical-nearest in
                # bse, so an order-of-magnitude approximation is sufficient.
                tile_pair = None
                if self._sparse_tile_by_bse is not None and self._num_bf16_experts > 0:
                    M_global = hidden_states.shape[0]   # host-known
                    top_k = group_ids.shape[1]          # host-known
                    m_per_expert = max(
                        1,
                        (M_global * top_k) // self._num_bf16_experts,
                    )
                    tile_pair = self._lookup_sparse_tile(m_per_expert)
                if tile_pair is not None:
                    from sglang.srt.layers.moe.fused_moe_triton import (
                        override_split_config,
                    )
                    up_tile, down_tile = tile_pair
                    with override_split_config(up_tile, down_tile):
                        group_out = outplace_fused_experts(
                            hidden_states,
                            self.w13_weight,
                            self.w2_weight,
                            group_weights,
                            group_ids,
                        )
                else:
                    group_out = outplace_fused_experts(
                        hidden_states,
                        self.w13_weight,
                        self.w2_weight,
                        group_weights,
                        group_ids,
                    )
            elif num_bits == 4:
                # Skip INT4 kernel if every expert in this layer is BF16-only
                # (layer-level sensitivity/invariance sweeps hit this when a
                # layer is 100% BF16 — no INT4 params were created).
                if self._all_bf16_only:
                    continue
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
                if self._num_bf16_experts == 0:
                    continue
                # INT8 shares w13_weight/w2_weight — same remap as BF16
                if self._bf16_id_remap is not None:
                    safe_ids = group_ids.clamp(min=0)
                    remapped = self._bf16_id_remap[safe_ids]
                    group_ids = torch.where(group_ids >= 0, remapped, group_ids)
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

        if output is None:
            output = torch.zeros_like(hidden_states)

        # EP: all-reduce across EP + MoE-TP ranks.
        if self.reduce_results and (self.moe_tp_size > 1 or self.ep_size > 1):
            from sglang.srt.distributed.communication_op import (
                tensor_model_parallel_all_reduce,
            )
            output = tensor_model_parallel_all_reduce(output)

        return output


def _build_gptq_marlin_config_from_ckpt(ckpt_path: str):
    """Read the GPTQ quantize config from a checkpoint dir, return GPTQMarlinConfig.

    Tries ``<ckpt>/quantize_config.json`` first, then falls back to the
    ``quantization_config`` field inside ``<ckpt>/config.json`` (Qwen GPTQ
    checkpoints use the latter).
    """
    from sglang.srt.layers.quantization.gptq import GPTQMarlinConfig

    qcfg = None
    qc_path = os.path.join(ckpt_path, "quantize_config.json")
    if os.path.isfile(qc_path):
        with open(qc_path) as f:
            qcfg = json.load(f)
    else:
        cfg_path = os.path.join(ckpt_path, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                qcfg = json.load(f).get("quantization_config")
    if qcfg is None:
        raise ValueError(
            f"GPTQ checkpoint {ckpt_path}: cannot find quantize_config.json "
            "nor config.json[quantization_config]; required to build "
            "GPTQMarlinConfig for INT4 attention swap."
        )
    return GPTQMarlinConfig.from_config(qcfg)


def _swap_attention_to_int4(
    model: nn.Module,
    gptq_checkpoint_path: str,
    device: torch.device,
) -> None:
    """Swap every layer's self_attn.{qkv_proj, o_proj} with INT4 GPTQ-Marlin.

    Reuses the production GPTQMarlinLinearMethod path:
      1. Build GPTQMarlinConfig from the GPTQ checkpoint's quantize config.
      2. For each transformer layer: construct a fresh INT4 QKVParallelLinear
         and RowParallelLinear matching the existing module's geometry
         (hidden_size, head counts, bias, dtype, TP rank/size).
      3. Stream q/k/v/o tensors from the checkpoint via the standard
         weight_loader_v2 (which fuses q/k/v along the output dim and applies
         TP sharding).
      4. process_weights_after_loading repacks GPTQ → Marlin in-place.
      5. Swap into the layer; free the old BF16 modules with explicit gc to
         keep peak VRAM flat (matches the per-layer MoE swap pattern).
    """
    from safetensors import safe_open

    from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear

    gptq_config = _build_gptq_marlin_config_from_ckpt(gptq_checkpoint_path)

    layers_module = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers_module = model.model.layers
    elif hasattr(model, "layers"):
        layers_module = model.layers
    if layers_module is None:
        raise ValueError("Cannot find model layers for INT4 attention swap")

    # Discover safetensors files (single-file or sharded directory)
    if os.path.isfile(gptq_checkpoint_path) and gptq_checkpoint_path.endswith(
        ".safetensors"
    ):
        st_files = [gptq_checkpoint_path]
    else:
        st_files = sorted(
            os.path.join(gptq_checkpoint_path, f)
            for f in os.listdir(gptq_checkpoint_path)
            if f.endswith(".safetensors")
        )
    if not st_files:
        raise ValueError(
            f"No .safetensors files under {gptq_checkpoint_path} for INT4 attention swap"
        )

    import gc

    attrs = ("qweight", "scales", "qzeros", "g_idx")
    num_swapped = 0
    for layer_id, layer in enumerate(layers_module):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            # Non-attention layer (uncommon for transformer stacks); skip cleanly.
            continue
        old_qkv = getattr(attn, "qkv_proj", None)
        old_o = getattr(attn, "o_proj", None)
        if old_qkv is None or old_o is None:
            continue
        if not isinstance(old_qkv, QKVParallelLinear):
            raise TypeError(
                f"Layer {layer_id} self_attn.qkv_proj is "
                f"{type(old_qkv).__name__}, expected QKVParallelLinear. "
                "INT4 attention swap only supports vanilla QKVParallelLinear."
            )
        if not isinstance(old_o, RowParallelLinear):
            raise TypeError(
                f"Layer {layer_id} self_attn.o_proj is "
                f"{type(old_o).__name__}, expected RowParallelLinear."
            )

        # Construct fresh INT4 modules on-device matching the existing geometry.
        # `with torch.device(device):` ensures empty param storage lands on GPU
        # (constructors here don't take a device argument).
        prefix_qkv = f"model.layers.{layer_id}.self_attn.qkv_proj"
        prefix_o = f"model.layers.{layer_id}.self_attn.o_proj"
        with torch.device(device):
            new_qkv = QKVParallelLinear(
                hidden_size=old_qkv.hidden_size,
                head_size=old_qkv.head_size,
                total_num_heads=old_qkv.total_num_heads,
                total_num_kv_heads=old_qkv.total_num_kv_heads,
                bias=old_qkv.bias is not None,
                params_dtype=old_qkv.params_dtype,
                quant_config=gptq_config,
                prefix=prefix_qkv,
                tp_rank=old_qkv.tp_rank,
                tp_size=old_qkv.tp_size,
            )
            new_o = RowParallelLinear(
                input_size=old_o.input_size,
                output_size=old_o.output_size,
                bias=old_o.bias is not None,
                input_is_parallel=getattr(old_o, "input_is_parallel", True),
                params_dtype=old_o.params_dtype,
                reduce_results=old_o.reduce_results,
                quant_config=gptq_config,
                prefix=prefix_o,
                tp_rank=old_o.tp_rank,
                tp_size=old_o.tp_size,
            )

        # Stream tensors. q/k/v share new_qkv, o is its own module.
        proj_to_target = {
            "q_proj": ("q", new_qkv),
            "k_proj": ("k", new_qkv),
            "v_proj": ("v", new_qkv),
            "o_proj": (None, new_o),
        }
        loaded: Dict[str, set] = {p: set() for p in proj_to_target}
        attn_prefix = f"model.layers.{layer_id}.self_attn."

        for st_file in st_files:
            with safe_open(st_file, framework="pt", device=str(device)) as f:
                for key in f.keys():
                    if not key.startswith(attn_prefix):
                        continue
                    suffix = key[len(attn_prefix):]
                    parts = suffix.split(".")
                    if len(parts) != 2:
                        continue
                    proj, attr = parts
                    if proj not in proj_to_target or attr not in attrs:
                        continue
                    shard_id, mod = proj_to_target[proj]
                    tensor = f.get_tensor(key)
                    param = getattr(mod, attr)
                    if shard_id is None:
                        mod.weight_loader_v2(param, tensor)
                    else:
                        mod.weight_loader_v2(param, tensor, shard_id)
                    loaded[proj].add(attr)

        # Loud failure if any tensor is missing — partial loads silently
        # corrupt outputs.
        for proj, got in loaded.items():
            missing = set(attrs) - got
            if missing:
                raise ValueError(
                    f"Layer {layer_id} self_attn.{proj}: missing tensors "
                    f"{sorted(missing)} in {gptq_checkpoint_path}"
                )

        # Repack GPTQ → Marlin (creates workspace, permutes scales, etc.)
        new_qkv.quant_method.process_weights_after_loading(new_qkv)
        new_o.quant_method.process_weights_after_loading(new_o)

        # Swap and drop the old BF16 modules so their VRAM is freed before
        # the next layer (mirrors the MoE per-layer free pattern).
        attn.qkv_proj = new_qkv
        attn.o_proj = new_o
        del old_qkv, old_o
        gc.collect()
        torch.cuda.empty_cache()
        num_swapped += 1

        if layer_id % 10 == 0:
            logger.info(
                f"Swapped layer {layer_id} self_attn.qkv_proj+o_proj → INT4"
            )

    gc.collect()
    torch.cuda.empty_cache()
    logger.info(
        f"Swapped attention to INT4 in {num_swapped} layers "
        f"(checkpoint: {gptq_checkpoint_path})"
    )


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
    5. (Optional) If ``attention_num_bits == 4``, swap every layer's
       qkv_proj+o_proj with INT4 GPTQ-Marlin linears (same checkpoint).
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    heter_config = _parse_heter_config(config_path)
    int4_only_by_layer = heter_config.get("_int4_only_by_layer", {})
    # bf16-only is a VRAM optimization for the per-layer sensitivity sweep;
    # not used in normal online serving (empty dict by default).
    bf16_only_by_layer = heter_config.get("_bf16_only_by_layer", {})

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

    import gc

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
        int4_only_experts = int4_only_by_layer.get(str(layer_id))
        bf16_only_experts = bf16_only_by_layer.get(str(layer_id))
        heter_moe = HeterFusedMoE.from_fused_moe(
            fused_moe, heter_config,
            int4_only_experts=int4_only_experts,
            bf16_only_experts=bf16_only_experts,
            layer_id=layer_id,
        )

        # Load INT4 expert weights (skip entirely if all experts are bf16-only)
        if not heter_moe._all_bf16_only:
            heter_moe.load_int4_weights(int4_checkpoint, layer_id)
            heter_moe.repack_int4_to_marlin()

        # Swap, then drop local/module refs to the old FusedMoE and release
        # its VRAM before moving on — otherwise all 48 layers' original BF16
        # weights stay pinned while we clone compact BF16 per layer, which
        # OOMs on 80 GB for vertical/random at n≥32.
        setattr(moe_parent, moe_attr, heter_moe)
        del fused_moe, candidate
        gc.collect()
        torch.cuda.empty_cache()
        num_swapped += 1

        if layer_id % 10 == 0:
            logger.info(f"Swapped layer {layer_id} FusedMoE → HeterFusedMoE")

    gc.collect()
    torch.cuda.empty_cache()

    logger.info(
        f"Applied heterogeneous precision to {num_swapped} MoE layers. "
        f"Groups: {[g['name'] for g in heter_config['groups']]}"
    )

    if heter_config.get("attention_num_bits", 16) == 4:
        _swap_attention_to_int4(model, int4_checkpoint, device)
