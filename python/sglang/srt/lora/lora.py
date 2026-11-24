# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Integrates "S-LoRA: Serving Thousands of Concurrent LoRA Adapters"
# and "Punica: Multi-Tenant LoRA Serving"

# LoRA layers class inheritance adapted from:
# https://github.com/vllm-project/vllm/blob/4abf6336ec65c270343eb895e7b18786e9274176/vllm/lora/layers.py

import logging
import re
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.hf_transformers import AutoConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.model_loader.loader import DefaultModelLoader

# Matches both per-expert keys ("...experts.0.<module>...") and shared-outer
# keys ("...experts.<module>..."), while excluding "shared_experts." (where the
# preceding char is "_", not ".").
_ROUTED_EXPERT_PATTERN = re.compile(r"\.experts\.")
_SHARED_EXPERT_MOE_PATTERN = re.compile(r"\.shared_experts\.")
_LAYER_REL_PATH_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.(.*))?$")
_LORA_AB_KEY_PATTERN = re.compile(
    r"^(?P<module_path>.+?)\.(?P<ab_kind>lora_A|lora_B)(?P<tail>(?:\.[^.]+)*\.weight)$"
)
# Models whose attention layer defaults `attn_output_gate` to True when omitted
# from the HF text config (doubling Q projection output width).
_GATED_QKV_MODEL_TYPE_PREFIXES = ("qwen3_5", "qwen3_next", "interns2_mobius")
_GATED_QKV_ARCH_PREFIXES = ("Qwen3_5", "Qwen3Next", "InternS2Mobius")

_BaseModuleIndex = Dict[int, Dict[str, torch.nn.Module]]
_MemberGroup = Dict[str, Tuple[str, torch.Tensor]]
_ABGroup = Dict[str, _MemberGroup]

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    def __init__(self, config: LoRAConfig, base_hf_config: AutoConfig):
        super().__init__()
        self.config: LoRAConfig = config
        self.base_hf_config: AutoConfig = base_hf_config

        # lora weights in cpu. The weights are loaded from checkpoint.
        self.weights: Dict[str, torch.Tensor] = {}
        self.pinned_weights: Dict[str, torch.Tensor] = {}


class LoRAAdapter(nn.Module):
    base_model: Optional[torch.nn.Module] = None

    def __init__(
        self,
        uid: str,
        config: LoRAConfig,
        base_hf_config: AutoConfig,
        load_config: LoadConfig,
        lora_backend: BaseLoRABackend,
        base_model: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.uid: str = uid
        self.config: LoRAConfig = config
        assert self.config.hf_config["peft_type"].lower() == "lora"
        self.base_hf_config: AutoConfig = base_hf_config
        self.load_config: LoadConfig = load_config
        self.lora_backend: BaseLoRABackend = lora_backend
        self.scaling: float = self.config.lora_alpha / self.config.r

        # Bypass nn.Module.__setattr__ so the base model is held as a plain
        # reference rather than auto-registered as a submodule (which would
        # leak its parameters into our state_dict / parameters() / .to()).
        object.__setattr__(self, "base_model", base_model)

        self.layers: List[LoRALayer] = nn.ModuleList(
            [
                LoRALayer(config, base_hf_config)
                for _ in range(base_hf_config.num_hidden_layers)
            ]
        )

        self.embedding_layers: Dict[str, torch.Tensor] = {}
        self.pinned_embedding_layers: Dict[str, torch.Tensor] = {}
        self.added_tokens_embeddings: Dict[str, torch.Tensor] = {}
        self.pinned_added_tokens_embeddings: Dict[str, torch.Tensor] = {}

    def _get_text_config(self):
        cfg = self.base_hf_config
        get_text_config = getattr(cfg, "get_text_config", None)
        if callable(get_text_config):
            return get_text_config()
        return cfg

    def _get_base_modules(
        self, base_modules: Optional[_BaseModuleIndex]
    ) -> _BaseModuleIndex:
        return (
            base_modules
            if base_modules is not None
            else _inspect_base_model(self.base_model)[0]
        )

    def _repeat_fused_lora_a(
        self,
        weights: Dict[str, torch.Tensor],
        orig_key: str,
        a_weight: torch.Tensor,
        b_weight: torch.Tensor,
        multiply: int,
    ) -> None:
        _validate_ab_pair(a_weight, b_weight, orig_key, check_rank=False)
        expected_rank = b_weight.shape[-1]
        if a_weight.shape[-2] != expected_rank:
            if multiply > 1 and a_weight.shape[-2] == multiply * expected_rank:
                raise ValueError(
                    f"Fused LoRA weight '{orig_key}' appears already stacked "
                    f"(rank dim {a_weight.shape[-2]} == {multiply} * {expected_rank}); "
                    f"expected unstacked rank {expected_rank}."
                )
            raise ValueError(
                f"Fused LoRA weight '{orig_key}' has rank dim {a_weight.shape[-2]}, "
                f"expected {expected_rank}."
            )
        if multiply > 1:
            repeat_dims = [1] * a_weight.dim()
            repeat_dims[-2] = multiply
            weights[orig_key] = a_weight.repeat(*repeat_dims)

    def initialize_weights(self):
        from sglang.srt.lora.utils import get_normalized_target_modules

        model_path = self.config.path
        loader = DefaultModelLoader(self.load_config)
        revision = getattr(self.config.hf_config, "revision", None)

        # Get normalized target modules for filtering
        normalized_target_modules = get_normalized_target_modules(
            self.config.target_modules
        )
        for name, loaded_weight in loader._get_weights_iterator(
            DefaultModelLoader.Source(
                model_path, revision=revision, fall_back_to_pt=True
            )
        ):
            self._process_weight(name, loaded_weight, normalized_target_modules)

        self._normalize_weights()

    def initialize_weights_from_tensors(self, tensors: Dict[str, torch.Tensor]):
        from sglang.srt.lora.utils import get_normalized_target_modules

        normalized_target_modules = get_normalized_target_modules(
            self.config.target_modules
        )
        for name, tensor in tensors.items():
            self._process_weight(name, tensor, normalized_target_modules)

        self._normalize_weights()

    def _process_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        normalized_target_modules: set[str],
    ):
        # Remap PEFT "unembed_tokens" key to "lm_head" so the weight is
        # recognized and loaded into the correct buffer.
        if "unembed_tokens" in name:
            name = name.replace("unembed_tokens", "lm_head")

        layer_id = get_layer_id(name)
        if layer_id is not None:
            self.layers[layer_id].weights[name] = loaded_weight.cpu()
        elif "embed_tokens" in name or "lm_head" in name:
            # Check if this module is declared in target_modules before loading.
            # When normalized_target_modules is {"all"} (e.g. target_modules was
            # "all-linear"), we allow loading since the server-level
            # --lora-target-modules will govern which modules are active.
            module_name = "embed_tokens" if "embed_tokens" in name else "lm_head"
            if (
                "all" in normalized_target_modules
                or module_name in normalized_target_modules
            ):
                self.embedding_layers[name] = loaded_weight.cpu()
            else:
                logger.debug(
                    f"Skipping {name} as '{module_name}' is not in adapter's target_modules: {self.config.target_modules}"
                )
        elif "input_embeddings" in name or "output_embeddings" in name:
            # added/extra token emb
            self.added_tokens_embeddings[name] = loaded_weight.cpu()
            assert loaded_weight.shape[0] == self.config.lora_added_tokens_size, (
                f"LoRA adapter {self.uid} has lora_added_tokens_size {self.config.lora_added_tokens_size} specified in the config, "
                f"but the loaded weight '{name}' has shape {loaded_weight.shape[0]} in first dimension"
            )

    def _normalize_weights(self):
        base_modules, pp_missing_layers = _inspect_base_model(self.base_model)
        for layer_id, layer in enumerate(self.layers):
            if layer_id in pp_missing_layers:
                # Off-stage PP layers hold PPMissingLayer on this rank and are
                # never loaded into GPU buffers here; drop them so downstream
                # normalization and mem_pool validation do not misclassify them.
                layer.weights.clear()
                continue
            weight_names = list(layer.weights.keys())
            self.normalize_qkv_proj(weight_names, layer.weights, base_modules)
            self.normalize_inkling_qkvr_proj(weight_names, layer.weights)
            self._normalize_shared_expert_moe(layer.weights)
            self._rename_expert_w_to_proj(layer.weights)
            # Stack gate_proj + x_proj → in_proj for Mamba layers (before gate_up normalization)
            self._normalize_in_proj(layer.weights, base_modules)
            # Stack in_proj_q + in_proj_k + in_proj_v + in_proj_z → in_proj_qkvz for GDN layers
            self._normalize_in_proj_qkvz(layer.weights, base_modules)
            weight_names = list(layer.weights.keys())
            self.normalize_gate_up_proj(weight_names, layer.weights, base_modules)
            weight_names = list(layer.weights.keys())
            self.normalize_fused_qkv_a_proj(weight_names, layer.weights, base_modules)

    def normalize_inkling_qkvr_proj(
        self, weight_names: List[str], weights: Dict[str, torch.Tensor]
    ):
        """Normalize Inkling split attention LoRA keys into the runtime qkvr layer.

        Inkling checkpoints expose separate attention projections
        (wq_du/wk_dv/wv_dv/wr_du), while the serving model uses a single
        MergedColumnParallelLinear named qkvr with those slices in that order.
        """
        for weight_name in weight_names:
            if "wq_du" not in weight_name:
                continue

            q_name = weight_name
            k_name = weight_name.replace("wq_du", "wk_dv")
            v_name = weight_name.replace("wq_du", "wv_dv")
            r_name = weight_name.replace("wq_du", "wr_du")
            qkvr_name = weight_name.replace("wq_du", "qkvr")

            if any(name not in weights for name in (k_name, v_name, r_name)):
                continue

            cat_dim = weights[q_name].dim() - 2
            weights[qkvr_name] = torch.cat(
                (
                    weights[q_name],
                    weights[k_name],
                    weights[v_name],
                    weights[r_name],
                ),
                cat_dim,
            )
            weights.pop(q_name)
            weights.pop(k_name)
            weights.pop(v_name)
            weights.pop(r_name)

        for weight_name in list(weights.keys()):
            if "qkvr" not in weight_name or "lora_A" not in weight_name:
                continue
            if weights[weight_name].shape[-2] != self.config.r:
                continue
            repeat_dims = [1] * weights[weight_name].dim()
            repeat_dims[-2] = 4
            weights[weight_name] = weights[weight_name].repeat(*repeat_dims)

    def _fuse_split_group(
        self,
        *,
        weights: Dict[str, torch.Tensor],
        base_modules: _BaseModuleIndex,
        prefix: str,
        tail: str,
        a_members: _MemberGroup,
        b_members: _MemberGroup,
        split_members: Tuple[str, ...],
        fused_name: str,
        config_dims_fn: Optional[
            Callable[[Optional[int], _MemberGroup], Optional[Tuple[Optional[int], ...]]]
        ] = None,
        extra_pop_keys: Tuple[str, ...] = (),
    ) -> None:
        for member, (_, a_t) in a_members.items():
            _validate_ab_pair(a_t, b_members[member][1], f"{prefix}.{member}")

        expected_dims = _get_fused_module_slice_dims(
            _lookup_base_module(base_modules, prefix, fused_name), len(split_members)
        )
        if (
            expected_dims is None
            and len(b_members) < len(split_members)
            and config_dims_fn is not None
        ):
            layer_info = _split_layer_path(prefix)
            expected_dims = config_dims_fn(
                layer_info[0] if layer_info is not None else None, b_members
            )

        member_out_dims = _resolve_slice_dims(
            prefix, fused_name, split_members, b_members, expected_dims
        )
        fused_a = _concat_split_parts(
            prefix, "lora_A", a_members, split_members, fused_name
        )
        fused_b = _concat_split_parts(
            prefix, "lora_B", b_members, split_members, fused_name, member_out_dims
        )
        for key in extra_pop_keys:
            weights.pop(key, None)
        for orig_key, _ in (*a_members.values(), *b_members.values()):
            weights.pop(orig_key, None)
        weights[_fused_key(prefix, fused_name, "lora_A", tail)] = fused_a
        weights[_fused_key(prefix, fused_name, "lora_B", tail)] = fused_b

    def _normalize_fused_projection(
        self,
        *,
        weight_names: List[str],
        weights: Dict[str, torch.Tensor],
        base_modules: _BaseModuleIndex,
        split_members: Tuple[str, ...],
        fused_name: str,
        skip_preserved: bool = False,
        exclude_member: Optional[Callable[[str, str], bool]] = None,
        should_fuse: Optional[Callable[[str, set[str]], bool]] = None,
        config_dims_fn: Optional[
            Callable[[Optional[int], _MemberGroup], Optional[Tuple[Optional[int], ...]]]
        ] = None,
    ) -> None:
        groups = _collect_fused_groups(
            weight_names,
            weights,
            split_members=split_members,
            fused_name=fused_name,
            skip_preserved=skip_preserved,
            exclude_member=exclude_member,
        )
        for (prefix, tail), ab_group in groups.items():
            a_members = ab_group.get("lora_A", {})
            b_members = ab_group.get("lora_B", {})
            if should_fuse is not None and not should_fuse(
                prefix, set(a_members) | set(b_members)
            ):
                continue

            if _check_group_structure(
                prefix, fused_name, split_members, a_members, b_members
            ):
                self._repeat_fused_lora_a(
                    weights,
                    a_members[fused_name][0],
                    a_members[fused_name][1],
                    b_members[fused_name][1],
                    len(split_members),
                )
                continue

            self._fuse_split_group(
                weights=weights,
                base_modules=base_modules,
                prefix=prefix,
                tail=tail,
                a_members=a_members,
                b_members=b_members,
                split_members=split_members,
                fused_name=fused_name,
                config_dims_fn=config_dims_fn,
            )

    def normalize_qkv_proj(
        self,
        weight_names: List[str],
        weights: Dict[str, torch.Tensor],
        base_modules: Optional[_BaseModuleIndex] = None,
    ):
        mods = self._get_base_modules(base_modules)

        def should_fuse(prefix: str, members: set[str]) -> bool:
            if "qkv_proj" in members or len(members) > 1:
                return True
            # Keep standalone q_proj intact on models without fused qkv_proj
            # (e.g., DeepSeek-V2-Lite when q_lora_rank is None).
            if (
                "q_proj" in members
                and _lookup_base_module(mods, prefix, "qkv_proj") is None
                and _lookup_base_module(mods, prefix, "q_proj") is not None
            ):
                return False
            return True

        self._normalize_fused_projection(
            weight_names=weight_names,
            weights=weights,
            base_modules=mods,
            split_members=("q_proj", "k_proj", "v_proj"),
            fused_name="qkv_proj",
            should_fuse=should_fuse,
            config_dims_fn=self._infer_qkv_dims_from_config,
        )

    def _has_default_attn_output_gate(self, text_cfg) -> bool:
        if hasattr(text_cfg, "attn_output_gate"):
            return bool(getattr(text_cfg, "attn_output_gate"))
        if getattr(text_cfg, "use_attn_output_gate", False):
            return True
        model_type = str(getattr(text_cfg, "model_type", "") or "").lower()
        if model_type.startswith(_GATED_QKV_MODEL_TYPE_PREFIXES):
            return True
        archs = list(getattr(self.base_hf_config, "architectures", None) or [])
        archs += list(getattr(text_cfg, "architectures", None) or [])
        return any(arch.startswith(_GATED_QKV_ARCH_PREFIXES) for arch in archs)

    def _infer_qkv_dims_from_config(
        self, layer_id: Optional[int], members: _MemberGroup
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        q_from_w = members["q_proj"][1].shape[-2] if "q_proj" in members else None
        kv_from_w = (
            members["k_proj"][1].shape[-2]
            if "k_proj" in members
            else (members["v_proj"][1].shape[-2] if "v_proj" in members else None)
        )
        total_qkv_out: Optional[int] = None
        get_hidden_dim: Optional[Callable[[str, int], Tuple[int, int]]] = getattr(
            self.base_model, "get_hidden_dim", None
        )
        if layer_id is not None and get_hidden_dim is not None:
            try:
                _, total_qkv_out = get_hidden_dim("qkv_proj", layer_id)
            except NotImplementedError:
                total_qkv_out = None

        text_cfg = self._get_text_config()
        num_heads = getattr(text_cfg, "num_attention_heads", None)
        num_kv_heads = getattr(text_cfg, "num_key_value_heads", None)
        head_dim = getattr(text_cfg, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(text_cfg, "hidden_size", None)
            if hidden_size is not None and num_heads:
                head_dim = hidden_size // num_heads

        if num_kv_heads is not None and head_dim is not None:
            kv_dim: Optional[int] = num_kv_heads * head_dim
        elif (
            total_qkv_out is not None
            and q_from_w is not None
            and total_qkv_out > q_from_w
            and (total_qkv_out - q_from_w) % 2 == 0
        ):
            kv_dim = (total_qkv_out - q_from_w) // 2
        else:
            kv_dim = kv_from_w

        if total_qkv_out is not None and kv_dim is not None:
            q_dim: Optional[int] = total_qkv_out - 2 * kv_dim
        elif num_heads is not None and head_dim is not None:
            attn_output_gate = self._has_default_attn_output_gate(text_cfg)
            q_dim = num_heads * head_dim * (2 if attn_output_gate else 1)
        else:
            q_dim = q_from_w

        return (q_dim, kv_dim, kv_dim)

    def _normalize_shared_expert_moe(self, weights: Dict[str, torch.Tensor]):
        """Reshape flat Inkling shared-sink factors to shared-outer 3D form."""
        # Gate on the architecture so other models with 2D shared experts keep
        # the stock path.
        cfg = self._get_text_config()
        archs = list(getattr(self.base_hf_config, "architectures", None) or [])
        archs += list(getattr(cfg, "architectures", None) or [])
        is_inkling = (
            any("Inkling" in arch for arch in archs)
            or "inkling" in str(getattr(cfg, "model_type", "")).lower()
        )
        if not is_inkling:
            return
        num_shared = getattr(cfg, "n_shared_experts", 0) or 0
        if num_shared <= 0:
            return
        for name in list(weights.keys()):
            if "shared_experts." not in name:
                continue
            if re.search(r"shared_experts\.\d+\.", name):
                # Preserve named per-expert factors so adapter validation can
                # reject outer factors that are not actually shared.
                continue
            is_down = any(k in name for k in (".w2.", ".down_proj."))
            is_gate_up = any(
                k in name
                for k in (
                    ".w1.",
                    ".w3.",
                    ".gate_proj.",
                    ".up_proj.",
                    ".gate_up_proj.",
                )
            )
            if not (is_down or is_gate_up):
                continue
            w = weights[name]
            if w.dim() != 2:
                continue
            if "lora_A" in name:
                if is_down:
                    r, flat = w.shape
                    weights[name] = w.reshape(r, num_shared, flat // num_shared)
                    weights[name] = weights[name].transpose(0, 1).contiguous()
                else:
                    weights[name] = w.unsqueeze(0)
            else:
                if is_down:
                    weights[name] = w.unsqueeze(0)
                else:
                    flat, r = w.shape
                    weights[name] = w.reshape(num_shared, flat // num_shared, r)

    def _rename_expert_w_to_proj(self, weights: Dict[str, torch.Tensor]):
        """Rename w1 -> gate_proj, w3 -> up_proj, w2 -> down_proj so that
        normalize_gate_up_proj can stack them into gate_up_proj."""
        renames = {}
        for name in list(weights.keys()):
            new_name = name
            if ".w1." in name:
                new_name = name.replace(".w1.", ".gate_proj.")
            elif ".w3." in name:
                new_name = name.replace(".w3.", ".up_proj.")
            elif ".w2." in name:
                new_name = name.replace(".w2.", ".down_proj.")
            if new_name != name:
                renames[name] = new_name
        for old_name, new_name in renames.items():
            weights[new_name] = weights.pop(old_name)

    def _normalize_in_proj(
        self,
        weights: Dict[str, torch.Tensor],
        base_modules: Optional[_BaseModuleIndex] = None,
    ):
        # Must run before normalize_gate_up_proj so Mamba gate_proj-only is not
        # merged into gate_up_proj.
        mods = self._get_base_modules(base_modules)

        def exclude_standalone_x_proj(prefix: str, leaf: str) -> bool:
            # On Mamba1, x_proj is an independent downstream projection
            # alongside in_proj — never treat it as a split member of in_proj.
            return (
                leaf == "x_proj"
                and _lookup_base_module(mods, prefix, "x_proj") is not None
            )

        def should_fuse(prefix: str, members: set[str]) -> bool:
            if "in_proj" in members or ("gate_proj" in members and "x_proj" in members):
                return True
            return (
                _lookup_base_module(mods, prefix, "in_proj") is not None
                and _lookup_base_module(mods, prefix, "x_proj") is None
            )

        def fallback_dims(
            _layer_id: Optional[int], members: _MemberGroup
        ) -> Optional[Tuple[int, int]]:
            ref_out = next(iter(members.values()))[1].shape[-2]
            return (ref_out, ref_out)

        self._normalize_fused_projection(
            weight_names=list(weights.keys()),
            weights=weights,
            base_modules=mods,
            split_members=("gate_proj", "x_proj"),
            fused_name="in_proj",
            skip_preserved=True,
            exclude_member=exclude_standalone_x_proj,
            should_fuse=should_fuse,
            config_dims_fn=fallback_dims,
        )

    def _normalize_in_proj_qkvz(
        self,
        weights: Dict[str, torch.Tensor],
        base_modules: Optional[_BaseModuleIndex] = None,
    ):
        def should_fuse(prefix: str, members: set[str]) -> bool:
            qkv_prefix = f"{prefix}.in_proj_qkv." if prefix else "in_proj_qkv."
            if any(k.startswith(qkv_prefix) for k in weights):
                raise ValueError(
                    f"Unsupported combination of 'in_proj_qkv' and "
                    f"{sorted(members)} at '{prefix}'; 2-way in_proj_qkv + "
                    f"in_proj_z normalization is not yet supported."
                )
            return True

        self._normalize_fused_projection(
            weight_names=list(weights.keys()),
            weights=weights,
            base_modules=self._get_base_modules(base_modules),
            split_members=("in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z"),
            fused_name="in_proj_qkvz",
            should_fuse=should_fuse,
            config_dims_fn=self._infer_qkvz_dims_from_config,
        )

    def _infer_qkvz_dims_from_config(
        self, _layer_id: Optional[int], members: _MemberGroup
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        text_cfg = self._get_text_config()
        k_head = getattr(text_cfg, "linear_key_head_dim", None)
        k_num = getattr(text_cfg, "linear_num_key_heads", None)
        v_head = getattr(text_cfg, "linear_value_head_dim", None)
        v_num = getattr(text_cfg, "linear_num_value_heads", None)
        key_dim = k_head * k_num if k_head is not None and k_num is not None else None
        val_dim = v_head * v_num if v_head is not None and v_num is not None else None
        if key_dim is None:
            for name in ("in_proj_q", "in_proj_k"):
                if name in members:
                    key_dim = members[name][1].shape[-2]
                    break
        if val_dim is None:
            for name in ("in_proj_v", "in_proj_z"):
                if name in members:
                    val_dim = members[name][1].shape[-2]
                    break
        return (key_dim, key_dim, val_dim, val_dim)

    def normalize_gate_up_proj(
        self,
        weight_names: List[str],
        weights: Dict[str, torch.Tensor],
        base_modules: Optional[_BaseModuleIndex] = None,
    ):
        mods = self._get_base_modules(base_modules)
        split_members = ("gate_proj", "up_proj")
        fused_name = "gate_up_proj"
        groups = _collect_fused_groups(
            weight_names,
            weights,
            split_members=split_members,
            fused_name=fused_name,
            skip_preserved=True,
        )
        for (prefix, tail), ab_group in groups.items():
            a_members = dict(ab_group.get("lora_A", {}))
            b_members = dict(ab_group.get("lora_B", {}))
            # PEFT can ship numel-zero placeholders when up_proj has no weight.
            empty_up_keys = tuple(
                m.pop("up_proj")[0]
                for m in (a_members, b_members)
                if "up_proj" in m and m["up_proj"][1].numel() == 0
            )
            if not a_members and not b_members:
                for key in empty_up_keys:
                    weights.pop(key, None)
                continue

            is_non_gated_moe = (
                bool(_ROUTED_EXPERT_PATTERN.search(f"{prefix}."))
                and _find_moe_is_gated(mods, prefix, shared=False) is False
            )
            if _check_group_structure(
                prefix, fused_name, split_members, a_members, b_members
            ):
                self._repeat_fused_lora_a(
                    weights,
                    a_members[fused_name][0],
                    a_members[fused_name][1],
                    b_members[fused_name][1],
                    1 if is_non_gated_moe else 2,
                )
                for key in empty_up_keys:
                    weights.pop(key, None)
                continue

            has_gate, has_up = "gate_proj" in a_members, "up_proj" in a_members
            if is_non_gated_moe and has_gate and has_up:
                raise ValueError(
                    f"Non-gated MoE layer at '{prefix}' does not support both "
                    f"gate_proj and up_proj LoRA weights."
                )
            if has_gate and not has_up and is_non_gated_moe:
                # Non-gated MoE uses stacked_multiply=1: rename without stacking.
                orig_a_key, a_t = a_members["gate_proj"]
                orig_b_key, b_t = b_members["gate_proj"]
                _validate_ab_pair(a_t, b_t, f"{prefix}.gate_proj")
                for key in (*empty_up_keys, orig_a_key, orig_b_key):
                    weights.pop(key, None)
                weights[_fused_key(prefix, fused_name, "lora_A", tail)] = a_t
                weights[_fused_key(prefix, fused_name, "lora_B", tail)] = b_t
                continue
            if (
                has_up
                and not has_gate
                and not _should_fuse_orphan_up_proj(mods, prefix)
            ):
                continue

            self._fuse_split_group(
                weights=weights,
                base_modules=mods,
                prefix=prefix,
                tail=tail,
                a_members=a_members,
                b_members=b_members,
                split_members=split_members,
                fused_name=fused_name,
                config_dims_fn=lambda _id, m: (
                    (d := next(iter(m.values()))[1].shape[-2]),
                    d,
                ),
                extra_pop_keys=empty_up_keys,
            )

    def normalize_fused_qkv_a_proj(
        self,
        weight_names: List[str],
        weights: Dict[str, torch.Tensor],
        base_modules: Optional[_BaseModuleIndex] = None,
    ):
        mods = self._get_base_modules(base_modules)

        def should_fuse(prefix: str, members: set[str]) -> bool:
            if "fused_qkv_a_proj_with_mqa" in members or "q_a_proj" in members:
                return True
            # Keep standalone kv_a_proj_with_mqa intact when q_lora_rank is None
            # (e.g., DeepSeek-V2-Lite without fused_qkv_a_proj_with_mqa).
            if (
                _lookup_base_module(mods, prefix, "fused_qkv_a_proj_with_mqa")
                is not None
            ):
                return True
            if _lookup_base_module(mods, prefix, "kv_a_proj_with_mqa") is not None:
                return False
            return getattr(self._get_text_config(), "q_lora_rank", None) is not None

        self._normalize_fused_projection(
            weight_names=weight_names,
            weights=weights,
            base_modules=mods,
            split_members=("q_a_proj", "kv_a_proj_with_mqa"),
            fused_name="fused_qkv_a_proj_with_mqa",
            should_fuse=should_fuse,
            config_dims_fn=self._infer_mla_qkv_a_dims_from_config,
        )

    def _infer_mla_qkv_a_dims_from_config(
        self, _layer_id: Optional[int], _members: _MemberGroup
    ) -> Tuple[Optional[int], Optional[int]]:
        text_cfg = self._get_text_config()
        q_a = getattr(text_cfg, "q_lora_rank", None)
        kv_lora = getattr(text_cfg, "kv_lora_rank", None)
        qk_rope = getattr(text_cfg, "qk_rope_head_dim", None)
        kv_a = (
            kv_lora + qk_rope if kv_lora is not None and qk_rope is not None else None
        )
        return (q_a, kv_a)


def _split_layer_path(path: str) -> Optional[Tuple[int, str]]:
    match = _LAYER_REL_PATH_PATTERN.search(path)
    if match is None:
        return None
    return int(match.group(1)), match.group(2) or ""


def _inspect_base_model(
    base_model: Optional[torch.nn.Module],
) -> Tuple[_BaseModuleIndex, set[int]]:
    if base_model is None:
        return {}, set()
    from sglang.srt.lora.layers import unwrap_lora_layer

    should_apply_lora = getattr(base_model, "should_apply_lora", None)
    modules: _BaseModuleIndex = {}
    pp_missing: set[int] = set()
    for name, module in base_model.named_modules():
        layer_info = _split_layer_path(name)
        if layer_info is None:
            continue
        layer_id, rel_path = layer_info
        if not rel_path:
            if isinstance(module, PPMissingLayer):
                pp_missing.add(layer_id)
            continue
        if should_apply_lora is not None and not should_apply_lora(name):
            continue
        if rel_path == "base_layer" or rel_path.endswith(".base_layer"):
            continue
        modules.setdefault(layer_id, {})[rel_path] = unwrap_lora_layer(module)
    return modules, pp_missing - modules.keys()


def _lookup_base_module(
    base_modules: _BaseModuleIndex, prefix: str, leaf: str
) -> Optional[torch.nn.Module]:
    layer_info = _split_layer_path(prefix)
    if layer_info is None:
        return None
    layer_mods = base_modules.get(layer_info[0])
    if not layer_mods:
        return None
    parent = layer_info[1]
    return layer_mods.get(f"{parent}.{leaf}" if parent else leaf)


def _find_moe_is_gated(
    base_modules: _BaseModuleIndex, prefix: str, *, shared: bool
) -> Optional[bool]:
    layer_info = _split_layer_path(prefix)
    if layer_info is None or not (layer_mods := base_modules.get(layer_info[0])):
        return None
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    exact = layer_mods.get(re.sub(r"\.\d+$", "", layer_info[1]))
    if isinstance(exact, FusedMoE):
        return bool(exact.moe_runner_config.is_gated)
    for mod in layer_mods.values():
        if isinstance(mod, FusedMoE) and bool(mod.is_shared_fused_moe) == shared:
            return bool(mod.moe_runner_config.is_gated)
    return None


def _should_fuse_orphan_up_proj(base_modules: _BaseModuleIndex, prefix: str) -> bool:
    if not base_modules:
        return False
    probe = f"{prefix}."
    if _ROUTED_EXPERT_PATTERN.search(probe):
        return _find_moe_is_gated(base_modules, prefix, shared=False) is True
    if _SHARED_EXPERT_MOE_PATTERN.search(probe):
        shared_gated = _find_moe_is_gated(base_modules, prefix, shared=True)
        if shared_gated is not None:
            return shared_gated
    if _lookup_base_module(base_modules, prefix, "up_proj") is not None:
        return False
    return _lookup_base_module(base_modules, prefix, "gate_up_proj") is not None


def _get_fused_module_slice_dims(
    module: Optional[torch.nn.Module], n_slices: int
) -> Optional[Tuple[int, ...]]:
    if module is None:
        return None
    from sglang.srt.layers.linear import MergedColumnParallelLinear, QKVParallelLinear

    if isinstance(module, QKVParallelLinear) and n_slices == 3:
        v_head_size = getattr(module, "v_head_size", module.head_size)
        return (
            module.total_num_heads * module.head_size,
            module.total_num_kv_heads * module.head_size,
            module.total_num_kv_heads * v_head_size,
        )
    if (
        isinstance(module, MergedColumnParallelLinear)
        and len(module.output_sizes) >= n_slices
    ):
        return tuple(int(s) for s in module.output_sizes[:n_slices])
    return None


def _resolve_slice_dims(
    prefix: str,
    fused_name: str,
    split_members: Tuple[str, ...],
    members: _MemberGroup,
    expected_dims: Optional[Tuple[Optional[int], ...]],
) -> Dict[str, int]:
    resolved: Dict[str, int] = {}
    for idx, member in enumerate(split_members):
        observed = members[member][1].shape[-2] if member in members else None
        expected = expected_dims[idx] if expected_dims is not None else None
        if observed is not None and expected is not None and observed != expected:
            raise ValueError(
                f"{member} lora_B output dim {observed} does not match expected "
                f"{expected} for '{fused_name}' at '{prefix}'."
            )
        dim = observed if observed is not None else expected
        if dim is None or dim <= 0:
            raise ValueError(
                f"Cannot determine output dim for missing '{member}' in partial "
                f"'{fused_name}' at '{prefix}'."
            )
        resolved[member] = dim
    return resolved


def _fused_key(prefix: str, leaf: str, ab_kind: str, tail: str) -> str:
    return f"{prefix}.{leaf}.{ab_kind}{tail}" if prefix else f"{leaf}.{ab_kind}{tail}"


def _collect_fused_groups(
    weight_names: List[str],
    weights: Dict[str, torch.Tensor],
    *,
    split_members: Tuple[str, ...],
    fused_name: str,
    skip_preserved: bool = False,
    exclude_member: Optional[Callable[[str, str], bool]] = None,
) -> Dict[Tuple[str, str], _ABGroup]:
    from sglang.srt.lora.utils import PRESERVED_LORA_TARGET_NAMES

    valid_leaves = frozenset((*split_members, fused_name))
    groups: Dict[Tuple[str, str], _ABGroup] = {}
    for weight_name in weight_names:
        if weight_name not in weights:
            continue
        match = _LORA_AB_KEY_PATTERN.match(weight_name)
        if match is None:
            continue
        prefix, _, leaf = match.group("module_path").rpartition(".")
        if leaf not in valid_leaves:
            continue
        if skip_preserved:
            full_path = f"{prefix}.{leaf}" if prefix else leaf
            if ".".join(full_path.split(".")[-2:]) in PRESERVED_LORA_TARGET_NAMES:
                continue
        if exclude_member is not None and exclude_member(prefix, leaf):
            continue
        ab_group = groups.setdefault((prefix, match.group("tail")), {})
        ab_group.setdefault(match.group("ab_kind"), {})[leaf] = (
            weight_name,
            weights[weight_name],
        )
    return groups


def _check_group_structure(
    prefix: str,
    fused_name: str,
    split_members: Tuple[str, ...],
    a_members: _MemberGroup,
    b_members: _MemberGroup,
) -> bool:
    a_keys, b_keys = set(a_members), set(b_members)
    if a_keys != b_keys:
        details = []
        if a_keys - b_keys:
            details.append(f"missing lora_B for {sorted(a_keys - b_keys)}")
        if b_keys - a_keys:
            details.append(f"missing lora_A for {sorted(b_keys - a_keys)}")
        raise ValueError(
            f"Unpaired LoRA A/B weights in fused group '{fused_name}' at "
            f"'{prefix}': {', '.join(details)}."
        )
    has_fused = fused_name in a_keys
    present_split = a_keys & set(split_members)
    if has_fused and present_split:
        raise ValueError(
            f"Conflicting split {sorted(present_split)} and fused '{fused_name}' "
            f"LoRA weights at '{prefix}'."
        )
    return has_fused


def _validate_ab_pair(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    label: str,
    *,
    check_rank: bool = True,
) -> None:
    if (
        a_tensor.dim() < 2
        or b_tensor.dim() < 2
        or a_tensor.dim() != b_tensor.dim()
        or a_tensor.dtype != b_tensor.dtype
        or a_tensor.device != b_tensor.device
    ):
        raise ValueError(f"Inconsistent LoRA A/B metadata for '{label}'.")
    if check_rank and a_tensor.shape[-2] != b_tensor.shape[-1]:
        raise ValueError(
            f"Inconsistent LoRA A/B ranks ({a_tensor.shape[-2]} vs "
            f"{b_tensor.shape[-1]}) for '{label}'."
        )


def _concat_split_parts(
    prefix: str,
    ab_kind: str,
    members: _MemberGroup,
    split_members: Tuple[str, ...],
    fused_name: str,
    member_out_dims: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    ref = next(iter(members.values()))[1]
    for member, (_, tensor) in members.items():
        if (
            tensor.dim() != ref.dim()
            or tensor.dtype != ref.dtype
            or tensor.device != ref.device
            or tensor.shape[:-2] != ref.shape[:-2]
        ):
            raise ValueError(
                f"Inconsistent tensor metadata for '{prefix}.{member}' in fused "
                f"group '{fused_name}'."
            )
        if ab_kind == "lora_A" and tensor.shape != ref.shape:
            raise ValueError(
                f"Inconsistent lora_A shapes ({tuple(ref.shape)} vs "
                f"{tuple(tensor.shape)}) across fused group '{prefix}'."
            )
        if ab_kind == "lora_B" and tensor.shape[-1] != ref.shape[-1]:
            raise ValueError(
                f"Inconsistent lora_B ranks ({ref.shape[-1]} vs "
                f"{tensor.shape[-1]}) across fused group '{prefix}'."
            )

    parts = []
    for member in split_members:
        if member in members:
            parts.append(members[member][1])
        elif ab_kind == "lora_A":
            parts.append(torch.zeros_like(ref))
        else:
            assert member_out_dims is not None
            parts.append(
                torch.zeros(
                    (*ref.shape[:-2], member_out_dims[member], ref.shape[-1]),
                    dtype=ref.dtype,
                    device=ref.device,
                )
            )
    return torch.cat(parts, dim=ref.dim() - 2)
