from __future__ import annotations

import logging
import re
from copy import copy, deepcopy
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    _NPULinearMethodBase,
)
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.modelslim.schemes import (
    ModelSlimMXFP4Scheme,
    ModelSlimMXFP4W4A8Scheme,
    ModelSlimMXFP8MoEScheme,
    ModelSlimMXFP8Scheme,
    ModelSlimQFP8DynamicKVFP8Scheme,
    ModelSlimW4A4Int4,
    ModelSlimW4A4Int4MoE,
    ModelSlimW4A4MXFP4MoE,
    ModelSlimW4A8Int8MoE,
    ModelSlimW4A8MXFP4MoE,
    ModelSlimW8A8Int8,
    ModelSlimW8A8Int8MoE,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import apply_module_patch

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.modelslim.schemes import (
        ModelSlimLinearScheme,
    )

logger = logging.getLogger(__name__)


# func refers to RMSNorm.__init__
def npu_wrapper_rmsnorm_init(func):
    def init(self, hidden_size: int, **extra_args) -> None:
        func(self, hidden_size, **extra_args)
        self.ignore_anti = True
        # The Ascend w8a8_int8 quantization requires adding a bias in rmsnorm
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)

    return init


# func refers to RMSNorm.forward_oot
def npu_wrapper_rmsnorm_forward(func):
    def _rmsnorm_forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            from sgl_kernel_npu.norm.add_rmsnorm_bias import add_rmsnorm_bias

            out, residual_out = add_rmsnorm_bias(
                x,
                residual,
                self.weight.data,
                self.bias,
                self.variance_epsilon,
            )
            return out.to(x.dtype), residual_out

        out = torch.ops.npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]
        out = out + self.bias
        return out.to(x.dtype)

    return _rmsnorm_forward_oot


def remap_dspark_weight_name_npu(
    name: str, *, for_quant_config: bool = False
) -> Optional[str]:
    """Map a checkpoint name without consulting model state.

    The loader selects vocabulary stages and filters disabled confidence-head
    weights. ModelSlim uses the same names to add quantization-config aliases.
    Config aliases retain scale suffixes and rotary entries and omit vocabulary
    weights, matching the original ModelSlim configuration semantics.
    """
    if not name.startswith("mtp."):
        return None
    if not for_quant_config and "rotary_emb.inv_freq" in name:
        return None
    parts = name.split(".", 2)
    if len(parts) < 3:
        return None
    stage_id, rest = parts[1], parts[2]
    if not stage_id.isdigit():
        return None
    if not for_quant_config:
        # Preserve the loader's integer-stage validation; isdigit() alone also
        # accepts characters such as superscript digits that int() rejects.
        int(stage_id)

    if rest.startswith(("embed.", "embed_tokens.", "head.", "lm_head.")):
        if not for_quant_config:
            if rest in ("embed.weight", "embed_tokens.weight"):
                return "embed_tokens.weight"
            if rest in ("head.weight", "lm_head.weight"):
                return "lm_head.weight"
        return None
    if rest.startswith(("markov_head.", "confidence_head.")):
        return rest

    mapped_rest = rest
    if mapped_rest.startswith("attn."):
        mapped_rest = "self_attn." + mapped_rest.removeprefix("attn.")
    elif mapped_rest.startswith("ffn."):
        mapped_rest = "mlp." + mapped_rest.removeprefix("ffn.")
    elif mapped_rest.startswith("attn_norm."):
        mapped_rest = "input_layernorm." + mapped_rest.removeprefix("attn_norm.")
    elif mapped_rest.startswith("ffn_norm."):
        mapped_rest = "post_attention_layernorm." + mapped_rest.removeprefix(
            "ffn_norm."
        )
    mapped_rest = mapped_rest.replace(".w1.", ".gate_proj.")
    mapped_rest = mapped_rest.replace(".w2.", ".down_proj.")
    mapped_rest = mapped_rest.replace(".w3.", ".up_proj.")
    mapped_rest = mapped_rest.replace(".gate.tid2eid", ".topk.tid2eid")
    mapped_rest = mapped_rest.replace(".gate.bias", ".gate.e_score_correction_bias")
    if not for_quant_config and mapped_rest.endswith(".scale"):
        mapped_rest = mapped_rest.removesuffix(".scale") + ".weight_scale_inv"
    return f"stages.{stage_id}.{mapped_rest}"


class ModelSlimConfig(QuantizationConfig):
    """
    Config class for ModelSlim Quantization, a NPU-specific quantization type.
    """

    supports_kimi_k3_quantized_latent_projections = True

    # Hardcoded compatibility for models missing an MLA fusion declaration.
    _HARDCODED_PACKED_MODULES = {
        "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
    }
    # Legacy vision aliases: runtime path fragment -> checkpoint path fragment.
    # Apply only within the matching vision scope and when the description exists.
    _LEGACY_VISION_ALIASES = {
        "vision_model": {"proj": "out_proj"},
        "visual": {"qkv_proj": "qkv"},
        "vision_tower": {"attn.qkv_proj": "wqkv", "attn.proj": "wo"},
        "mm_projector": {"attn.qkv_proj": "wqkv", "attn.proj": "wo"},
    }

    def __init__(self, quant_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.requested_kv_cache_dtype = "auto"
        quant_config = dict(quant_config or {})
        keys = [k for k in quant_config if isinstance(k, str)]
        is_dsv4 = any(k.startswith("hc_head_") for k in keys)
        if is_dsv4:
            from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

            remap = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
            quant_config = {
                (remap(k) if isinstance(k, str) else k): v
                for k, v in quant_config.items()
            }

        # Add an mlp.* alias for each block_sparse_moe.* key but KEEP the original,
        # so both module namings resolve
        for k in list(quant_config.keys()):
            if not isinstance(k, str):
                continue
            if "block_sparse_moe" in k:
                quant_config[
                    k.replace("block_sparse_moe.experts", "mlp.experts").replace(
                        "block_sparse_moe.shared_experts", "mlp.shared_experts"
                    )
                ] = quant_config[k]

        self.quant_description = quant_config
        ignore = cast(List[str], quant_config.get("ignore", []))
        self.ignore = ignore if ignore is not None else []
        packed_modules_mapping = quant_config.get("packed_modules_mapping", {})
        self.packed_modules_mapping = (
            packed_modules_mapping if packed_modules_mapping is not None else {}
        )

        for name in self.quant_description.keys():
            if "norm.bias" in name:
                apply_module_patch(
                    "sglang.srt.layers.layernorm.RMSNorm",
                    "__init__",
                    [npu_wrapper_rmsnorm_init],
                )
                apply_module_patch(
                    "sglang.srt.layers.layernorm.RMSNorm",
                    "forward_npu",
                    [npu_wrapper_rmsnorm_forward],
                )
        # Share DSpark path remapping while retaining config-specific suffixes,
        # checkpoint keys, and the precedence of explicitly configured aliases.
        dspark_quant_aliases = {}
        if any(
            isinstance(name, str) and name.startswith("mtp.") for name in quant_config
        ):
            for name, scheme in quant_config.items():
                if not isinstance(name, str):
                    continue
                alias = remap_dspark_weight_name_npu(name, for_quant_config=True)
                if alias is not None:
                    dspark_quant_aliases[alias] = scheme

        quant_config = {
            **dspark_quant_aliases,
            **quant_config,
        }

        self.quant_description = quant_config

    def configure_kv_cache_quantization(self, dtype):
        """Set this model's requested KV cache dtype."""
        self.requested_kv_cache_dtype = dtype

    def get_modelslim_kv_cache_layout(self, model, *, use_mla_backend):
        """Return the layout of this worker's installed ModelSlim KV scheme."""
        for module in model.modules():
            if isinstance(
                getattr(module, "quant_method", None), ModelSlimKVCacheMethod
            ):
                return module.scheme.get_cache_layout(use_mla_backend=use_mla_backend)
        return None

    @staticmethod
    def _kv_scheme_class(quant_type):
        # Map supported ModelSlim KV formats to their scheme classes.
        kv_quant_schemes = {
            "Q_FP8_DYNAMIC_KV_FP8": ModelSlimQFP8DynamicKVFP8Scheme,
        }
        scheme_class = kv_quant_schemes.get(quant_type)
        if scheme_class is None:
            raise ValueError(f"Unsupported ModelSlim KV quant_type: {quant_type!r}")
        return scheme_class

    def get_modelslim_kv_scheme(self, prefix):
        if self.requested_kv_cache_dtype in ("auto", "bf16", "bfloat16"):
            return None
        # Map the model's layer prefix to checkpoint-name candidates, e.g.
        # with or without "language_model."; try the original name first.
        resolved_prefix = prefix
        candidates = self._quant_prefix_candidates(prefix)
        kv_suffixes = (
            "quant_type",
            "fa_k.scale",
            "fa_k.offset",
            "fa_v.scale",
            "fa_v.offset",
        )
        has_layer_metadata = False
        # Resolve to the first candidate with a quant_type or fa_k/fa_v key.
        for candidate in candidates:
            if any(
                f"{candidate}.{suffix}" in self.quant_description
                for suffix in kv_suffixes
            ):
                resolved_prefix = candidate
                has_layer_metadata = True
                break
        quant_type = self.quant_description.get(f"{resolved_prefix}.quant_type")
        if quant_type == "FLOAT" or any(
            fnmatchcase(resolved_prefix, ignored)
            or resolved_prefix.startswith(ignored + ".")
            for ignored in self.ignore
        ):
            raise ValueError(
                "ModelSlim quantized KV cache cannot share a pool "
                f"with unquantized attention layer {resolved_prefix}. "
                "Remove that layer's ignore/FLOAT declaration or use "
                "--kv-cache-dtype auto for a BF16 cache."
            )
        if not has_layer_metadata:
            raise ValueError(
                f"{resolved_prefix}: ModelSlim quantized KV cache requires "
                "a per-layer FA quantization declaration"
            )
        try:
            scheme_class = self._kv_scheme_class(quant_type)
        except ValueError as error:
            raise ValueError(
                f"{resolved_prefix}: unsupported ModelSlim KV format "
                f"(quant_type={quant_type!r})"
            ) from error
        return scheme_class(resolved_prefix, self.requested_kv_cache_dtype)

    @staticmethod
    def _is_quant_entry(name: Any, value: Any) -> bool:
        """Check whether a key-value pair has the format of a quantization entry.

        Args:
            name: A key from quant_description, e.g. "attn.qkv.weight".
            value: Its value, e.g. "W8A8" or a packed_modules_mapping dict.

        Returns:
            True for a string key containing a dot and a string value.
            ("attn.qkv.weight", "W8A8") -> True.
            ("quant_method", "modelslim") -> False.
            ("packed_modules_mapping", {"qkv_proj": ["q_proj", "k_proj", "v_proj"]})
                -> False.
            This checks entry format, not scheme support.
        """
        return isinstance(name, str) and "." in name and isinstance(value, str)

    @staticmethod
    def _map_quant_entries(entries, mapper) -> List[Tuple[str, str]]:
        """Rename entry keys without changing their values.

        Apply substring/prefix rules, then try suffix rules on the full key.
        If no full-key suffix matches, try the layer name and add the field back.

        Args:
            entries: Entry dict, e.g. {"attn.qkv.weight": "W8A8"}.
            mapper: The model's checkpoint-to-runtime name rules.
                Example: suffix ".qkv" -> ".qkv_proj".

        Returns:
            A list of pairs, e.g. [("attn.qkv_proj.weight", "W8A8")].
            None rules omit entries from this result; unmatched names stay.
            Duplicate names stay in the list so the caller can check conflicts.
        """
        non_suffix_mapper = copy(mapper)
        non_suffix_mapper.orig_to_new_suffix = {}
        suffix_mapper = copy(mapper)
        suffix_mapper.orig_to_new_prefix = {}
        suffix_mapper.orig_to_new_substr = {}

        mapped_entries = []
        for name, value in entries.items():
            mapped = non_suffix_mapper.apply_list([name])
            # An empty result means a rule explicitly skips this name.
            # If no rule matches, apply_list returns the original name.
            if not mapped:
                continue
            name = mapped[0]

            matches_suffix = any(
                name.endswith(suffix) for suffix in mapper.orig_to_new_suffix
            )
            if matches_suffix:
                mapped = suffix_mapper.apply_list([name])
                if not mapped:
                    continue
                mapped_name = mapped[0]
            elif "." in name:
                layer_name, field = name.rsplit(".", 1)
                mapped = suffix_mapper.apply_list([layer_name])
                if not mapped:
                    continue
                mapped_name = f"{mapped[0]}.{field}"
            else:
                mapped_name = name

            mapped_entries.append((mapped_name, value))

        return mapped_entries

    def apply_weight_name_mapper(self, hf_to_sglang_mapper):
        """Add renamed keys to quant_description; keep the original keys.

        Example: keep attn.qkv.weight and add attn.qkv_proj.weight with the same value.
        Always map from the first-call snapshot to avoid repeated renaming.

        Args:
            hf_to_sglang_mapper: The model's checkpoint-to-runtime name rules.
                None rules add no alias; they do not delete original entries.

        Returns:
            None. Replaces quant_description after all conflict checks pass.

        Raises:
            ValueError: An existing or newly mapped key has a different value.
        """
        # Reuse original keys on each call: avoid qkv -> qkv_proj -> qkv_proj_proj.
        if not hasattr(self, "_unmapped_quant_description"):
            self._unmapped_quant_description = dict(self.quant_description)

        # Select quantization entries; leave settings like quant_method alone.
        quant_entries = {
            name: value
            for name, value in self._unmapped_quant_description.items()
            if self._is_quant_entry(name, value)
        }
        mapped_entries = self._map_quant_entries(quant_entries, hf_to_sglang_mapper)

        # Keep existing entries; reject different values for the same name.
        updated_description = dict(self.quant_description)
        for name, value in mapped_entries:
            if name in updated_description and updated_description[name] != value:
                raise ValueError(f"Conflicting ModelSlim descriptions for {name}")
            updated_description[name] = value
        # Save only after all checks pass.
        self.quant_description = updated_description

    def update_packed_modules_mapping(
        self, mapping: Dict[str, List[str] | Dict[str, List[str]]]
    ) -> None:
        """Update fusion rules without changing the caller's tables or lists.

        Args:
            mapping: Rules to add or replace, e.g.
                {"gate_up_proj": ["gate_proj", "up_proj"]}.
                Also accepts Kimi's {"model": {...}} table.
                Same-name rules are replaced, including whole nested tables.

        Returns:
            None. Stores a deep-copied, updated packed_modules_mapping.
        """
        # Preserve dict.update semantics without modifying shared tables or lists.
        updated = deepcopy(self.packed_modules_mapping)
        updated.update(deepcopy(mapping))
        self.packed_modules_mapping = updated

    def _quant_prefix_candidates(self, prefix: str) -> List[str]:
        """List possible layer names without looking them up in the description.

        Args:
            prefix: Layer path without .weight or other parameter suffixes.
                Example: model.layers.0.mlp.gate_proj.

        Returns:
            Candidate paths with duplicates removed; the input comes first.
            Example: model.layers.0.mlp.gate_proj -> [
                "model.layers.0.mlp.gate_proj",
                "model.layers.0.block_sparse_moe.gate_proj",
                "model.language_model.layers.0.mlp.gate_proj",
                "language_model.model.layers.0.mlp.gate_proj",
                "model.language_model.layers.0.block_sparse_moe.gate_proj",
                "language_model.model.layers.0.block_sparse_moe.gate_proj",
            ]
        """
        candidates = [prefix]
        for candidate in list(candidates):
            if candidate.startswith(".") and not candidate.startswith(".."):
                candidates.append(candidate[1:])
        # Kimi-K3 uses mlp in code and block_sparse_moe in the checkpoint.
        for candidate in list(candidates):
            if ".mlp." in candidate:
                candidates.append(candidate.replace(".mlp.", ".block_sparse_moe."))

        for candidate in list(candidates):
            if candidate.startswith("model.language_model."):
                candidates.append(
                    "model." + candidate.removeprefix("model.language_model.")
                )
            elif candidate.startswith("model."):
                candidates.append(
                    "model.language_model." + candidate.removeprefix("model.")
                )
            if candidate.startswith("model.visual."):
                candidates.append(candidate.removeprefix("model."))
            elif candidate.startswith("visual."):
                candidates.append("model." + candidate)
            if candidate.startswith("language_model."):
                candidates.append(candidate.removeprefix("language_model."))
            else:
                candidates.append(f"language_model.{candidate}")

        return list(dict.fromkeys(candidates))

    def _resolve_quant_prefix(self, prefix: str) -> str:
        """Find a layer name whose .weight entry exists in quant_description.

        Args:
            prefix: Layer path without .weight or other parameter suffixes.
                Example: visual.blocks.0.attn.qkv.

        Returns:
            A layer path without .weight or other parameter suffixes.
            Example: model.visual.blocks.0.attn.qkv.
            Returns the input unchanged if no candidate matches.
        """
        for candidate in self._quant_prefix_candidates(prefix):
            if candidate + ".weight" in self.quant_description:
                return candidate
        return prefix

    def get_linear_method(self) -> ModelSlimLinearMethod:
        return ModelSlimLinearMethod(self)

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_name(cls) -> str:
        return "modelslim"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        filenames = ["quant_model_description.json"]
        return filenames

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ModelSlimConfig:
        return cls(config)

    def _get_packed_components(self, name: str) -> List[str]:
        """Find fusion components using model rules before defaults.

        Args:
            name: The last part of a layer name, e.g. gate_up_proj.

        Returns:
            A copied list, e.g. ["gate_proj", "up_proj"].
            Returns [] for an explicit empty rule or when no rule exists.

        Raises:
            ValueError: A nested mapping group other than "model" is present.
        """
        from sglang.srt.layers.quantization.utils import _FALLBACK_FUSED_SHARDS

        mapping = self.packed_modules_mapping

        # KimiK3LinearForCausalLM uses {"model": {...}} instead of a flat table.
        # Reject other groups: ignoring {"visual": {...}} could select wrong defaults.
        for group, parts in mapping.items():
            if isinstance(parts, Mapping) and group != "model":
                raise ValueError(
                    "ModelSlim supports flat packed mappings and the Kimi-K3 "
                    f"'model' group only; found group {group!r}"
                )
        model_mapping = mapping.get("model", {})
        if isinstance(model_mapping, Mapping) and name in model_mapping:
            return list(model_mapping[name])

        for candidates in (
            mapping,
            _FALLBACK_FUSED_SHARDS,
            self._HARDCODED_PACKED_MODULES,
        ):
            if name not in candidates:
                continue
            parts = candidates[name]
            if isinstance(parts, Mapping):
                continue
            return list(parts)
        return []

    @staticmethod
    def _replace_path_fragment(prefix: str, old: str, new: str) -> str:
        """Replace whole path segments, including at the start or end.

        Args:
            prefix: Layer path without .weight or other parameter suffixes.
                Example: vision_tower.blocks.0.attn.proj.
            old: Path fragment to replace, e.g. attn.proj.
            new: Replacement fragment, e.g. wo.

        Returns:
            The renamed path, e.g. vision_tower.blocks.0.wo.
            Keeps self_attn.proj and attn.projection unchanged.
        """
        # Match attn.proj, but not self_attn.proj or attn.projection.
        # Keep the shared dot: proj.proj -> out_proj.out_proj, not out_proj.proj.
        padded = "." + prefix + "."
        pattern = r"(?<=\.)" + re.escape(old) + r"(?=\.)"
        renamed = re.sub(pattern, lambda match: new, padded)
        return renamed[1:-1]

    def _resolve_legacy_vision_quant_prefix(self, prefix: str) -> Optional[str]:
        """Look up an old vision name missing from the model's mapper.

        Args:
            prefix: Layer path without .weight or other parameter suffixes.
                Example: model.visual.blocks.0.attn.qkv_proj.

        Returns:
            A layer path without .weight or other parameter suffixes.
            Example: model.visual.blocks.0.attn.qkv.
            Returns None if the alias does not apply or its .weight entry is absent.
        """
        path_parts = prefix.split(".")
        aliases = {}

        # Pick one table: vision_model before visual if both appear in the path.
        for scope, scope_aliases in self._LEGACY_VISION_ALIASES.items():
            if scope in path_parts:
                aliases = scope_aliases
                break

        # Rename whole path segments, then check for a .weight description.
        for runtime_name, checkpoint_name in aliases.items():
            candidate = self._replace_path_fragment(
                prefix, runtime_name, checkpoint_name
            )
            if candidate == prefix:
                continue
            quant_prefix = self._resolve_quant_prefix(candidate)
            if quant_prefix + ".weight" in self.quant_description:
                return quant_prefix
        return None

    def _resolve_linear_quant_prefixes(
        self,
        prefix: str,
        output_sizes: Optional[List[int]] = None,
    ) -> List[str]:
        """Find .weight descriptions through three supported paths:

        - Original name: attn.qkv_proj.
        - Legacy vision name: visual.attn.qkv_proj -> visual.attn.qkv.
        - Split original name: qkv_proj -> q_proj, k_proj, v_proj.

        Do not split legacy names. Check split entries even if a whole entry exists.

        Args:
            prefix: Layer path without .weight or other parameter suffixes.
                Example: model.layers.0.self_attn.qkv_proj.
            output_sizes: Optional sizes in component order, e.g. [8, 4, 0].
                Skip zero-size parts only when the number of sizes matches the parts.

        Returns:
            Layer paths without .weight or other parameter suffixes.
            One whole-layer path or a list of component paths.
            If no description matches, returns [prefix] for the caller to handle.

        Raises:
            ValueError: Some components are missing, their schemes differ,
                whole and split layouts both exist, or a mapping group is invalid.
        """
        # Original name: look for attn.qkv_proj.weight, including mapper aliases.
        direct = self._resolve_quant_prefix(prefix)
        whole_quant_prefix = (
            direct if direct + ".weight" in self.quant_description else None
        )
        # If absent, try a legacy vision name: visual.attn.qkv_proj -> visual.attn.qkv.
        if whole_quant_prefix is None:
            whole_quant_prefix = self._resolve_legacy_vision_quant_prefix(prefix)

        # Split the original name: qkv_proj -> q_proj, k_proj, v_proj.
        # Use the runtime name to look up packed rules, not the legacy name.
        parent, separator, name = prefix.rpartition(".")
        shards = self._get_packed_components(name)
        # Skip zero-size parts, e.g. [Q, K, V] with sizes [8, 4, 0] uses Q and K.
        # Sizes must have one entry per part.
        if output_sizes is not None and len(output_sizes) == len(shards):
            shards = [shard for shard, size in zip(shards, output_sizes) if size != 0]
        components = [
            self._resolve_quant_prefix(parent + separator + shard) for shard in shards
        ]
        present = [p + ".weight" in self.quant_description for p in components]
        if any(present):
            if not all(present):
                missing = [
                    p + ".weight" for p, found in zip(components, present) if not found
                ]
                raise ValueError(
                    f"Missing ModelSlim quantization descriptions for some components "
                    f"of fused layer {prefix}. Missing keys: {missing}"
                )
            schemes = {self.quant_description[p + ".weight"] for p in components}
            if len(schemes) != 1:
                entries = {
                    p + ".weight": self.quant_description[p + ".weight"]
                    for p in components
                }
                raise ValueError(
                    f"ModelSlim requires all components of fused layer {prefix} "
                    f"to use the same quantization scheme. Found: {entries}"
                )

        # Reject whole + split descriptions: qkv_proj and separate q/k/v.
        # A single part equal to the whole name is the same entry, not a conflict.
        if any(present):
            if whole_quant_prefix is not None and components != [whole_quant_prefix]:
                raise ValueError(
                    f"Ambiguous ModelSlim layout for {prefix}: whole tensor "
                    f"{whole_quant_prefix}.weight and component tensors "
                    f"{[p + '.weight' for p in components]} are both described"
                )
            return components
        if whole_quant_prefix is not None:
            return [whole_quant_prefix]
        # No description found; let the caller handle the missing .weight key.
        return [direct]

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        """Find the layer's quantization scheme and return a method object.

        Save the chosen scheme on layer for later use.

        Args:
            layer: The layer being built, e.g. QKVParallelLinear.
            prefix: Layer path without .weight or other parameter suffixes.
                Example: model.layers.0.self_attn.qkv_proj.

        Returns:
            A method object, e.g. W8A8 -> ModelSlimLinearMethod.
            For linear layers, FLOAT or a missing description -> unquantized.
            None for layers other than LinearBase or FusedMoE.
        """
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            layer.scheme = self.get_linear_scheme(layer, prefix)
            if layer.scheme is None:
                return UnquantizedLinearMethod()
            return ModelSlimLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            moe_schemes = self.get_moe_scheme(layer, prefix)
            if moe_schemes is None:
                raise ValueError(f"No ModelSlim MoE scheme found for layer {prefix}")
            layer.w13_scheme, layer.w2_scheme = moe_schemes
            layer.w13_kernel, layer.w2_kernel = (
                layer.w13_scheme.kernel,
                layer.w2_scheme.kernel,
            )
            return ModelSlimFusedMoEMethod(self)
        return None

    def get_linear_scheme(
        self, layer: torch.nn.Module, prefix: Optional[str] = None
    ) -> Optional[ModelSlimLinearScheme]:
        """Find the description and create this linear layer's scheme.

        Args:
            layer: The layer being built, e.g. QKVParallelLinear.
                Its output_sizes supplies the sizes of fused components when available.
            prefix: Layer path without .weight or other parameter suffixes.
                Example: model.layers.0.self_attn.qkv_proj.

        Returns:
            A scheme, e.g. ModelSlimW8A8Int8; None for FLOAT or no description.

        Raises:
            ValueError: Fusion descriptions conflict or are incomplete,
                a mapping group is invalid, or the declared scheme is unsupported.
        """

        # Mapper aliases are already in quant_description. Check whole and split names.
        quant_prefixes = self._resolve_linear_quant_prefixes(
            prefix, output_sizes=getattr(layer, "output_sizes", None)
        )
        prefix = quant_prefixes[
            0
        ]  # All components were checked to use the same scheme.

        linear_quant_schemes = [
            ("W4A4_DYNAMIC", ModelSlimW4A4Int4),
            ("W8A8", ModelSlimW8A8Int8),
            ("W8A8S", ModelSlimW8A8Int8),
            ("W8A8_DYNAMIC", ModelSlimW8A8Int8),
            ("W8A8_MXFP8", ModelSlimMXFP8Scheme),
            ("W4A8_MXFP", ModelSlimMXFP4W4A8Scheme),
            ("W4A4_MXFP4", ModelSlimMXFP4Scheme),
        ]

        key = prefix + ".weight"
        if key not in self.quant_description:
            logger.warning(
                "No ModelSlim description for layer %s; using unquantized weights. "
                "Tried: %s",
                prefix,
                [
                    candidate + ".weight"
                    for candidate in self._quant_prefix_candidates(prefix)
                ],
            )
            return None
        scheme_name = self.quant_description[key]
        if scheme_name == "FLOAT":
            return None

        for supported_name, scheme_class in linear_quant_schemes:
            if scheme_name == supported_name:
                logger.info_once(f"Using {scheme_class.__name__}")
                return scheme_class(quant_config=self.quant_description, prefix=prefix)

        raise ValueError(
            f"Unsupported ModelSlim Linear scheme {scheme_name!r} for layer {prefix} "
            f"(description key: {key})"
        )

    def get_moe_scheme(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ):
        moe_quant_schemes = [
            ("W4A4_MXFP4", ModelSlimW4A4MXFP4MoE),
            ("W4A8_MXFP", ModelSlimW4A8MXFP4MoE),
            ("W4A4_DYNAMIC", ModelSlimW4A4Int4MoE),
            ("W4A8_DYNAMIC", ModelSlimW4A8Int8MoE),
            ("W8A8_DYNAMIC", ModelSlimW8A8Int8MoE),
            ("W8A8_MXFP8", ModelSlimMXFP8MoEScheme),
        ]

        # Try multiple naming conventions:
        #   (gate_proj, up_proj, down_proj) – standard compressed-tensors format
        #   (w1, w3, w2)                      – MiniMax-M2.5 / some other models
        naming_conventions = [
            ("gate_proj", "up_proj", "down_proj"),
            ("w1", "w3", "w2"),
        ]

        w13_scheme_name = None
        w2_scheme_name = None
        resolved_prefix = prefix
        for candidate in self._quant_prefix_candidates(prefix):
            for gate_name, up_name, down_name in naming_conventions:
                w13_keys = [
                    f"{candidate}.0.{gate_name}.weight",
                    f"{candidate}.0.{up_name}.weight",
                ]
                w2_key = f"{candidate}.0.{down_name}.weight"
                w13_entries = {
                    key: self.quant_description[key]
                    for key in w13_keys
                    if key in self.quant_description
                }
                if w13_entries and w2_key in self.quant_description:
                    w13_names = list(w13_entries.values())
                    # For w13, both projections must agree on the scheme
                    unique_w13 = set(w13_names)
                    if len(unique_w13) > 1:
                        raise ValueError(
                            "Mismatched ModelSlim quantization for W13 in layer "
                            f"{prefix}: {w13_entries}"
                        )
                    w13_scheme_name = w13_names[0]
                    w2_scheme_name = self.quant_description[w2_key]
                    resolved_prefix = candidate
                    break
            if w13_scheme_name is not None:
                break

        if w13_scheme_name is None:
            # Build a helpful error message listing all attempted key patterns
            all_attempted = []
            for candidate in self._quant_prefix_candidates(prefix):
                for gate_name, up_name, down_name in naming_conventions:
                    w13_keys = [
                        f"{candidate}.0.{gate_name}.weight",
                        f"{candidate}.0.{up_name}.weight",
                    ]
                    w2_key = f"{candidate}.0.{down_name}.weight"
                    w13_found = any(k in self.quant_description for k in w13_keys)
                    w2_found = w2_key in self.quant_description
                    status = (
                        f"{candidate} "
                        f"({gate_name}/{up_name}="
                        f"{'found' if w13_found else 'missing'}, "
                        f"{down_name}={'found' if w2_found else 'missing'})"
                    )
                    all_attempted.append(status)
            raise ValueError(
                f"Missing ModelSlim MoE quantization description for layer {prefix}: "
                + "; ".join(all_attempted)
            )

        # Map scheme names to classes
        scheme_map = dict(
            moe_quant_schemes
        )  # dict: "W4A4_DYNAMIC" -> ModelSlimW4A4Int4MoE, etc.

        # Instantiate the schemes
        def instantiate(name, weight_group):
            cls = scheme_map.get(name)
            if cls is None:
                logger.warning(
                    f"Unsupported scheme '{name}' for layer {resolved_prefix}"
                )
                return None
            return cls(self, weight_group)

        w13_scheme = instantiate(w13_scheme_name, weight_group="w13")
        w2_scheme = instantiate(w2_scheme_name, weight_group="w2")
        if w13_scheme is None or w2_scheme is None:
            raise ValueError(
                f"Unsupported ModelSlim MoE schemes for layer {prefix}: "
                f"W13='{w13_scheme_name}', W2='{w2_scheme_name}'"
            )
        logger.info_once(f"Using {type(w13_scheme).__name__} for W13")
        logger.info_once(f"Using {type(w2_scheme).__name__} for W2")

        return w13_scheme, w2_scheme

    def get_scaled_act_names(self) -> List[str]:
        return []


class ModelSlimKVCacheMethod(QuantizeMethodBase):
    """Use the attention layer's scheme without GPU per-tensor KV fallbacks."""

    def create_weights(self, layer, **kwargs):
        layer.scheme.create_weights(layer, **kwargs)

    def process_weights_after_loading(self, layer):
        layer.scheme.process_weights_after_loading(layer)

    def restore_weights_before_loading(self, layer):
        for parameter in layer.parameters(recurse=False):
            parameter.modelslim_loaded = False

    def apply(self, layer, k, v):
        return layer.scheme.quantize_cache(k, v)


class ModelSlimLinearMethod(_NPULinearMethodBase):
    def __init__(self, quantization_config: ModelSlimConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the ModelSlimLinearScheme associated with the layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Use the output of create_weights and the ModelSlimLinearScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)


class ModelSlimFusedMoEMethod(FusedMoEMethodBase):
    """
    Fused MoE method for ModelSlim quantization on Ascend NPU.

    Delegates routing, activation, and finalization to the modular NPU MoE
    components introduced in the hardware backend refactoring.
    """

    def __init__(self, quantization_config: ModelSlimConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_scheme.process_weights_after_loading(layer)
        layer.w2_scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the ModelSlimMoEScheme associated with the layer to create
        the necessary parameters for the layer. See FusedMoEMethodBase for param
        details
        """
        layer.w13_scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            weight_prefix="w13",
            **extra_weight_attrs,
        )
        layer.w2_scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            weight_prefix="w2",
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        moe_runner_config.layer = layer
        self.moe_runner_config = moe_runner_config
        backend = get_moe_runner_backend()
        if backend.is_auto():
            backend = MoeRunnerBackend.ASCEND
        self.runner = MoeRunner(backend, moe_runner_config)

    # ------------------------------------------------------------------
    # Main apply()
    # ------------------------------------------------------------------
    def apply(
        self,
        layer,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

        quant_info = AscendQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight_scale=layer.w2_weight_scale,
            w13_weight_offset=getattr(layer, "w13_weight_offset", None),
            w2_weight_offset=getattr(layer, "w2_weight_offset", None),
            w13_scale_bias=getattr(layer, "w13_scale_bias", None),
            w2_scale_bias=getattr(layer, "w2_scale_bias", None),
            w13_weight_bias=getattr(layer, "w13_weight_bias", None),
            w2_weight_bias=getattr(layer, "w2_weight_bias", None),
        )
        return self.runner.run(dispatch_output, quant_info)
