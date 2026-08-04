"""Fail-closed GGUF loading adapter for the Kimi K3 text model.

The Kimi K3 llama.cpp converter does more than rename tensors.  It packs the
routed experts, folds the KDA ``A_log`` exponential, splits MLA ``kv_b_proj``,
fuses attention-residual score weights, and reshapes the KDA convolutions.
The generic GGUF loader cannot safely infer those operations from a Hugging
Face state dict.  This module owns the exact inverse contract for the
language-only SGLang model.

Vision is intentionally out of scope.  Kimi K3 stores its vision tower in a
separate mmproj GGUF, which this loader neither opens nor silently ignores in
multimodal mode.
"""

from __future__ import annotations

import json
import mmap
import os
import re
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


KIMI_K3_HF_MODEL_TYPE = "kimi_k3"
KIMI_K3_GGUF_ARCH = "kimi-k3"
KIMI_K3_ARCHITECTURE = "KimiK3ForConditionalGeneration"
KIMI_K3_SITU = (4.0, 25.0)

_EXPERT_GGUF_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<projection>gate|down|up)_exps\.weight$"
)
_EXPERT_SOURCE_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\."
    r"(?P<expert>\d+)\.w[123]\.weight_(?:packed|scale|aster_quant)$"
)
_LAYER_SOURCE_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.(?P<tail>.+)$")

_EXPERT_PROJECTION_TO_CKPT = {
    "gate": "w1",
    "down": "w2",
    "up": "w3",
}


def is_kimi_k3_gguf_config(model_config: ModelConfig) -> bool:
    return getattr(
        model_config.hf_config, "model_type", None
    ) == KIMI_K3_HF_MODEL_TYPE and str(model_config.model_path).lower().endswith(
        ".gguf"
    )


class KimiK3GGUFQuantConfig:
    """Factory for the hybrid runtime quantization policy used by K3 GGUF.

    ASTER quantizes only routed experts.  All protected tensors in the text
    GGUF are F32/F16/BF16 and Kimi K3's implementation directly accesses their
    ``.weight`` parameters.  Applying :class:`GGUFLinearMethod` to those
    modules would replace ``.weight`` with ``.qweight`` and break model
    construction/post-load hooks.
    """

    @staticmethod
    def create():
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.gguf import GGUFConfig, GGUFMoEMethod
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedEmbeddingMethod,
            UnquantizedLinearMethod,
        )
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

        class _KimiK3GGUFConfig(GGUFConfig):
            stream_moe_weights_to_final_device = True

            def get_quant_method(self, layer: torch.nn.Module, prefix: str):
                if isinstance(layer, FusedMoE):
                    return GGUFMoEMethod(self)
                if isinstance(layer, LinearBase):
                    return UnquantizedLinearMethod()
                if isinstance(layer, VocabParallelEmbedding):
                    return UnquantizedEmbeddingMethod()
                return None

        return _KimiK3GGUFConfig()


@dataclass(frozen=True)
class _Transform:
    kind: str
    source_name: str | None = None
    partner_name: str | None = None


class _MMapRangeReleaser:
    """Bound the resident working set of the monolithic K3 GGUF mmap.

    The adapter deliberately yields zero-copy views into one 851 GiB mapping.
    A yielded view is consumed synchronously by SGLang's weight loader before
    the generator is resumed.  At that resume boundary it is safe to discard
    the source pages: all retained expert data already belongs to the CUDA
    parameter device.

    ``MADV_DONTNEED`` is mandatory.  ``POSIX_FADV_DONTNEED`` is an additional
    best-effort page-cache hint; failure of that optional hint falls back to
    the successful mmap advice rather than silently disabling eviction.
    """

    def __init__(self, path: str, reader) -> None:
        mapped = getattr(getattr(reader, "data", None), "_mmap", None)
        madvise = getattr(mapped, "madvise", None)
        if (
            mapped is None
            or not callable(madvise)
            or not hasattr(mmap, "MADV_DONTNEED")
        ):
            raise RuntimeError(
                "Kimi K3 GGUF loading requires mmap MADV_DONTNEED support to "
                "bound host page-cache residency"
            )
        try:
            mapped_size = int(mapped.size())
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError("Kimi K3 GGUF mmap size is unavailable") from error
        if mapped_size <= 0:
            raise RuntimeError("Kimi K3 GGUF mmap is empty")

        self._path = path
        self._mapped = mapped
        self._mapped_size = mapped_size
        self._page_size = int(mmap.PAGESIZE)
        self._fd: int | None = None
        self._fadvise_enabled = hasattr(os, "posix_fadvise") and hasattr(
            os, "POSIX_FADV_DONTNEED"
        )

    def _aligned_range(self, offset: int, length: int) -> tuple[int, int]:
        offset = int(offset)
        length = int(length)
        if offset < 0 or length <= 0 or offset + length > self._mapped_size:
            raise ValueError(
                "Kimi K3 GGUF release range is outside the mmap: "
                f"offset={offset}, length={length}, size={self._mapped_size}"
            )
        start = offset - (offset % self._page_size)
        raw_end = offset + length
        end = min(
            self._mapped_size,
            ((raw_end + self._page_size - 1) // self._page_size) * self._page_size,
        )
        return start, end - start

    def release(self, offset: int, length: int) -> None:
        start, aligned_length = self._aligned_range(offset, length)
        try:
            self._mapped.madvise(mmap.MADV_DONTNEED, start, aligned_length)
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Kimi K3 GGUF could not evict a consumed mmap range"
            ) from error

        if not self._fadvise_enabled:
            return
        try:
            if self._fd is None:
                self._fd = os.open(self._path, os.O_RDONLY)
            os.posix_fadvise(self._fd, start, aligned_length, os.POSIX_FADV_DONTNEED)
        except OSError:
            # The mmap advice above already discarded this process's pages.
            # Some FUSE filesystems reject fadvise; remember that proven
            # fallback instead of retrying a rejected hint for every tensor.
            self._fadvise_enabled = False
            self.close()

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def _strip_language_prefix(name: str) -> str:
    return name.removeprefix("language_model.")


def _field_value(reader, key: str):
    field = reader.fields.get(key)
    if field is None:
        raise ValueError(f"Kimi K3 GGUF metadata is missing {key!r}")
    try:
        value = field.contents()
    except Exception as error:
        raise ValueError(f"Kimi K3 GGUF metadata {key!r} is unreadable") from error
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if hasattr(value, "item") and not isinstance(value, (str, list, tuple, dict)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    return value


def _find_arch(gguf_module):
    names = {value: key for key, value in gguf_module.MODEL_ARCH_NAMES.items()}
    if KIMI_K3_GGUF_ARCH in names:
        return names[KIMI_K3_GGUF_ARCH]
    # Released gguf packages can parse the K3 container and all of its stock
    # Kimi-linear tensor names but may predate the dedicated KIMI_K3 enum.  K3
    # additions are mapped explicitly below, so the Kimi-linear table is a
    # complete and deterministic base rather than a best-effort fallback.
    if "kimi-linear" in names:
        return names["kimi-linear"]
    raise RuntimeError(
        "The installed gguf package has neither kimi-k3 nor kimi-linear "
        "tensor-name maps"
    )


def _read_weight_index(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise ValueError(
            "Kimi K3 GGUF requires the adjacent model.safetensors.index.json "
            "to bind every protected tensor to the source checkpoint"
        )
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Kimi K3 source weight index is unreadable: {path}"
        ) from error
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Kimi K3 source weight index has no non-empty weight_map")
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise ValueError("Kimi K3 source weight index contains non-string entries")
    return tuple(weight_map)


def _dense_tensor(tensor) -> torch.Tensor:
    """Return a zero-copy torch view for a protected GGUF dense tensor."""

    type_name = tensor.tensor_type.name
    if type_name not in {"F32", "F16", "BF16"}:
        raise ValueError(
            f"protected Kimi K3 tensor {tensor.name!r} unexpectedly uses {type_name}"
        )
    value = torch.from_numpy(tensor.data)
    if type_name == "F32":
        if value.dtype != torch.float32:
            raise ValueError(
                f"F32 tensor {tensor.name!r} has storage dtype {value.dtype}"
            )
        return value
    if type_name == "F16":
        if value.dtype == torch.float16:
            return value
        if value.element_size() != 2:
            raise ValueError(f"F16 tensor {tensor.name!r} has invalid storage")
        return value.view(torch.float16)
    # GGUFReader exposes BF16 as its raw uint8 byte rows (unlike F16, which
    # it exposes as float16). Reinterpret pairs of bytes without dequantizing
    # or copying; ``view`` restores the logical final dimension.
    if value.dtype != torch.uint8 or value.shape[-1] % 2:
        raise ValueError(f"BF16 tensor {tensor.name!r} has invalid storage")
    return value.view(torch.bfloat16)


class KimiK3GGUFAdapter:
    """Validated tensor inventory and inverse converter for one K3 text GGUF."""

    def __init__(self, gguf_file: str, model_config: ModelConfig):
        import gguf

        self.gguf_file = os.fspath(gguf_file)
        self.model_config = model_config
        self.config = model_config.hf_config
        self.text_config = model_config.hf_text_config
        self.reader = gguf.GGUFReader(self.gguf_file)
        self._range_releaser = _MMapRangeReleaser(self.gguf_file, self.reader)
        self._verify_config_and_metadata()

        arch = _find_arch(gguf)
        self.name_map = gguf.get_tensor_name_map(
            arch, int(self.text_config.num_hidden_layers)
        )
        self.source_names = _read_weight_index(
            Path(self.gguf_file).parent / "model.safetensors.index.json"
        )
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        if len(self.tensors) != len(self.reader.tensors):
            raise ValueError("Kimi K3 GGUF contains duplicate tensor names")

        self.regular: dict[str, str] = {}
        self.transforms: dict[str, _Transform] = {}
        self.experts: list[tuple[object, int, str]] = []
        self._build_inventory()

    def _verify_config_and_metadata(self) -> None:
        architectures = getattr(self.config, "architectures", None)
        if architectures != [KIMI_K3_ARCHITECTURE]:
            raise ValueError(
                "Kimi K3 GGUF requires architectures exactly "
                f"[{KIMI_K3_ARCHITECTURE!r}], got {architectures!r}"
            )
        if getattr(self.config, "language_only", False) is not True:
            raise ValueError(
                "Kimi K3 GGUF currently supports --language-only; the separate "
                "vision mmproj is not loaded by this adapter"
            )
        if _field_value(self.reader, "general.architecture") != KIMI_K3_GGUF_ARCH:
            raise ValueError("adjacent config does not match GGUF general.architecture")

        num_layers = int(self.text_config.num_hidden_layers)
        if int(_field_value(self.reader, "kimi-k3.block_count")) != num_layers:
            raise ValueError("adjacent Kimi K3 layer count differs from GGUF metadata")

        config_situ = (
            float(self.text_config.activation_situ_beta),
            float(self.text_config.activation_situ_linear_beta),
        )
        gguf_situ = (
            float(_field_value(self.reader, "kimi-k3.activation.situ_beta")),
            float(_field_value(self.reader, "kimi-k3.activation.situ_linear_beta")),
        )
        if config_situ != KIMI_K3_SITU or gguf_situ != KIMI_K3_SITU:
            raise ValueError(
                "Kimi K3 SiTU contract requires beta=4.0 and linear_beta=25.0; "
                f"config={config_situ!r}, gguf={gguf_situ!r}"
            )

    def _full_attention_layers(self) -> set[int]:
        config = self.text_config.linear_attn_config or {}
        # The checkpoint list is one-indexed; GGUF block ids are zero-indexed.
        return {int(value) - 1 for value in config.get("full_attn_layers", [])}

    def _record_regular(self, gguf_name: str, source_name: str) -> None:
        previous = self.regular.setdefault(gguf_name, source_name)
        if previous != source_name:
            raise ValueError(
                f"multiple Kimi K3 source tensors map to {gguf_name!r}: "
                f"{previous!r}, {source_name!r}"
            )

    def _record_transform(self, name: str, transform: _Transform) -> None:
        previous = self.transforms.setdefault(name, transform)
        if previous != transform:
            raise ValueError(f"conflicting Kimi K3 transforms for {name!r}")

    def _classify_special_source(self, source_name: str, stripped: str) -> bool:
        layer_match = _LAYER_SOURCE_RE.match(stripped)
        if layer_match is None:
            if stripped in {
                "model.output_attn_res_norm.weight",
                "model.output_attn_res_proj.weight",
            }:
                self._record_transform(
                    "output_res_score.weight", _Transform("residual_score")
                )
                return True
            return False

        layer = int(layer_match.group("layer"))
        tail = layer_match.group("tail")
        prefix = f"blk.{layer}"
        if tail in {
            "self_attention_res_norm.weight",
            "self_attention_res_proj.weight",
        }:
            self._record_transform(
                f"{prefix}.attn_res_score.weight",
                _Transform("residual_score", f"model.layers.{layer}.self_attention"),
            )
            return True
        if tail in {"mlp_res_norm.weight", "mlp_res_proj.weight"}:
            self._record_transform(
                f"{prefix}.ffn_res_score.weight",
                _Transform("residual_score", f"model.layers.{layer}.mlp"),
            )
            return True
        if tail == "self_attn.A_log":
            self._record_transform(f"{prefix}.ssm_a", _Transform("a_log", source_name))
            return True
        if tail == "self_attn.dt_bias":
            self._record_transform(
                f"{prefix}.ssm_dt.bias", _Transform("direct", source_name)
            )
            return True
        if tail in {
            "self_attn.q_conv1d.weight",
            "self_attn.k_conv1d.weight",
            "self_attn.v_conv1d.weight",
        }:
            role = tail.removeprefix("self_attn.")[0]
            self._record_transform(
                f"{prefix}.ssm_conv1d_{role}.weight",
                _Transform("conv1d", source_name),
            )
            return True
        if tail == "self_attn.g_proj.weight":
            gguf_role = (
                "attn_gate" if layer in self._full_attention_layers() else "ssm_g"
            )
            self._record_transform(
                f"{prefix}.{gguf_role}.weight", _Transform("direct", source_name)
            )
            return True
        if tail == "self_attn.kv_b_proj.weight":
            k_name = f"{prefix}.attn_k_b.weight"
            v_name = f"{prefix}.attn_v_b.weight"
            self._record_transform(k_name, _Transform("kv_b", source_name, v_name))
            self._record_transform(
                v_name, _Transform("kv_b_partner", source_name, k_name)
            )
            return True
        routed = {
            "block_sparse_moe.routed_expert_down_proj.weight": "ffn_routed_down",
            "block_sparse_moe.routed_expert_up_proj.weight": "ffn_routed_up",
            "block_sparse_moe.routed_expert_norm.weight": "ffn_routed_norm",
        }.get(tail)
        if routed is not None:
            self._record_transform(
                f"{prefix}.{routed}.weight", _Transform("direct", source_name)
            )
            return True
        if tail == "block_sparse_moe.gate.e_score_correction_bias":
            self._record_transform(
                f"{prefix}.exp_probs_b.bias", _Transform("direct", source_name)
            )
            return True
        return False

    def _build_inventory(self) -> None:
        for source_name in self.source_names:
            stripped = _strip_language_prefix(source_name)
            if stripped.startswith(("vision_tower.", "mm_projector.")):
                continue
            if _EXPERT_SOURCE_RE.match(stripped):
                continue
            if self._classify_special_source(source_name, stripped):
                continue
            gguf_name = self.name_map.get_name(
                stripped, try_suffixes=(".weight", ".bias")
            )
            if gguf_name is None:
                raise ValueError(
                    f"Kimi K3 source tensor has no GGUF mapping: {source_name!r}"
                )
            self._record_regular(gguf_name, source_name)

        expected_moe_layers = {
            layer
            for layer in range(int(self.text_config.num_hidden_layers))
            if getattr(self.text_config, "num_experts", None) is not None
            and layer >= int(self.text_config.first_k_dense_replace)
            and layer % int(self.text_config.moe_layer_freq) == 0
        }
        expert_roles: dict[int, set[str]] = {}
        for tensor in self.reader.tensors:
            match = _EXPERT_GGUF_RE.match(tensor.name)
            if match is None:
                continue
            layer = int(match.group("layer"))
            projection = match.group("projection")
            if tensor.tensor_type.name not in {"IQ2_XXS", "IQ2_XS"}:
                raise ValueError(
                    f"ASTER routed tensor {tensor.name!r} has unsupported "
                    f"type {tensor.tensor_type.name}"
                )
            if int(tensor.data.shape[0]) != int(self.text_config.num_experts):
                raise ValueError(
                    f"ASTER routed tensor {tensor.name!r} expert count differs "
                    "from the adjacent config"
                )
            expert_roles.setdefault(layer, set()).add(projection)
            self.experts.append((tensor, layer, projection))
        if set(expert_roles) != expected_moe_layers or any(
            roles != {"gate", "down", "up"} for roles in expert_roles.values()
        ):
            raise ValueError(
                "Kimi K3 GGUF routed-expert tensor inventory is incomplete"
            )

        allowed = (
            set(self.regular)
            | set(self.transforms)
            | {tensor.name for tensor, _, _ in self.experts}
        )
        actual = set(self.tensors)
        missing = sorted(allowed - actual)
        extra = sorted(actual - allowed)
        if missing or extra:
            raise ValueError(
                "Kimi K3 GGUF tensor inventory differs from the exact adapter plan: "
                f"missing={missing[:8]!r} ({len(missing)} total), "
                f"extra={extra[:8]!r} ({len(extra)} total)"
            )

    def quant_config(self):
        return KimiK3GGUFQuantConfig.create()

    def _release_tensor_range(
        self, tensor, *, relative_offset: int = 0, length: int | None = None
    ) -> None:
        relative_offset = int(relative_offset)
        tensor_bytes = int(tensor.n_bytes)
        if length is None:
            length = tensor_bytes - relative_offset
        length = int(length)
        if (
            relative_offset < 0
            or length <= 0
            or relative_offset + length > tensor_bytes
        ):
            raise ValueError(
                f"invalid release subrange for K3 GGUF tensor {tensor.name!r}: "
                f"offset={relative_offset}, length={length}, bytes={tensor_bytes}"
            )
        self._range_releaser.release(int(tensor.data_offset) + relative_offset, length)

    def _iter_expert_types(self) -> Iterable[tuple[str, torch.Tensor]]:
        num_experts = int(self.text_config.num_experts)
        for tensor, layer, projection in self.experts:
            ckpt_projection = _EXPERT_PROJECTION_TO_CKPT[projection]
            for expert in range(num_experts):
                name = (
                    f"language_model.model.layers.{layer}.block_sparse_moe."
                    f"experts.{expert}.{ckpt_projection}.qweight_type"
                )
                yield name, torch.tensor(int(tensor.tensor_type), dtype=torch.int64)

    def _iter_dense(self) -> Iterable[tuple[str, torch.Tensor]]:
        for tensor in self.reader.tensors:
            if tensor.name in self.regular:
                try:
                    yield self.regular[tensor.name], _dense_tensor(tensor)
                finally:
                    self._release_tensor_range(tensor)
                continue
            transform = self.transforms.get(tensor.name)
            if transform is None:
                continue
            if transform.kind == "kv_b_partner":
                continue
            release_tensors = [tensor]
            try:
                value = _dense_tensor(tensor)
                if transform.kind == "direct":
                    assert transform.source_name is not None
                    yield transform.source_name, value
                elif transform.kind == "a_log":
                    if not torch.isfinite(value).all() or not torch.all(value < 0):
                        raise ValueError(
                            f"encoded K3 A_log is invalid: {tensor.name!r}"
                        )
                    assert transform.source_name is not None
                    yield transform.source_name, torch.log(-value.float())
                elif transform.kind == "conv1d":
                    if value.ndim != 4 or value.shape[0] != 1 or value.shape[2] != 1:
                        raise ValueError(
                            f"K3 conv tensor has unexpected shape: {tensor.name!r} "
                            f"{tuple(value.shape)!r}"
                        )
                    assert transform.source_name is not None
                    yield (
                        transform.source_name,
                        value.reshape(value.shape[1], 1, value.shape[3]),
                    )
                elif transform.kind == "kv_b":
                    assert transform.partner_name is not None
                    partner_tensor = self.tensors[transform.partner_name]
                    release_tensors.append(partner_tensor)
                    partner = _dense_tensor(partner_tensor)
                    n_heads = int(self.text_config.num_key_value_heads)
                    kv_rank = int(self.text_config.kv_lora_rank)
                    qk_dim = int(self.text_config.qk_nope_head_dim)
                    v_dim = int(self.text_config.v_head_dim)
                    if tuple(value.shape) != (n_heads, kv_rank, qk_dim):
                        raise ValueError(
                            f"K3 k_b shape differs from config: {tuple(value.shape)!r}"
                        )
                    if tuple(partner.shape) != (n_heads, v_dim, kv_rank):
                        raise ValueError(
                            f"K3 v_b shape differs from config: "
                            f"{tuple(partner.shape)!r}"
                        )
                    combined = torch.cat((value.transpose(1, 2), partner), dim=1)
                    assert transform.source_name is not None
                    yield (
                        transform.source_name,
                        combined.reshape(n_heads * (qk_dim + v_dim), kv_rank),
                    )
                elif transform.kind == "residual_score":
                    if value.numel() != int(self.text_config.hidden_size):
                        raise ValueError(
                            f"K3 residual score has unexpected size: {tensor.name!r}"
                        )
                    if not torch.isfinite(value).all():
                        raise ValueError(
                            f"K3 residual score is non-finite: {tensor.name!r}"
                        )
                    if tensor.name == "output_res_score.weight":
                        name = "language_model.model.output_attn_res_score_gguf"
                    elif tensor.name.endswith(".attn_res_score.weight"):
                        layer = int(tensor.name.split(".")[1])
                        name = (
                            f"language_model.model.layers.{layer}."
                            "self_attention_res_score_gguf"
                        )
                    else:
                        layer = int(tensor.name.split(".")[1])
                        name = f"language_model.model.layers.{layer}.mlp_res_score_gguf"
                    yield name, value.float().reshape(-1)
                else:
                    # Inventory construction owns this enum.
                    raise AssertionError(
                        f"unknown K3 GGUF transform {transform.kind!r}"
                    )
            finally:
                for release_tensor in release_tensors:
                    self._release_tensor_range(release_tensor)

    def _iter_expert_weights(self) -> Iterable[tuple[str, torch.Tensor]]:
        num_experts = int(self.text_config.num_experts)
        for tensor, layer, projection in self.experts:
            ckpt_projection = _EXPERT_PROJECTION_TO_CKPT[projection]
            weight = tensor.data
            for expert in range(num_experts):
                name = (
                    f"language_model.model.layers.{layer}.block_sparse_moe."
                    f"experts.{expert}.{ckpt_projection}.qweight"
                )
                expert_weight = weight[expert]
                if not expert_weight.flags.c_contiguous:
                    raise ValueError(
                        f"K3 GGUF expert tensor is not row-contiguous: {tensor.name!r}"
                    )
                relative_offset = expert * int(weight.strides[0])
                # Zero-copy mmap view. FusedMoE's GGUF loader immediately owns
                # only the local TP shard on the parameter device. The finally
                # runs when that synchronous consumer requests the next weight.
                try:
                    yield name, torch.from_numpy(expert_weight)
                finally:
                    self._release_tensor_range(
                        tensor,
                        relative_offset=relative_offset,
                        length=int(expert_weight.nbytes),
                    )

    def weights_iterator(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        # Qtypes must precede packed weights; the FusedMoE loader uses them to
        # compute the byte-axis TP shard without materializing the full expert.
        try:
            yield from self._iter_expert_types()
            yield from self._iter_dense()
            yield from self._iter_expert_weights()
        finally:
            self._range_releaser.close()
