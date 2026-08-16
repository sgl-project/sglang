# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

import msgspec
import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn

from sglang.multimodal_gen.configs.models.dits.magi2 import (
    Magi2PreviewArchConfig,
    Magi2PreviewConfig,
    Magi2RefinerArchConfig,
    Magi2RefinerConfig,
)
from sglang.multimodal_gen.configs.models.vaes.magi2 import (
    Magi2AudioVAEConfig,
    Magi2TurboVAEConfig,
    Magi2VideoVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.magi2 import Magi2PipelineConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.attention.magi2_block_grid_attention import (
    Magi2BlockGridAttention,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.models.dits.magi2_common import Magi2ModalityRMSNorm
from sglang.multimodal_gen.runtime.models.dits.magi2_preview import Magi2PreviewDiT
from sglang.multimodal_gen.runtime.models.dits.magi2_refiner import Magi2RefinerDiT
from sglang.multimodal_gen.runtime.models.vaes.magi2_audio_vae import Magi2AudioVAE
from sglang.multimodal_gen.runtime.models.vaes.magi2_turbo_vae import (
    Magi2TurboVAE,
    strip_turbo_vae_state_dict_prefix,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

_INDEX_NAME = "model.safetensors.index.json"
_SINGLE_NAME = "model.safetensors"

_PREVIEW_DIR = "preview"
_REFINER_DIR = "refiner"
_TEXT_ENCODER_DIR = "text_encoder"
_VIDEO_VAE_FILE = os.path.join("vae", "Wan2.2_VAE.pth")
_TURBO_VAE_FILE = os.path.join("turbo_vae", "checkpoint.ckpt")
_AUDIO_VAE_DIR = "stable-audio-open-1.0"

# Training state that would trip the strict load.
_TURBO_VAE_TRAINING_ONLY_PREFIXES = ("aligned_feature_projection_heads.",)

# W_gate and W_up fuse into w13_weight, and all three need a transpose.
_CKPT_W_GATE = "W_gate"
_CKPT_W_UP = "W_up"
_CKPT_W_DOWN = "W_down"

# Checkpoint spelling: the W_* entries become w13_weight/w2 in _relayout_experts.
_EP_SHARDED_SUFFIXES = (
    "moe_mlp.router.gate",
    "moe_mlp.router.expert_bias",
    f"moe_mlp.{_CKPT_W_GATE}",
    f"moe_mlp.{_CKPT_W_UP}",
    f"moe_mlp.{_CKPT_W_DOWN}",
)

# Composed with the arch config's mapping through sglang's mapping engine.
_PREVIEW_RENAMES: dict[str, str] = {
    r"^(.*)\.mhc_phi_fused_(attn|mlp)$": r"\1.mhc_\2.phi_fused",
    r"^(.*)\.mhc_(alpha|bias)_(pre|post|res)_(attn|mlp)$": r"\1.mhc_\4.\2_\3",
    r"^(.*)\.attention\.sinks$": r"\1.attention.attn.sinks",
    r"^(.*)\.moe_mlp\.gate$": r"\1.moe_mlp.router.gate",
}

_REFINER_RENAMES: dict[str, str] = {}


class Magi2ExpertShard(msgspec.Struct, frozen=True):
    """Expert tensor rows are ``head`` major, ``expert`` minor."""

    ep_size: int
    ep_rank: int
    num_heads: int
    num_experts: int

    @classmethod
    def from_group(
        cls,
        *,
        ep_group: dist.ProcessGroup | None,
        num_heads: int,
        num_experts: int,
    ) -> Magi2ExpertShard:
        ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
        ep_rank = 0 if ep_group is None else dist.get_rank(ep_group)
        if num_heads % ep_size:
            raise ValueError(
                f"MAGI-2 expert parallelism needs num_heads ({num_heads}) "
                f"divisible by ep_size ({ep_size}); a rank owns whole heads"
            )
        return cls(
            ep_size=ep_size,
            ep_rank=ep_rank,
            num_heads=num_heads,
            num_experts=num_experts,
        )

    @property
    def row_range(self) -> tuple[int, int] | None:
        """Rows of dim 0 this rank owns, or ``None`` when it owns all of them."""
        if self.ep_size == 1:
            return None
        rows = (self.num_heads // self.ep_size) * self.num_experts
        return self.ep_rank * rows, (self.ep_rank + 1) * rows


def _read_weight_map(checkpoint_dir: str) -> dict[str, str]:
    index_path = os.path.join(checkpoint_dir, _INDEX_NAME)
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return {
            name: os.path.join(checkpoint_dir, shard)
            for name, shard in weight_map.items()
        }

    single_path = os.path.join(checkpoint_dir, _SINGLE_NAME)
    if not os.path.exists(single_path):
        raise FileNotFoundError(
            f"No {_INDEX_NAME} or {_SINGLE_NAME} under {checkpoint_dir}"
        )
    with safe_open(single_path, framework="pt") as handle:
        return {name: single_path for name in handle.keys()}  # noqa: SIM118


def _target_name(source_name: str, mapping_fn) -> str:
    # Empty target means the mapping drops the key.
    target, _merge_index, _num_to_merge = mapping_fn(source_name)
    return target


def _expert_target(source_name: str, mapping_fn) -> tuple[str, str] | None:
    for role in (_CKPT_W_GATE, _CKPT_W_UP, _CKPT_W_DOWN):
        suffix = f".moe_mlp.{role}"
        if source_name.endswith(suffix):
            stem = source_name[: -len(suffix)]
            # Sibling key reuses the engine's layer rename; roles have no 1:1 name.
            mapped = _target_name(f"{stem}.moe_mlp.gate", mapping_fn)
            return mapped[: -len(".moe_mlp.router.gate")], role
    return None


def _read_rows(handle, name: str, *, row_range: tuple[int, int] | None) -> torch.Tensor:
    """``get_slice`` keeps the read lazy: only the requested byte range leaves disk."""
    if row_range is None:
        return handle.get_tensor(name)

    start, end = row_range
    view = handle.get_slice(name)
    total_rows = view.get_shape()[0]
    if end > total_rows:
        raise ValueError(
            f"{name}: expert shard rows [{start}, {end}) exceed the checkpoint's "
            f"{total_rows} rows; ep_size does not match the checkpoint"
        )
    return view[start:end]


def _fuse_w13(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Disk layout is ``(E, head_dim, intermediate)`` (magi2_core/model/magi2_preview.py:2547-2566); the kernel wants the transpose, gate first."""
    fused = torch.cat((gate.transpose(1, 2), up.transpose(1, 2)), dim=1)
    # fused_experts' Triton path asserts contiguity.
    assert fused.is_contiguous(), "w13_weight must be contiguous"
    return fused


def _transpose_w2(down: torch.Tensor) -> torch.Tensor:
    w2 = down.transpose(1, 2).contiguous()
    assert w2.is_contiguous(), "w2 must be contiguous"
    return w2


def _rms_norm_param_names(model: nn.Module) -> set[str]:
    """Stored as an offset from 1 (magi2_core/model/magi2_preview.py:524, :263, refiner :265/:275), folded in at load time."""
    return {
        f"{module_name}.weight"
        for module_name, module in model.named_modules()
        if isinstance(module, Magi2ModalityRMSNorm)
    }


def _fp32_param_names(model: nn.Module) -> set[str]:
    """Enforced, not assumed: a single fp32 ULP in the rotary path cost 61 dB."""
    names: set[str] = set()
    for name, _tensor in _named_tensors(model):
        if (
            name.startswith(("pre_adapter.", "post_adapter."))
            or ".mhc_" in name
            or name.endswith((".router.gate", ".router.expert_bias", ".sinks"))
        ):
            names.add(name)
    # Norms are fp32 in the reference and upcast per forward anyway.
    return names | _rms_norm_param_names(model)


def _named_tensors(model: nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    yield from model.named_parameters()
    yield from model.named_buffers()


def mark_magi2_params_required(model: nn.Module) -> None:
    """Without this a missed ``expert_bias`` arrives as zeros (fsdp_load.py:827); same trick as minimax_h3.py:1204-1206."""
    for _name, param in model.named_parameters():
        param.missing_param_init = "error"


def assert_magi2_fp32_params(model: nn.Module) -> None:
    tensors = dict(_named_tensors(model))
    wrong = {
        name: tensors[name].dtype
        for name in sorted(_fp32_param_names(model))
        if tensors[name].dtype != torch.float32
    }
    if wrong:
        raise ValueError(
            f"{type(model).__name__}: these tensors must stay fp32 after load, "
            f"got {wrong}"
        )


def _assert_full_coverage(model: nn.Module, loaded: set[str]) -> None:
    """Stronger than ``missing_param_init``, which skips buffers such as the router's ``expert_bias``."""
    required = {name for name, _ in _named_tensors(model)}
    missing = sorted(required - loaded)
    if missing:
        raise ValueError(
            f"{type(model).__name__}: {len(missing)} checkpoint tensor(s) missing, "
            f"first few: {missing[:8]}"
        )
    unexpected = sorted(loaded - required)
    if unexpected:
        raise ValueError(
            f"{type(model).__name__}: checkpoint has {len(unexpected)} tensor(s) "
            f"with no model counterpart, first few: {unexpected[:8]}"
        )


def _plan_reads(
    *, weight_map: dict[str, str], mapping_fn
) -> dict[str, list[tuple[str, str]]]:
    """Grouped by shard so each file opens once; dropped keys never enter the plan, so their bytes are never read."""
    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    dropped = 0
    for source_name, path in weight_map.items():
        expert = _expert_target(source_name, mapping_fn)
        if expert is not None:
            prefix, role = expert
            by_shard[path].append((source_name, f"{prefix}.moe_mlp.{role}"))
            continue
        target = _target_name(source_name, mapping_fn)
        if not target:
            dropped += 1
            continue
        by_shard[path].append((source_name, target))

    if dropped:
        # Expected: one raw expert_bias per MoE layer, superseded by its EMA twin.
        logger.info(
            "[magi2] dropped %d checkpoint tensor(s) by param_names_mapping",
            dropped,
        )
    return by_shard


def _is_ep_sharded(target_name: str) -> bool:
    return target_name.endswith(_EP_SHARDED_SUFFIXES)


def _read_shard(
    item: tuple[str, list[tuple[str, str]]], *, shard: Magi2ExpertShard
) -> dict[str, torch.Tensor]:
    path, entries = item
    out: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as handle:
        for source_name, target_name in entries:
            out[target_name] = _read_rows(
                handle,
                source_name,
                row_range=(shard.row_range if _is_ep_sharded(target_name) else None),
            )
    return out


def _relayout_experts(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    gate_suffix = f".moe_mlp.{_CKPT_W_GATE}"
    for name, tensor in raw.items():
        if name.endswith(gate_suffix):
            prefix = name[: -len(gate_suffix)]
            state[f"{prefix}.moe_mlp.experts.w13_weight"] = _fuse_w13(
                tensor, raw[f"{prefix}.moe_mlp.{_CKPT_W_UP}"]
            )
        elif name.endswith(f".moe_mlp.{_CKPT_W_DOWN}"):
            prefix = name[: -len(f".moe_mlp.{_CKPT_W_DOWN}")]
            state[f"{prefix}.moe_mlp.experts.w2"] = _transpose_w2(tensor)
        elif not name.endswith(f".moe_mlp.{_CKPT_W_UP}"):
            state[name] = tensor
    return state


def _cast_state_dict(
    state: dict[str, torch.Tensor],
    *,
    fp32_names: set[str],
    rms_norm_names: set[str],
    param_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        dtype = torch.float32 if name in fp32_names else param_dtype
        tensor = tensor.to(device=device, dtype=dtype)
        if name in rms_norm_names:
            tensor = tensor + 1.0
        out[name] = tensor
    return out


def _load_dit_from_shards(
    model: nn.Module,
    *,
    checkpoint_dir: str,
    renames: dict[str, str],
    config_mapping: dict[str, str],
    shard: Magi2ExpertShard,
    param_dtype: torch.dtype,
    device: torch.device,
    max_workers: int,
) -> nn.Module:
    mapping_fn: Callable[[str], tuple[str, object, object]] = get_param_names_mapping(
        {**config_mapping, **renames}
    )
    weight_map = _read_weight_map(checkpoint_dir)
    plan = _plan_reads(weight_map=weight_map, mapping_fn=mapping_fn)

    raw: dict[str, torch.Tensor] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for shard_state in pool.map(
            lambda item: _read_shard(item, shard=shard), plan.items()
        ):
            raw.update(shard_state)

    state = _relayout_experts(raw)
    state = _cast_state_dict(
        state,
        fp32_names=_fp32_param_names(model),
        rms_norm_names=_rms_norm_param_names(model),
        param_dtype=param_dtype,
        device=device,
    )

    _assert_full_coverage(model, set(state))
    # assign=True: built on meta, so there is nothing to copy into.
    model.load_state_dict(state, strict=True, assign=True)
    assert_magi2_fp32_params(model)

    for param in model.parameters():
        param.requires_grad = False
    return model.eval()


def load_magi2_preview_dit(
    *,
    checkpoint_root: str,
    config: Magi2PreviewConfig,
    ep_group: dist.ProcessGroup | None,
    device: torch.device,
    param_dtype: torch.dtype,
    max_workers: int = 8,
) -> Magi2PreviewDiT:
    arch: Magi2PreviewArchConfig = config.arch_config
    shard = Magi2ExpertShard.from_group(
        ep_group=ep_group,
        num_heads=arch.moe_num_heads,
        num_experts=arch.moe_num_experts,
    )
    logger.info(
        "[magi2] preview DiT: ep_size=%d ep_rank=%d local_experts=%d",
        shard.ep_size,
        shard.ep_rank,
        (arch.moe_num_heads // shard.ep_size) * arch.moe_num_experts,
    )

    with set_default_torch_dtype(param_dtype), torch.device("meta"):
        model = Magi2PreviewDiT(config=config, ep_group=ep_group)

    mark_magi2_params_required(model)
    return _load_dit_from_shards(
        model,
        checkpoint_dir=os.path.join(checkpoint_root, _PREVIEW_DIR),
        renames=_PREVIEW_RENAMES,
        config_mapping=arch.param_names_mapping,
        shard=shard,
        param_dtype=param_dtype,
        device=device,
        max_workers=max_workers,
    )


def load_magi2_refiner_dit(
    *,
    checkpoint_root: str,
    config: Magi2RefinerConfig,
    device: torch.device,
    param_dtype: torch.dtype,
    max_workers: int = 4,
) -> Magi2RefinerDiT:
    """No experts, so the refiner is replicated on every rank and the shard degenerates to a full read."""
    arch: Magi2RefinerArchConfig = config.arch_config
    attention = Magi2BlockGridAttention(
        num_heads=arch.num_attention_heads,
        head_dim=arch.head_dim,
        num_kv_heads=arch.num_query_groups,
    )

    with set_default_torch_dtype(param_dtype), torch.device("meta"):
        model = Magi2RefinerDiT(config=config, attention=attention)

    mark_magi2_params_required(model)
    return _load_dit_from_shards(
        model,
        checkpoint_dir=os.path.join(checkpoint_root, _REFINER_DIR),
        renames=_REFINER_RENAMES,
        config_mapping=arch.param_names_mapping,
        shard=Magi2ExpertShard.from_group(ep_group=None, num_heads=1, num_experts=1),
        param_dtype=param_dtype,
        device=device,
        max_workers=max_workers,
    )


def _torch_load(path: str) -> dict:
    """``weights_only=True`` covers the video VAE but rejects the turbo decoder's pickled optimizer state."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, AttributeError) as e:
        logger.warning(
            "[magi2] %s needs weights_only=False (%s); it is a training "
            "checkpoint carrying non-tensor objects",
            os.path.basename(path),
            type(e).__name__,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def _wan_pth_residual_block_key(tail: str) -> str:
    """Sequential indices 0/2/3/6 plus ``shortcut`` (vae2_2.py:225-236) become named members (wanvae.py:702-711)."""
    if tail.startswith("shortcut."):
        return "conv_shortcut." + tail[len("shortcut.") :]
    index, rest = tail[len("residual.") :].split(".", 1)
    member = {"0": "norm1", "2": "conv1", "3": "norm2", "6": "conv2"}[index]
    return f"{member}.{rest}"


def _wan_pth_head_key(tail: str) -> str:
    # head Sequential -> norm_out / conv_out (vae2_2.py:565-569).
    index, rest = tail[len("head.") :].split(".", 1)
    return {"0": f"norm_out.{rest}", "2": f"conv_out.{rest}"}[index]


def _wan_pth_stack_key(tail: str, *, block_attr: str, sampler_attr: str) -> str:
    """Reads resample-vs-residual off the key (vae2_2.py:101-124), so the per-stage residual count is not hardcoded."""
    outer, rest = tail.split(".", 1)
    if rest.startswith("avg_shortcut."):
        # AvgDown3D / DupUp3D hold no parameters.
        raise ValueError(f"unexpected parameterized avg_shortcut key: {tail}")
    inner_stack, inner = rest.split(".", 1)
    assert inner_stack == block_attr, f"unexpected Wan VAE key: {tail}"
    inner_index, inner_tail = inner.split(".", 1)
    if inner_tail.startswith(("resample.", "time_conv.")):
        return f"{outer}.{sampler_attr}.{inner_tail}"
    return f"{outer}.resnets.{inner_index}.{_wan_pth_residual_block_key(inner_tail)}"


def _wan_pth_middle_key(tail: str) -> str:
    # middle Sequential -> mid_block.resnets / .attentions.
    index, rest = tail[len("middle.") :].split(".", 1)
    if index == "1":
        return f"mid_block.attentions.0.{rest}"
    member = {"0": "resnets.0", "2": "resnets.1"}[index]
    return f"mid_block.{member}.{_wan_pth_residual_block_key(rest)}"


def remap_wan_pth_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """The upstream (vae2_2.py WanVAE_) and diffusers (wanvae.py) trees are structurally identical; only key spelling differs."""
    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.startswith("conv1."):
            remapped["quant_conv." + key[len("conv1.") :]] = tensor
            continue
        if key.startswith("conv2."):
            remapped["post_quant_conv." + key[len("conv2.") :]] = tensor
            continue

        side, tail = key.split(".", 1)
        assert side in ("encoder", "decoder"), f"unexpected Wan VAE key: {key}"
        if tail.startswith("conv1."):
            target = "conv_in." + tail[len("conv1.") :]
        elif tail.startswith("head."):
            target = _wan_pth_head_key(tail)
        elif tail.startswith("middle."):
            target = _wan_pth_middle_key(tail)
        elif tail.startswith("downsamples."):
            target = "down_blocks." + _wan_pth_stack_key(
                tail[len("downsamples.") :],
                block_attr="downsamples",
                sampler_attr="downsampler",
            )
        elif tail.startswith("upsamples."):
            target = "up_blocks." + _wan_pth_stack_key(
                tail[len("upsamples.") :],
                block_attr="upsamples",
                sampler_attr="upsampler",
            )
        else:
            raise ValueError(f"unexpected Wan VAE key: {key}")
        remapped[f"{side}.{target}"] = tensor
    return remapped


def load_magi2_video_vae(
    *,
    checkpoint_root: str,
    config: Magi2VideoVAEConfig,
    device: torch.device,
    param_dtype: torch.dtype,
) -> AutoencoderKLWan:
    """``conv1`` is the quant conv, ``conv2`` the post-quant conv (vae2_2.py:868-870)."""
    path = os.path.join(checkpoint_root, _VIDEO_VAE_FILE)
    with set_default_torch_dtype(param_dtype), torch.device("meta"):
        vae = AutoencoderKLWan(config)

    state = remap_wan_pth_state_dict(_torch_load(path))
    state = {
        name: tensor.to(device=device, dtype=param_dtype)
        for name, tensor in state.items()
    }
    _assert_full_coverage(vae, set(state))
    vae.load_state_dict(state, strict=True, assign=True)
    return vae.eval()


def load_magi2_turbo_vae(
    *,
    checkpoint_root: str,
    config: Magi2TurboVAEConfig,
    device: torch.device,
    param_dtype: torch.dtype,
) -> Magi2TurboVAE:
    """Three payload shapes are normalized by ``strip_turbo_vae_state_dict_prefix`` (magi2_core/model/turbo_vaed.py:1038-1061)."""
    path = os.path.join(checkpoint_root, _TURBO_VAE_FILE)
    with set_default_torch_dtype(param_dtype), torch.device("meta"):
        vae = Magi2TurboVAE(config)

    state = strip_turbo_vae_state_dict_prefix(_torch_load(path))
    state = {
        name: tensor.to(device=device, dtype=param_dtype)
        for name, tensor in state.items()
        if not name.startswith(_TURBO_VAE_TRAINING_ONLY_PREFIXES)
    }
    _assert_full_coverage(vae, set(state))
    vae.load_state_dict(state, strict=True, assign=True)
    return vae.eval()


def load_magi2_audio_vae(
    *,
    checkpoint_root: str,
    config: Magi2AudioVAEConfig,
    device: torch.device,
    param_dtype: torch.dtype,
) -> Magi2AudioVAE:
    """Weights sit under ``pretransform.model.`` (magi2_core/pipeline/audio_decoder.py:104-108); the encoder half is dropped."""
    directory = os.path.join(checkpoint_root, _AUDIO_VAE_DIR)
    audio_vae = Magi2AudioVAE(config)

    weight_map = _read_weight_map(directory)
    raw: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, path in weight_map.items():
        by_shard[path].append(name)
    for path, names in by_shard.items():
        with safe_open(path, framework="pt") as handle:
            for name in names:
                raw[name] = handle.get_tensor(name)

    # The module remaps and loads strict, so key drift either way raises there.
    audio_vae.load_weights(raw.items())
    return audio_vae.to(device=device, dtype=param_dtype).eval()


# MAGI-2 trained with each of these split into its own pre-token; Qwen's grouping
# silently shifts token boundaries for every non-Latin prompt.
_CJK_PATTERN = (
    "([\u1100-\u11ff\u2e80-\ua4cf\ua840-\ud7af\uf900-\ufaff"
    "\ufe30-\ufe4f\uff65-\uffdc\U00020000-\U0002ffff])"
)


def _isolate_cjk_characters(tokenizer) -> None:
    from tokenizers import Regex, pre_tokenizers

    backend = tokenizer.backend_tokenizer
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(pattern=Regex(_CJK_PATTERN), behavior="isolated"),
            backend.pre_tokenizer,
        ]
    )


def load_magi2_text_encoder(
    *, checkpoint_root: str, device: torch.device, param_dtype: torch.dtype
):
    """``AutoModel`` here resolves the multimodal wrapper, so the text model is named explicitly (magi2_core/model/qwen35.py:250)."""
    from transformers import AutoTokenizer

    try:
        from transformers import Qwen3_5TextModel
    except ImportError as e:
        raise RuntimeError(
            "MAGI-2's text encoder needs a transformers build providing "
            "Qwen3_5TextModel; AutoModel would silently load a different "
            "module and a different hidden state"
        ) from e

    path = os.path.join(checkpoint_root, _TEXT_ENCODER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="right")
    _isolate_cjk_characters(tokenizer)
    text_encoder = Qwen3_5TextModel.from_pretrained(path, torch_dtype=param_dtype)
    text_encoder = text_encoder.to(device).eval()
    text_encoder.requires_grad_(False)
    return text_encoder, tokenizer


def load_magi2_modules(
    *,
    server_args: ServerArgs,
    pipeline,
    ep_group: dist.ProcessGroup | None,
) -> dict[str, nn.Module | None]:
    """Bypasses ``PipelineComponentLoader``: its diffusers retry (component_loader.py:264-297) masks real bugs, and its subclass registry (:85-91) is process-wide."""
    config: Magi2PipelineConfig = server_args.pipeline_config
    checkpoint_root = pipeline.model_path
    device = get_local_torch_device()
    dit_dtype = PRECISION_TO_TYPE[config.dit_precision]

    # The text encoder runs once per request before either DiT, and as a plain
    # transformers module sglang's layerwise offload cannot hook it. Staged in by
    # Magi2TextEncodingStage.
    text_encoder_device = torch.device("cpu")

    transformer = load_magi2_preview_dit(
        checkpoint_root=checkpoint_root,
        config=config.dit_config,
        ep_group=ep_group,
        device=device,
        param_dtype=dit_dtype,
    )
    transformer_2 = (
        load_magi2_refiner_dit(
            checkpoint_root=checkpoint_root,
            config=config.refiner_dit_config,
            device=device,
            param_dtype=dit_dtype,
        )
        if config.enable_refiner
        else None
    )

    text_encoder, tokenizer = load_magi2_text_encoder(
        checkpoint_root=checkpoint_root,
        device=text_encoder_device,
        param_dtype=PRECISION_TO_TYPE[config.text_encoder_precisions[0]],
    )

    vae = load_magi2_video_vae(
        checkpoint_root=checkpoint_root,
        config=config.vae_config,
        device=device,
        param_dtype=PRECISION_TO_TYPE[config.vae_precision],
    )
    turbo_vae = (
        load_magi2_turbo_vae(
            checkpoint_root=checkpoint_root,
            config=config.turbo_vae_config,
            device=device,
            param_dtype=dit_dtype,
        )
        if config.use_turbo_vae
        else None
    )
    audio_vae = load_magi2_audio_vae(
        checkpoint_root=checkpoint_root,
        config=config.audio_vae_config,
        device=device,
        param_dtype=PRECISION_TO_TYPE[config.audio_vae_precision],
    )

    # transformer_2 and turbo_vae must be present as keys even when disabled: the
    # pipeline reads them through get_module() as eager add_stage_if arguments.
    return {
        "transformer": transformer,
        "transformer_2": transformer_2,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "vae": vae,
        "turbo_vae": turbo_vae,
        "audio_vae": audio_vae,
    }
