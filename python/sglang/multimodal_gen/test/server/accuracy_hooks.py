from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from sglang.multimodal_gen.test.server.accuracy_config import (
    DEFAULT_TIMESTEP,
    I2V_IMAGE_DIM,
    TIMESTEP_NORMALIZATION_FACTOR,
    ComponentType,
)
from sglang.multimodal_gen.test.server.accuracy_utils import (
    extract_output_tensor,
    seed_and_broadcast,
)

Inputs = Dict[str, Any]
BuildInputsFn = Callable[[Any, nn.Module, str, Optional[nn.Module]], Inputs]
PrepareCallFn = Callable[[nn.Module, Inputs], "HookCall"]
NormalizeFn = Callable[[Any], torch.Tensor]

# These are harness defaults for synthetic accuracy inputs.
# They are not checkpoint truth. We use them only when the model config or
# forward signature does not expose a more specific shape or channel count.
DEFAULT_TEXT_SEQ_LEN = 64
DEFAULT_TOKEN_LAYOUT_SIZE = 32
REDUCED_TOKEN_LAYOUT_SIZE = 16
DEFAULT_VIDEO_FRAME_COUNT = 4
DEFAULT_AUDIO_FRAME_COUNT = 16
DEFAULT_IMAGE_TOKEN_COUNT = 257
ALIAS_ROTARY_TEXT_PAD_MULTIPLE = 32
DEFAULT_TRANSFORMER_IN_CHANNELS = 16
DEFAULT_TRANSFORMER_TEXT_CHANNELS = 4096
DEFAULT_TRANSFORMER_POOLED_CHANNELS = 768
DEFAULT_VAE_LATENT_CHANNELS = 16
DEFAULT_VAE_LATENT_SPATIAL_SIZE = 32
LARGE_CHANNEL_LAYOUT_THRESHOLD = 128


@dataclass(frozen=True)
class TransformerHookCompat:
    normalize_reference_timestep: bool = False
    negate_reference_output: bool = False
    omit_reference_guidance: bool = False
    use_2d_hidden_states: bool = False


def _resolve_transformer_hook_compat(case: Any) -> TransformerHookCompat:
    model_path = case.server_args.model_path.lower()
    if "z-image" in model_path:
        return TransformerHookCompat(
            normalize_reference_timestep=True,
            negate_reference_output=True,
        )
    if "qwen" in model_path:
        return TransformerHookCompat(
            normalize_reference_timestep=True,
            omit_reference_guidance=True,
        )
    if "sana" in model_path:
        return TransformerHookCompat(
            omit_reference_guidance=True,
            use_2d_hidden_states=True,
        )
    if "flux" in model_path:
        return TransformerHookCompat(normalize_reference_timestep=True)
    return TransformerHookCompat()


@dataclass
class HookCall:
    module: nn.Module
    args: tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    negate_output: bool = False


@dataclass(frozen=True)
class NativeHookProfile:
    build_inputs: BuildInputsFn
    prepare_sglang_call: PrepareCallFn
    prepare_reference_call: PrepareCallFn
    normalize_sglang_output: NormalizeFn = extract_output_tensor
    normalize_reference_output: NormalizeFn = extract_output_tensor


class _DeterministicRNG:
    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def randn(
        self, shape: tuple[int, ...], device: str, dtype: torch.dtype
    ) -> torch.Tensor:
        torch.manual_seed(self._seed)
        tensor = torch.randn(shape, device="cpu", dtype=dtype).to(device)
        seed_and_broadcast(self._seed, tensor)
        self._seed += 1
        return tensor


def _resolve_nested_attr(obj: Any, path: str) -> Any:
    current = obj
    for name in path.split("."):
        if current is None or not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def _read_config_value(model: nn.Module, keys: list[str], default: int) -> int:
    config = getattr(model, "config", None)
    for key in keys:
        for root in (model, config):
            value = _resolve_nested_attr(root, key) if root is not None else None
            if isinstance(value, int) and value > 0:
                return value
    return default


def _forward_parameter_names(module: nn.Module) -> set[str]:
    return set(inspect.signature(module.forward).parameters.keys())


def _infer_transformer_layout(param_names: set[str]) -> str:
    if "img_shapes" in param_names or "txt_seq_lens" in param_names:
        return "token_shapes"
    if "img_ids" in param_names or "txt_ids" in param_names:
        return "token_ids"
    if "x" in param_names or "cap_feats" in param_names:
        return "alias"
    return "video"


def _build_position_ids(
    height: int, width: int, dims: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    img_len = height * width
    txt_len = DEFAULT_TEXT_SEQ_LEN
    if dims == 4:
        img_ids = torch.zeros(img_len, 4, device=device, dtype=torch.bfloat16)
        img_ids[:, 1] = torch.arange(height).repeat_interleave(width)
        img_ids[:, 2] = torch.arange(width).repeat(height)
        txt_ids = torch.zeros(txt_len, 4, device=device, dtype=torch.bfloat16)
    else:
        img_ids = torch.zeros(img_len, 3, device=device, dtype=torch.bfloat16)
        img_ids[:, 0] = torch.arange(height).repeat_interleave(width)
        img_ids[:, 1] = torch.arange(width).repeat(height)
        txt_ids = torch.zeros(txt_len, 3, device=device, dtype=torch.bfloat16)
    return img_ids, txt_ids


def _build_alias_rotary_freqs(
    model: nn.Module, device: str, height: int, width: int
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    cap_len = DEFAULT_TEXT_SEQ_LEN
    cap_pad_len = (-cap_len) % ALIAS_ROTARY_TEXT_PAD_MULTIPLE
    cap_ids = (
        torch.stack(
            torch.meshgrid(
                torch.arange(cap_len + cap_pad_len),
                torch.arange(1),
                torch.arange(1),
                indexing="ij",
            ),
            dim=-1,
        )
        .flatten(0, 2)
        .to(device)
    )
    img_ids = (
        torch.stack(
            torch.meshgrid(
                torch.arange(1),
                torch.arange(height // 2),
                torch.arange(width // 2),
                indexing="ij",
            ),
            dim=-1,
        )
        .flatten(0, 2)
        .to(device)
    )
    cos_cap, sin_cap = model.rotary_emb(cap_ids)
    cos_img, sin_img = model.rotary_emb(img_ids)
    return ((cos_cap, sin_cap), (cos_img, sin_img))


def _supports_image_conditioning(module: nn.Module) -> bool:
    image_embedder = _resolve_nested_attr(module, "condition_embedder.image_embedder")
    if image_embedder is not None:
        return True
    image_dim = _read_config_value(
        module, ["arch_config.image_dim", "image_dim"], default=0
    )
    return image_dim > 0


def _build_transformer_hook_inputs(
    case: Any, model: nn.Module, device: str, ref_model: Optional[nn.Module] = None
) -> Inputs:
    """Build one synthetic input bundle that both transformer variants can consume."""
    compat = _resolve_transformer_hook_compat(case)
    param_names = _forward_parameter_names(model)
    if ref_model is not None:
        # The input bundle has to satisfy both call signatures.
        param_names.update(_forward_parameter_names(ref_model))

    rng = _DeterministicRNG()
    layout = _infer_transformer_layout(param_names)
    requires_audio_stream_inputs = (
        "audio_hidden_states" in param_names
        and "audio_encoder_hidden_states" in param_names
    )
    requires_audio_video_shape_inputs = requires_audio_stream_inputs and all(
        key in param_names for key in ("num_frames", "height", "width")
    )
    in_channels = _read_config_value(
        model,
        [
            "arch_config.in_channels",
            "in_channels",
            "transformer_config.in_channels",
        ],
        default=DEFAULT_TRANSFORMER_IN_CHANNELS,
    )
    text_channels = _read_config_value(
        model,
        [
            "text_states_dim",
            "arch_config.cap_feat_dim",
            "cap_feat_dim",
            "caption_channels",
            "arch_config.text_dim",
            "text_dim",
            "arch_config.text_embed_dim",
            "text_embed_dim",
            "arch_config.joint_attention_dim",
            "joint_attention_dim",
            "cross_attention_dim",
            "hidden_size",
            "dim",
        ],
        default=DEFAULT_TRANSFORMER_TEXT_CHANNELS,
    )
    audio_in_channels = _read_config_value(
        model,
        [
            "arch_config.audio_in_channels",
            "audio_in_channels",
            "arch_config.audio_out_channels",
            "audio_out_channels",
        ],
        default=in_channels,
    )
    pooled_channels = _read_config_value(
        model,
        [
            "text_states_dim_2",
            "arch_config.pooled_projection_dim",
            "pooled_projection_dim",
            "pooled_embed_dim",
            "text_embed_dim",
            "projection_dim",
        ],
        default=DEFAULT_TRANSFORMER_POOLED_CHANNELS,
    )
    image_channels = _read_config_value(
        model,
        ["arch_config.image_dim", "image_dim", "cross_attention_dim"],
        default=I2V_IMAGE_DIM,
    )

    if requires_audio_video_shape_inputs:
        patch_size = getattr(model, "patch_size", None)
        if not (
            isinstance(patch_size, tuple)
            and len(patch_size) == 3
            and all(isinstance(dim, int) and dim > 0 for dim in patch_size)
        ):
            patch_size = (1, 2, 2)
        patch_t, patch_h, patch_w = patch_size
        num_frames = DEFAULT_VIDEO_FRAME_COUNT * patch_t
        height = REDUCED_TOKEN_LAYOUT_SIZE * patch_h
        width = REDUCED_TOKEN_LAYOUT_SIZE * patch_w
        seq_len = (num_frames // patch_t) * (height // patch_h) * (width // patch_w)
        hidden_states = rng.randn((1, seq_len, in_channels), device, torch.bfloat16)
    elif layout == "token_shapes":
        height, width = DEFAULT_TOKEN_LAYOUT_SIZE, DEFAULT_TOKEN_LAYOUT_SIZE
        seq_len = (height // 2) * (width // 2)
        hidden_states = rng.randn((1, seq_len, in_channels), device, torch.bfloat16)
    elif layout == "token_ids":
        height, width = REDUCED_TOKEN_LAYOUT_SIZE, REDUCED_TOKEN_LAYOUT_SIZE
        seq_len = height * width
        hidden_states = rng.randn((1, seq_len, in_channels), device, torch.bfloat16)
    elif layout == "alias":
        height, width = DEFAULT_TOKEN_LAYOUT_SIZE, DEFAULT_TOKEN_LAYOUT_SIZE
        hidden_states = rng.randn(
            (1, in_channels, 1, height, width), device, torch.bfloat16
        )
    elif compat.use_2d_hidden_states:
        spatial_size = (
            REDUCED_TOKEN_LAYOUT_SIZE
            if "encoder_attention_mask" in param_names
            or "encoder_hidden_states_mask" in param_names
            else DEFAULT_TOKEN_LAYOUT_SIZE
        )
        height, width = spatial_size, spatial_size
        hidden_states = rng.randn(
            (1, in_channels, height, width),
            device,
            torch.bfloat16,
        )
    else:
        spatial_size = (
            REDUCED_TOKEN_LAYOUT_SIZE
            if "encoder_attention_mask" in param_names
            or "encoder_hidden_states_mask" in param_names
            else DEFAULT_TOKEN_LAYOUT_SIZE
        )
        height, width = spatial_size, spatial_size
        hidden_states = rng.randn(
            (1, in_channels, DEFAULT_VIDEO_FRAME_COUNT, height, width),
            device,
            torch.bfloat16,
        )

    inputs: Inputs = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": rng.randn(
            (1, DEFAULT_TEXT_SEQ_LEN, text_channels), device, torch.bfloat16
        ),
        "timestep": torch.tensor(
            [DEFAULT_TIMESTEP], device=device, dtype=torch.bfloat16
        ),
        "guidance": torch.tensor([1.0], device=device, dtype=torch.bfloat16),
    }

    if requires_audio_stream_inputs:
        inputs["audio_hidden_states"] = rng.randn(
            (1, DEFAULT_AUDIO_FRAME_COUNT, audio_in_channels),
            device,
            torch.bfloat16,
        )
        inputs["audio_encoder_hidden_states"] = rng.randn(
            (1, DEFAULT_TEXT_SEQ_LEN, text_channels),
            device,
            torch.bfloat16,
        )
        inputs["audio_timestep"] = inputs["timestep"].clone()
        inputs["audio_num_frames"] = DEFAULT_AUDIO_FRAME_COUNT
        if requires_audio_video_shape_inputs:
            inputs["num_frames"] = num_frames
            inputs["height"] = height
            inputs["width"] = width

    if "pooled_projections" in param_names:
        inputs["pooled_projections"] = rng.randn(
            (1, pooled_channels), device, torch.bfloat16
        )
    if (
        "encoder_attention_mask" in param_names
        or "encoder_hidden_states_mask" in param_names
    ):
        attention_mask = torch.ones(
            1, DEFAULT_TEXT_SEQ_LEN, device=device, dtype=torch.bool
        )
        inputs["encoder_attention_mask"] = attention_mask
        inputs["encoder_hidden_states_mask"] = attention_mask
    if "audio_encoder_attention_mask" in param_names:
        inputs["audio_encoder_attention_mask"] = torch.ones(
            1, DEFAULT_TEXT_SEQ_LEN, device=device, dtype=torch.bool
        )
    if "encoder_hidden_states_image" in param_names and _supports_image_conditioning(
        model
    ):
        inputs["encoder_hidden_states_image"] = rng.randn(
            (1, DEFAULT_IMAGE_TOKEN_COUNT, image_channels), device, torch.bfloat16
        )
    if "additional_t_cond" in param_names:
        inputs["additional_t_cond"] = torch.zeros((1,), device=device, dtype=torch.long)
    if "img_shapes" in param_names:
        inputs["img_shapes"] = [[(1, height // 2, width // 2)]]
    if "txt_seq_lens" in param_names:
        inputs["txt_seq_lens"] = [DEFAULT_TEXT_SEQ_LEN]
    if "img_ids" in param_names or "txt_ids" in param_names:
        id_dims = 4 if in_channels >= LARGE_CHANNEL_LAYOUT_THRESHOLD else 3
        img_ids, txt_ids = _build_position_ids(height, width, id_dims, device)
        inputs["img_ids"] = img_ids
        inputs["txt_ids"] = txt_ids

    if "freqs_cis" in param_names and hasattr(model, "rotary_emb"):
        if "img_shapes" in inputs and "txt_seq_lens" in inputs:
            img_freqs, txt_freqs = model.rotary_emb(
                inputs["img_shapes"],
                inputs["txt_seq_lens"],
                device=hidden_states.device,
            )
            if torch.is_complex(img_freqs) and torch.is_complex(txt_freqs):
                inputs["freqs_cis"] = (
                    torch.cat([img_freqs.real.float(), img_freqs.imag.float()], dim=-1),
                    torch.cat([txt_freqs.real.float(), txt_freqs.imag.float()], dim=-1),
                )
            else:
                inputs["freqs_cis"] = (img_freqs, txt_freqs)
        elif "img_ids" in inputs and "txt_ids" in inputs:
            ids = torch.cat([inputs["txt_ids"], inputs["img_ids"]], dim=0)
            inputs["freqs_cis"] = model.rotary_emb(ids)
        elif inputs["hidden_states"].ndim == 5:
            inputs["freqs_cis"] = _build_alias_rotary_freqs(
                model, device, height, width
            )

    inputs["hook_compat"] = compat
    return inputs


def _get_transformer_hook_compat(inputs: Inputs) -> TransformerHookCompat:
    compat = inputs.get("hook_compat")
    assert isinstance(compat, TransformerHookCompat)
    return compat


def _supports_guidance_embedding(module: nn.Module) -> bool:
    time_text_embed = getattr(module, "time_text_embed", None)
    if time_text_embed is None:
        return True

    parameters = list(inspect.signature(time_text_embed.forward).parameters.values())

    if any(param.kind is inspect.Parameter.VAR_POSITIONAL for param in parameters):
        return True

    accepted_args = [
        param
        for param in parameters
        if param.name != "self"
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    return len(accepted_args) >= 3


def _prepare_transformer_hook_call(
    module: nn.Module, inputs: Inputs, side: str
) -> HookCall:
    param_names = _forward_parameter_names(module)
    signature = inspect.signature(module.forward)
    compat = _get_transformer_hook_compat(inputs)
    kwargs: Dict[str, Any] = {}
    negate_output = side == "reference" and compat.negate_reference_output

    if "hidden_states" in param_names:
        kwargs["hidden_states"] = inputs["hidden_states"]
    if "x" in param_names:
        kwargs["x"] = [inputs["hidden_states"].squeeze(0)]
    if "encoder_hidden_states" in param_names:
        encoder_value: Any = inputs["encoder_hidden_states"]
        if (
            side == "sglang"
            and "pooled_projections" in inputs
            and "pooled_projections" not in param_names
            and "encoder_attention_mask" not in param_names
        ):
            encoder_value = [
                inputs["encoder_hidden_states"],
                inputs["pooled_projections"],
            ]
        kwargs["encoder_hidden_states"] = encoder_value
    if "cap_feats" in param_names:
        kwargs["cap_feats"] = [inputs["encoder_hidden_states"].squeeze(0)]

    if "timestep" in param_names:
        timestep = inputs["timestep"]
        if side == "reference" and compat.normalize_reference_timestep:
            timestep = timestep / TIMESTEP_NORMALIZATION_FACTOR
        kwargs["timestep"] = timestep
    if "t" in param_names:
        timestep = inputs["timestep"]
        if side == "reference" and compat.normalize_reference_timestep:
            timestep = timestep / TIMESTEP_NORMALIZATION_FACTOR
        kwargs["t"] = timestep

    if "guidance" in param_names and "guidance" in inputs:
        if side == "reference" and compat.omit_reference_guidance:
            pass
        else:
            skip_guidance_for_image_context = (
                "encoder_hidden_states_image" in param_names
                and "img_ids" not in param_names
                and "img_shapes" not in param_names
            )
            supports_guidance_embedding = _supports_guidance_embedding(module)
            requires_guidance_arg = (
                signature.parameters["guidance"].default is inspect._empty
            )
            should_include_guidance = (
                not skip_guidance_for_image_context and supports_guidance_embedding
            )
            if should_include_guidance or requires_guidance_arg:
                guidance_value = inputs["guidance"]
                if side == "sglang":
                    guidance_value = guidance_value * TIMESTEP_NORMALIZATION_FACTOR
                kwargs["guidance"] = guidance_value

    if (
        "encoder_hidden_states_image" in param_names
        and "encoder_hidden_states_image" in inputs
    ):
        value = inputs["encoder_hidden_states_image"]
        kwargs["encoder_hidden_states_image"] = [value] if side == "sglang" else value

    for key in (
        "pooled_projections",
        "img_ids",
        "txt_ids",
        "img_shapes",
        "txt_seq_lens",
        "freqs_cis",
        "additional_t_cond",
        "audio_hidden_states",
        "audio_encoder_hidden_states",
        "audio_timestep",
        "encoder_attention_mask",
        "encoder_hidden_states_mask",
        "audio_encoder_attention_mask",
        "num_frames",
        "height",
        "width",
        "audio_num_frames",
    ):
        if key in param_names and key in inputs:
            kwargs[key] = inputs[key]

    if "return_dict" in param_names:
        kwargs["return_dict"] = True

    return HookCall(module=module, kwargs=kwargs, negate_output=negate_output)


def _prepare_transformer_sglang_call(module: nn.Module, inputs: Inputs) -> HookCall:
    return _prepare_transformer_hook_call(module, inputs, side="sglang")


def _prepare_transformer_reference_call(module: nn.Module, inputs: Inputs) -> HookCall:
    return _prepare_transformer_hook_call(module, inputs, side="reference")


def _normalize_transformer_reference_output(output: Any) -> torch.Tensor:
    sample = getattr(output, "sample", None)
    if (
        isinstance(sample, (list, tuple))
        and sample
        and all(isinstance(item, torch.Tensor) for item in sample)
    ):
        return torch.stack(list(sample), dim=0)
    return extract_output_tensor(output)


class _VAEDecodeModule(nn.Module):
    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if (
            any(
                isinstance(module, (nn.Conv3d, nn.ConvTranspose3d))
                for module in self.vae.modules()
            )
            and z.ndim == 4
        ):
            z = z.unsqueeze(2)
        output = self.vae.decode(z)
        tensor = output.sample if hasattr(output, "sample") else output
        if isinstance(tensor, (list, tuple)):
            tensor = tensor[0]
        return tensor.squeeze(2) if tensor.ndim == 5 else tensor


def _infer_vae_latent_channels(model: nn.Module) -> int:
    for path in ("post_quant_conv.in_channels", "post_quant_conv.conv.in_channels"):
        value = _resolve_nested_attr(model, path)
        if isinstance(value, int) and value > 0:
            return value
    return _read_config_value(
        model,
        [
            "z_dim",
            "arch_config.z_dim",
            "latent_channels",
            "arch_config.latent_channels",
            "num_channels_latents",
            "arch_config.num_channels_latents",
            "latent_dim",
            "z_channels",
            "arch_config.z_channels",
        ],
        default=DEFAULT_VAE_LATENT_CHANNELS,
    )


def _build_vae_hook_inputs(
    case: Any, model: nn.Module, device: str, ref_model: Optional[nn.Module] = None
) -> Inputs:
    del case, ref_model
    latent_channels = _infer_vae_latent_channels(model)
    rng = _DeterministicRNG()
    return {
        "z": rng.randn(
            (
                1,
                latent_channels,
                DEFAULT_VAE_LATENT_SPATIAL_SIZE,
                DEFAULT_VAE_LATENT_SPATIAL_SIZE,
            ),
            device,
            torch.bfloat16,
        )
    }


def _prepare_vae_decode_call(module: nn.Module, inputs: Inputs) -> HookCall:
    return HookCall(module=_VAEDecodeModule(module), args=(inputs["z"],))


TRANSFORMER_NATIVE_PROFILE = NativeHookProfile(
    build_inputs=_build_transformer_hook_inputs,
    prepare_sglang_call=_prepare_transformer_sglang_call,
    prepare_reference_call=_prepare_transformer_reference_call,
    normalize_reference_output=_normalize_transformer_reference_output,
)

VAE_NATIVE_PROFILE = NativeHookProfile(
    build_inputs=_build_vae_hook_inputs,
    prepare_sglang_call=_prepare_vae_decode_call,
    prepare_reference_call=_prepare_vae_decode_call,
)


def resolve_component_native_profile(component: ComponentType) -> NativeHookProfile:
    if component == ComponentType.TRANSFORMER:
        return TRANSFORMER_NATIVE_PROFILE
    if component == ComponentType.VAE:
        return VAE_NATIVE_PROFILE
    raise KeyError(f"Unsupported native accuracy component: {component.value}")
