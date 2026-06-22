# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weight-layout builders for OmniDreams native VAE kernels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    orig = getattr(module, "_orig_mod", None)
    return orig if isinstance(orig, torch.nn.Module) else module


def _module_device(module: torch.nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    raise ValueError(f"{module.__class__.__name__} has no parameters or buffers")


def _module_attr(obj: object, name: str) -> torch.nn.Module:
    value = getattr(obj, name)
    if not isinstance(value, torch.nn.Module):
        raise TypeError(f"{obj.__class__.__name__}.{name} must be a torch module")
    return value


def _conv3d_attr(obj: object, name: str) -> torch.nn.Conv3d:
    value = getattr(obj, name)
    if not isinstance(value, torch.nn.Conv3d):
        raise TypeError(f"{obj.__class__.__name__}.{name} must be a Conv3d module")
    return value


def _tensor_attr(obj: object, name: str) -> torch.Tensor:
    value = getattr(obj, name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{obj.__class__.__name__}.{name} must be a tensor")
    return value


def _sequence_attr(obj: object, name: str) -> Any:
    value = getattr(obj, name)
    if not hasattr(value, "__getitem__") or not hasattr(value, "__len__"):
        raise TypeError(f"{obj.__class__.__name__}.{name} must be an indexed sequence")
    return value


def _conv_bias(conv: torch.nn.Conv2d | torch.nn.Conv3d) -> torch.Tensor:
    bias = conv.bias
    if bias is None:
        raise ValueError(f"{conv.__class__.__name__} must have a bias tensor")
    return bias


def _pad_1d(tensor: torch.Tensor, size: int, *, device: torch.device) -> torch.Tensor:
    src = tensor.detach().to(device=device, dtype=torch.float16).flatten().contiguous()
    if src.numel() == size:
        return src
    out = torch.zeros((size,), device=device, dtype=torch.float16)
    out[: src.numel()].copy_(src)
    return out.contiguous()


def _lightvae_gamma(
    norm: torch.nn.Module,
    real_c: int,
    pad_c: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    gamma = (
        _tensor_attr(norm, "gamma")
        .detach()
        .reshape(real_c)
        .to(
            device=device,
            dtype=torch.float16,
        )
    )
    if pad_c == real_c:
        return gamma.contiguous()
    out = torch.zeros((pad_c,), device=device, dtype=torch.float16)
    out[:real_c].copy_(gamma)
    return out.contiguous()


def _lightvae_causal3_weight(
    conv: torch.nn.Conv3d,
    *,
    in_real: int,
    in_pad: int,
    out_real: int,
    out_pad: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w = conv.weight.detach().to(device=device, dtype=torch.float16).contiguous()
    b = _conv_bias(conv).detach().to(device=device, dtype=torch.float16).contiguous()
    kt, kh, kw = int(w.shape[2]), int(w.shape[3]), int(w.shape[4])
    if (kt, kh, kw) != (3, 3, 3):
        raise ValueError(f"expected LightVAE 3x3x3 weight, got {tuple(w.shape)}")
    packed = torch.zeros(
        (out_pad, 3, 3, 3 * in_pad), device=device, dtype=torch.float16
    )
    packed3d = torch.zeros(
        (out_pad, 3, 3, 3, in_pad), device=device, dtype=torch.float16
    )
    for dt in range(3):
        wt = w[:out_real, :in_real, dt].permute(0, 2, 3, 1).contiguous()
        packed[:out_real, :, :, dt * in_pad : dt * in_pad + in_real].copy_(wt)
        packed3d[:out_real, dt, :, :, :in_real].copy_(wt)
    return (
        packed.contiguous(),
        _pad_1d(b[:out_real], out_pad, device=device),
        packed3d.contiguous(),
    )


def _lightvae_spatial3_weight(
    conv: torch.nn.Conv2d,
    *,
    in_real: int,
    in_pad: int,
    out_real: int,
    out_pad: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    w = conv.weight.detach().to(device=device, dtype=torch.float16).contiguous()
    b = _conv_bias(conv).detach().to(device=device, dtype=torch.float16).contiguous()
    packed = torch.zeros((out_pad, 3, 3, in_pad), device=device, dtype=torch.float16)
    packed[:out_real, :, :, :in_real].copy_(w[:out_real, :in_real].permute(0, 2, 3, 1))
    return packed.contiguous(), _pad_1d(b[:out_real], out_pad, device=device)


def _lightvae_temporal3_weight(
    conv: torch.nn.Conv3d,
    *,
    channels: int,
    real_channels: int | None = None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    w = conv.weight.detach().to(device=device, dtype=torch.float16).contiguous()
    b = _conv_bias(conv).detach().to(device=device, dtype=torch.float16).contiguous()
    real = int(real_channels if real_channels is not None else channels)
    if real > channels:
        raise ValueError(
            f"real_channels {real} cannot exceed padded channels {channels}"
        )
    packed = torch.zeros((channels, 3 * channels), device=device, dtype=torch.float16)
    for dt in range(3):
        packed[:real, dt * channels : dt * channels + real].copy_(
            w[:real, :real, dt, 0, 0]
        )
    return packed.contiguous(), _pad_1d(b[:real], channels, device=device)


def _lightvae_conv1_weight(
    conv: torch.nn.Module,
    *,
    in_real: int,
    in_pad: int,
    out_real: int,
    out_pad: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    w = _tensor_attr(conv, "weight").detach().to(device=device, dtype=torch.float16)
    w = w.contiguous()
    b = _tensor_attr(conv, "bias").detach().to(device=device, dtype=torch.float16)
    b = b.contiguous()
    if w.dim() == 5:
        w = w[:, :, 0, 0, 0]
    elif w.dim() == 4:
        w = w[:, :, 0, 0]
    packed = torch.zeros((out_pad, in_pad), device=device, dtype=torch.float16)
    packed[:out_real, :in_real].copy_(w[:out_real, :in_real])
    return packed.contiguous(), _pad_1d(b[:out_real], out_pad, device=device)


def _lightvae_resblock_state(
    block: torch.nn.Module,
    *,
    in_real: int,
    in_pad: int,
    out_real: int,
    out_pad: int,
    device: torch.device,
) -> dict[str, Any]:
    residual = _sequence_attr(block, "residual")
    conv1_w, conv1_b, conv1_w3d = _lightvae_causal3_weight(
        residual[2],
        in_real=in_real,
        in_pad=in_pad,
        out_real=out_real,
        out_pad=out_pad,
        device=device,
    )
    conv2_w, conv2_b, conv2_w3d = _lightvae_causal3_weight(
        residual[6],
        in_real=out_real,
        in_pad=out_pad,
        out_real=out_real,
        out_pad=out_pad,
        device=device,
    )
    shortcut_w = shortcut_b = None
    shortcut = _module_attr(block, "shortcut")
    if not isinstance(shortcut, torch.nn.Identity):
        shortcut_w, shortcut_b = _lightvae_conv1_weight(
            shortcut,
            in_real=in_real,
            in_pad=in_pad,
            out_real=out_real,
            out_pad=out_pad,
            device=device,
        )
    return {
        "in_real": in_real,
        "in_pad": in_pad,
        "out_real": out_real,
        "out_pad": out_pad,
        "norm1_gamma": _lightvae_gamma(residual[0], in_real, in_pad, device=device),
        "conv1_w": conv1_w,
        "conv1_w3d": conv1_w3d,
        "conv1_b": conv1_b,
        "norm2_gamma": _lightvae_gamma(residual[3], out_real, out_pad, device=device),
        "conv2_w": conv2_w,
        "conv2_w3d": conv2_w3d,
        "conv2_b": conv2_b,
        "shortcut_w": shortcut_w,
        "shortcut_b": shortcut_b,
    }


def _require_lightvae_layout(model: torch.nn.Module) -> None:
    if not hasattr(model, "encoder") or not hasattr(model, "conv1"):
        raise TypeError("native LightVAE encoder requires a WanVAE encoder model")
    enc = _unwrap_module(_module_attr(model, "encoder"))
    downs = _sequence_attr(enc, "downsamples")
    if len(downs) != 11:
        raise ValueError(
            "native LightVAE encoder supports the pruned Wan LightVAE layout only "
            f"(expected 11 downsample blocks, got {len(downs)})"
        )
    conv1 = _module_attr(enc, "conv1")
    conv1_out_channels = getattr(conv1, "out_channels", None)
    if conv1_out_channels != 24:
        raise ValueError(
            "native LightVAE encoder supports the lightvae checkpoint only "
            f"(encoder.conv1.out_channels={conv1_out_channels})"
        )


_LIGHTVAE_FP8_STAGED_STATE_CACHE: dict[tuple[int, int, int, str], dict[str, Any]] = {}


def load_lightvae_fp8_state(path: str) -> dict[str, torch.Tensor]:
    """Load a LightVAE FP8 calibration state from a local torch checkpoint."""

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, Mapping) and "fp8_state" in obj:
        obj = obj["fp8_state"]
    if not isinstance(obj, Mapping):
        raise TypeError(
            "native LightVAE fp8 state must be a mapping or contain an 'fp8_state' mapping"
        )

    state: dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        if isinstance(value, torch.Tensor):
            state[str(key)] = value.detach()
    if not state:
        raise ValueError("native LightVAE fp8 state did not contain tensor entries")
    return state


def _ds_key(idx: int, suffix: str) -> str:
    return f"encoder.downsamples.{idx}{suffix}"


def _mid_key(idx: int, suffix: str) -> str:
    return f"encoder.middle.{idx}{suffix}"


def _pad_scale_tensor(
    scale: torch.Tensor,
    real_channels: int,
    padded_channels: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    s = scale.detach().to(device=device, dtype=torch.float16).flatten().contiguous()
    if s.numel() in (1, padded_channels):
        return s
    if s.numel() != real_channels:
        raise ValueError(
            f"scale has {s.numel()} channels, expected 1, {real_channels}, or {padded_channels}"
        )
    out = torch.ones((padded_channels,), device=device, dtype=torch.float16)
    out[:real_channels].copy_(s)
    return out.contiguous()


def _repeat_scale_tensor(scale: torch.Tensor, copies: int) -> torch.Tensor:
    if scale.numel() == 1 or copies == 1:
        return scale.contiguous()
    return scale.repeat(copies).contiguous()


def _scale_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return (
        (numerator.float() / denominator.float().clamp_min(1.0e-12))
        .to(dtype=torch.float16)
        .contiguous()
    )


def _bias_over_scale(
    bias: torch.Tensor | None,
    output_scale: torch.Tensor,
) -> torch.Tensor | None:
    if bias is None:
        return None
    if output_scale.numel() == 1:
        return (
            (bias.float() / output_scale.flatten()[0].float().clamp_min(1.0e-12))
            .to(dtype=torch.float16)
            .contiguous()
        )
    if bias.numel() != output_scale.numel():
        raise ValueError(
            f"bias has {bias.numel()} values, expected {output_scale.numel()} for scaled FP8 epilogue"
        )
    return _scale_ratio(bias, output_scale)


def _scale_prefix(scale: torch.Tensor, real_channels: int) -> torch.Tensor:
    if scale.numel() == 1 or scale.numel() == real_channels:
        return scale.contiguous()
    if scale.numel() < real_channels:
        raise ValueError(
            f"scale has {scale.numel()} values, expected at least {real_channels}"
        )
    return scale[:real_channels].contiguous()


def _scale_prefix_float(scale: torch.Tensor, real_channels: int) -> torch.Tensor:
    return _scale_prefix(scale, real_channels).to(dtype=torch.float32).contiguous()


def _state_scale_any(
    fp8_state: Mapping[str, torch.Tensor],
    keys: tuple[str, ...],
    real_channels: int,
    padded_channels: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    for key in keys:
        scale = fp8_state.get(key)
        if scale is not None:
            return _pad_scale_tensor(
                scale, real_channels, padded_channels, device=device
            )
    raise KeyError(f"LightVAE fp8 state missing scale. Tried: {', '.join(keys)}")


def _activation_scale_any(
    fp8_state: Mapping[str, torch.Tensor],
    keys: tuple[str, ...],
    real_channels: int,
    padded_channels: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    return _state_scale_any(
        fp8_state, keys, real_channels, padded_channels, device=device
    )


def _activation_scale_scalar_any(
    fp8_state: Mapping[str, torch.Tensor],
    keys: tuple[str, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    for key in keys:
        scale = fp8_state.get(key)
        if scale is not None:
            return (
                scale.detach()
                .to(device=device, dtype=torch.float32)
                .flatten()
                .amax()
                .reshape(1)
                .to(torch.float16)
                .contiguous()
            )
    raise KeyError(
        f"LightVAE fp8 state missing scalar activation_scale. Tried: {', '.join(keys)}"
    )


def _float_scalar(scale: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    return (
        scale.detach()
        .to(device=device, dtype=torch.float32)
        .flatten()
        .amax()
        .reshape(1)
        .contiguous()
    )


def _inverse_float_scalar(scale: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    return (1.0 / _float_scalar(scale, device=device).clamp(min=1.0e-12)).contiguous()


def _weight_scale_any(
    fp8_state: Mapping[str, torch.Tensor],
    key: str,
    real_channels: int,
    padded_channels: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    return _state_scale_any(
        fp8_state, (key,), real_channels, padded_channels, device=device
    )


def _lightvae_fp8_prepare_krsc(
    native_ext: Any,
    weight_krsc: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    return native_ext.lightvae_fp8_prepare_conv2d_weight_krsc(
        weight_krsc.contiguous(),
        input_scale.contiguous(),
        output_scale.contiguous(),
    )


def _lightvae_fp8_prepare_conv1(
    native_ext: Any,
    weight_kc: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    return _lightvae_fp8_prepare_krsc(
        native_ext,
        weight_kc.reshape(weight_kc.shape[0], 1, 1, weight_kc.shape[1]).contiguous(),
        input_scale,
        output_scale,
    )


def _lightvae_fp8_causal_weight(
    native_ext: Any,
    conv: torch.nn.Conv3d,
    *,
    in_real: int,
    in_pad: int,
    out_real: int,
    out_pad: int,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    warp_mma_scaled_epilogue: bool,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    weight, bias, _weight3d = _lightvae_causal3_weight(
        conv,
        in_real=in_real,
        in_pad=in_pad,
        out_real=out_real,
        out_pad=out_pad,
        device=device,
    )
    if warp_mma_scaled_epilogue:
        return (
            _lightvae_fp8_prepare_krsc(
                native_ext,
                weight,
                _repeat_scale_tensor(input_scale, 3),
                weight_scale,
            ),
            bias,
            weight_scale,
            _scale_ratio(weight_scale, output_scale),
            _bias_over_scale(bias, output_scale),
        )
    return (
        _lightvae_fp8_prepare_krsc(
            native_ext,
            weight,
            _repeat_scale_tensor(input_scale, 3),
            output_scale,
        ),
        bias,
        None,
        None,
        None,
    )


def _lightvae_fp8_spatial_weight(
    native_ext: Any,
    conv: torch.nn.Conv2d,
    *,
    in_real: int,
    in_pad: int,
    out_real: int,
    out_pad: int,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    warp_mma_scaled_epilogue: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    weight, bias = _lightvae_spatial3_weight(
        conv,
        in_real=in_real,
        in_pad=in_pad,
        out_real=out_real,
        out_pad=out_pad,
        device=device,
    )
    if warp_mma_scaled_epilogue:
        return (
            _lightvae_fp8_prepare_krsc(native_ext, weight, input_scale, weight_scale),
            bias,
            _scale_ratio(weight_scale, output_scale),
            _bias_over_scale(bias, output_scale),
        )
    return (
        _lightvae_fp8_prepare_krsc(native_ext, weight, input_scale, output_scale),
        bias,
        None,
        None,
    )


def _lightvae_fp8_temporal_weight(
    native_ext: Any,
    conv: torch.nn.Conv3d,
    *,
    channels: int,
    real_channels: int | None = None,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight, bias = _lightvae_temporal3_weight(
        conv,
        channels=channels,
        real_channels=real_channels,
        device=device,
    )
    _ = weight_scale
    return (
        _lightvae_fp8_prepare_conv1(
            native_ext,
            weight,
            _repeat_scale_tensor(input_scale, 3),
            output_scale,
        ),
        bias,
    )


def _lightvae_fp8_conv1_weight(
    native_ext: Any,
    conv: torch.nn.Module,
    *,
    in_real: int,
    in_pad: int,
    out_real: int,
    out_pad: int,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight, bias = _lightvae_conv1_weight(
        conv,
        in_real=in_real,
        in_pad=in_pad,
        out_real=out_real,
        out_pad=out_pad,
        device=device,
    )
    _ = weight_scale
    return _lightvae_fp8_prepare_conv1(
        native_ext, weight, input_scale, output_scale
    ), bias


def _build_lightvae_fp8_resblock_state(
    native_ext: Any,
    block: torch.nn.Module,
    *,
    path: str,
    input_scale: torch.Tensor,
    norm1_scale: torch.Tensor,
    conv1_scale: torch.Tensor,
    conv1_weight_scale: torch.Tensor,
    norm2_scale: torch.Tensor,
    conv2_weight_scale: torch.Tensor,
    shortcut_scale: torch.Tensor | None,
    shortcut_weight_scale: torch.Tensor | None,
    output_scale: torch.Tensor,
    in_real: int,
    in_pad: int,
    out_real: int,
    out_pad: int,
    warp_mma_scaled_epilogue: bool,
    device: torch.device,
) -> dict[str, Any]:
    residual = _sequence_attr(block, "residual")
    conv1_w, conv1_b, conv1_source_scale, conv1_epilogue_scale, conv1_bias_scaled = (
        _lightvae_fp8_causal_weight(
            native_ext,
            residual[2],
            in_real=in_real,
            in_pad=in_pad,
            out_real=out_real,
            out_pad=out_pad,
            input_scale=norm1_scale,
            output_scale=conv1_scale,
            weight_scale=conv1_weight_scale,
            warp_mma_scaled_epilogue=warp_mma_scaled_epilogue,
            device=device,
        )
    )
    conv2_w, conv2_b, conv2_source_scale, conv2_epilogue_scale, conv2_bias_scaled = (
        _lightvae_fp8_causal_weight(
            native_ext,
            residual[6],
            in_real=out_real,
            in_pad=out_pad,
            out_real=out_real,
            out_pad=out_pad,
            input_scale=norm2_scale,
            output_scale=output_scale,
            weight_scale=conv2_weight_scale,
            warp_mma_scaled_epilogue=warp_mma_scaled_epilogue,
            device=device,
        )
    )
    shortcut = None
    shortcut_module = _module_attr(block, "shortcut")
    if not isinstance(shortcut_module, torch.nn.Identity):
        if shortcut_scale is None:
            raise ValueError(f"{path} shortcut requires shortcut_scale")
        sw, sb = _lightvae_fp8_conv1_weight(
            native_ext,
            shortcut_module,
            in_real=in_real,
            in_pad=in_pad,
            out_real=out_real,
            out_pad=out_pad,
            input_scale=input_scale,
            output_scale=shortcut_scale,
            weight_scale=shortcut_weight_scale
            if shortcut_weight_scale is not None
            else shortcut_scale,
            device=device,
        )
        shortcut = {"w": sw, "b": sb, "scale": shortcut_scale}
    return {
        "path": path,
        "in_real": in_real,
        "out_real": out_real,
        "input_scale": input_scale,
        "input_scale_rms": _scale_prefix_float(input_scale, in_real),
        "norm1_scale": norm1_scale,
        "norm1_scale_rms": _scale_prefix_float(norm1_scale, in_real),
        "conv1_scale": conv1_scale,
        "conv1_scale_rms": _scale_prefix_float(conv1_scale, out_real),
        "norm2_scale": norm2_scale,
        "norm2_scale_rms": _scale_prefix_float(norm2_scale, out_real),
        "output_scale": output_scale,
        "norm1_gamma": _lightvae_gamma(residual[0], in_real, in_pad, device=device),
        "norm2_gamma": _lightvae_gamma(residual[3], out_real, out_pad, device=device),
        "conv1_w": conv1_w,
        "conv1_b": conv1_b,
        "conv1_source_scale": conv1_source_scale,
        "conv1_epilogue_scale": conv1_epilogue_scale,
        "conv1_bias_scaled": conv1_bias_scaled,
        "conv2_w": conv2_w,
        "conv2_b": conv2_b,
        "conv2_source_scale": conv2_source_scale,
        "conv2_epilogue_scale": conv2_epilogue_scale,
        "conv2_bias_scaled": conv2_bias_scaled,
        "shortcut": shortcut,
    }


def build_lightvae_encoder_fp8_staged_state(
    model: torch.nn.Module,
    fp8_state: Mapping[str, torch.Tensor],
    native_ext: Any,
) -> dict[str, Any]:
    """Return the prepared TIN16 FP8 LightVAE encoder state."""

    _require_lightvae_layout(model)
    required_prepare = getattr(
        native_ext, "lightvae_fp8_prepare_conv2d_weight_krsc", None
    )
    if not callable(required_prepare):
        raise TypeError(
            "native extension missing lightvae_fp8_prepare_conv2d_weight_krsc"
        )

    device = _module_device(model)
    state_key = (id(model), id(fp8_state), id(native_ext), str(device))
    cached = _LIGHTVAE_FP8_STAGED_STATE_CACHE.get(state_key)
    if cached is not None:
        return cached

    enc = _unwrap_module(_module_attr(model, "encoder")).eval()
    enc_conv1 = _conv3d_attr(enc, "conv1")
    downs = _sequence_attr(enc, "downsamples")
    middle = _sequence_attr(enc, "middle")
    head = _sequence_attr(enc, "head")
    fs = {str(k): v for k, v in fp8_state.items()}
    use_warp_mma_scaled_epilogue_causal_conv3 = True
    use_warp_mma_scaled_epilogue_spatial_conv3 = True

    def act(keys: tuple[str, ...], real: int, pad: int) -> torch.Tensor:
        return _activation_scale_any(fs, keys, real, pad, device=device)

    def ws(key: str, real: int, pad: int) -> torch.Tensor:
        return _weight_scale_any(fs, key, real, pad, device=device)

    scale_input = act(
        (
            "encoder.conv1.input.activation_scale",
            "encoder.input.activation_scale",
            "input.activation_scale",
        ),
        3,
        32,
    )
    scale_conv1 = act(("encoder.conv1.activation_scale",), 24, 32)

    scale_s0_0_n1 = act(
        (
            _ds_key(0, ".residual.1.activation_scale"),
            _ds_key(0, ".residual.0.activation_scale"),
        ),
        24,
        32,
    )
    scale_s0_0_c1 = act((_ds_key(0, ".residual.2.activation_scale"),), 24, 32)
    scale_s0_0_n2 = act(
        (
            _ds_key(0, ".residual.4.activation_scale"),
            _ds_key(0, ".residual.3.activation_scale"),
        ),
        24,
        32,
    )
    scale_s0_0 = act((_ds_key(0, ".activation_scale"),), 24, 32)
    scale_s0_1_n1 = act(
        (
            _ds_key(1, ".residual.1.activation_scale"),
            _ds_key(1, ".residual.0.activation_scale"),
        ),
        24,
        32,
    )
    scale_s0_1_c1 = act((_ds_key(1, ".residual.2.activation_scale"),), 24, 32)
    scale_s0_1_n2 = act(
        (
            _ds_key(1, ".residual.4.activation_scale"),
            _ds_key(1, ".residual.3.activation_scale"),
        ),
        24,
        32,
    )
    scale_s0_1 = act((_ds_key(1, ".activation_scale"),), 24, 32)
    scale_ds0 = act(
        (_ds_key(2, ".activation_scale"), _ds_key(2, ".resample.1.activation_scale")),
        24,
        32,
    )

    scale_s1_0_n1 = act(
        (
            _ds_key(3, ".residual.1.activation_scale"),
            _ds_key(3, ".residual.0.activation_scale"),
        ),
        24,
        32,
    )
    scale_s1_0_c1 = act((_ds_key(3, ".residual.2.activation_scale"),), 48, 64)
    scale_s1_0_n2 = act(
        (
            _ds_key(3, ".residual.4.activation_scale"),
            _ds_key(3, ".residual.3.activation_scale"),
        ),
        48,
        64,
    )
    scale_s1_0_short = act((_ds_key(3, ".shortcut.activation_scale"),), 48, 64)
    scale_s1_0 = act((_ds_key(3, ".activation_scale"),), 48, 64)
    scale_s1_1_n1 = act(
        (
            _ds_key(4, ".residual.1.activation_scale"),
            _ds_key(4, ".residual.0.activation_scale"),
        ),
        48,
        64,
    )
    scale_s1_1_c1 = act((_ds_key(4, ".residual.2.activation_scale"),), 48, 64)
    scale_s1_1_n2 = act(
        (
            _ds_key(4, ".residual.4.activation_scale"),
            _ds_key(4, ".residual.3.activation_scale"),
        ),
        48,
        64,
    )
    scale_s1_1 = act((_ds_key(4, ".activation_scale"),), 48, 64)
    scale_ds1_spatial = act((_ds_key(5, ".resample.1.activation_scale"),), 48, 64)
    scale_ds1 = act(
        (_ds_key(5, ".activation_scale"), _ds_key(5, ".time_conv.activation_scale")),
        48,
        64,
    )

    scale_s2_0_n1 = act(
        (
            _ds_key(6, ".residual.1.activation_scale"),
            _ds_key(6, ".residual.0.activation_scale"),
        ),
        48,
        64,
    )
    scale_s2_0_c1 = act((_ds_key(6, ".residual.2.activation_scale"),), 96, 96)
    scale_s2_0_n2 = act(
        (
            _ds_key(6, ".residual.4.activation_scale"),
            _ds_key(6, ".residual.3.activation_scale"),
        ),
        96,
        96,
    )
    scale_s2_0_short = act((_ds_key(6, ".shortcut.activation_scale"),), 96, 96)
    scale_s2_0 = act((_ds_key(6, ".activation_scale"),), 96, 96)
    scale_s2_1_n1 = act(
        (
            _ds_key(7, ".residual.1.activation_scale"),
            _ds_key(7, ".residual.0.activation_scale"),
        ),
        96,
        96,
    )
    scale_s2_1_c1 = act((_ds_key(7, ".residual.2.activation_scale"),), 96, 96)
    scale_s2_1_n2 = act(
        (
            _ds_key(7, ".residual.4.activation_scale"),
            _ds_key(7, ".residual.3.activation_scale"),
        ),
        96,
        96,
    )
    scale_s2_1 = act((_ds_key(7, ".activation_scale"),), 96, 96)
    scale_ds2_spatial = act((_ds_key(8, ".resample.1.activation_scale"),), 96, 96)
    scale_ds2 = act(
        (_ds_key(8, ".activation_scale"), _ds_key(8, ".time_conv.activation_scale")),
        96,
        96,
    )

    scale_s3_0_n1 = act(
        (
            _ds_key(9, ".residual.1.activation_scale"),
            _ds_key(9, ".residual.0.activation_scale"),
        ),
        96,
        96,
    )
    scale_s3_0_c1 = act((_ds_key(9, ".residual.2.activation_scale"),), 96, 96)
    scale_s3_0_n2 = act(
        (
            _ds_key(9, ".residual.4.activation_scale"),
            _ds_key(9, ".residual.3.activation_scale"),
        ),
        96,
        96,
    )
    scale_s3_0 = act((_ds_key(9, ".activation_scale"),), 96, 96)
    scale_s3_1_n1 = act(
        (
            _ds_key(10, ".residual.1.activation_scale"),
            _ds_key(10, ".residual.0.activation_scale"),
        ),
        96,
        96,
    )
    scale_s3_1_c1 = act((_ds_key(10, ".residual.2.activation_scale"),), 96, 96)
    scale_s3_1_n2 = act(
        (
            _ds_key(10, ".residual.4.activation_scale"),
            _ds_key(10, ".residual.3.activation_scale"),
        ),
        96,
        96,
    )
    scale_s3_1 = act((_ds_key(10, ".activation_scale"),), 96, 96)

    scale_mid0_n1 = act(
        (
            _mid_key(0, ".residual.1.activation_scale"),
            _mid_key(0, ".residual.0.activation_scale"),
        ),
        96,
        96,
    )
    scale_mid0_c1 = act((_mid_key(0, ".residual.2.activation_scale"),), 96, 96)
    scale_mid0_n2 = act(
        (
            _mid_key(0, ".residual.4.activation_scale"),
            _mid_key(0, ".residual.3.activation_scale"),
        ),
        96,
        96,
    )
    scale_mid0 = act((_mid_key(0, ".activation_scale"),), 96, 96)
    scale_mid_attn_norm = act(
        (
            _mid_key(1, ".norm.activation_scale"),
            _mid_key(1, ".to_qkv.input.activation_scale"),
            _mid_key(0, ".activation_scale"),
        ),
        96,
        96,
    )
    scale_mid_qkv = _activation_scale_scalar_any(
        fs,
        (
            _mid_key(1, ".to_qkv.activation_scale"),
            _mid_key(1, ".qkv.activation_scale"),
            _mid_key(1, ".activation_scale"),
            _mid_key(0, ".activation_scale"),
        ),
        device=device,
    )
    scale_mid_sdpa = _activation_scale_scalar_any(
        fs,
        (
            _mid_key(1, ".sdpa.activation_scale"),
            _mid_key(1, ".attention.activation_scale"),
            _mid_key(1, ".to_qkv.activation_scale"),
            _mid_key(1, ".activation_scale"),
            _mid_key(0, ".activation_scale"),
        ),
        device=device,
    )
    scale_mid_attn = act(
        (_mid_key(1, ".activation_scale"), _mid_key(0, ".activation_scale")), 96, 96
    )
    scale_mid1_n1 = act(
        (
            _mid_key(2, ".residual.1.activation_scale"),
            _mid_key(2, ".residual.0.activation_scale"),
        ),
        96,
        96,
    )
    scale_mid1_c1 = act((_mid_key(2, ".residual.2.activation_scale"),), 96, 96)
    scale_mid1_n2 = act(
        (
            _mid_key(2, ".residual.4.activation_scale"),
            _mid_key(2, ".residual.3.activation_scale"),
        ),
        96,
        96,
    )
    scale_mid1 = act((_mid_key(2, ".activation_scale"),), 96, 96)
    scale_head_norm = act(
        ("encoder.head.1.activation_scale", "encoder.head.0.activation_scale"), 96, 96
    )
    scale_head_conv = act(("encoder.head.2.activation_scale",), 32, 32)
    scale_post = act(("conv1.activation_scale",), 32, 32)

    conv1_w, conv1_b, conv1_source_scale, conv1_epilogue_scale, conv1_bias_scaled = (
        _lightvae_fp8_causal_weight(
            native_ext,
            enc_conv1,
            in_real=3,
            in_pad=32,
            out_real=24,
            out_pad=32,
            input_scale=scale_input,
            output_scale=scale_conv1,
            weight_scale=ws("encoder.conv1.weight_scale", 24, 32),
            warp_mma_scaled_epilogue=use_warp_mma_scaled_epilogue_causal_conv3,
            device=device,
        )
    )

    def rb(
        idx: int,
        block: torch.nn.Module,
        in_real: int,
        in_pad: int,
        out_real: int,
        out_pad: int,
        input_scale: torch.Tensor,
        n1: torch.Tensor,
        c1: torch.Tensor,
        n2: torch.Tensor,
        output_scale: torch.Tensor,
        shortcut_scale: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        path = _ds_key(idx, "")
        shortcut_weight_scale = (
            ws(f"{path}.shortcut.weight_scale", out_real, out_pad)
            if shortcut_scale is not None
            else None
        )
        return _build_lightvae_fp8_resblock_state(
            native_ext,
            block,
            path=path,
            input_scale=input_scale,
            norm1_scale=n1,
            conv1_scale=c1,
            conv1_weight_scale=ws(f"{path}.residual.2.weight_scale", out_real, out_pad),
            norm2_scale=n2,
            conv2_weight_scale=ws(f"{path}.residual.6.weight_scale", out_real, out_pad),
            shortcut_scale=shortcut_scale,
            shortcut_weight_scale=shortcut_weight_scale,
            output_scale=output_scale,
            in_real=in_real,
            in_pad=in_pad,
            out_real=out_real,
            out_pad=out_pad,
            warp_mma_scaled_epilogue=use_warp_mma_scaled_epilogue_causal_conv3,
            device=device,
        )

    def mid_rb(
        idx: int,
        block: torch.nn.Module,
        input_scale: torch.Tensor,
        n1: torch.Tensor,
        c1: torch.Tensor,
        n2: torch.Tensor,
        output_scale: torch.Tensor,
    ) -> dict[str, Any]:
        path = _mid_key(idx, "")
        return _build_lightvae_fp8_resblock_state(
            native_ext,
            block,
            path=path,
            input_scale=input_scale,
            norm1_scale=n1,
            conv1_scale=c1,
            conv1_weight_scale=ws(f"{path}.residual.2.weight_scale", 96, 96),
            norm2_scale=n2,
            conv2_weight_scale=ws(f"{path}.residual.6.weight_scale", 96, 96),
            shortcut_scale=None,
            shortcut_weight_scale=None,
            output_scale=output_scale,
            in_real=96,
            in_pad=96,
            out_real=96,
            out_pad=96,
            warp_mma_scaled_epilogue=use_warp_mma_scaled_epilogue_causal_conv3,
            device=device,
        )

    ds0_w, ds0_b, ds0_epilogue_scale, ds0_bias_scaled = _lightvae_fp8_spatial_weight(
        native_ext,
        downs[2].resample[1],
        in_real=24,
        in_pad=32,
        out_real=24,
        out_pad=32,
        input_scale=scale_s0_1,
        output_scale=scale_ds0,
        weight_scale=ws(_ds_key(2, ".resample.1.weight_scale"), 24, 32),
        warp_mma_scaled_epilogue=use_warp_mma_scaled_epilogue_spatial_conv3,
        device=device,
    )
    ds1_w, ds1_b, ds1_epilogue_scale, ds1_bias_scaled = _lightvae_fp8_spatial_weight(
        native_ext,
        downs[5].resample[1],
        in_real=48,
        in_pad=64,
        out_real=48,
        out_pad=64,
        input_scale=scale_s1_1,
        output_scale=scale_ds1_spatial,
        weight_scale=ws(_ds_key(5, ".resample.1.weight_scale"), 48, 64),
        warp_mma_scaled_epilogue=use_warp_mma_scaled_epilogue_spatial_conv3,
        device=device,
    )
    ds1_tw, ds1_tb = _lightvae_fp8_temporal_weight(
        native_ext,
        downs[5].time_conv,
        channels=64,
        real_channels=48,
        input_scale=scale_ds1_spatial,
        output_scale=scale_ds1,
        weight_scale=ws(_ds_key(5, ".time_conv.weight_scale"), 48, 64),
        device=device,
    )
    ds2_w, ds2_b, ds2_epilogue_scale, ds2_bias_scaled = _lightvae_fp8_spatial_weight(
        native_ext,
        downs[8].resample[1],
        in_real=96,
        in_pad=96,
        out_real=96,
        out_pad=96,
        input_scale=scale_s2_1,
        output_scale=scale_ds2_spatial,
        weight_scale=ws(_ds_key(8, ".resample.1.weight_scale"), 96, 96),
        warp_mma_scaled_epilogue=use_warp_mma_scaled_epilogue_spatial_conv3,
        device=device,
    )
    ds2_tw, ds2_tb = _lightvae_fp8_temporal_weight(
        native_ext,
        downs[8].time_conv,
        channels=96,
        input_scale=scale_ds2_spatial,
        output_scale=scale_ds2,
        weight_scale=ws(_ds_key(8, ".time_conv.weight_scale"), 96, 96),
        device=device,
    )
    head_w, head_b, head_source_scale, head_epilogue_scale, head_bias_scaled = (
        _lightvae_fp8_causal_weight(
            native_ext,
            head[2],
            in_real=96,
            in_pad=96,
            out_real=32,
            out_pad=32,
            input_scale=scale_head_norm,
            output_scale=scale_head_conv,
            weight_scale=ws("encoder.head.2.weight_scale", 32, 32),
            warp_mma_scaled_epilogue=use_warp_mma_scaled_epilogue_causal_conv3,
            device=device,
        )
    )
    post_w, post_b = _lightvae_fp8_conv1_weight(
        native_ext,
        _module_attr(model, "conv1"),
        in_real=32,
        in_pad=32,
        out_real=32,
        out_pad=32,
        input_scale=scale_head_conv,
        output_scale=scale_post,
        weight_scale=ws("conv1.weight_scale", 32, 32),
        device=device,
    )
    mid_attn_qkv_w, mid_attn_qkv_b = _lightvae_fp8_conv1_weight(
        native_ext,
        middle[1].to_qkv,
        in_real=96,
        in_pad=96,
        out_real=288,
        out_pad=288,
        input_scale=scale_mid_attn_norm,
        output_scale=scale_mid_qkv,
        weight_scale=ws(_mid_key(1, ".to_qkv.weight_scale"), 288, 288),
        device=device,
    )
    mid_attn_proj_w, mid_attn_proj_b = _lightvae_fp8_conv1_weight(
        native_ext,
        middle[1].proj,
        in_real=96,
        in_pad=96,
        out_real=96,
        out_pad=96,
        input_scale=scale_mid_sdpa,
        output_scale=scale_mid_attn,
        weight_scale=ws(_mid_key(1, ".proj.weight_scale"), 96, 96),
        device=device,
    )

    mean = (
        _tensor_attr(model, "mean")
        .detach()
        .to(device=device, dtype=torch.float16)
        .reshape(-1)
        .contiguous()
    )
    inv_std = (
        _tensor_attr(model, "inv_std")
        .detach()
        .to(device=device, dtype=torch.float16)
        .reshape(-1)
        .contiguous()
    )
    if mean.numel() != 16 or inv_std.numel() != 16:
        raise ValueError(
            "native LightVAE fp8 encoder expects 16 latent mean/std channels"
        )

    staged = {
        "warp_mma_scaled_epilogue": use_warp_mma_scaled_epilogue_causal_conv3
        and use_warp_mma_scaled_epilogue_spatial_conv3,
        "warp_mma_scaled_epilogue_causal_conv3": use_warp_mma_scaled_epilogue_causal_conv3,
        "warp_mma_scaled_epilogue_spatial_conv3": use_warp_mma_scaled_epilogue_spatial_conv3,
        "scale_input": scale_input,
        "scale_conv1": scale_conv1,
        "conv1_w": conv1_w,
        "conv1_b": conv1_b,
        "conv1_source_scale": conv1_source_scale,
        "conv1_epilogue_scale": conv1_epilogue_scale,
        "conv1_bias_scaled": conv1_bias_scaled,
        "blocks": [
            rb(
                0,
                downs[0],
                24,
                32,
                24,
                32,
                scale_conv1,
                scale_s0_0_n1,
                scale_s0_0_c1,
                scale_s0_0_n2,
                scale_s0_0,
            ),
            rb(
                1,
                downs[1],
                24,
                32,
                24,
                32,
                scale_s0_0,
                scale_s0_1_n1,
                scale_s0_1_c1,
                scale_s0_1_n2,
                scale_s0_1,
            ),
            rb(
                3,
                downs[3],
                24,
                32,
                48,
                64,
                scale_ds0,
                scale_s1_0_n1,
                scale_s1_0_c1,
                scale_s1_0_n2,
                scale_s1_0,
                scale_s1_0_short,
            ),
            rb(
                4,
                downs[4],
                48,
                64,
                48,
                64,
                scale_s1_0,
                scale_s1_1_n1,
                scale_s1_1_c1,
                scale_s1_1_n2,
                scale_s1_1,
            ),
            rb(
                6,
                downs[6],
                48,
                64,
                96,
                96,
                scale_ds1,
                scale_s2_0_n1,
                scale_s2_0_c1,
                scale_s2_0_n2,
                scale_s2_0,
                scale_s2_0_short,
            ),
            rb(
                7,
                downs[7],
                96,
                96,
                96,
                96,
                scale_s2_0,
                scale_s2_1_n1,
                scale_s2_1_c1,
                scale_s2_1_n2,
                scale_s2_1,
            ),
            rb(
                9,
                downs[9],
                96,
                96,
                96,
                96,
                scale_ds2,
                scale_s3_0_n1,
                scale_s3_0_c1,
                scale_s3_0_n2,
                scale_s3_0,
            ),
            rb(
                10,
                downs[10],
                96,
                96,
                96,
                96,
                scale_s3_0,
                scale_s3_1_n1,
                scale_s3_1_c1,
                scale_s3_1_n2,
                scale_s3_1,
            ),
            mid_rb(
                0,
                middle[0],
                scale_s3_1,
                scale_mid0_n1,
                scale_mid0_c1,
                scale_mid0_n2,
                scale_mid0,
            ),
            mid_rb(
                2,
                middle[2],
                scale_mid_attn,
                scale_mid1_n1,
                scale_mid1_c1,
                scale_mid1_n2,
                scale_mid1,
            ),
        ],
        "ds0": {
            "w": ds0_w,
            "b": ds0_b,
            "scale": scale_ds0,
            "epilogue_scale": ds0_epilogue_scale,
            "bias_scaled": ds0_bias_scaled,
            "real": 24,
        },
        "ds1": {
            "spatial_w": ds1_w,
            "spatial_b": ds1_b,
            "spatial_scale": scale_ds1_spatial,
            "spatial_epilogue_scale": ds1_epilogue_scale,
            "spatial_bias_scaled": ds1_bias_scaled,
            "temporal_w": ds1_tw,
            "temporal_b": ds1_tb,
            "scale": scale_ds1,
            "real": 48,
        },
        "ds2": {
            "spatial_w": ds2_w,
            "spatial_b": ds2_b,
            "spatial_scale": scale_ds2_spatial,
            "spatial_epilogue_scale": ds2_epilogue_scale,
            "spatial_bias_scaled": ds2_bias_scaled,
            "temporal_w": ds2_tw,
            "temporal_b": ds2_tb,
            "scale": scale_ds2,
            "real": 96,
        },
        "scale_mid0": scale_mid0,
        "scale_mid_attn": scale_mid_attn,
        "mid_attn": {
            "input_scale_rms": _scale_prefix_float(scale_mid0, 96),
            "norm_gamma": _lightvae_gamma(middle[1].norm, 96, 96, device=device),
            "norm_scale": scale_mid_attn_norm,
            "norm_scale_rms": _scale_prefix_float(scale_mid_attn_norm, 96),
            "qkv_w": mid_attn_qkv_w,
            "qkv_b": mid_attn_qkv_b,
            "qkv_scale": scale_mid_qkv,
            "qkv_scale_float": _float_scalar(scale_mid_qkv, device=device),
            "sdpa_inverse_scale_float": _inverse_float_scalar(
                scale_mid_sdpa, device=device
            ),
            "unit_float": torch.ones(
                (1,), device=device, dtype=torch.float32
            ).contiguous(),
            "proj_w": mid_attn_proj_w,
            "proj_b": mid_attn_proj_b,
        },
        "scale_mid1": scale_mid1,
        "scale_mid1_rms": _scale_prefix_float(scale_mid1, 96),
        "head_norm_gamma": _lightvae_gamma(head[0], 96, 96, device=device),
        "scale_head_norm": scale_head_norm,
        "scale_head_norm_rms": _scale_prefix_float(scale_head_norm, 96),
        "scale_head_conv": scale_head_conv,
        "head_w": head_w,
        "head_b": head_b,
        "head_source_scale": head_source_scale,
        "head_epilogue_scale": head_epilogue_scale,
        "head_bias_scaled": head_bias_scaled,
        "scale_post": scale_post,
        "post_w": post_w,
        "post_b": post_b,
        "mean": mean,
        "inv_std": inv_std,
    }
    _LIGHTVAE_FP8_STAGED_STATE_CACHE[state_key] = staged
    return staged
