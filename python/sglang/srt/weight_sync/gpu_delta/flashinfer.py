# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""FlashInfer expert layouts expressed as views of live consumer bytes.

The apply kernel knows only affine byte coordinates. These admission-time
helpers factor the inference backend's permutations into those coordinates.
They never transform a delta, requantize a weight, or allocate a weight image.
A segment selects a rectangle of one canonical projection; its destinations
have the same flattened byte order, and may cover several independent images.
Padding is deliberately outside every destination.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


@dataclass(frozen=True)
class FlashInferLayout:
    format: str
    backend: str
    gated: bool
    quant_mode: str = ""
    up_first: bool = False


@dataclass(frozen=True)
class DestinationSegment:
    # Coordinates are canonical ELEMENTS, not bytes. Only the returned views
    # use byte coordinates, so BF16 source selections retain their element size.
    slices: tuple[tuple[int, int], tuple[int, int]]
    destinations: tuple[torch.Tensor, ...]


def _backend(backend):
    backend = backend.value if isinstance(backend, Enum) else backend
    if backend in {
        "trtllm",
        "flashinfer_trtllm",
        "flashinfer_trtllm_routed",
        "flashinfer_mxfp4",
    }:
        resolved = "trtllm"
    elif backend in {"cutedsl", "flashinfer_cutedsl"}:
        resolved = "cutedsl"
    elif backend in {"megamoe", "flashinfer_megamoe"}:
        resolved = "megamoe"
    else:
        raise ValueError(f"unsupported FlashInfer expert backend: {backend}")
    return resolved


def identify_layout(layer, backend):
    """Identify an existing inference path; do not select or construct one."""
    backend = _backend(backend)
    method = layer.quant_method
    if method.runner is not None and _backend(method.runner.runner_backend) != backend:
        raise ValueError("FlashInfer selection does not match the live expert runner")
    name = type(method).__name__
    gated = bool(layer.moe_runner_config.is_gated)
    if name == "UnquantizedFusedMoEMethod":
        if backend != "trtllm" or not method.use_flashinfer_trtllm_moe:
            raise ValueError("no native FlashInfer BF16 expert path for this backend")
        layout = FlashInferLayout(
            "bf16", backend, gated, up_first=method.load_up_proj_weight_first
        )
    elif name == "ModelOptNvFp4FusedMoEMethod":
        if not method.quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError("GPU delta does not admit on-load NVFP4 quantization")
        if backend == "trtllm":
            layout = FlashInferLayout("nvfp4", backend, gated, "w4a4", up_first=False)
        elif backend == "cutedsl":
            if not method._is_cutedsl_v2_standard:
                raise ValueError("GPU delta requires the standard CuTe DSL layout")
            if layer._cutedsl_wrapper is None:
                raise ValueError("CuTe DSL warmup must initialize its wrapper")
            mode = layer._cutedsl_wrapper.quant_mode
            if mode not in {"w4a4", "w4a16"}:
                raise ValueError(f"unsupported CuTe DSL quant mode: {mode}")
            layout = FlashInferLayout("nvfp4", backend, gated, mode, up_first=True)
        elif backend == "megamoe":
            layout = FlashInferLayout("nvfp4", backend, gated, "w4a4", up_first=False)
        else:
            raise ValueError(f"unsupported NVFP4 expert backend: {backend}")
    elif name == "Mxfp4FlashinferTrtllmMoEMethod":
        if backend != "trtllm" or method._fp8.dequant_fp4_to_fp8:
            raise ValueError("MXFP4 TRTLLM requires native packed expert weights")
        layout = FlashInferLayout("mxfp4", backend, gated, up_first=True)
    elif name == "Mxfp4MoEMethod":
        if backend != "trtllm" or method._fi_kernel != "trtllm_sm100":
            raise ValueError("unsupported native MXFP4 FlashInfer expert path")
        layout = FlashInferLayout("mxfp4", backend, gated, up_first=True)
    elif name == "Fp8MoEMethod":
        if backend == "cutedsl":
            raise ValueError("ordinary CuTe DSL MoE has no native FP8/MXFP4 path")
        if method.dequant_fp4_to_fp8 or method.convert_mxfp8_to_block:
            raise ValueError("GPU delta does not admit weight requantization")
        config = method.quant_config
        if method.use_mxfp8:
            if not config.is_checkpoint_fp8_serialized:
                raise ValueError("GPU delta requires serialized MXFP8 weights")
            layout = FlashInferLayout("mxfp8", backend, gated)
        elif method.is_fp4_expert:
            if backend != "megamoe":
                raise ValueError("native FP4 TRTLLM experts require its MXFP4 method")
            layout = FlashInferLayout("mxfp4", backend, gated)
        elif (
            backend == "trtllm"
            and config.scale_fmt == "ue8m0"
            and config.weight_block_size == [128, 128]
            and config.is_checkpoint_fp8_serialized
        ):
            layout = FlashInferLayout("fp8-block128-ue8m0", backend, gated)
        else:
            raise ValueError(
                "unsupported FlashInfer expert format; native UE8M0 is required"
            )
    else:
        raise ValueError(f"unsupported FlashInfer expert quantization: {name}")
    return layout


def _bytes(tensor):
    if tensor.ndim == 0:
        tensor = tensor.view(1)
    if tensor.stride(-1) != 1:
        raise ValueError("FlashInfer image requires contiguous innermost elements")
    return tensor.detach().view(torch.uint8)


def _same_image(a, b):
    return (
        a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
        and a.storage_offset() * a.element_size()
        == b.storage_offset() * b.element_size()
        and a.numel() * a.element_size() == b.numel() * b.element_size()
    )


def _unique_images(images):
    unique = []
    for image in images:
        if not any(_same_image(image, current) for current in unique):
            unique.append(image)
    return unique


def _cutedsl_scale_images(layer, stem, expert):
    primary = getattr(layer, f"{stem}_blockscale_swizzled")[expert]
    images = [primary]
    mma = getattr(layer, f"{stem}_blockscale_mma", None)
    if mma is not None:
        # Public MMA axes: (32,4,mtiles,4,ktiles,E). This permutation
        # recovers physical order and creates no temporary scale tensor.
        physical = mma[..., expert].permute(2, 4, 0, 1, 3)
        if not physical.is_contiguous():
            raise ValueError("unsupported nonstandard CuTe DSL MMA physical strides")
        images.append(physical.view(primary.shape).view(primary.dtype))
    images = _unique_images(images)
    source = getattr(layer, f"{stem}_weight_scale")[expert]
    if not any(_same_image(source, image) for image in images):
        raise ValueError("NVFP4 source scale does not alias a prepared scale image")
    return images


def _row_pieces(rows):
    """One full-128-row piece and at most seven power-of-two tail pieces."""
    start = 0
    for unit in (128, 64, 32, 16, 8, 4, 2, 1):
        count = (rows - start) // unit
        if count:
            yield start, start + count * unit, unit
            start += count * unit


def _row_map(row, group, half, shuffle):
    # These are the backend's fixed bit permutations, not a per-row index map:
    # gated groups, then TRTLLM's 32-row transpose (4 x 8 -> 8 x 4).
    if group:
        row = row // group * (2 * group) + half * group + row % group
    if shuffle:
        row = row // 32 * 32 + row % 4 * 8 + row % 32 // 4
    return row


def _factored_segments(
    images,
    rows,
    cols,
    element_size,
    row_offset,
    column_block,
    column_stride,
    block_stride,
):
    """Factor fixed row-bit permutations into affine axes, excluding padding.

    A 128-row bulk plus binary tail pieces gives at most eight row rectangles.
    Columns need at most a full-block rectangle and one tail. No tensor data,
    index tensor, or weight-sized intermediate is constructed.
    """
    byte_cols = cols * element_size
    full = byte_cols // column_block * column_block
    columns = [(0, full)] if full else []
    if full < byte_cols:
        columns.append((full, byte_cols))
    images = tuple(_bytes(image) for image in _unique_images(images))
    result = []
    for r0, r1, unit in _row_pieces(rows):
        bits = [1 << bit for bit in reversed(range(unit.bit_length() - 1))]
        shape = ((r1 - r0) // unit, *(2 for _ in bits))
        origin = row_offset(r0)
        strides = (
            row_offset(r0 + unit) - origin,
            *(row_offset(r0 + bit) - origin for bit in bits),
        )
        for c0, c1 in columns:
            if (c1 - c0) % column_block == 0:
                cshape = ((c1 - c0) // column_block, column_block)
                cstrides = (block_stride, column_stride)
            else:
                cshape, cstrides = (c1 - c0,), (column_stride,)
            offset = origin + c0 // column_block * block_stride
            destinations = tuple(
                image.as_strided(
                    (*shape, *cshape),
                    (*strides, *cstrides),
                    image.storage_offset() + offset,
                )
                for image in images
            )
            result.append(
                DestinationSegment(
                    ((r0, r1), (c0 // element_size, c1 // element_size)), destinations
                )
            )
    return tuple(result)


def _scale_parameter(layer, stem, layout):
    if layout.backend == "trtllm":
        if layout.format == "nvfp4":
            suffix = "weight_scale"
        elif layout.format == "mxfp4":
            method = type(layer.quant_method).__name__
            if method == "Mxfp4MoEMethod":
                suffix = "weight_scale"
            elif method == "Mxfp4FlashinferTrtllmMoEMethod":
                suffix = "weight_scale_inv_shuffled"
            else:
                raise ValueError(f"unsupported TRTLLM MXFP4 method: {method}")
        elif layout.format in {"mxfp8", "fp8-block128-ue8m0"}:
            suffix = "weight_scale_inv"
        else:
            raise ValueError(f"unsupported TRTLLM scale format: {layout.format}")
    elif layout.backend == "cutedsl":
        if layout.format == "nvfp4":
            suffix = "weight_scale"
        else:
            raise ValueError(f"unsupported CuTe DSL scale format: {layout.format}")
    elif layout.backend == "megamoe":
        if layout.format == "nvfp4":
            suffix = "weight_scale"
        elif layout.format == "mxfp4":
            # Retain checkpoint scales separately; target the shuffled consumer.
            suffix = "weight_scale_inv_shuffled"
        elif layout.format == "mxfp8":
            suffix = "weight_scale_inv"
        else:
            raise ValueError(f"unsupported MegaMoE scale format: {layout.format}")
    else:
        raise ValueError(f"unsupported FlashInfer expert backend: {layout.backend}")
    return getattr(layer, f"{stem}_{suffix}")


def _weight_image(layer, stem, expert, layout):
    image = getattr(layer, f"{stem}_weight")[expert]
    if layout.format == "bf16":
        if image.dtype != torch.bfloat16 or image.ndim != 3 or image.shape[-1] != 64:
            raise ValueError("unexpected TRTLLM BF16 blocked weight geometry")
    elif image.ndim != 2 or image.element_size() != 1:
        raise ValueError("unexpected FlashInfer packed weight geometry")

    if layout.backend == "trtllm":
        if layout.format == "bf16":
            rows, cols = image.shape[1], image.shape[0] * 128
        elif layout.format in {"nvfp4", "mxfp4", "mxfp8", "fp8-block128-ue8m0"}:
            rows, cols = image.shape[0], image.shape[1]
        else:
            raise ValueError(f"unsupported TRTLLM weight format: {layout.format}")
    elif layout.backend == "cutedsl":
        if layout.format == "nvfp4":
            rows, cols = image.shape[0], image.shape[1]
        else:
            raise ValueError(f"unsupported CuTe DSL weight format: {layout.format}")
    elif layout.backend == "megamoe":
        if layout.format == "nvfp4":
            # NVFP4 retains the transpose as a view; recover physical packed rows.
            image = image.transpose(0, 1)
            rows, cols = image.shape[0], image.shape[1]
        elif layout.format == "mxfp8":
            # MXFP8 materializes the transpose; its physical rows are K-major.
            rows, cols = image.shape[1], image.shape[0]
        elif layout.format == "mxfp4":
            rows, cols = image.shape[0], image.shape[1]
        else:
            raise ValueError(f"unsupported MegaMoE weight format: {layout.format}")
    else:
        raise ValueError(f"unsupported FlashInfer expert backend: {layout.backend}")
    return image, rows, cols


def _mapping(layout, projection):
    if layout.backend == "trtllm":
        if layout.format in {"bf16", "nvfp4", "mxfp4", "mxfp8"}:
            group, shuffle = 1, True
        elif layout.format == "fp8-block128-ue8m0":
            # Block-128 projections use plain halves, handled separately below.
            group, shuffle = 0, False
        else:
            raise ValueError(f"unsupported TRTLLM expert format: {layout.format}")
    elif layout.backend == "cutedsl":
        if layout.format == "nvfp4":
            # Standard W4A4/W4A16: Up64, Gate64, Up64, Gate64, ...
            group, shuffle = 64, False
        else:
            raise ValueError(f"unsupported CuTe DSL expert format: {layout.format}")
    elif layout.backend == "megamoe":
        if layout.format == "nvfp4":
            # NVFP4 MegaMoE: Gate16, Up16, Gate16, Up16, ...
            group, shuffle = 16, False
        elif layout.format == "mxfp8":
            group, shuffle = 32, False
        elif layout.format == "mxfp4":
            group, shuffle = 8, False
        else:
            raise ValueError(f"unsupported MegaMoE expert format: {layout.format}")
    else:
        raise ValueError(f"unsupported FlashInfer expert backend: {layout.backend}")
    if projection == "down" or not layout.gated:
        group, half = 0, 0
    else:
        half = int((projection == "up") != layout.up_first)
    return group, half, shuffle


def expert_scale_target(layer, layout, local_expert, projection, canonical_shape):
    """Existing FP32 API storage for lossless block-128 UE8M0 expansion.

    The binding prepares complete FP32 values separately; XOR byte views cannot
    express the change of representation from an exponent byte to FP32.
    """
    if layout.format != "fp8-block128-ue8m0" or layout.backend != "trtllm":
        raise ValueError("FP32 scale target requires native block-128 UE8M0")
    if (
        projection not in {"gate", "up", "down"}
        or (projection == "gate" and not layout.gated)
        or len(canonical_shape) != 2
        or any(type(n) is not int or n <= 0 for n in canonical_shape)
        or type(local_expert) is not int
        or local_expert < 0
    ):
        raise ValueError("invalid block-128 scale projection")
    stem = "w2" if projection == "down" else "w13"
    target = getattr(layer, f"{stem}_weight_scale_inv")[local_expert]
    if target.dtype != torch.float32 or target.ndim != 2:
        raise ValueError("unexpected block-128 FP32 scale image")
    rows, cols = canonical_shape
    if layout.gated and projection != "down":
        if target.shape[0] % 2:
            raise ValueError("invalid block-128 gate/up scale geometry")
        half_rows = target.shape[0] // 2
        start = half_rows if projection == "up" else 0
        if rows > half_rows:
            raise ValueError("block-128 scale projection exceeds its half")
        target = target[start : start + rows]
    if rows > target.shape[0] or cols > target.shape[1]:
        raise ValueError("block-128 scale projection exceeds live storage")
    return target[:rows, :cols]


def expert_destinations(
    layer, layout, local_expert, projection, suffix, canonical_shape, canonical_dtype
):
    """Return disjoint canonical rectangles and live bit-preserving views.

    Canonical shapes are the local logical projection, excluding padding.
    Scalar calibration and lossless FP32 scale expansion use a separate complete
    value refresh; they cannot be represented by XOR destinations.
    """
    if projection not in {"gate", "up", "down"}:
        raise ValueError(f"unknown expert projection: {projection}")
    if not layout.gated and projection == "gate":
        raise ValueError("non-gated experts have no gate projection")
    if (
        len(canonical_shape) != 2
        or any(type(n) is not int or n <= 0 for n in canonical_shape)
        or type(local_expert) is not int
        or local_expert < 0
    ):
        raise ValueError(
            "expert byte layout requires a nonempty matrix and local expert"
        )
    group, half, shuffle = _mapping(layout, projection)
    rows, cols = canonical_shape
    stem = "w2" if projection == "down" else "w13"
    if (
        layout.backend == "cutedsl"
        and layout.format == "nvfp4"
        and suffix == "weight_scale"
    ):
        if canonical_dtype != "F8_E4M3":
            raise ValueError("NVFP4 block scales require canonical F8_E4M3 bytes")
        images = _cutedsl_scale_images(layer, stem, local_expert)
        primary = images[0]
        if primary.ndim != 2 or any(
            image.element_size() != 1
            or not image.is_contiguous()
            or image.numel() != primary.numel()
            for image in images
        ):
            raise ValueError("unexpected CuTe DSL block-scale padding geometry")
        padded_rows, padded_cols = primary.shape
        if padded_rows % 128 or padded_cols % 4 or cols > padded_cols:
            raise ValueError("unexpected CuTe DSL block-scale padding geometry")
        max_rows = padded_rows // 2 if group else padded_rows
        if rows > max_rows:
            raise ValueError(
                "canonical scale projection exceeds FlashInfer row storage"
            )

        def scale_row_offset(row):
            row = _row_map(row, group, half, shuffle)
            return row // 128 * padded_cols * 128 + row % 32 * 16 + row % 128 // 32 * 4

        segments = _factored_segments(
            images, rows, cols, 1, scale_row_offset, 4, 1, 512
        )
    else:
        weight, physical_rows, physical_cols = _weight_image(
            layer, stem, local_expert, layout
        )
        if not weight.is_contiguous():
            raise ValueError("unexpected noncontiguous FlashInfer physical storage")
        half_rows = physical_rows // (2 if layout.gated and projection != "down" else 1)
        if rows > half_rows:
            raise ValueError("canonical projection exceeds FlashInfer row storage")
        if group and half_rows % group:
            raise ValueError("invalid FlashInfer gate/up group alignment")
        if shuffle and physical_rows % 32:
            raise ValueError("TRTLLM weight row shuffle requires a multiple of 32")
        mapped = lambda row: _row_map(row, group, half, shuffle)
        if layout.format == "fp8-block128-ue8m0":
            if suffix != "weight" or canonical_dtype != "F8_E4M3":
                raise ValueError("block-128 scales require complete-value expansion")
            start = half_rows if layout.gated and projection == "up" else 0
            if cols > physical_cols:
                raise ValueError("canonical weight exceeds FlashInfer column storage")
            segments = (
                DestinationSegment(
                    ((0, rows), (0, cols)),
                    (_bytes(weight[start : start + rows, :cols]),),
                ),
            )
        elif suffix == "weight":
            if layout.format == "bf16":
                dtype = "BF16"
            elif layout.format in {"nvfp4", "mxfp4"}:
                dtype = "U8"
            elif layout.format == "mxfp8":
                dtype = "F8_E4M3"
            else:
                raise ValueError(
                    f"unsupported FlashInfer weight format: {layout.format}"
                )
            if canonical_dtype != dtype:
                raise ValueError(f"{layout.format} weights require canonical {dtype}")
            element_size = 2 if layout.format == "bf16" else 1
            if cols * element_size > physical_cols:
                raise ValueError("canonical weight exceeds FlashInfer column storage")
            if layout.backend == "trtllm":
                if layout.format == "bf16":
                    row_offset = lambda row: mapped(row) * 128
                    block, stride, block_stride = 128, 1, physical_rows * 128
                elif layout.format in {"nvfp4", "mxfp4", "mxfp8"}:
                    row_offset = lambda row: mapped(row) * physical_cols
                    block, stride, block_stride = 1, 1, 1
                else:
                    raise ValueError(
                        f"unsupported TRTLLM weight format: {layout.format}"
                    )
            elif layout.backend == "cutedsl":
                if layout.format == "nvfp4":
                    row_offset = lambda row: mapped(row) * physical_cols
                    block, stride, block_stride = 1, 1, 1
                else:
                    raise ValueError(
                        f"unsupported CuTe DSL weight format: {layout.format}"
                    )
            elif layout.backend == "megamoe":
                if layout.format == "mxfp8":
                    row_offset = mapped
                    block, stride, block_stride = 1, physical_rows, physical_rows
                elif layout.format in {"nvfp4", "mxfp4"}:
                    row_offset = lambda row: mapped(row) * physical_cols
                    block, stride, block_stride = 1, 1, 1
                else:
                    raise ValueError(
                        f"unsupported MegaMoE weight format: {layout.format}"
                    )
            else:
                raise ValueError(
                    f"unsupported FlashInfer expert backend: {layout.backend}"
                )
            segments = _factored_segments(
                [weight],
                rows,
                cols,
                element_size,
                row_offset,
                block,
                stride,
                block_stride,
            )
        elif suffix == "weight_scale":
            if layout.format == "nvfp4":
                dtype, scale_divisor = "F8_E4M3", 8
            elif layout.format == "mxfp4":
                dtype, scale_divisor = "U8", 16
            elif layout.format == "mxfp8":
                dtype, scale_divisor = "U8", 32
            else:
                raise ValueError(
                    f"unsupported FlashInfer block-scale format: {layout.format}"
                )
            if canonical_dtype != dtype:
                raise ValueError(
                    f"{layout.format} block scales require canonical {dtype}"
                )
            if physical_cols % scale_divisor:
                raise ValueError("weight columns do not cover complete scale groups")
            scale_cols = physical_cols // scale_divisor
            if cols > scale_cols:
                raise ValueError("canonical scales exceed FlashInfer column storage")
            images = [_scale_parameter(layer, stem, layout)[local_expert]]
            if layout.backend == "megamoe" and layout.format == "mxfp4":
                # DeepGEMM packs four UE8M0 bytes per int32 in MN-major storage;
                # MegaMoE then transposes each [4,32] row group for UTCCP.
                scale = images[0]
                if (
                    scale.dtype != torch.int32
                    or scale.ndim != 2
                    or tuple(scale.shape) != (physical_rows, (scale_cols + 3) // 4)
                    or scale.stride() != (1, physical_rows)
                    or physical_rows % 128
                ):
                    raise ValueError("unexpected DeepGEMM packed UE8M0 scale geometry")

                def row_offset(row):
                    row = mapped(row)
                    return (row // 128 * 128 + row % 32 * 4 + row % 128 // 32) * 4

                segments = _factored_segments(
                    [scale.transpose(0, 1)],
                    rows,
                    cols,
                    1,
                    row_offset,
                    4,
                    1,
                    physical_rows * 4,
                )
            elif (
                layout.backend == "trtllm"
                and layout.format in {"nvfp4", "mxfp4", "mxfp8"}
            ) or (layout.backend == "megamoe" and layout.format in {"nvfp4", "mxfp8"}):
                padded_rows = (physical_rows + 127) // 128 * 128
                padded_cols = (scale_cols + 3) // 4 * 4
                for image in images:
                    if (
                        image.element_size() != 1
                        or not image.is_contiguous()
                        or image.numel() != padded_rows * padded_cols
                    ):
                        raise ValueError(
                            "unexpected FlashInfer block-scale padding geometry"
                        )

                def row_offset(row):
                    row = mapped(row)
                    return (
                        row // 128 * padded_cols * 128
                        + row % 32 * 16
                        + row % 128 // 32 * 4
                    )

                segments = _factored_segments(
                    images, rows, cols, 1, row_offset, 4, 1, 512
                )
            else:
                raise ValueError(
                    f"unsupported FlashInfer scale layout: {layout.backend}/{layout.format}"
                )
        else:
            raise ValueError(f"unsupported FlashInfer expert suffix: {suffix}")
    return segments


def consumer_objects(layer, layout):
    """Snapshot selected consumer objects and non-parameter tensor identities."""
    method = layer.quant_method
    objects = [
        method,
        method.runner,
        layer.moe_runner_config,
        layer.moe_runner_config.is_gated,
        layer.w13_weight,
        layer.w2_weight,
    ]
    if layout.format != "bf16":
        objects.extend(
            (
                _scale_parameter(layer, "w13", layout),
                _scale_parameter(layer, "w2", layout),
            )
        )
    if layout.format == "nvfp4":
        objects.extend(
            (
                layer.g1_alphas,
                layer.g1_alphas_up,
                layer.g2_alphas,
                layer.w13_input_scale_quant,
                layer.w2_input_scale_quant,
            )
        )
    if layout.backend == "cutedsl":
        if layout.format != "nvfp4":
            raise ValueError(f"unsupported CuTe DSL consumer format: {layout.format}")
        objects.extend(
            (
                layer._cutedsl_wrapper,
                layer._cutedsl_wrapper.quant_mode,
                layer._cutedsl_scales,
                *layer._cutedsl_scales,
                layer._cutedsl_input_scale,
                layer.w13_blockscale_swizzled,
                layer.w2_blockscale_swizzled,
            )
        )
        # MMA images are created by the standard v2 path when required.
        objects.extend(
            (
                getattr(layer, "w13_blockscale_mma", None),
                getattr(layer, "w2_blockscale_mma", None),
            )
        )
    elif layout.backend == "megamoe":
        if layout.format not in {"nvfp4", "mxfp4", "mxfp8"}:
            raise ValueError(f"unsupported MegaMoE consumer format: {layout.format}")
        mega = layer._flashinfer_megamoe_layer
        if mega is None:
            raise ValueError("FlashInfer MegaMoE must be warmed before admission")
        objects.extend(
            (
                mega,
                mega._megakernel_config,
                mega._kernel,
                mega._transformed,
                layer._flashinfer_megamoe_forward,
            )
        )
        for weights, scales in mega._transformed:
            objects.extend((weights, scales))
        if layout.format == "nvfp4":
            config = mega._megakernel_config
            objects.extend(
                (
                    config.input_norm_const,
                    layer._flashinfer_megamoe_input_norm_const,
                )
            )
    elif layout.backend == "trtllm":
        if layout.format == "nvfp4":
            # Activation-dependent optional tensors; absence is itself captured.
            objects.extend(
                getattr(layer, name, None)
                for name in (
                    "gemm1_alpha",
                    "gemm1_beta",
                    "gemm1_clamp_limit",
                    "g1_scale_c",
                )
            )
        elif layout.format in {"mxfp8", "fp8-block128-ue8m0"}:
            objects.extend(
                (
                    layer._flashinfer_trtllm_gemm1_alpha,
                    layer._flashinfer_trtllm_gemm1_beta,
                    layer._flashinfer_trtllm_gemm1_clamp_limit,
                )
            )
        elif layout.format == "mxfp4":
            if type(method).__name__ == "Mxfp4FlashinferTrtllmMoEMethod":
                objects.append(layer._gemm1_clamp_limit_tensor)
            elif type(method).__name__ == "Mxfp4MoEMethod":
                pass
            else:
                raise ValueError("unsupported TRTLLM MXFP4 consumer")
        elif layout.format == "bf16":
            pass
        else:
            raise ValueError(f"unsupported TRTLLM consumer format: {layout.format}")
    else:
        raise ValueError(f"unsupported FlashInfer expert backend: {layout.backend}")
    return tuple(objects)
