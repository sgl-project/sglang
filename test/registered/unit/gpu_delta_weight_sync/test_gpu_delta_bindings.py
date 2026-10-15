"""CPU canonical bindings, physical byte layouts, and format admission contracts."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.weight_sync.gpu_delta import flashinfer as fi
from sglang.srt.weight_sync.gpu_delta.bindings import (
    TensorBinding,
    _moe_binding,
    validate_ue8m0_payload,
)
from sglang.srt.weight_sync.gpu_delta.flashinfer import (
    DestinationSegment,
    FlashInferLayout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _block_scale_layer():
    return SimpleNamespace(
        _map_global_expert_id_to_local_expert_id=lambda expert: (
            0 if expert == 2 else -1
        ),
        w2_weight_scale_inv=torch.full((1, 17, 15), -3.0),
    )


@pytest.mark.parametrize("checkpoint_dtype", ["F32", "U8"])
def test_block_scale_negotiation_is_identical_on_local_and_foreign_ranks(
    checkpoint_dtype,
):
    layout = FlashInferLayout("fp8-block128-ue8m0", "trtllm", True)
    layer = _block_scale_layer()
    local, foreign = ({"dtype": checkpoint_dtype, "shape": [17, 15]} for _ in range(2))
    binding = _moe_binding("scale", local, layer, 2, "down", "weight_scale_inv", layout)
    assert (
        _moe_binding("scale", foreign, layer, 3, "down", "weight_scale_inv", layout)
        is None
    )
    expected = {
        "dtype": "U8",
        "shape": [17, 15],
        "format": "fp8-block128-ue8m0",
        "encoding": "raw_bytes",
    }
    if checkpoint_dtype == "F32":
        expected["checkpoint_dtype"] = "F32"
    assert local == foreign == expected
    description = binding.describe()
    assert all(description[key] == value for key, value in expected.items())
    assert len(description["views"]) == 1
    if checkpoint_dtype == "U8":
        assert "checkpoint_dtype" not in description
    with pytest.raises(ValueError, match="exact UE8M0"):
        _moe_binding(
            "scale",
            {"dtype": "I32", "shape": [17, 15]},
            layer,
            2,
            "down",
            "weight_scale_inv",
            layout,
        )


@pytest.mark.parametrize("experts", [1, 2])
def test_ue8m0_expansion_is_exact_shared_and_private(experts):
    layer = SimpleNamespace(
        _map_global_expert_id_to_local_expert_id=lambda expert: expert,
        w2_weight_scale_inv=torch.full((experts, 17, 15), -3.0),
    )
    original = layer.w2_weight_scale_inv.clone()
    payload = torch.arange(255, dtype=torch.uint8).repeat(experts)
    validate_ue8m0_payload(payload)
    expanded = payload.view(torch.float8_e8m0fnu).float()
    expected_bits = torch.arange(255, dtype=torch.int32) << 23
    expected_bits[0] = 0x00400000
    for expert in range(experts):
        binding = _moe_binding(
            f"scale{expert}",
            {"dtype": "U8", "shape": [17, 15]},
            layer,
            expert,
            "down",
            "weight_scale_inv",
            FlashInferLayout("fp8-block128-ue8m0", "trtllm", True),
        )
        start = expert * 255
        incoming = payload[start : start + 255]
        ((target, source),) = binding.raw_sources(
            incoming, expanded[start : start + 255]
        )
        assert torch.equal(source.view(torch.int32).flatten(), expected_bits)
        assert (
            source.untyped_storage().data_ptr() == expanded.untyped_storage().data_ptr()
        )
        assert source.storage_offset() == start
        assert target.data_ptr() == layer.w2_weight_scale_inv[expert].data_ptr()
        assert source.data_ptr() != target.data_ptr()
        with pytest.raises(ValueError, match="prepared FP32"):
            binding.raw_sources(incoming)
    with pytest.raises(ValueError, match="NaN"):
        validate_ue8m0_payload(torch.full_like(payload, 255))
    assert torch.equal(layer.w2_weight_scale_inv, original)


def test_apply_segments_keep_one_canonical_view_and_select_original_payload():
    target = torch.zeros((3, 5), dtype=torch.uint8)
    segments = (
        DestinationSegment(((0, 3), (0, 4)), (target[:, :4],)),
        DestinationSegment(((0, 3), (4, 5)), (target[:, 4:],)),
    )
    binding = TensorBinding(
        "weight", "U8", (3, 5), [[0, 3], [0, 5]], (target,), segments=segments
    )
    description = binding.describe()
    assert "format" not in description and "checkpoint_dtype" not in description
    payload = torch.arange(15, dtype=torch.uint8)
    proxies = binding.apply_bindings()
    assert binding.apply_bindings() is proxies
    assert len(proxies) == 2
    for proxy in proxies:
        assert not proxy.segments
        for destination in proxy.destinations:
            destination.bitwise_xor_(
                proxy.selected_bytes(payload).reshape(destination.shape)
            )
    assert torch.equal(target, payload.view(3, 5))
    assert binding.describe() == description
    assert len(description["views"]) == 1


def test_nvfp4_activation_calibration_is_global_but_weight_scales_are_local():
    layer = SimpleNamespace(
        _map_global_expert_id_to_local_expert_id=lambda expert: (
            0 if expert == 2 else -1
        ),
        quant_config=SimpleNamespace(use_per_token_activation=False),
        w13_input_scale=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        w13_weight_scale_2=torch.ones(1, 2),
    )
    layout = FlashInferLayout("nvfp4", "trtllm", True, "w4a4")
    meta = {"dtype": "F32", "shape": []}
    binding = _moe_binding("activation", meta, layer, 3, "up", "input_scale", layout)
    assert binding.storage[0].data_ptr() == layer.w13_input_scale[3, 1].data_ptr()
    assert (
        _moe_binding("weight", meta, layer, 3, "up", "weight_scale_2", layout) is None
    )
    changed = torch.tensor([0.5]).view(torch.uint8)
    ((target, source),) = binding.raw_sources(changed)
    assert target.item() == 7.0 and source.item() == 0.5
    layer.quant_config.use_per_token_activation = True
    assert (
        _moe_binding("activation", meta, layer, 3, "up", "input_scale", layout) is None
    )


def apply_segments(segments, mask):
    coverage = torch.zeros(mask.shape, dtype=torch.int32, device=mask.device)
    for segment in segments:
        (r0, r1), (c0, c1) = segment.slices
        coverage[r0:r1, c0:c1] += 1
        source = mask[r0:r1, c0:c1].contiguous().view(torch.uint8)
        for target in segment.destinations:
            target.bitwise_xor_(source.reshape(target.shape))
    assert torch.all(coverage == 1)


def test_cutedsl_bindings_preserve_scale_images_padding_and_aliases():
    # Independently derive physical indices, including row/column tails. This
    # catches accidental reshapes that clone, alias double-XOR, and padding writes.
    for projection, rows, cols in (
        ("gate", 64, 3),
        ("up", 192, 19),
        ("down", 17, 3),
        ("down", 129, 8),
        ("down", 191, 19),
        ("down", 256, 64),
    ):
        full_rows = rows if projection == "down" else 2 * rows
        pr, pc = (full_rows + 127) // 128 * 128, (cols + 3) // 4 * 4
        for independent in (False, True):
            primary = torch.randint(256, (pr, pc), dtype=torch.uint8)
            physical = primary.view(pr // 128, pc // 4, 32, 4, 4)
            mma = physical.permute(2, 3, 0, 4, 1).unsqueeze(-1)
            if independent:
                mma = mma.clone(memory_format=torch.preserve_format)
            stem = "w2" if projection == "down" else "w13"
            layer = SimpleNamespace(
                **{
                    f"{stem}_blockscale_swizzled": primary.unsqueeze(0),
                    f"{stem}_blockscale_mma": mma,
                    f"{stem}_weight_scale": primary.unsqueeze(0),
                    f"{stem}_weight": torch.zeros(
                        (1, full_rows, cols * 8), dtype=torch.uint8
                    ),
                }
            )
            layout = fi.FlashInferLayout("nvfp4", "cutedsl", True, "w4a4", True)
            layer._map_global_expert_id_to_local_expert_id = lambda expert: expert
            binding = _moe_binding(
                "scale",
                {"dtype": "F8_E4M3", "shape": [rows, cols]},
                layer,
                0,
                projection,
                "weight_scale",
                layout,
            )
            segments = binding.segments
            assert len(binding.storage) == 1 + independent
            images = [primary]
            if independent:
                images.append(mma[..., 0].permute(2, 4, 0, 1, 3).view_as(primary))
            before = [image.clone() for image in images]
            pointers = [image.data_ptr() for image in images]
            mask = torch.randint(256, (rows, cols), dtype=torch.uint8)
            row = torch.arange(rows)[:, None]
            if projection != "down":
                row = row // 64 * 128 + row % 64
                if projection == "gate":
                    row += 64
            col = torch.arange(cols)[None, :]
            offset = (
                (row // 128 * (pc // 4) + col // 4) * 512
                + row % 32 * 16
                + row % 128 // 32 * 4
                + col % 4
            )
            apply_segments(segments, mask)
            for image, original in zip(images, before):
                expected = original.flatten().clone()
                expected[offset.flatten()] ^= mask.flatten()
                assert torch.equal(image.flatten(), expected)
            apply_segments(segments, mask)
            for image, original, pointer in zip(images, before, pointers):
                assert torch.equal(image, original)
                assert image.data_ptr() == pointer


def test_weight_views_share_one_layout_for_both_activation_modes():
    for mode in ("w4a4", "w4a16"):
        original = torch.randint(256, (2, 256, 97), dtype=torch.uint8)
        layer = SimpleNamespace(w13_weight=original.clone())
        expected = original.clone()
        for projection, half in (("up", 0), ("gate", 1)):
            mask = torch.randint(256, (128, 97), dtype=torch.uint8)
            segments = fi.expert_destinations(
                layer,
                fi.FlashInferLayout("nvfp4", "cutedsl", True, mode, True),
                1,
                projection,
                "weight",
                mask.shape,
                "U8",
            )
            apply_segments(segments, mask)
            for start in range(0, 128, 64):
                expected[1, 2 * start + half * 64 : 2 * start + (half + 1) * 64] ^= (
                    mask[start : start + 64]
                )
        assert torch.equal(layer.w13_weight, expected)


def test_layout_admission_rejects_conversion_and_unsupported_pairs():
    def layer(name, **attrs):
        config = attrs.pop("quant_config", SimpleNamespace())
        method = type(name, (), {})()
        method.__dict__.update(
            runner=None,
            dequant_fp4_to_fp8=False,
            convert_mxfp8_to_block=False,
            use_mxfp8=False,
            is_fp4_expert=False,
        )
        method.__dict__.update(attrs, quant_config=config)
        if not hasattr(config, "scale_fmt"):
            config.scale_fmt = None
            config.weight_block_size = None
            if not hasattr(config, "is_checkpoint_fp8_serialized"):
                config.is_checkpoint_fp8_serialized = False
        return SimpleNamespace(
            quant_method=method,
            quant_config=config,
            moe_runner_config=SimpleNamespace(is_gated=True),
            _cutedsl_wrapper=SimpleNamespace(quant_mode="w4a4"),
        )

    nv = layer(
        "ModelOptNvFp4FusedMoEMethod",
        _is_cutedsl_v2_standard=True,
        quant_config=SimpleNamespace(is_checkpoint_nvfp4_serialized=True),
    )
    assert fi.identify_layout(nv, "flashinfer_trtllm") == fi.identify_layout(
        nv, "flashinfer_trtllm_routed"
    )
    assert fi.identify_layout(nv, "flashinfer_cutedsl").quant_mode == "w4a4"
    nv._cutedsl_wrapper.quant_mode = "w4a16"
    assert fi.identify_layout(nv, "flashinfer_cutedsl").quant_mode == "w4a16"
    nv.quant_config.is_checkpoint_nvfp4_serialized = False
    with pytest.raises(ValueError, match="on-load"):
        fi.identify_layout(nv, "flashinfer_cutedsl")
    mx = layer(
        "Fp8MoEMethod",
        use_mxfp8=True,
        quant_config=SimpleNamespace(is_checkpoint_fp8_serialized=True),
    )
    assert fi.identify_layout(mx, "flashinfer_megamoe").format == "mxfp8"
    with pytest.raises(ValueError, match="no native"):
        fi.identify_layout(mx, "flashinfer_cutedsl")
    mx.quant_method.convert_mxfp8_to_block = True
    with pytest.raises(ValueError, match="requantization"):
        fi.identify_layout(mx, "flashinfer_trtllm")
    with pytest.raises(ValueError, match="native UE8M0"):
        fi.identify_layout(layer("Fp8MoEMethod"), "flashinfer_trtllm")


def test_nonalias_scale_image_and_incorrect_geometry_fail_closed():
    layer = SimpleNamespace(
        w2_weight=torch.zeros((1, 128, 32), dtype=torch.uint8),
        w2_weight_scale=torch.zeros((1, 128, 4), dtype=torch.uint8),
        w2_blockscale_swizzled=torch.zeros((1, 128, 4), dtype=torch.uint8),
    )
    layout = fi.FlashInferLayout("nvfp4", "cutedsl", True, "w4a4", True)
    with pytest.raises(ValueError, match="does not alias"):
        fi.expert_destinations(
            layer, layout, 0, "down", "weight_scale", (128, 4), "F8_E4M3"
        )
    layer.w2_weight_scale = layer.w2_blockscale_swizzled
    with pytest.raises(ValueError, match="row storage"):
        fi.expert_destinations(
            layer, layout, 0, "down", "weight_scale", (129, 4), "F8_E4M3"
        )


def shuffled_rows(value):
    # Independent equivalent of FlashInfer's dstToSrcBlk32RowMap index_select.
    return value.reshape(-1, 8, 4, value.shape[-1]).transpose(1, 2).reshape_as(value)


def interleave(value, group):
    rows, cols = value.shape
    return (
        value.reshape(2, rows // (2 * group), group, cols)
        .transpose(0, 1)
        .reshape(rows, cols)
    )


def blocked_scales(value):
    # FlashInfer to_blocked: pad, block [128,4], transpose [4,32] row factors.
    rows, cols = value.shape
    padded = torch.full(
        ((rows + 127) // 128 * 128, (cols + 3) // 4 * 4), 173, dtype=torch.uint8
    )
    padded[:rows, :cols] = value
    pr, pc = padded.shape
    return (
        padded.reshape(pr // 128, 128, pc // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .contiguous()
        .flatten()
    )


@pytest.mark.parametrize(
    "fmt,backend,group,up_first",
    [
        ("nvfp4", "trtllm", 1, False),
        ("mxfp8", "trtllm", 1, False),
        ("mxfp4", "trtllm", 1, True),
        ("bf16", "trtllm", 1, False),
        ("nvfp4", "megamoe", 16, False),
        ("mxfp8", "megamoe", 32, False),
    ],
)
def test_factored_destinations_match_loader_permutations(fmt, backend, group, up_first):
    # Full independent physical replays prove gate/up order, bytes of BF16,
    # scale row shuffling, both Mega transpose policies, and excluded padding.
    rows, physical_cols = 192, 384
    dtype = "BF16" if fmt == "bf16" else "F8_E4M3" if fmt == "mxfp8" else "U8"
    elem = 2 if fmt == "bf16" else 1
    layout = fi.FlashInferLayout(fmt, backend, True, up_first=up_first)
    divisor = {"nvfp4": 8, "mxfp8": 32, "mxfp4": 16}.get(fmt)
    original = torch.randint(256, (2 * rows, physical_cols), dtype=torch.uint8)
    scales = (
        torch.randint(256, (2 * rows, physical_cols // divisor), dtype=torch.uint8)
        if divisor
        else None
    )

    def weight_transform(value):
        value = interleave(value, group)
        if backend == "trtllm":
            value = shuffled_rows(value)
        if fmt == "bf16":
            return (
                value.reshape(2 * rows, physical_cols // 128, 128)
                .transpose(0, 1)
                .contiguous()
                .view(torch.bfloat16)
            )
        if backend == "megamoe":
            value = value.T
            if fmt == "mxfp8":
                value = value.contiguous()
        return value

    def scale_transform(value):
        value = interleave(value, group)
        if backend == "trtllm":
            value = shuffled_rows(value)
        return blocked_scales(value)

    actual = weight_transform(original.clone())
    layer = SimpleNamespace(
        w13_weight=actual.unsqueeze(0), quant_method=SimpleNamespace()
    )
    if scales is not None:
        setattr(
            layer,
            "w13_weight_scale" if fmt == "nvfp4" else "w13_weight_scale_inv",
            scale_transform(scales).unsqueeze(0),
        )
    for projection in ("gate", "up"):
        half = int((projection == "up") != up_first)
        # Tail rows and columns create coarse affine rectangles, never padding writes.
        nr, nc = rows - 3, physical_cols // elem - 3
        mask = torch.randint(256, (nr, nc * elem), dtype=torch.uint8)
        segments = fi.expert_destinations(
            layer, layout, 0, projection, "weight", (nr, nc), dtype
        )
        if elem == 2:
            apply_segments(segments, mask.view(torch.bfloat16))
        else:
            apply_segments(segments, mask)
        original[half * rows : half * rows + nr, : nc * elem] ^= mask
        assert torch.equal(
            actual.view(torch.uint8), weight_transform(original).view(torch.uint8)
        )
        if scales is not None:
            nc = scales.shape[1] - 1
            mask = torch.randint(256, (nr, nc), dtype=torch.uint8)
            segments = fi.expert_destinations(
                layer,
                layout,
                0,
                projection,
                "weight_scale",
                (nr, nc),
                "F8_E4M3" if fmt == "nvfp4" else "U8",
            )
            apply_segments(segments, mask)
            scales[half * rows : half * rows + nr, :nc] ^= mask
            target = fi._scale_parameter(layer, "w13", layout)[0]
            assert torch.equal(target.flatten(), scale_transform(scales))


def test_block128_complete_scale_view_and_consumer_snapshot():
    layer = SimpleNamespace(
        quant_method=SimpleNamespace(runner=None),
        moe_runner_config=SimpleNamespace(is_gated=True),
        w13_weight=torch.zeros((1, 512, 256), dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros((1, 256, 256), dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.ones((1, 4, 2), dtype=torch.float32),
        w2_weight_scale_inv=torch.ones((1, 2, 2), dtype=torch.float32),
        _flashinfer_trtllm_gemm1_alpha=None,
        _flashinfer_trtllm_gemm1_beta=None,
        _flashinfer_trtllm_gemm1_clamp_limit=None,
    )
    layout = fi.FlashInferLayout("fp8-block128-ue8m0", "trtllm", True)
    target = fi.expert_scale_target(layer, layout, 0, "up", (2, 2))
    target.fill_(3)
    assert torch.equal(layer.w13_weight_scale_inv[0, :2], torch.ones((2, 2)))
    assert torch.equal(layer.w13_weight_scale_inv[0, 2:], torch.full((2, 2), 3.0))
    before = tuple(map(id, fi.consumer_objects(layer, layout)))
    layer._flashinfer_trtllm_gemm1_beta = torch.ones(1)
    assert tuple(map(id, fi.consumer_objects(layer, layout))) != before
    with pytest.raises(ValueError, match="complete-value"):
        fi.expert_destinations(layer, layout, 0, "up", "weight_scale", (2, 2), "U8")


def test_megamoe_mxfp4_packed_ue8m0_views():
    # The pre-Mega input is four exponent bytes per I32, column-major by MN.
    # Compose the actual two tensor transforms, independent of address formulas.
    rows, cols = 256, 512
    weight = torch.randint(256, (2 * rows, cols), dtype=torch.uint8)
    scales = torch.randint(256, (2 * rows, cols // 16), dtype=torch.uint8)

    def transform_scale(value):
        packed = value.contiguous().view(torch.int32)
        packed = packed.T.contiguous().T
        interleaved = interleave(packed, 8)
        transposed = (
            interleaved.reshape(-1, 4, 32, packed.shape[1])
            .transpose(1, 2)
            .reshape_as(packed)
        )
        return transposed.T.contiguous().T

    actual_weight = interleave(weight, 8)
    actual_scale = transform_scale(scales)
    layer = SimpleNamespace(
        w13_weight=actual_weight.unsqueeze(0),
        w13_weight_scale_inv=actual_scale.unsqueeze(0),
        quant_method=SimpleNamespace(),
    )
    layout = fi.FlashInferLayout("mxfp4", "megamoe", True)
    for projection, half in (("gate", 0), ("up", 1)):
        for suffix, original, live, transform in (
            ("weight", weight, actual_weight, lambda x: interleave(x, 8)),
            ("weight_scale", scales, actual_scale, transform_scale),
        ):
            r, c = rows - 5, original.shape[1] - 3
            mask = torch.randint(256, (r, c), dtype=torch.uint8)
            segments = fi.expert_destinations(
                layer, layout, 0, projection, suffix, (r, c), "U8"
            )
            apply_segments(segments, mask)
            original[half * rows : half * rows + r, :c] ^= mask
            assert torch.equal(live, transform(original))
