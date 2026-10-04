"""Exact-byte mapping checks against the normal SGLang/FlashInfer loader helpers.

Run with CUDA and the serving image's FlashInfer. These checks are outside the
update timing path and deliberately use the normal value-layout implementation
as an independent oracle for the feature-owned byte-mask implementation.
"""

import math
import sys
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _apply_prepared_masks(bindings, masks, all_configs=False):
    """Exercise the prepared grouped path; nvCOMP itself has a separate oracle."""
    from sglang.srt.weight_sync.gpu_delta_apply import (
        _CONFIGS,
        _compiled,
        _grid,
        _parameters,
        _resident_ctas,
        plan_apply,
        prepare_status_check,
    )
    from sglang.srt.weight_sync.gpu_delta_layout import PreparedDelta, _PreparedBatch

    outputs, size = [], 0
    for binding, mask in zip(bindings, masks):
        offset = (size + 15) // 16 * 16
        outputs.append((binding, offset, mask.numel()))
        size = offset + mask.numel()
    payload = torch.zeros(size, dtype=torch.uint8, device="cuda")
    for (_, offset, length), mask in zip(outputs, masks):
        payload[offset : offset + length].copy_(mask.reshape(-1))
    prepared = PreparedDelta.__new__(PreparedDelta)
    prepared.error = torch.zeros(1, dtype=torch.int32, device="cuda")
    prepared.timing_enabled = False
    decoded = torch.empty(
        size * (4 if all_configs else 1), dtype=torch.uint8, device="cuda"
    )
    group, transformed = plan_apply(outputs)
    saved = [
        (value, value.clone()) for binding in bindings for value in binding.storage
    ]
    apply = None
    if group is not None:
        tuned, _, _, reused, skipped = group.prepare(decoded, prepared.error)
        if all_configs:
            assert tuned or reused
            assert not skipped
        pointers = torch.tensor(
            list(group.pointer_rows(decoded.data_ptr())),
            dtype=torch.int64,
            device="cuda",
        )
        apply = partial(group.launch, pointers, prepared.error)
        # A second plan with identical compiled geometry reuses the device query.
        misses = _resident_ctas.cache_info().misses
        repeated, _ = plan_apply(outputs)
        tuned, footprint, _, reused, skipped = repeated.prepare(decoded, prepared.error)
        assert tuned == footprint == skipped == 0
        assert reused == 1
        assert repeated.kernel is group.kernel
        assert _resident_ctas.cache_info().misses == misses
    # Compilation and module loading must not execute against live weights.
    for value, before in saved:
        torch.testing.assert_close(
            value.view(torch.uint8), before.view(torch.uint8), rtol=0, atol=0
        )
    decoder = SimpleNamespace(
        enqueue=lambda: decoded[:size].copy_(payload),
        statuses=torch.zeros(1, dtype=torch.int32, device="cuda"),
        actual_sizes=torch.tensor([size], device="cuda"),
        expected_sizes=torch.tensor([size], device="cuda"),
    )
    batch = _PreparedBatch(
        [],
        decoder,
        apply,
        [
            (binding.xor, binding.selected_bytes(decoded[offset : offset + length]))
            for binding, offset, length in transformed
        ],
        [],
        prepare_status_check(decoder, prepared.error),
    )
    # Prior errors, a new decoder error and a wrong decoded size must suppress
    # every mapping, including the explicit transformed padded scale path.
    for prior, status, actual in ((1, 0, size), (0, 1, size), (0, 0, size - 1)):
        prepared.error.fill_(prior)
        decoder.statuses.fill_(status)
        decoder.actual_sizes.fill_(actual)
        prepared._decode_batch(batch)
        prepared._apply_batch(batch)
        assert prepared.error.item() == 1
        for value, before in saved:
            torch.testing.assert_close(
                value.view(torch.uint8), before.view(torch.uint8), rtol=0, atol=0
            )
    prepared.error.zero_()
    decoder.statuses.zero_()
    decoder.actual_sizes.fill_(size)
    prepared._decode_batch(batch)
    prepared._apply_batch(batch)
    if all_configs:
        expected = [value.clone() for value, _ in saved]
        for config in _CONFIGS:
            for value, before in saved:
                value.copy_(before)
            prepared._decode_batch(batch)
            if group is not None:
                kernel = _compiled(
                    group.contracts, group.alignments, decoded.device.index, config
                )
                _, _, tiles = _parameters(group.contracts, group.alignments, config[0])
                grid = _grid(tiles, kernel, decoded.device.index, config)
                kernel[grid](pointers, prepared.error)
            for xor, source in batch.transformed:
                xor(source)
            for (value, _), reference in zip(saved, expected):
                torch.testing.assert_close(
                    value.view(torch.uint8), reference.view(torch.uint8), rtol=0, atol=0
                )
    return group


def _reference_layer(values, independent_mma=False):
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
        interleave_w13_halves,
        resolve_cutedsl_standard_scales,
    )
    from sglang.srt.layers.quantization.modelopt_quant import _compute_gemm1_alphas
    from sglang.srt.layers.quantization.utils import swizzle_blockscale
    from sglang.srt.layers.utils import (
        alias_or_bind_derived_param,
        copy_or_rebind_param,
    )

    layer = torch.nn.Module()
    layer.moe_tp_size = 1
    layer.use_presharded_weights = False
    layer.moe_ep_rank = 1
    layer.num_experts, layer.num_local_experts = 4, 2
    layer.quant_method = SimpleNamespace(_is_cutedsl_v2_standard=True)
    layer.moe_runner_config = SimpleNamespace(is_gated=True)
    layer.quant_config = SimpleNamespace(use_per_token_activation=False)
    layer._map_global_expert_id_to_local_expert_id = lambda expert: (
        expert - 2 if 2 <= expert < 4 else -1
    )
    layer._cutedsl_wrapper = SimpleNamespace(quant_mode="w4a16")
    for suffix in ("weight", "weight_scale"):
        # The actual checkpoint loader puts Up before Gate for CuTe DSL.
        fused = torch.cat([values["up", suffix], values["gate", suffix]], dim=1)
        copy_or_rebind_param(layer, "w13_" + suffix, interleave_w13_halves(fused))
        copy_or_rebind_param(layer, "w2_" + suffix, values["down", suffix].clone())
    for stem in ("w13", "w2"):
        scale = getattr(layer, stem + "_weight_scale")
        alias_or_bind_derived_param(
            layer,
            stem + "_weight_scale",
            stem + "_blockscale_swizzled",
            swizzle_blockscale(scale),
        )
        weight = getattr(layer, stem + "_weight")
        mma = convert_sf_to_mma_layout(
            getattr(layer, stem + "_blockscale_swizzled")
            .contiguous()
            .view(torch.uint8)
            .reshape(-1),
            m=weight.shape[1],
            k=2 * weight.shape[2],
            num_groups=2,
            sf_vec_size=16,
        )
        # Both public-loader aliasing and an independent existing MMA consumer
        # image must receive exactly one mask application.
        if independent_mma:
            mma = mma.clone(memory_format=torch.preserve_format)
        copy_or_rebind_param(layer, stem + "_blockscale_mma", mma)
    copy_or_rebind_param(
        layer,
        "w13_weight_scale_2",
        torch.stack(
            [values["gate", "weight_scale_2"], values["up", "weight_scale_2"]], dim=1
        ),
    )
    copy_or_rebind_param(
        layer, "w2_weight_scale_2", values["down", "weight_scale_2"].clone()
    )
    # W4A16 ignores canonical activation scales for its GEMM alphas. Keep their
    # global checkpoint buffers distinct from the neutralized derived values.
    copy_or_rebind_param(layer, "w13_input_scale", torch.ones(4, 2, device="cuda"))
    copy_or_rebind_param(layer, "w2_input_scale", torch.ones(4, device="cuda"))
    neutral = torch.ones(2, device="cuda")
    gate, up = _compute_gemm1_alphas(layer.w13_weight_scale_2, neutral, True)
    copy_or_rebind_param(layer, "g1_alphas", gate)
    copy_or_rebind_param(layer, "g1_alphas_up", up)
    copy_or_rebind_param(layer, "g2_alphas", neutral * layer.w2_weight_scale_2)
    copy_or_rebind_param(layer, "w13_input_scale_quant", neutral.clone())
    copy_or_rebind_param(layer, "w2_input_scale_quant", neutral.clone())
    a, b, c, d = resolve_cutedsl_standard_scales(layer)
    layer._cutedsl_scales = tuple(x.clone() for x in (a, b, c))
    layer._cutedsl_input_scale = d.clone()
    return layer


def _canonical(seed):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    values = {}
    for projection, rows, cols in (
        ("gate", 128, 256),
        ("up", 128, 256),
        ("down", 256, 128),
    ):
        values[projection, "weight"] = torch.randint(
            256, (2, rows, cols // 2), dtype=torch.uint8, device="cuda", generator=rng
        )
        # Legal positive scale values; the resulting XOR masks are never
        # interpreted as FP8 numeric values by the delta mapper.
        values[projection, "weight_scale"] = torch.randint(
            1,
            120,
            (2, rows, cols // 16),
            dtype=torch.uint8,
            device="cuda",
            generator=rng,
        ).view(torch.float8_e4m3fn)
        values[projection, "weight_scale_2"] = (
            torch.rand(2, device="cuda", generator=rng) + 0.5
        )
    return values


@pytest.mark.parametrize("independent_mma", [False, True])
def test_expert_delta_matches_full_loader_layout_and_scale_refresh(
    monkeypatch, independent_mma
):
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
        refresh_cutedsl_standard_scales_for_weight_update,
    )
    from sglang.srt.weight_sync.gpu_delta_layout import GpuDeltaLayout, _moe_binding

    monkeypatch.setenv("SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16", "1")
    old, new = _canonical(13), _canonical(29)
    live, expected = (
        _reference_layer(old, independent_mma=independent_mma),
        _reference_layer(new),
    )
    images = {
        name: parameter
        for name, parameter in live.named_parameters(remove_duplicate=False)
    }
    images.update(
        {f"wrapper.{i}": tensor for i, tensor in enumerate(live._cutedsl_scales)}
    )
    pointers = {name: tensor.data_ptr() for name, tensor in images.items()}
    bindings, masks = [], []
    for (projection, suffix), before in old.items():
        after = new[projection, suffix]
        for local in range(2):
            value = before[local]
            name = f"model.layers.3.mlp.experts.{local + 2}.{projection}_proj.{suffix}"
            dtype = {
                torch.uint8: "U8",
                torch.float8_e4m3fn: "F8_E4M3",
                torch.float32: "F32",
            }[value.dtype]
            binding = _moe_binding(
                name,
                {"dtype": dtype, "shape": list(value.shape)},
                live,
                local + 2,
                projection,
                suffix,
            )
            mask = value.reshape(-1).view(torch.uint8) ^ after[local].reshape(-1).view(
                torch.uint8
            )
            if binding.encoding == "raw_bytes":
                torch._foreach_copy_([binding.storage[0]], [after[local]])
            else:
                bindings.append(binding)
                masks.append(mask)
    _apply_prepared_masks(bindings, masks)
    from sglang.srt.weight_sync.gpu_delta_layout import _direct_binding

    for dtype in (torch.uint8, torch.bfloat16, torch.float32):
        width = torch.empty((), dtype=dtype).element_size()
        canonical = torch.randint(
            256, (12, 20 * width), dtype=torch.uint8, device="cuda"
        )
        live_tp = canonical.view(dtype)[2:10, 5:13].clone()
        mask = torch.randint(256, canonical.shape, dtype=torch.uint8, device="cuda")
        expected_tp = live_tp.view(torch.uint8) ^ mask.reshape(12, 20, width)[
            2:10, 5:13
        ].reshape(8, -1)
        binding = _direct_binding(
            "tp",
            {
                "dtype": {
                    torch.uint8: "U8",
                    torch.bfloat16: "BF16",
                    torch.float32: "F32",
                }[dtype],
                "shape": [12, 20],
            },
            live_tp,
            [[2, 10], [5, 13]],
        )
        _apply_prepared_masks([binding], [mask.reshape(-1)])
        torch.testing.assert_close(
            live_tp.view(torch.uint8), expected_tp, rtol=0, atol=0
        )

    # Mixed linear lengths, a column-strided source and strided routed targets.
    # Tiny items, exact tile ends and partial tails share only useful CTAs.
    # Guard bytes catch stores that escape an item's final mask.
    dense, guards = [], []
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    max_grid = (
        4 * device.multi_processor_count * device.max_threads_per_multi_processor // 128
    )
    # Exceed even the thread-limited occupancy grid to exercise repeated tiles
    # and descriptor boundaries, as well as byte-tail masking.
    for index, size in enumerate(
        (1, 4095, 4096, 4097, 4100, 9001, (max_grid + 1) * 4096 + 17)
    ):
        padding = 4 if size == 4100 else 1
        storage = torch.full(
            (size + 2 * padding,), 0x5A, dtype=torch.uint8, device="cuda"
        )
        guards.append(storage)
        target = storage[padding:-padding].view(1, size)
        dense.append(
            _direct_binding(
                f"model.layers.3.dense{index}.weight",
                {"dtype": "U8", "shape": [1, size]},
                target,
            )
        )
    routed = [b for b in bindings if b.name.endswith(".down_proj.weight")]
    for dense_subset in (dense, dense[1:4]):
        selected = dense_subset + routed + [binding]
        masks = [
            torch.zeros(
                math.prod(b.shape) * b.torch_dtype.itemsize,
                dtype=torch.uint8,
                device="cuda",
            )
            if b.name in {item.name for item in routed}
            else torch.randint(
                256,
                (math.prod(b.shape) * b.torch_dtype.itemsize,),
                dtype=torch.uint8,
                device="cuda",
            )
            for b in selected
        ]
        expected_values = [
            b.destinations[0].clone()
            ^ b.selected_bytes(mask).reshape(b.destinations[0].shape)
            for b, mask in zip(selected, masks)
        ]
        group = _apply_prepared_masks(selected, masks, all_configs=True)
        assert len(group.sources) == len(selected)
        assert group.tiles == sum(
            count * math.ceil(contract[0] / group.config[0])
            for contract, count in group.contracts
        )
        if dense_subset is dense:
            assert group.word32_contracts > 0
        for b, expected_value in zip(selected, expected_values):
            torch.testing.assert_close(
                b.destinations[0], expected_value, rtol=0, atol=0
            )
        assert all(
            storage[0].item() == storage[-1].item() == 0x5A for storage in guards
        )

    plan = GpuDeltaLayout.__new__(GpuDeltaLayout)
    plan.derived = []
    plan._add_moe_derived("model.layers.3.mlp.experts", live)
    for image in plan.derived:
        torch.where(
            torch.tensor(True, device="cuda"),
            image.source,
            image.destination,
            out=image.destination,
        )
    refresh_cutedsl_standard_scales_for_weight_update(expected)
    for name, target in expected.named_parameters(remove_duplicate=False):
        actual = dict(live.named_parameters(remove_duplicate=False))[name]
        torch.testing.assert_close(
            actual.view(torch.uint8), target.view(torch.uint8), rtol=0, atol=0
        )
    for actual, target in zip(live._cutedsl_scales, expected._cutedsl_scales):
        torch.testing.assert_close(
            actual.view(torch.uint8), target.view(torch.uint8), rtol=0, atol=0
        )
    assert pointers == {name: tensor.data_ptr() for name, tensor in images.items()}
    assert live.w13_weight_scale.data_ptr() == live.w13_blockscale_swizzled.data_ptr()
    assert live.w2_weight_scale.data_ptr() == live.w2_blockscale_swizzled.data_ptr()


def test_feature_scale_permutation_matches_loader_padding():
    from sglang.srt.layers.quantization.utils import swizzle_blockscale
    from sglang.srt.weight_sync.gpu_delta_layout import swizzle_scale_bytes

    for shape in ((17, 3), (128, 64), (2, 256, 19)):
        values = torch.randint(1, 120, shape, dtype=torch.uint8, device="cuda")
        expected = swizzle_blockscale(values.view(torch.float8_e4m3fn)).view(
            torch.uint8
        )
        torch.testing.assert_close(
            swizzle_scale_bytes(values), expected, rtol=0, atol=0
        )
        if len(shape) == 2:
            from sglang.srt.weight_sync.gpu_delta_layout import _moe_binding

            live = expected.clone().view(torch.float8_e4m3fn).unsqueeze(0)
            layer = SimpleNamespace(
                moe_tp_size=1,
                use_presharded_weights=False,
                quant_method=SimpleNamespace(_is_cutedsl_v2_standard=True),
                moe_runner_config=SimpleNamespace(is_gated=True),
                _map_global_expert_id_to_local_expert_id=lambda _: 0,
                w2_blockscale_swizzled=live,
                w2_weight_scale=live,
            )
            binding = _moe_binding(
                "model.layers.0.mlp.experts.0.down_proj.weight_scale",
                {"dtype": "F8_E4M3", "shape": list(shape)},
                layer,
                0,
                "down",
                "weight_scale",
            )
            mask = torch.randint(256, shape, dtype=torch.uint8, device="cuda")
            _apply_prepared_masks([binding], [mask.reshape(-1)])
            torch.testing.assert_close(
                live[0].view(torch.uint8),
                expected ^ swizzle_scale_bytes(mask),
                rtol=0,
                atol=0,
            )


def test_cached_mla_refresh_survives_graph_replay_and_failure_gate():
    from sglang.srt.weight_sync.gpu_delta_layout import GpuDeltaLayout

    weight = torch.zeros((2 * (4 + 6), 8), dtype=torch.bfloat16, device="cuda")
    key, value = weight.unflatten(0, (2, 10)).split([4, 6], dim=1)
    attn = SimpleNamespace(
        kv_b_proj=SimpleNamespace(weight=weight),
        qk_nope_head_dim=4,
        v_head_dim=6,
        w_kc=key.transpose(1, 2).contiguous().transpose(1, 2),
        w_vc=value.contiguous().transpose(1, 2),
    )
    plan = GpuDeltaLayout.__new__(GpuDeltaLayout)
    plan.derived = []
    plan._add_mla_derived("attention", attn)
    identity = [
        (d.destination.data_ptr(), d.destination.stride()) for d in plan.derived
    ]
    succeeded = torch.tensor(True, device="cuda")

    def refresh():
        for image in plan.derived:
            torch.where(
                succeeded, image.source, image.destination, out=image.destination
            )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        refresh()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        refresh()
    for success in (True, True, False):
        before = [
            d.destination.contiguous().view(torch.uint8).clone() for d in plan.derived
        ]
        weight.view(torch.uint8).random_(256)
        succeeded.fill_(success)
        graph.replay()
        for image, original in zip(plan.derived, before):
            expected = (
                image.source.contiguous().view(torch.uint8) if success else original
            )
            torch.testing.assert_close(
                image.destination.contiguous().view(torch.uint8),
                expected,
                rtol=0,
                atol=0,
            )
    assert identity == [
        (d.destination.data_ptr(), d.destination.stride()) for d in plan.derived
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
