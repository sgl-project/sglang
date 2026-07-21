"""CPU coverage for Kimi-K2.5/K2.7 encoder-DP wiring."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.mm.process import normalize_and_patchify
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.models import kimi_k3_vl, kimi_k25
from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.multimodal.processors.kimi_k3 import (
    KimiK3GPUProcessorWrapper,
    KimiK3ImageProcessor,
)
from sglang.srt.multimodal.processors.kimi_k25 import (
    KimiGPUProcessorWrapper,
    KimiK2_5VLImageProcessor,
    _expand_image_token_ids,
    _grid_thw_from_resize_config,
    _resize_bicubic_if_needed,
    _resize_images_by_source_shape,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.cuda_ipc_transport_utils import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
    CudaIpcTensorTransportProxy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Tokenizer:
    def encode(self, _text):
        return []


def test_kimi_shared_concat_avoids_single_image_copy():
    first = torch.arange(12).view(3, 4)
    second = first + 12

    assert kimi_k25.concat_or_single is kimi_k3_vl.concat_or_single
    assert kimi_k25.concat_or_single([first]) is first
    torch.testing.assert_close(
        kimi_k25.concat_or_single([first, second]), torch.cat([first, second])
    )


@pytest.mark.parametrize("frames", [1, 2])
def test_kimi_tpool_shared_fast_path_matches_temporal_mean(frames):
    grid_thws = torch.tensor([[frames, 4, 6]], dtype=torch.int64)
    hidden_states = torch.arange(frames * 4 * 6 * 3, dtype=torch.float32).view(-1, 3)

    reference = hidden_states.view(frames, 2, 2, 3, 2, 3)
    reference = reference.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
    reference = reference.view(6, 4, 3)

    assert kimi_k25.tpool_patch_merger is kimi_k3_vl.tpool_patch_merger
    actual = kimi_k25.tpool_patch_merger(hidden_states, grid_thws)
    explicit_metadata = kimi_k25.tpool_patch_merger(
        hidden_states, grid_thws, grid_thw_list=grid_thws.tolist()
    )
    assert len(actual) == len(explicit_metadata) == 1
    torch.testing.assert_close(actual[0], reference, rtol=0, atol=0)
    torch.testing.assert_close(explicit_metadata[0], reference, rtol=0, atol=0)


class _HFProcessor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.image_processor = SimpleNamespace()
        self.media_processor = SimpleNamespace(
            media_proc_cfg={
                "patch_size": 14,
                "merge_kernel_size": 2,
                "in_patch_limit": 16384,
                "patch_limit_on_one_side": 256,
                "fixed_output_tokens": None,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "transparent_bg_config": None,
            }
        )


def test_kimi_processors_precompute_hash_before_cpu_transfer():
    assert KimiK2_5VLImageProcessor.precompute_hash_before_cpu_transfer
    assert KimiK3ImageProcessor.precompute_hash_before_cpu_transfer


def test_kimi_k3_builds_grid_metadata_on_cpu_from_resize_config():
    config = {
        "new_height": 224,
        "new_width": 322,
        "pad_height": 0,
        "pad_width": 14,
    }

    grid_thw = torch.tensor(
        [_grid_thw_from_resize_config(config, patch_size=14)], dtype=torch.int64
    )

    assert grid_thw.device.type == "cpu"
    assert grid_thw.tolist() == [[1, 16, 24]]


def test_kimi_normalize_and_patchify_matches_reference_on_cpu():
    image = torch.linspace(0.0, 255.0, 2 * 3 * 15 * 16).reshape(2, 3, 15, 16)
    image_mean_values = [0.48145466, 0.4578275, 0.40821073]
    image_std_values = [0.26862954, 0.26130258, 0.27577711]
    wrapper = KimiGPUProcessorWrapper(
        _HFProcessor(),
        image_token="<|media_pad|>",
        image_token_id=42,
        patch_size=14,
        merge_kernel_size=2,
        in_patch_limit=16384,
        patch_limit_on_one_side=256,
        fixed_output_tokens=None,
        image_mean=image_mean_values,
        image_std=image_std_values,
    )
    image_scale, image_bias = wrapper._get_gpu_norm_tensors(device="cpu")
    image_mean = torch.tensor(image_mean_values).view(1, 3, 1, 1)
    image_std = torch.tensor(image_std_values).view(1, 3, 1, 1)

    actual = normalize_and_patchify(image, image_scale, image_bias, 14, 28, 28)
    expected = F.pad(image, (0, 12, 0, 13), value=0.0)
    expected = (expected / 255.0 - image_mean) / image_std
    expected = expected.view(2, 3, 2, 14, 2, 14)
    expected = expected.permute(0, 2, 4, 1, 3, 5).reshape(2, 4, 3, 14, 14)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("processor_cls", "wrapper_cls"),
    [
        (KimiK2_5VLImageProcessor, KimiGPUProcessorWrapper),
        (KimiK3ImageProcessor, KimiK3GPUProcessorWrapper),
    ],
)
def test_kimi_processor_workers_clone_the_gpu_wrapper(processor_cls, wrapper_cls):
    server_args = SimpleNamespace(
        keep_mm_feature_on_device=True,
        mm_feature_transport="cpu",
        disable_fast_image_processor=False,
        skip_tokenizer_init=False,
        mm_process_config={},
        mm_io_worker_num=0,
        mm_processor_worker_num=0,
        tokenizer_worker_num=1,
        base_gpu_id=0,
    )
    processor = processor_cls(
        hf_config=SimpleNamespace(media_placeholder_token_id=42),
        server_args=server_args,
        _processor=_HFProcessor(),
        transport_mode=None,
    )
    try:
        worker_processor = asyncio.run(
            processor.mm_processor_executor.run(lambda *, processor: processor)
        )
        assert processor.mm_processor_worker_num == 2
        assert processor.mm_io_worker_num == 16
        assert isinstance(processor._processor, wrapper_cls)
        assert isinstance(worker_processor, wrapper_cls)
        assert worker_processor is not processor._processor
    finally:
        processor.mm_processor_executor.shutdown()
        processor.io_executor.shutdown()
        processor.cpu_executor.shutdown()


class _MoonViT3dTower:
    device = torch.device("cpu")
    merge_kernel_size = (2, 2)

    def __init__(self):
        self.config = SimpleNamespace(hidden_size=2)
        self.patch_embed = SimpleNamespace(
            proj=SimpleNamespace(weight=torch.empty(1, dtype=torch.float32))
        )
        self.grid_thws = None

    def __call__(self, pixel_values, grid_thws):
        self.grid_thws = grid_thws
        # MoonViT3d returns a list of [tokens, merge_area, hidden] tensors.
        return [pixel_values.reshape(-1, 4, pixel_values.shape[-1])]


class _Projector:
    def __call__(self, image_embeds):
        return image_embeds


def _image_item(feature, grid_thw):
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(0, 1)],
        feature=feature,
        model_specific_data={"image_grid_thw": torch.tensor(grid_thw)},
    )


def test_kimi_gpu_preprocess_batches_only_source_compatible_images():
    torch.manual_seed(0)
    indexed_images = [
        (0, torch.randn(3, 32, 24)),
        (1, torch.randn(3, 32, 24)),
        (2, torch.randn(3, 28, 20)),
    ]
    expected = [
        F.interpolate(
            image.unsqueeze(0), size=(16, 12), mode="bicubic", align_corners=False
        )
        for _, image in indexed_images
    ]
    real_interpolate = F.interpolate
    input_shapes = []

    def record_interpolate(image, *args, **kwargs):
        input_shapes.append(tuple(image.shape))
        return real_interpolate(image, *args, **kwargs)

    with patch(
        "sglang.srt.multimodal.processors.kimi_k25.F.interpolate",
        side_effect=record_interpolate,
    ):
        actual = _resize_images_by_source_shape(indexed_images, 16, 12)

    assert input_shapes == [(2, 3, 32, 24), (1, 3, 28, 20)]
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference)


def test_kimi_skips_noop_bicubic_resize():
    image = torch.arange(2 * 3 * 16 * 12, dtype=torch.uint8).reshape(2, 3, 16, 12)
    expected = F.interpolate(
        image.float(), size=(16, 12), mode="bicubic", align_corners=False
    )

    with patch(
        "sglang.srt.multimodal.processors.kimi_k25.F.interpolate"
    ) as interpolate:
        actual = _resize_bicubic_if_needed(image, 16, 12)

    interpolate.assert_not_called()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "input_ids",
    ([1, 99, 2, 99, 3], torch.tensor([[1, 99, 2, 99, 3]], dtype=torch.int32)),
)
def test_kimi_expands_pre_tokenized_image_placeholders(input_ids):
    actual = _expand_image_token_ids(input_ids, 99, [3, 2])

    assert actual.dtype == torch.long
    assert actual.device.type == "cpu"
    assert actual.tolist() == [[1, 99, 99, 99, 2, 99, 99, 3]]


def test_kimi_rejects_mismatched_pre_tokenized_image_placeholders():
    with pytest.raises(ValueError, match="Expected 2 image placeholder"):
        _expand_image_token_ids([1, 99, 2], 99, [3, 2])


def test_kimi_k3_rejects_silently_dropped_images():
    processor = object.__new__(KimiK3ImageProcessor)
    processor.mm_tokens = Mock()
    processor.load_mm_data = AsyncMock(
        return_value=SimpleNamespace(images=[object()])
    )

    with pytest.raises(ValueError, match="expected 2, loaded 1"):
        asyncio.run(
            processor.process_mm_data_async(
                image_data=["image-1", "image-2"],
                input_text="<|media_pad|><|media_pad|>",
                request_obj=SimpleNamespace(video_data=None),
            )
        )


def test_kimi_k3_uses_token_ids_to_preserve_media_boundaries():
    processor = object.__new__(KimiK3ImageProcessor)
    processor.mm_tokens = SimpleNamespace(image_token_id=99)
    processor.fast_load_mm_data = AsyncMock(
        return_value=SimpleNamespace(
            images=[object(), object()], input_ids=[1, 99, 2, 99, 3]
        )
    )
    processor.load_mm_data = AsyncMock()
    processor.process_and_combine_mm_data_async = AsyncMock(
        return_value=([], torch.tensor([[1, 2]]), None)
    )

    asyncio.run(
        processor.process_mm_data_async(
            image_data=["image-1", "image-2"],
            input_text=[1, 99, 2, 99, 3],
            request_obj=SimpleNamespace(video_data=None),
        )
    )

    processor.fast_load_mm_data.assert_awaited_once()
    processor.load_mm_data.assert_not_awaited()


def test_kimi_k3_rejects_tokenized_placeholder_mismatch():
    processor = object.__new__(KimiK3ImageProcessor)
    processor.mm_tokens = SimpleNamespace(image_token_id=99)
    processor.fast_load_mm_data = AsyncMock()
    processor.load_mm_data = AsyncMock()

    with pytest.raises(ValueError, match=r"expected 2, found 1 token\(s\)"):
        asyncio.run(
            processor.process_mm_data_async(
                image_data=["image-1", "image-2"],
                input_text=torch.tensor([[1, 99, 2]]),
                request_obj=SimpleNamespace(video_data=None),
            )
        )

    processor.fast_load_mm_data.assert_not_awaited()
    processor.load_mm_data.assert_not_awaited()


def test_kimi_k25_uses_token_ids_to_preserve_media_boundaries():
    processor = object.__new__(KimiK2_5VLImageProcessor)
    processor.mm_tokens = SimpleNamespace(image_token_id=99)
    processor.fast_load_mm_data = AsyncMock(
        return_value=SimpleNamespace(
            images=[object(), object()], input_ids=[1, 99, 2, 99, 3]
        )
    )
    processor.load_mm_data = AsyncMock()
    processor.process_and_combine_mm_data_async = AsyncMock(
        return_value=([], torch.tensor([[1, 2]]), None)
    )
    processor.use_cuda_ipc = False

    asyncio.run(
        processor.process_mm_data_async(
            image_data=["image-1", "image-2"],
            input_text=[1, 99, 2, 99, 3],
            request_obj=SimpleNamespace(),
        )
    )

    processor.fast_load_mm_data.assert_awaited_once()
    processor.load_mm_data.assert_not_awaited()


@pytest.mark.parametrize(
    ("request_obj", "extra_kwargs"),
    [
        (SimpleNamespace(video_data=["video"]), {}),
        (SimpleNamespace(video_data=None), {"audio_data": ["audio"]}),
    ],
)
def test_kimi_k3_rejects_unsupported_modalities(request_obj, extra_kwargs):
    processor = object.__new__(KimiK3ImageProcessor)
    processor.mm_tokens = Mock()
    processor.load_mm_data = AsyncMock()

    with pytest.raises(ValueError, match="supports image input only"):
        asyncio.run(
            processor.process_mm_data_async(
                image_data=[],
                input_text="prompt",
                request_obj=request_obj,
                **extra_kwargs,
            )
        )
    processor.load_mm_data.assert_not_awaited()


def test_dp_helper_supports_moonvit3d_packed_embeddings_on_tp1():
    tower = _MoonViT3dTower()
    pixel_values = torch.randn(4, 2)

    with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0):
        output = run_dp_sharded_mrope_vision_model(
            tower, pixel_values, [[1, 2, 2]], rope_type="rope_2d_packed"
        )

    assert torch.equal(output, pixel_values.reshape(1, 4, 2))
    assert torch.equal(tower.grid_thws, torch.tensor([[1, 2, 2]]))


def test_dp_helper_can_lazily_load_kimi_features_on_tp1():
    tower = _MoonViT3dTower()
    pixel_values = torch.randn(4, 2)
    loader = Mock(return_value=pixel_values)

    with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0):
        output = run_dp_sharded_mrope_vision_model(
            tower,
            None,
            [[1, 2, 2]],
            rope_type="rope_2d_packed",
            load_local_pixel_values=loader,
            pixel_values_device=pixel_values.device,
            pixel_values_dtype=pixel_values.dtype,
        )

    assert torch.equal(output, pixel_values.reshape(1, 4, 2))
    loader.assert_called_once_with([0])


def test_dp_helper_uses_config_hidden_size_for_empty_moonvit3d_rank():
    class _GatherGroup:
        def all_gather(self, tensor, dim):
            return torch.cat([torch.ones_like(tensor), tensor], dim=dim)

    tower = _MoonViT3dTower()
    parallel = SimpleNamespace(
        attn_tp_size=2,
        attn_tp_rank=1,
        attn_tp_group=_GatherGroup(),
    )

    with patch("sglang.srt.multimodal.mm_utils.get_parallel", return_value=parallel):
        output = run_dp_sharded_mrope_vision_model(
            tower,
            torch.randn(4, 2),
            [[1, 2, 2]],
            rope_type="rope_2d_packed",
        )

    assert output.shape == (1, 4, 2)
    assert tower.grid_thws is None


def test_dp_helper_lazily_loads_only_its_local_image_shard():
    class _GatherGroup:
        def all_gather(self, tensor, dim):
            # Rank one's embedding is irrelevant to this rank's loader call;
            # retain the expected gathered shape for output reconstruction.
            return torch.cat([tensor, torch.zeros_like(tensor)], dim=dim)

    tower = _MoonViT3dTower()
    features = [torch.full((4, 2), 1.0), torch.full((4, 2), 2.0)]
    loader = Mock(side_effect=lambda indices: torch.cat([features[i] for i in indices]))
    parallel = SimpleNamespace(
        attn_tp_size=2,
        attn_tp_rank=0,
        attn_tp_group=_GatherGroup(),
    )

    with patch("sglang.srt.multimodal.mm_utils.get_parallel", return_value=parallel):
        output = run_dp_sharded_mrope_vision_model(
            tower,
            None,
            [[1, 2, 2], [1, 2, 2]],
            rope_type="rope_2d_packed",
            load_local_pixel_values=loader,
            pixel_values_device=torch.device("cpu"),
            pixel_values_dtype=torch.float32,
        )

    loader.assert_called_once_with([0])
    assert output.shape == (2, 4, 2)


def test_kimi_k25_encoder_dp_selects_packed_moonvit_contract():
    model = KimiK25ForConditionalGeneration.__new__(KimiK25ForConditionalGeneration)
    nn.Module.__init__(model)
    model.use_data_parallel = True
    model.vision_tower = _MoonViT3dTower()
    model.mm_projector = _Projector()
    items = [_image_item(torch.randn(4, 2), [[1, 2, 2]])]
    sharded_embeddings = torch.randn(1, 2)

    with patch(
        "sglang.srt.models.kimi_k25.run_dp_sharded_mrope_vision_model",
        return_value=sharded_embeddings,
    ) as run_dp:
        output = model.get_image_feature(items)

    assert output is sharded_embeddings
    tower, pixel_values, grid_thws = run_dp.call_args.args
    assert tower is model.vision_tower
    assert pixel_values is None
    assert grid_thws == [[1, 2, 2]]
    assert run_dp.call_args.kwargs["rope_type"] == "rope_2d_packed"
    assert callable(run_dp.call_args.kwargs["load_local_pixel_values"])


def test_kimi_lazy_ipc_feature_skips_scheduler_reconstruction():
    proxy = CudaIpcTensorTransportProxy.__new__(CudaIpcTensorTransportProxy)
    proxy.reconstruct_on_target_device = Mock()
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=123,
        pad_value=456,
        offsets=[(0, 1)],
        feature=proxy,
        model_specific_data={DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY: True},
    )

    with patch(
        "sglang.srt.managers.schedule_batch.torch.cuda.current_device", return_value=0
    ):
        mm_inputs = MultimodalInputs.from_processor_output(
            MultimodalProcessorOutput(mm_items=[item])
        )

    assert mm_inputs.mm_items[0].feature is proxy
    proxy.reconstruct_on_target_device.assert_not_called()


def test_kimi_lazy_ipc_feature_acknowledges_all_tp_consumers():
    proxy = CudaIpcTensorTransportProxy.__new__(CudaIpcTensorTransportProxy)
    proxy.reconstruct_on_target_device = Mock(return_value=torch.randn(1, 2))
    item = MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)

    item.reconstruct(0, ipc_consumer_count=8)

    proxy.reconstruct_on_target_device.assert_called_once_with(0, consumer_count=8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
