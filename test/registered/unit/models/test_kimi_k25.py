"""CPU coverage for Kimi-K2.5/K2.7 encoder-DP wiring."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.models.kimi_k25 import (
    KimiK25ForConditionalGeneration,
    mm_projection_auto,
)
from sglang.srt.models.kimi_vl_moonvit import tpool_patch_merger
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin
from sglang.srt.multimodal.processors.kimi_k3 import (
    KimiK3GPUProcessorWrapper,
    KimiK3ImageProcessor,
    _expand_k3_image_prompt_text,
    _expand_k3_image_prompt_token_ids,
)
from sglang.srt.multimodal.processors.kimi_k25 import (
    KimiGPUProcessorWrapper,
    KimiK2_5VLImageProcessor,
    _ensure_chw_rgb,
    _expand_image_token_ids,
    _resize_bicubic_if_needed,
    _resize_images_by_source_shape,
)
from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
    CudaIpcTensorTransportProxy,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


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
        (0, torch.randint(0, 256, (3, 32, 24), dtype=torch.uint8)),
        (1, torch.randint(0, 256, (3, 32, 24), dtype=torch.uint8)),
        (2, torch.randint(0, 256, (3, 28, 20), dtype=torch.uint8)),
    ]
    expected = [
        _resize_bicubic_if_needed(image.unsqueeze(0), 16, 12)
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


def test_kimi_resize_tracks_the_checkpoint_processors_pil_bicubic():
    # Plain F.interpolate skips PIL's implicit antialiasing on downscale and
    # drifts far outside 8-bit rounding; photo-like content, not pure noise.
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
    plane = np.clip(
        128
        + 90 * np.sin(xx / 40) * np.cos(yy / 55)
        + 40 * ((xx // 37 + yy // 41) % 2)
        + rng.normal(0, 6, (512, 512)),
        0,
        255,
    )
    array = np.stack([plane, np.roll(plane, 7, 0), np.roll(plane, 13, 1)], -1).astype(
        np.uint8
    )
    pil = torch.from_numpy(
        np.asarray(Image.fromarray(array).resize((252, 252), Image.BICUBIC)).astype(
            np.float32
        )
    ).permute(2, 0, 1)
    source = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    resized = _resize_bicubic_if_needed(source, 252, 252)

    assert resized.shape == (1, 3, 252, 252)
    torch.testing.assert_close(resized, resized.round())
    assert resized.min() >= 0.0 and resized.max() <= 255.0
    # Within a couple of 8-bit levels of PIL; the non-antialiased resize is off
    # by an order of magnitude more, which is the regression this guards.
    assert (resized[0] - pil).abs().max() <= 4.0
    naive = F.interpolate(
        source.float(), size=(252, 252), mode="bicubic", align_corners=False
    )
    assert (naive[0] - pil).abs().max() > 20.0


def test_kimi_resize_is_a_dtype_only_cast_when_already_at_target():
    image = torch.randint(0, 256, (1, 3, 16, 12), dtype=torch.uint8)

    resized = _resize_bicubic_if_needed(image, 16, 12)

    assert resized.dtype == torch.float32
    torch.testing.assert_close(resized, image.float())


def test_kimi_expands_one_placeholder_per_image_from_existing_ids():
    # 7 is the placeholder; the two images claim 3 and 2 tokens.
    input_ids = [1, 7, 2, 7, 3]

    expanded = _expand_image_token_ids(
        input_ids, image_token_id=7, image_token_counts=[3, 2]
    )

    assert expanded.tolist() == [[1, 7, 7, 7, 2, 7, 7, 3]]


def test_kimi_expansion_rejects_a_placeholder_count_mismatch():
    with pytest.raises(ValueError, match="placeholder"):
        _expand_image_token_ids([1, 7, 2], image_token_id=7, image_token_counts=[3, 2])


def test_kimi_expansion_matches_the_base_retokenize_avoidance_rebuild():
    # preserve_processor_input_ids skips the base rebuild, which is only safe
    # while both produce the same sequence. Reference is the original loop.
    def reference(original_ids, counts, placeholder):
        rebuilt, next_image = [], 0
        for token_id in original_ids:
            if token_id == placeholder:
                rebuilt.extend([placeholder] * counts[next_image])
                next_image += 1
            else:
                rebuilt.append(token_id)
        return rebuilt

    rng = np.random.default_rng(0)
    for n_images in (1, 3, 8):
        # Placeholder 7 is below the random range, so only the inserted
        # positions count as placeholders.
        ids = rng.integers(100, 5000, 400).tolist()
        for slot in range(n_images):
            ids.insert(slot * 37 + 5, 7)
        counts = rng.integers(1, 400, n_images).tolist()
        expected = reference(ids, counts, 7)

        assert BaseMultimodalProcessor._expand_input_ids(ids, counts, 7) == expected
        wrapper = _expand_image_token_ids(
            ids, image_token_id=7, image_token_counts=counts
        )
        assert wrapper.flatten().tolist() == expected


def test_kimi_cpu_fallback_keeps_the_request_tokens():
    # preserve_processor_input_ids disables the base rebuild on every path.
    hf_processor = Mock()
    hf_processor.media_processor.media_tokens_calculator = Mock(return_value=3)
    hf_processor.return_value = {"input_ids": torch.tensor([[99, 99, 99]])}

    wrapper = KimiGPUProcessorWrapper.__new__(KimiGPUProcessorWrapper)
    wrapper._hf_processor = hf_processor
    wrapper._image_token = "<|media_pad|>"
    wrapper._image_token_id = 7

    out = wrapper._cpu_call(
        "a<|media_pad|>b", ["img"], original_input_ids=[1, 7, 2], medias=None
    )

    # Not the [99, 99, 99] the HF processor returned.
    assert out["input_ids"].flatten().tolist() == [1, 7, 7, 7, 2]


def test_kimi_cpu_fallback_falls_back_to_the_hf_tokens_without_request_ids():
    hf_processor = Mock()
    hf_processor.media_processor.media_tokens_calculator = Mock(return_value=3)
    hf_processor.return_value = {"input_ids": torch.tensor([[99, 99, 99]])}

    wrapper = KimiGPUProcessorWrapper.__new__(KimiGPUProcessorWrapper)
    wrapper._hf_processor = hf_processor
    wrapper._image_token = "<|media_pad|>"
    wrapper._image_token_id = 7

    out = wrapper._cpu_call("a<|media_pad|>b", ["img"], medias=None)

    assert out["input_ids"].flatten().tolist() == [99, 99, 99]


def test_kimi_refuses_already_normalized_float_pixels():
    with pytest.raises(ValueError, match="uint8"):
        _ensure_chw_rgb(torch.rand(3, 8, 8))


def test_kimi_placeholder_count_only_reads_real_token_ids():
    count = KimiGridMMDataMixin.count_image_placeholders

    assert count([1, 7, 2, 7], 7) == 2
    assert count(torch.tensor([[1, 7, 2]]), 7) == 1
    assert count([1, 2, 3], 7) == 0
    # A prompt string carries no token IDs, so the caller must not take the
    # tokenized fast path.
    assert count("<|media_pad|>", 7) is None


def test_kimi_single_frame_pool_matches_the_temporal_mean():
    torch.manual_seed(0)
    x = torch.randn(1 * 4 * 4, 8)
    grid_thws = torch.tensor([[1, 4, 4]])

    (merged,) = tpool_patch_merger(x, grid_thws)

    # t == 1 skips the mean; it must stay bit-identical to averaging one frame.
    reference = (
        x.view(1, 2, 2, 2, 2, 8).permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
    )
    assert torch.equal(merged, reference.view(4, 4, 8))


def test_kimi_multi_frame_pool_still_averages_across_frames():
    torch.manual_seed(0)
    x = torch.randn(3 * 4 * 4, 8)
    grid_thws = torch.tensor([[3, 4, 4]])

    (merged,) = tpool_patch_merger(x, grid_thws)

    reference = (
        x.view(3, 2, 2, 2, 2, 8).permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
    )
    assert merged.shape == (4, 4, 8)
    torch.testing.assert_close(merged, reference.view(4, 4, 8))


class _IdentityProjector(nn.Module):
    """Stands in for K2VLMultiModalProjector, which is never None in production."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x):
        self.seen = x
        return x


def test_kimi_projection_returns_one_flattened_feature_tensor():
    torch.manual_seed(0)
    per_image = [torch.randn(4, 2, 8), torch.randn(6, 2, 8)]

    packed = mm_projection_auto(_IdentityProjector(), per_image)

    assert packed.shape == (20, 8)
    torch.testing.assert_close(packed, torch.cat(per_image, dim=0).reshape(-1, 8))


def test_kimi_projection_does_not_copy_a_single_image():
    single = torch.randn(4, 2, 8)
    projector = _IdentityProjector()

    packed = mm_projection_auto(projector, [single])

    # The projector must receive the tensor itself, not a one-element cat of it.
    assert projector.seen.data_ptr() == single.data_ptr()
    assert packed.data_ptr() == single.data_ptr()


def test_dp_helper_supports_moonvit3d_packed_embeddings_on_tp1():
    tower = _MoonViT3dTower()
    pixel_values = torch.randn(4, 2)

    # The IPC consumer count asks for the *configured* TP size (matching
    # MmItemMemoryPool.try_to_recycle), so the double publishes one too.
    with get_context().override_server_args(tp_size=1), get_parallel().override(
        tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0
    ):
        output = run_dp_sharded_mrope_vision_model(
            tower, pixel_values, [[1, 2, 2]], rope_type="rope_2d_packed"
        )

    assert torch.equal(output, pixel_values.reshape(1, 4, 2))
    assert torch.equal(tower.grid_thws, torch.tensor([[1, 2, 2]]))


def test_dp_helper_can_lazily_load_kimi_features_on_tp1():
    tower = _MoonViT3dTower()
    pixel_values = torch.randn(4, 2)
    loader = Mock(return_value=pixel_values)

    # The IPC consumer count asks for the *configured* TP size (matching
    # MmItemMemoryPool.try_to_recycle), so the double publishes one too.
    with get_context().override_server_args(tp_size=1), get_parallel().override(
        tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0
    ):
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
    # Single image, so this empty rank takes the broadcast fast path: the
    # buffer it allocates is shaped from config.hidden_size, then filled by
    # the owner rank.
    owner_embedding = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    broadcast_src = []

    class _GatherGroup:
        def all_gather(self, tensor, dim):
            return torch.cat([torch.ones_like(tensor), tensor], dim=dim)

        def broadcast(self, tensor, src):
            broadcast_src.append(src)
            tensor.copy_(owner_embedding)
            return tensor

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
    assert torch.equal(output, owner_embedding)
    assert broadcast_src == [0]
    assert tower.grid_thws is None


def test_dp_helper_broadcasts_a_single_image_from_its_owner_rank():
    broadcast_src = []

    class _GatherGroup:
        def all_gather(self, tensor, dim):
            raise AssertionError("a single image must not reach the all-gather")

        def broadcast(self, tensor, src):
            broadcast_src.append(src)
            return tensor

    tower = _MoonViT3dTower()
    pixel_values = torch.randn(4, 2)
    parallel = SimpleNamespace(
        attn_tp_size=2,
        attn_tp_rank=0,
        attn_tp_group=_GatherGroup(),
    )

    with patch("sglang.srt.multimodal.mm_utils.get_parallel", return_value=parallel):
        output = run_dp_sharded_mrope_vision_model(
            tower,
            pixel_values,
            [[1, 2, 2]],
            rope_type="rope_2d_packed",
        )

    assert torch.equal(output, pixel_values.reshape(1, 4, 2))
    assert broadcast_src == [0]


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


def test_kimi_non_dp_keeps_grid_thws_on_the_host():
    model = KimiK25ForConditionalGeneration.__new__(KimiK25ForConditionalGeneration)
    nn.Module.__init__(model)
    model.use_data_parallel = False
    model.vision_tower = _MoonViT3dTower()
    # Not the host, so a stray .to(tower.device) shows up without a GPU.
    model.vision_tower.device = torch.device("meta")
    model.mm_projector = _IdentityProjector()
    items = [_image_item(torch.randn(4, 2), [[1, 2, 2]])]

    # The IPC consumer count asks for the *configured* TP size (matching
    # MmItemMemoryPool.try_to_recycle), so the double publishes one too.
    with get_context().override_server_args(tp_size=1), get_parallel().override(
        tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0
    ):
        model.get_image_feature(items)

    # A device copy would cost one sync per .tolist() inside MoonViT3d.
    assert model.vision_tower.grid_thws.device.type == "cpu"


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


def test_kimi_lazy_vmm_feature_uses_proxy_consumer_count():
    proxy = CudaIpcTensorTransportProxy.__new__(CudaIpcTensorTransportProxy)
    proxy.consumer_count = 2
    proxy.reconstruct_on_target_device = Mock(return_value=torch.randn(1, 2))
    item = MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)

    item.reconstruct(0, ipc_consumer_count=8)

    proxy.reconstruct_on_target_device.assert_called_once_with(0, consumer_count=2)


def test_kimi_lazy_vmm_cache_hit_uses_proxy_consumer_count():
    proxy = CudaIpcTensorTransportProxy.__new__(CudaIpcTensorTransportProxy)
    proxy.consumer_count = 2
    proxy.acknowledge_consumption = Mock()
    item = MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)

    item.acknowledge_deferred_cuda_ipc_feature(consumer_count=8)

    proxy.acknowledge_consumption.assert_called_once_with(2)


class _Tokenizer:
    def encode(self, text, allowed_special=None):
        tokens = {
            "<|media_begin|>image 1536x1024<|media_content|>": [10, 11],
            "<|media_begin|>image 1024x1536<|media_content|>": [12, 13],
            "<|media_end|>": [14],
        }
        return tokens.get(text, [])


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


@pytest.mark.parametrize(
    ("processor_cls", "wrapper_cls"),
    [
        (KimiK2_5VLImageProcessor, KimiGPUProcessorWrapper),
        (KimiK3ImageProcessor, KimiK3GPUProcessorWrapper),
    ],
)
def test_kimi_processor_workers_clone_the_gpu_wrapper(processor_cls, wrapper_cls):
    server_args = SimpleNamespace(
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
        assert isinstance(processor._processor, wrapper_cls)
        assert isinstance(worker_processor, wrapper_cls)
        assert worker_processor is not processor._processor
    finally:
        processor.mm_processor_executor.shutdown()
        processor.io_executor.shutdown()
        processor.cpu_executor.shutdown()


def test_kimi_k3_expands_image_placeholders_with_original_dimensions():
    actual = _expand_k3_image_prompt_token_ids(
        [1, 99, 2, 99, 3],
        99,
        [3, 2],
        [(1536, 1024), (1024, 1536)],
        _Tokenizer(),
    )

    assert actual.tolist() == [[1, 10, 11, 99, 99, 99, 14, 2, 12, 13, 99, 99, 14, 3]]


def test_kimi_k3_cpu_prompt_uses_the_same_media_contract():
    actual = _expand_k3_image_prompt_text(
        "before<|media_pad|>between<|media_pad|>after",
        "<|media_pad|>",
        [3, 2],
        [(1536, 1024), (1024, 1536)],
    )

    assert actual == (
        "before<|media_begin|>image 1536x1024<|media_content|>"
        "<|media_pad|><|media_pad|><|media_pad|><|media_end|>between"
        "<|media_begin|>image 1024x1536<|media_content|>"
        "<|media_pad|><|media_pad|><|media_end|>after"
    )


def test_kimi_k3_epd_rebuild_uses_the_same_media_contract():
    processor = object.__new__(KimiK3ImageProcessor)
    processor.hf_config = SimpleNamespace(
        vision_config=SimpleNamespace(merge_kernel_size=(2, 2))
    )
    processor.mm_tokens = SimpleNamespace(image_token_id=99)
    processor._tokenizer = _Tokenizer()
    embeddings = {Modality.IMAGE: torch.arange(20, dtype=torch.float32).reshape(5, 4)}

    output = processor.get_mm_data(
        [1, 99, 2, 99, 3],
        embeddings,
        img_grid_thw=torch.tensor([[1, 2, 6], [1, 2, 4]]),
        original_image_sizes=[[1536, 1024], [1024, 1536]],
    )

    assert output.input_ids == [
        1,
        10,
        11,
        99,
        99,
        99,
        14,
        2,
        12,
        13,
        99,
        99,
        14,
        3,
    ]
    assert [item.offsets for item in output.mm_items] == [[(3, 5)], [(10, 11)]]
    torch.testing.assert_close(
        output.mm_items[0].precomputed_embeddings, embeddings[Modality.IMAGE][:3]
    )
    torch.testing.assert_close(
        output.mm_items[1].precomputed_embeddings, embeddings[Modality.IMAGE][3:]
    )


def test_kimi_k3_cpu_transport_defers_gpu_preprocessing():
    from sglang.srt.multimodal.kimi_k3_image_processing import (
        DEFERRED_PREPROCESSING_KEY,
    )

    processor = object.__new__(KimiK3ImageProcessor)
    processor.mm_tokens = SimpleNamespace(image_token_id=99)
    processor.mm_feature_transport = "cpu"
    processor.use_cuda_ipc = False
    processor._processor = SimpleNamespace(
        _patch_size=2,
        prepare_deferred=Mock(
            return_value=(
                torch.tensor([[1, 99, 99, 2, 99, 3]]),
                [
                    {
                        "num_tokens": 2,
                        "new_width": 4,
                        "new_height": 2,
                        "pad_width": 0,
                        "pad_height": 2,
                    },
                    {
                        "num_tokens": 1,
                        "new_width": 2,
                        "new_height": 2,
                        "pad_width": 2,
                        "pad_height": 2,
                    },
                ],
                {
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                    "transparent_bg_config": None,
                },
            )
        ),
    )
    images = [
        torch.arange(3 * 2 * 4, dtype=torch.uint8).reshape(3, 2, 4),
        torch.arange(3 * 2 * 2, dtype=torch.uint8).reshape(3, 2, 2),
    ]
    base_output = SimpleNamespace(
        input_text="prompt", images=images, input_ids=[1, 99, 2, 99, 3]
    )

    output = processor._build_deferred_output(base_output)

    assert output.input_ids == [1, 99, 99, 2, 99, 3]
    assert [item.offsets for item in output.mm_items] == [[(1, 2)], [(4, 4)]]
    assert [item.feature.dtype for item in output.mm_items] == [
        torch.uint8,
        torch.uint8,
    ]
    assert [item.feature.shape for item in output.mm_items] == [
        torch.Size([3, 2, 4]),
        torch.Size([3, 2, 2]),
    ]
    assert [item.image_grid_thw.tolist() for item in output.mm_items] == [
        [[1, 2, 2]],
        [[1, 2, 2]],
    ]
    assert all(item.hash is not None for item in output.mm_items)
    assert all(item.pad_value is not None for item in output.mm_items)
    assert all(
        DEFERRED_PREPROCESSING_KEY in item.model_specific_data
        for item in output.mm_items
    )


@pytest.mark.parametrize(
    ("image_shape", "in_patch_limit", "expected"),
    [((3, 32, 32), 65536, True), ((3, 1024, 1024), 1, False)],
)
def test_kimi_k3_defers_only_when_raw_transport_is_smaller(
    image_shape, in_patch_limit, expected
):
    processor = object.__new__(KimiK3ImageProcessor)
    processor.mm_feature_transport = "cpu"
    processor._processor = SimpleNamespace(
        _patch_size=14,
        _merge_kernel_size=2,
        _in_patch_limit=in_patch_limit,
        _patch_limit_on_one_side=512,
        _fixed_output_tokens=None,
    )
    image = torch.zeros(image_shape, dtype=torch.uint8)

    with patch("sglang.srt.multimodal.processors.kimi_k3.is_cuda", return_value=True):
        assert processor._should_defer_gpu_preprocessing([image]) is expected


def test_kimi_k3_does_not_defer_non_uint8_tensor_preprocessing():
    processor = object.__new__(KimiK3ImageProcessor)
    processor.mm_feature_transport = "cpu"

    with patch("sglang.srt.multimodal.processors.kimi_k3.is_cuda", return_value=True):
        assert not processor._should_defer_gpu_preprocessing(
            [torch.zeros((3, 32, 32), dtype=torch.float32)]
        )


def test_kimi_k3_does_not_defer_empty_image_batch():
    processor = object.__new__(KimiK3ImageProcessor)
    processor.mm_feature_transport = "cpu"

    with patch("sglang.srt.multimodal.processors.kimi_k3.is_cuda", return_value=True):
        assert not processor._should_defer_gpu_preprocessing([])


def test_kimi_k3_eager_preprocessing_preserves_float_tensor_support():
    from sglang.srt.multimodal.processors.kimi_k3 import _k3_to_cuda_chw

    image = torch.zeros((1, 4, 4), dtype=torch.float32)
    with patch.object(torch.Tensor, "cuda", lambda self: self):
        output = _k3_to_cuda_chw(image)

    assert output.dtype == torch.float32
    assert output.shape == (3, 4, 4)


@pytest.mark.parametrize("transport", ["cuda_ipc", "fabric"])
def test_kimi_k3_keeps_gpu_transport_preprocessing_eager(transport):
    processor = object.__new__(KimiK3ImageProcessor)
    processor.mm_feature_transport = transport

    with patch("sglang.srt.multimodal.processors.kimi_k3.is_cuda", return_value=True):
        assert not processor._should_defer_gpu_preprocessing(
            [torch.zeros((3, 32, 32), dtype=torch.uint8)]
        )


def test_kimi_k3_rejects_silently_dropped_images():
    processor = object.__new__(KimiK3ImageProcessor)
    processor.mm_tokens = Mock()
    processor.load_mm_data = AsyncMock(return_value=SimpleNamespace(images=[object()]))

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
    processor.mm_feature_transport = "cpu"
    processor.mm_tokens = SimpleNamespace(image_token_id=99)
    processor.mm_feature_transport = "cuda_ipc"
    processor.use_cuda_ipc = True
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
