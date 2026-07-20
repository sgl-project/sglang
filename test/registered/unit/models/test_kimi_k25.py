"""CPU coverage for Kimi-K2.5/K2.7 encoder-DP wiring."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.multimodal.processors.kimi_k25 import (
    _resize_images_by_source_shape,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.cuda_ipc_transport_utils import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
    CudaIpcTensorTransportProxy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
