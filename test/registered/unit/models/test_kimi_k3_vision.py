from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.vision import (
    prepare_flashinfer_cudnn_vision_attention_metadata,
)
from sglang.srt.models import kimi_k3_vl
from sglang.srt.models.kimi_k3_vl import (
    KimiK3VisionTower,
    MoonViT3dEncoder,
    _resolve_mm_attention_backend,
    interpolate_pos_emb,
    sdpa_varlen_attention,
)
from sglang.srt.multimodal import kimi_k3_vit_cuda_graph_runner
from sglang.srt.multimodal.kimi_k3_vit_cuda_graph_runner import (
    KimiK3ViTCudaGraphRunner,
)
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    (
        "configured_backend",
        "device_type",
        "capability",
        "max_seqlen",
        "total_tokens",
        "fa4_available",
        "expected",
    ),
    [
        ("sdpa", "cuda", (10, 3), 8192, 8192, True, "sdpa"),
        ("auto", "cpu", None, 1024, 1024, True, "sdpa"),
        ("auto", "cuda", (10, 0), 1024, 1024, True, "sdpa"),
        ("auto", "cuda", (10, 3), 1536, 1536, True, "triton_attn"),
        ("auto", "cuda", (10, 3), 1600, 1600, True, "fa4"),
        ("auto", "cuda", (10, 3), 1024, 4096, True, "fa4"),
        ("auto", "cuda", (10, 3), 1536, 1536, False, "triton_attn"),
        ("auto", "cuda", (10, 3), 1600, 1600, False, "sdpa"),
    ],
)
def test_kimi_k3_resolves_shape_aware_attention_backend(
    monkeypatch,
    configured_backend,
    device_type,
    capability,
    max_seqlen,
    total_tokens,
    fa4_available,
    expected,
):
    if capability is not None:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: capability)

    actual = _resolve_mm_attention_backend(
        configured_backend,
        max_seqlen=max_seqlen,
        total_tokens=total_tokens,
        device=torch.device(device_type),
        fa4_available=fa4_available,
    )

    assert actual == expected


def test_kimi_k3_skips_attention_precompile_on_cpu():
    encoder = MoonViT3dEncoder(
        hidden_dim=8,
        num_layers=1,
        block_cfg={
            "num_heads": 1,
            "hidden_dim": 8,
            "qkv_hidden_size": 8,
            "mlp_dim": 16,
            "norm_type": "rmsnorm",
            "activation": F.gelu,
            "attn_bias": False,
            "linear_bias": False,
        },
    )

    assert not encoder.precompile_attention_backend(torch.bfloat16, torch.device("cpu"))


def test_kimi_k3_sdpa_reuses_prepared_segment_bounds():
    class SeqlensThatMustNotSync:
        def tolist(self):
            raise AssertionError("prepared segment bounds must avoid tensor.tolist()")

    q = torch.randn(4, 1, 8)
    k = torch.randn(4, 1, 8)
    v = torch.randn(4, 1, 8)
    bounds = ((0, 2), (2, 4))
    expected = sdpa_varlen_attention(q, k, v, torch.tensor([0, 2, 4]))
    actual = sdpa_varlen_attention(
        q,
        k,
        v,
        SeqlensThatMustNotSync(),
        segment_bounds=bounds,
    )

    assert torch.equal(actual, expected)


def test_kimi_k3_vision_tower_reuses_prepared_forward_metadata(monkeypatch):
    config = SimpleNamespace(
        patch_size=2,
        init_pos_emb_height=2,
        init_pos_emb_width=2,
        init_pos_emb_time=1,
        pos_emb_type="divided_fixed",
        pos_emb_interpolation_mode="bilinear",
        patch_embed_proj_bias=False,
        merge_kernel_size=(1, 1),
        merge_type="sd2_tpool",
        vt_hidden_size=8,
        vt_num_attention_heads=1,
        vt_num_hidden_layers=0,
        num_hidden_layers=0,
        vt_intermediate_size=16,
        qkv_hidden_size=8,
        norm_type="rmsnorm",
        activation_func="gelu_pytorch_tanh",
        attn_bias=False,
        linear_bias=False,
    )
    tower = KimiK3VisionTower(config).eval()
    pixel_values = torch.randn(4, 3, 2, 2)
    grid_thws = torch.tensor([[1, 2, 2]])
    grid_thw_list = ((1, 2, 2),)
    reference = tower(pixel_values, grid_thws)
    metadata = tower.prepare_forward_metadata(
        grid_thws,
        grid_thw_list=grid_thw_list,
        total_tokens=pixel_values.shape[0],
        dtype=pixel_values.dtype,
    )
    assert metadata.position_embeddings is not None

    def fail_reprepare(**_):
        raise AssertionError("forward metadata must be reused")

    def fail_position_recompute(*_args, **_kwargs):
        raise AssertionError("prepared position embeddings must be reused")

    monkeypatch.setattr(tower.encoder, "prepare_forward_metadata", fail_reprepare)
    monkeypatch.setattr(
        tower.patch_embed.pos_emb,
        "position_embeddings",
        fail_position_recompute,
    )
    actual = tower(
        pixel_values,
        grid_thws,
        grid_thw_list=grid_thw_list,
        forward_metadata=metadata,
    )

    assert len(actual) == len(reference) == 1
    torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0, equal_nan=True)


def test_kimi_k3_dp_helper_passes_host_grid_list_to_capable_tower():
    class RecordingTower:
        def __call__(
            self,
            pixel_values,
            *,
            grid_hw,
            max_seqlen,
            grid_thw_list,
        ):
            self.grid_hw = grid_hw
            self.max_seqlen = max_seqlen
            self.grid_thw_list = grid_thw_list
            return [pixel_values.unsqueeze(0)]

    tower = RecordingTower()
    pixels = torch.randn(4, 2)
    grids = [[1, 2, 2]]
    with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0):
        output = run_dp_sharded_mrope_vision_model(
            tower,
            pixels,
            grids,
            rope_type="rope_2d",
            pass_grid_thw_list=True,
        )

    assert tower.grid_hw.device == pixels.device
    assert tower.max_seqlen == 4
    assert tower.grid_thw_list is grids
    assert torch.equal(output, pixels.unsqueeze(0))


def test_kimi_k3_vit_graph_runner_bounds_shape_observations():
    runner = KimiK3ViTCudaGraphRunner(object(), capacity=2, min_hits=2)

    for index in range(20):
        runner._record_hit(((1, index + 1, 2),))

    assert len(runner.seen) == 16
    assert ((1, 1, 2),) not in runner.seen
    assert ((1, 20, 2),) in runner.seen


def test_kimi_k3_vit_graph_runner_skips_eager_on_capture(monkeypatch):
    class Tower:
        def prepare_forward_metadata(self, *_args, **_kwargs):
            return object()

    runner = KimiK3ViTCudaGraphRunner(Tower(), capacity=1, min_hits=1)
    pixels = torch.randn(4, 2)
    grids = torch.tensor([[1, 2, 2]])
    replayed = []

    def fail_eager(*_args, **_kwargs):
        raise AssertionError("the capture request must not run eager first")

    def capture(_key, pixel_values, _grid_thws, _grid_thw_list, metadata):
        return SimpleNamespace(
            graph=SimpleNamespace(replay=lambda: replayed.append(True)),
            input_buffer=torch.empty_like(pixel_values),
            outputs=(pixel_values.clone(),),
            metadata=metadata,
        )

    monkeypatch.setattr(runner, "_run_eager", fail_eager)
    monkeypatch.setattr(runner, "_capture", capture)
    outputs = runner.run(pixels, grids, ((1, 2, 2),))

    assert replayed == [True]
    assert torch.equal(outputs[0], pixels)


def test_kimi_k3_vit_graph_runner_uses_eager_above_max_seqlen(monkeypatch):
    runner = KimiK3ViTCudaGraphRunner(object(), capacity=1, min_hits=1, max_seqlen=4)
    pixels = torch.randn(8, 2)
    grids = torch.tensor([[1, 2, 4]])
    eager_calls = []

    def run_eager(pixel_values, *_args, **_kwargs):
        eager_calls.append(True)
        return [pixel_values.clone()], object()

    monkeypatch.setattr(runner, "_run_eager", run_eager)
    monkeypatch.setattr(
        runner,
        "_capture",
        lambda *_args, **_kwargs: pytest.fail("large shapes must not be captured"),
    )

    for _ in range(3):
        outputs = runner.run(pixels, grids, ((1, 2, 4),))

    assert eager_calls == [True, True, True]
    assert torch.equal(outputs[0], pixels)
    assert not runner.seen
    assert not runner.graphs


def test_kimi_k3_vit_graph_runner_reuses_global_graph_pool(monkeypatch):
    class Tower:
        def _forward_eager(self, pixel_values, *_args, **_kwargs):
            return [pixel_values.clone()]

    pool = object()
    graph = object()
    pool_creations = []
    capture_pools = []

    def get_pool(device_module):
        pool_creations.append(device_module)
        return pool

    def graph_context(captured_graph, *, pool):
        assert captured_graph is graph
        capture_pools.append(pool)
        return nullcontext()

    monkeypatch.setattr(
        kimi_k3_vit_cuda_graph_runner,
        "get_or_create_global_graph_memory_pool",
        get_pool,
    )
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(torch.cuda, "graph", graph_context)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    runner = KimiK3ViTCudaGraphRunner(Tower(), capacity=2, min_hits=1)
    pixels = torch.randn(4, 2)
    grids = torch.tensor([[1, 2, 2]])
    metadata = object()
    runner._capture(((1, 2, 2),), pixels, grids, ((1, 2, 2),), metadata)
    runner._capture(((1, 1, 4),), pixels, grids, ((1, 1, 4),), metadata)

    assert pool_creations == [torch.cuda]
    assert capture_pools == [pool, pool]


@pytest.mark.parametrize(
    ("capacity", "min_hits", "max_seqlen"),
    [(0, 1, None), (1, 0, None), (1, 1, 0)],
)
def test_kimi_k3_vit_graph_runner_rejects_invalid_limits(
    capacity, min_hits, max_seqlen
):
    with pytest.raises(ValueError):
        KimiK3ViTCudaGraphRunner(
            object(),
            capacity=capacity,
            min_hits=min_hits,
            max_seqlen=max_seqlen,
        )


def test_kimi_k3_position_interpolation_uses_contiguous_chw(monkeypatch):
    weight = torch.randn(4, 5, 8)
    output_size = (3, 7)
    expected = (
        F.interpolate(
            weight.permute(2, 0, 1).unsqueeze(0),
            size=output_size,
            mode="bilinear",
        )
        .squeeze(0)
        .permute(1, 2, 0)
        .flatten(end_dim=1)
    )
    original_interpolate = F.interpolate
    captured = {}

    def capture_layout(input_tensor, *args, **kwargs):
        captured["is_contiguous"] = input_tensor.is_contiguous()
        captured["stride"] = input_tensor.stride()
        return original_interpolate(input_tensor, *args, **kwargs)

    monkeypatch.setattr(kimi_k3_vl.F, "interpolate", capture_layout)
    actual = interpolate_pos_emb(weight, "bilinear", output_size)

    assert captured == {
        "is_contiguous": True,
        "stride": (160, 20, 5, 1),
    }
    assert torch.equal(actual, expected)


def test_kimi_k3_prepares_shared_attention_metadata_once(monkeypatch, request):
    metadata_ids = []
    values_are_contiguous = []

    class FakeAttention(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, q, k, v, *, forward_metadata, **kwargs):
            metadata_ids.append(id(forward_metadata))
            values_are_contiguous.append(v.is_contiguous())
            return q

    # Force the backend through the context, not by patching an import binding:
    # production reads the published config bag (get_mm().mm_attention_backend).
    override = get_context().override_server_args(
        mm_attention_backend="flashinfer_cudnn"
    )
    override.install()
    request.addfinalizer(override.restore)
    monkeypatch.setitem(kimi_k3_vl.QKV_BACKEND_IMPL, "flashinfer_cudnn", FakeAttention)

    encoder = MoonViT3dEncoder(
        hidden_dim=8,
        num_layers=2,
        block_cfg={
            "num_heads": 1,
            "hidden_dim": 8,
            "qkv_hidden_size": 8,
            "mlp_dim": 16,
            "norm_type": "rmsnorm",
            "activation": F.gelu,
            "attn_bias": False,
            "linear_bias": False,
        },
    )
    output = encoder(torch.randn(4, 8), torch.tensor([[1, 2, 2]]))

    assert output.shape == (4, 8)
    assert len(metadata_ids) == 2
    assert len(set(metadata_ids)) == 1
    assert values_are_contiguous == [True, True]


def test_flashinfer_cudnn_metadata_uses_bucketed_element_indptrs():
    metadata = prepare_flashinfer_cudnn_vision_attention_metadata(
        torch.tensor([0, 480, 1200], dtype=torch.int32),
        device=torch.device("cpu"),
        elem_per_token=1536,
    )

    expected_indptr = torch.tensor(
        [0, 480 * 1536, 1200 * 1536] + [1200 * 1536] * 6,
        dtype=torch.int32,
    )
    assert torch.equal(metadata.packed_indptrs, expected_indptr.repeat(3))
    assert metadata.sequence_lengths.flatten().tolist() == [480, 720] + [0] * 6
    assert metadata.flashinfer_max_seqlen == 4096


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


class _K3TowerStub:
    device = torch.device("cpu")
    merge_kernel_size = (2, 2)
    patch_size = 2

    def __init__(self):
        self.config = SimpleNamespace(hidden_size=2)
        self.patch_embed = SimpleNamespace(
            proj=SimpleNamespace(weight=torch.empty(1, dtype=torch.float32))
        )


def test_kimi_k3_encoder_dp_defers_feature_materialization(monkeypatch):
    """K3 vision is image-wise DP: the DP runner must receive lazy features
    (pixel_values=None + a loader), and the loader must materialize exactly
    the requested images on the owner rank with the tower dtype."""
    from unittest.mock import patch as mock_patch

    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
    from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

    model = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.use_data_parallel = True
    model.vision_tower = _K3TowerStub()
    model.mm_projector = lambda image_embeds: image_embeds

    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 1)],
            feature=torch.randn(4, 2, dtype=torch.float64),
            model_specific_data={"grid_thws": torch.tensor([[1, 2, 2]])},
        ),
        MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(1, 2)],
            feature=torch.randn(4, 2, dtype=torch.float64),
            model_specific_data={"grid_thws": torch.tensor([[1, 2, 2]])},
        ),
    ]
    sharded_embeddings = torch.randn(2, 2)

    # The IPC consumer count asks for the *configured* TP size (matching
    # MmItemMemoryPool.try_to_recycle), so publish it; the live topology the
    # sharding helper reads is forced through the context's own override.
    with mock_patch(
        "sglang.srt.multimodal.mm_utils.run_dp_sharded_mrope_vision_model",
        return_value=sharded_embeddings,
    ) as run_dp, get_context().override_server_args(tp_size=1), get_parallel().override(
        tp_size=1, attn_tp_size=1
    ):
        output = model.get_image_feature(items)
        # Exercise the loader while the runtime topology is forced.
        loader_in_scope = run_dp.call_args.kwargs["load_local_pixel_values"]
        local = loader_in_scope([1])
        both = loader_in_scope([0, 1])

    assert output is sharded_embeddings
    tower, pixel_values, grid_thws = run_dp.call_args.args
    assert tower is model.vision_tower
    assert pixel_values is None
    assert grid_thws == [[1, 2, 2], [1, 2, 2]]
    assert run_dp.call_args.kwargs["rope_type"] == "rope_2d"
    assert run_dp.call_args.kwargs["pass_grid_thw_list"] is True
    assert run_dp.call_args.kwargs["pool_temporal_dimension"] is True
    loader = run_dp.call_args.kwargs["load_local_pixel_values"]
    assert callable(loader)

    # Owner-rank materialization: only the requested image, tower dtype.
    assert local.shape == (4, 2)
    assert local.dtype == torch.float32
    assert torch.equal(local, items[1].feature.to(torch.float32))

    assert both.shape == (8, 2)
    assert torch.equal(
        both,
        torch.cat([items[0].feature, items[1].feature]).to(torch.float32),
    )


def test_kimi_k3_preprocesses_only_dp_owner_images(monkeypatch):
    """A vision-DP owner uses each assigned image's grid when preprocessing."""
    from unittest.mock import patch as mock_patch

    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
    from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
    from sglang.srt.multimodal.kimi_k3_image_processing import (
        DEFERRED_PREPROCESSING_KEY,
        KimiK3DeferredPreprocessing,
    )

    model = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.use_data_parallel = True
    model.vision_tower = _K3TowerStub()
    model.mm_projector = lambda image_embeds: image_embeds

    deferred_config = KimiK3DeferredPreprocessing(
        backend="gpu",
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        transparent_bg_config=None,
        resize_config={
            "num_tokens": 1,
            "new_width": 2,
            "new_height": 2,
            "pad_width": 0,
            "pad_height": 0,
        },
    )
    grids = [[1, 1, 1], [1, 1, 2]]
    patch_counts = [grid[0] * grid[1] * grid[2] for grid in grids]
    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(index, index)],
            feature=torch.full((3, 2, 2), index, dtype=torch.uint8),
            model_specific_data={
                "image_grid_thw": torch.tensor([grids[index]]),
                DEFERRED_PREPROCESSING_KEY: deferred_config,
            },
        )
        for index in range(2)
    ]
    calls = []

    def fake_preprocess(images, resize_configs, *args, **kwargs):
        ids = [int(image[0, 0, 0]) for image in images]
        calls.append(ids)
        pixel_values = torch.cat(
            [torch.full(size=(patch_counts[i], 2), fill_value=float(i)) for i in ids]
        )
        return pixel_values, torch.tensor([grids[i] for i in ids])

    # Configured TP size (the IPC consumer count) comes from the published
    # bags; the live topology is forced through the context's own override.
    with mock_patch(
        "sglang.srt.multimodal.mm_utils.run_dp_sharded_mrope_vision_model",
        return_value=torch.zeros(1, 2),
    ) as run_dp, get_context().override_server_args(tp_size=1), get_parallel().override(
        tp_size=1, attn_tp_size=1
    ), mock_patch(
        "sglang.srt.multimodal.processors.kimi_k25._gpu_preprocess_images",
        side_effect=fake_preprocess,
    ):
        model.get_image_feature(items)
        loader = run_dp.call_args.kwargs["load_local_pixel_values"]
        one = loader([1])

    assert calls == [[1]]
    assert one.dtype == torch.float32
    assert one.shape == (2, 2)
    assert (one == 1.0).all()


def test_kimi_k3_scheduler_leaves_feature_placement_to_dp_owner():
    from sglang.srt.managers.mm_schedule import _can_skip_pre_embed_feature_move
    from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

    model = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
    torch.nn.Module.__init__(model)

    assert _can_skip_pre_embed_feature_move(model.get_image_feature)


def test_kimi_k3_rejects_aggregated_items():
    """One item must carry exactly one logical image: the DP owner
    assignment and the bounded CUDA-IPC lease accounting are per-item, so
    aggregated encoder inputs must be split upstream (EPD encode server)."""
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
    from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

    model = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.use_data_parallel = True
    model.vision_tower = _K3TowerStub()
    model.mm_projector = lambda image_embeds: image_embeds

    aggregated = MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(0, 2)],
        feature=torch.arange(12 * 2, dtype=torch.float64).reshape(12, 2),
        model_specific_data={"grid_thws": torch.tensor([[1, 2, 2], [1, 2, 4]])},
    )

    with pytest.raises(ValueError, match="one vision grid per MultimodalDataItem"):
        model.get_image_feature([aggregated])
