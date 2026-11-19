import argparse
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.models.deepseek_v41_vit import ViT
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


@pytest.fixture(autouse=True)
def runtime_context():
    with (
        get_context().override_server_args(),
        get_parallel().override(tp_group=None),
    ):
        yield


@pytest.mark.parametrize("backend", ["triton_attn", None])
@torch.inference_mode()
def test_vision_graph_matches_eager_and_preserves_cached_outputs(backend) -> None:
    torch.manual_seed(0)
    args = SimpleNamespace(
        vision_patch_size=14,
        vision_dim=128,
        vision_n_heads=2,
        vision_inter_dim=256,
        vision_n_layers=2,
        vision_rope_theta=10000.0,
    )
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with get_context().override_server_args(mm_attention_backend=backend):
            vit = ViT(args).cuda().eval()
    finally:
        torch.set_default_dtype(dtype)
    # Parallel linear layers allocate empty weights for checkpoint loading.
    for name, parameter in vit.named_parameters():
        if "norm" in name:
            parameter.fill_(1)
        else:
            parameter.normal_(std=0.02)

    # Exercise padding, exact buckets, changed 2D RoPE at the same patch count,
    # reuse after a larger graph, and eager fallback above the largest bucket.
    grids = [(7, 9), (8, 8), (4, 16), (8, 16), (7, 9), (9, 15)]
    patches = [
        torch.randn(h * w, 3, 14, 14, device="cuda", dtype=torch.bfloat16)
        for h, w in grids
    ]
    expected = [vit(p, h, w) for p, (h, w) in zip(patches, grids, strict=True)]
    vit.capture_cuda_graphs([64, 128])
    results = [vit(p, h, w) for p, (h, w) in zip(patches, grids, strict=True)]
    for actual, reference in zip(results, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0.01, atol=0.01)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        vit(patches[0], *grids[0])
    events = {event.key: event.count for event in profile.key_averages()}
    assert events["cudaGraphLaunch"] == 1
    assert "aten::linear" not in events
    # Subsequent replays must not overwrite results already handed to the cache.
    for actual, reference in zip(results, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0.01, atol=0.01)


@pytest.mark.parametrize(
    "overrides,device,draft,backend,captures",
    [
        ({}, "cuda", False, "fa4", True),
        ({"disable_vision_cuda_graph": True}, "cuda", False, "fa4", False),
        ({"disaggregation_mode": "decode"}, "cuda", False, "fa4", False),
        ({}, "cpu", False, "fa4", False),
        ({}, "cuda", True, "fa4", False),
        ({}, "cuda", False, "sdpa", False),
    ],
)
def test_vision_graph_startup(overrides, device, draft, backend, captures):
    from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
        maybe_capture_vision_encoder_graphs,
    )

    vision = Mock(spec=ViT)
    vision.cuda_graph_runner = None
    vision.blocks = [SimpleNamespace(attn=SimpleNamespace(qkv_backend_name=backend))]
    runner = SimpleNamespace(
        device=device,
        is_draft_worker=draft,
        model=SimpleNamespace(
            vision=vision,
            config=SimpleNamespace(vision_max_n_token=2304, vision_downsample_ratio=2),
        ),
    )
    with get_context().override_server_args(
        vision_cuda_graph_sizes=[64, 128], **overrides
    ):
        maybe_capture_vision_encoder_graphs(runner)
        if captures:
            vision.capture_cuda_graphs.assert_called_once_with([64, 128, 9216])
            vision.cuda_graph_runner = object()
            maybe_capture_vision_encoder_graphs(runner)
            assert vision.capture_cuda_graphs.call_count == 1
        else:
            vision.capture_cuda_graphs.assert_not_called()


def test_vision_graph_cli_and_bucket_validation():
    from sglang.srt.arg_groups.fields.mm import Mm
    from sglang.srt.arg_groups.serving_hook import handle_multimodal
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model-path",
            "unused",
            "--disable-vision-cuda-graph",
            "--vision-cuda-graph-sizes",
            "64",
            "128",
        ]
    )
    assert args.disable_vision_cuda_graph
    assert args.vision_cuda_graph_sizes == [64, 128]
    handle_multimodal(Mm(vision_cuda_graph_sizes=[64, 128]))
    for sizes in ([], [0], [-1], [128, 64], [64, 64]):
        with pytest.raises(ValueError, match="vision_cuda_graph_sizes"):
            handle_multimodal(Mm(vision_cuda_graph_sizes=sizes))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
