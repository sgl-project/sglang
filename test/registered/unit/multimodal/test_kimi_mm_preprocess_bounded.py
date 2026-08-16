"""Bounded GPU image preprocessing (Kimi K2.5/K3).

``_gpu_preprocess_images`` processes size-groups in byte-capped sub-batches
and streams each image's patches through an optional sink instead of
materializing a request-wide concat, so peak GPU memory no longer scales
with the number of images in a request. These tests pin bit-exact parity
with the unchunked pipeline, hash stability, and the memory bound itself.
"""

import types
from collections import defaultdict

import pytest
import torch
from PIL import Image

from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.multimodal.processors.kimi_k3 import (
    _fill_transparent_bg,
    _k3_to_cuda_chw,
)
from sglang.srt.multimodal.processors.kimi_k25 import (
    MMFeatureStreamSink,
    _gpu_preprocess_images,
    _grid_thw_from_resize_config,
    _process_single_image,
    _resize_images_by_source_shape,
    _to_cuda_chw,
    navit_resize_config,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

PATCH_SIZE = 14


def _norm_tensors():
    scale = torch.full((3, 1, 1), 1.0 / 127.5, device="cuda")
    bias = torch.full((3, 1, 1), -1.0, device="cuda")
    return scale, bias


def _image(width: int, height: int, seed: int, mode: str = "RGB") -> Image.Image:
    generator = torch.Generator().manual_seed(seed)
    arr = torch.randint(
        0, 256, (height, width, len(mode)), generator=generator, dtype=torch.uint8
    )
    return Image.fromarray(arr.numpy(), mode=mode)


def _configs(images):
    return [
        navit_resize_config(*img.size, PATCH_SIZE, 2, 65536, 512, None)
        for img in images
    ]


def _unchunked_reference(images, configs, scale, bias, to_chw, post_resize=None):
    """The previous pipeline: whole-group batches, request-wide concat."""
    from sglang.kernels.ops.mm.process import normalize_and_patchify

    groups = defaultdict(list)
    for idx, (image, config) in enumerate(zip(images, configs)):
        padded_h = config["new_height"] + config["pad_height"]
        padded_w = config["new_width"] + config["pad_width"]
        key = (config["new_height"], config["new_width"], padded_h, padded_w)
        groups[key].append((idx, image, config))

    all_patches = [None] * len(images)
    all_grids = [None] * len(images)
    for (target_h, target_w, padded_h, padded_w), group in groups.items():
        if len(group) == 1:
            idx, image, config = group[0]
            all_patches[idx] = _process_single_image(
                image, config, scale, bias, PATCH_SIZE, to_chw, post_resize
            )
            all_grids[idx] = _grid_thw_from_resize_config(config, PATCH_SIZE)
            continue
        indexed = [(idx, to_chw(image)) for idx, image, _ in group]
        resized = _resize_images_by_source_shape(indexed, target_h, target_w)
        if post_resize is not None:
            resized = [post_resize(part) for part in resized]
        batch = normalize_and_patchify(
            torch.cat(resized, dim=0), scale, bias, PATCH_SIZE, padded_h, padded_w
        )
        grid = (1, padded_h // PATCH_SIZE, padded_w // PATCH_SIZE)
        for i, (idx, _, _) in enumerate(group):
            all_patches[idx] = batch[i]
            all_grids[idx] = grid
    return torch.cat(all_patches, dim=0), torch.tensor(all_grids, dtype=torch.int64)


def _assert_parity(images, to_chw=_to_cuda_chw, post_resize=None, **kwargs):
    configs = _configs(images)
    scale, bias = _norm_tensors()
    reference, ref_grids = _unchunked_reference(
        images, configs, scale, bias, to_chw, post_resize
    )
    entries, grids = _gpu_preprocess_images(
        images,
        configs,
        scale,
        bias,
        PATCH_SIZE,
        to_chw=to_chw,
        post_resize=post_resize,
        **kwargs,
    )
    assert torch.equal(torch.cat(entries, dim=0), reference)
    assert torch.equal(grids, ref_grids)
    for entry in entries:
        # Independent copies, not views pinning sub-batch storage.
        assert entry.untyped_storage().nbytes() == entry.nbytes
    return reference, grids


def test_parity_mixed_size_groups():
    _assert_parity(
        [
            _image(640, 480, seed=0),
            _image(640, 480, seed=1),
            _image(1024, 768, seed=2),
            _image(333, 517, seed=3),
            _image(640, 480, seed=4),
        ]
    )


def test_parity_single_group_chunked():
    images = [_image(896, 896, seed=i) for i in range(7)]
    config = _configs(images)[0]
    per_image_bytes = (
        (config["new_height"] + config["pad_height"])
        * (config["new_width"] + config["pad_width"])
        * 12
    )
    _assert_parity(images, chunk_bytes=2 * per_image_bytes)


def test_parity_rgba_compositing():
    bg_config = {
        "pattern": "chessboard",
        "chessboard_square_size": 8,
        "chessboard_square_on_top_left": True,
        "chessboard_white_value": 255,
        "chessboard_gray_value": 180,
    }
    _assert_parity(
        [
            _image(512, 512, seed=10, mode="RGBA"),
            _image(512, 512, seed=11),
            _image(512, 512, seed=12, mode="RGBA"),
        ],
        to_chw=_k3_to_cuda_chw,
        post_resize=lambda x: _fill_transparent_bg(x, bg_config),
        chunk_bytes=1,  # one image per sub-batch: the strictest chunking
    )


def test_sink_streams_in_order_with_stable_hashes():
    """Sink outputs land at their original indices, off-GPU, and the
    production-time hashes equal hashing the post-split slices they replace."""
    images = [_image(448, 448, seed=i) for i in range(5)] + [_image(640, 480, seed=9)]
    configs = _configs(images)
    scale, bias = _norm_tensors()

    sink = MMFeatureStreamSink(types.SimpleNamespace(use_cuda_ipc=False))
    entries, grids = _gpu_preprocess_images(
        images, configs, scale, bias, PATCH_SIZE, per_image_sink=sink
    )
    assert all(not entry.is_cuda for entry in entries)

    reference, _ = _unchunked_reference(images, configs, scale, bias, _to_cuda_chw)
    assert torch.equal(torch.cat([e.cuda() for e in entries], dim=0), reference)

    hashes = sink.hash_list(len(images))
    assert hashes is not None
    start = 0
    for i, grid in enumerate(grids):
        count = int(torch.prod(grid).item())
        assert hashes[i] == hash_feature(reference[start : start + count])
        start += count


def test_peak_memory_bounded():
    """With an off-GPU sink, peak allocation tracks the chunk size; the
    unchunked pipeline's peak scales with 2x the whole request."""
    images = [_image(1344, 1344, seed=i) for i in range(8)]
    configs = _configs(images)
    scale, bias = _norm_tensors()
    per_image_bytes = (
        (configs[0]["new_height"] + configs[0]["pad_height"])
        * (configs[0]["new_width"] + configs[0]["pad_width"])
        * 12
    )
    chunk_bytes = 2 * per_image_bytes

    def measure(fn):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        result = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        del result
        return peak

    unchunked_peak = measure(
        lambda: _unchunked_reference(images, configs, scale, bias, _to_cuda_chw)
    )
    bounded_peak = measure(
        lambda: _gpu_preprocess_images(
            images,
            configs,
            scale,
            bias,
            PATCH_SIZE,
            per_image_sink=lambda index, patches: patches.cpu(),
            chunk_bytes=chunk_bytes,
        )
    )

    assert unchunked_peak >= 2 * per_image_bytes * len(images)
    assert bounded_peak <= 4 * chunk_bytes
