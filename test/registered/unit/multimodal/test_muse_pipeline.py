import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.multimodal.media_artifacts.base import MediaArtifactInput
from sglang.srt.multimodal.processors import muse_spark_omni as muse
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class Tokenizer:
    tokens = {
        1: "<image>",
        2: "<video>",
        3: "<audio>",
        10: "<|patch|>",
        11: "<|image_start|>",
        12: "<|image_end|>",
        99: " text ",
        4: " <",
        5: "image",
        6: ">\n",
        7: " ",
        8: "\n",
    }

    def encode(self, text, **kwargs):
        return [{" ": 7, "\n": 8}[char] for char in text]

    def decode(self, ids):
        return "".join(self.tokens[i] for i in ids)


def processor():
    p = object.__new__(muse.MuseSpark1xOmniMultimodalProcessor)
    p._tokenizer = Tokenizer()
    p.image_start_id, p.image_end_id, p.patch_id = 11, 12, 10
    p.video_start_id, p.video_end_id = 21, 22
    p.audio_start_id, p.audio_end_id = 31, 32
    return p


@pytest.mark.parametrize(
    "ids,expected",
    [
        ([11, 10, 12, 99, 1], [11, 10, 12, 99, 11, 10, 10, 12]),
        ([1, 99, 11, 10, 10, 12], [11, 10, 12, 99, 11, 10, 10, 12]),
        ([11, 10, 12, 99, 11, 10, 10, 12], [11, 10, 12, 99, 11, 10, 10, 12]),
        ([1, 99, 1], [11, 10, 12, 99, 11, 10, 10, 12]),
    ],
)
def test_mixed_expanded_media_order(ids, expected):
    result = processor()._expand_modality(
        ids, "<image>", [1], [[11, 10, 12], [11, 10, 10, 12]]
    )
    assert result == expected
    assert muse._find_runs(result, 10, [[1], [2]]) == [[(1, 1)], [(5, 6)]]


@pytest.mark.parametrize("height", [1, 3, 4, 8])
def test_numpy_hwc_matches_pil(height):
    image = np.arange(height * 100 * 3, dtype=np.uint8).reshape(height, 100, 3)
    assert torch.equal(
        muse._image_to_uint8(image), muse._image_to_uint8(Image.fromarray(image))
    )


def test_presampled_video_uses_source_duration():
    frames, timestamps = muse._video_frames(
        (
            list(range(96)),
            {"fps": 30, "frames_indices": np.linspace(0, 1499, 96, dtype=int).tolist()},
        ),
        muse.PreprocessConfig(),
    )
    assert len(frames) == 96
    assert timestamps[-1] == pytest.approx(1499 / 30)


def test_precomputed_input_has_clear_error():
    with pytest.raises(ValueError, match="does not yet support"):
        asyncio.run(
            processor().process_mm_data_async(
                image_data=[
                    {"format": "precomputed_embedding", "feature": torch.zeros(1, 8)}
                ]
            )
        )


@pytest.mark.parametrize(
    "device,expected", [(None, True), ("cuda", True), ("cpu", False)]
)
def test_video_loader_selects_device(device, expected):
    with (
        patch.object(BaseMultimodalProcessor, "video_preprocessing_device", device),
        patch("sglang.srt.multimodal.processors.base_processor.load_video") as load,
    ):
        BaseMultimodalProcessor._load_single_item("video.mp4", Modality.VIDEO, None)
    load.assert_called_once_with("video.mp4", use_gpu=expected)


def test_muse_video_decode_stays_on_cpu():
    with patch("sglang.srt.multimodal.processors.base_processor.load_video") as load:
        muse.MuseSpark1xOmniMultimodalProcessor._load_single_item(
            "video.mp4", Modality.VIDEO, 96
        )
    load.assert_called_once_with("video.mp4", use_gpu=False)


def test_video_outside_chunk_does_not_encode():
    item = SimpleNamespace(
        hash=8, offsets=[(100, 101), (105, 106)], feature=torch.zeros(2, 3, 4, 4)
    )
    encoder = Mock()
    with patch.object(mm_schedule, "embedding_cache", MultiModalStaticCache(10000)):
        result, _ = mm_schedule._get_chunked_prefill_embedding(
            encoder,
            [item],
            [0, 1],
            [0],
            [10],
            [item.offsets],
            torch.zeros(10, dtype=torch.long),
        )
    encoder.assert_not_called()
    assert result is None or result.numel() == 0


def test_batched_cache_hits_acknowledge_each_consumer():
    cache = MultiModalStaticCache(10000)
    cache.set(7, EmbeddingResult(embedding=torch.zeros(2, 4)))
    items = [SimpleNamespace(hash=7, offsets=[(0, 1)], feature=None) for _ in range(2)]
    requests = [
        mm_schedule.PerImageRequestInfo(i, [item], [(0, 1)], 0, 2)
        for i, item in enumerate(items)
    ]
    with (
        patch.object(mm_schedule, "embedding_cache", cache),
        patch.object(mm_schedule, "_acknowledge_deferred_cuda_ipc_cache_hits") as ack,
    ):
        mm_schedule._batch_encode_per_image_misses(
            Mock(side_effect=AssertionError), requests, torch.device("cpu")
        )
    assert [call.args[0][0] for call in ack.call_args_list] == items


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_cache_miss_keeps_device_feature():
    p = processor()
    p.preprocess_config = muse.PreprocessConfig(
        use_gpu_preprocess=True, use_rust_resize=False
    )
    p.server_args = SimpleNamespace(base_gpu_id=0)
    p.mm_feature_transport = "cuda_ipc"
    entry = MediaArtifactInput(
        "sha256:" + "0" * 64,
        "sha256:" + "1" * 64,
        Modality.IMAGE,
        Image.new("RGB", (56, 56), "red"),
    )
    (artifact,) = p.prepare_artifact_batch([entry])
    assert artifact.feature.is_cuda
    cached = artifact.cache_value()
    assert cached.feature.device.type == "cpu"
    assert torch.equal(artifact.feature.cpu(), cached.feature)
    assert artifact.token_count == cached.token_count


def test_bpe_merged_placeholder_precedes_exact_pattern():
    result = processor()._expand_modality(
        [4, 5, 6, 99, 1], "<image>", [1], [[11, 10, 12], [11, 10, 10, 12]]
    )
    assert result == [7, 11, 10, 12, 8, 99, 11, 10, 10, 12]


def test_cache_decode_preserves_numpy_layout():
    image = np.zeros((4, 100, 3), dtype=np.uint8)
    decoded = processor().decode_media_snapshot(
        SimpleNamespace(data=image), Modality.IMAGE
    )
    actual, count = muse.preprocess_image(
        decoded, muse.PreprocessConfig(use_rust_resize=False)
    )
    expected, expected_count = muse.preprocess_image(
        Image.fromarray(image), muse.PreprocessConfig(use_rust_resize=False)
    )
    assert count == expected_count
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("markers", [1, 16])
def test_exact_placeholders_decode_segment_once(markers):
    p = processor()
    p._tokenizer.decode = Mock(wraps=p._tokenizer.decode)
    ids = [99] * 1000 + [1, 99] * markers
    replacements = [[11, 10, 12]] * markers
    result = p._expand_modality(ids, "<image>", [1], replacements)
    assert result == [99] * 1000 + [11, 10, 12, 99] * markers
    assert p._tokenizer.decode.call_count == 1


def test_multiple_merged_placeholders_preserve_surrounding_tokens():
    ids = [99, 4, 5, 6, 99, 4, 5, 6, 99]
    result = processor()._expand_modality(
        ids, "<image>", [1], [[11, 10, 12], [11, 10, 10, 12]]
    )
    assert result == [99, 7, 11, 10, 12, 8, 99, 7, 11, 10, 10, 12, 8, 99]


@pytest.mark.parametrize("ids", [[1, 1], [11, 10, 12, 1], [11, 10]])
def test_invalid_placeholder_count_or_unterminated_span(ids):
    with pytest.raises(ValueError):
        processor()._expand_modality(ids, "<image>", [1], [[11, 10, 12]])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
