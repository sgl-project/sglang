"""MiMo-V2 fixtures for the native Rust multimodal parity suites.

Named `_mimo_fixtures` (not `_fixtures`) so it cannot collide with the qwen
suite's helper on the process-global ``sys.path``.

Builds a ``MiMoV2Processor`` on a tiny word-level tokenizer plus a
``SimpleNamespace`` hf_config shaped like MiMo-V2.5's (``processor_config`` /
``vision_config`` dicts, no ``rope_scaling.mrope_section`` → 1-D rope), so the
production spec-resolution path runs without the real checkpoint.
"""

from types import SimpleNamespace

import numpy as np
from tokenizers import Tokenizer, decoders
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.multimodal.processors.mimo_v2 import MiMoV2Processor  # noqa: E402

register_cpu_ci(est_time=0, suite="base-a-test-cpu", disabled="MiMo test fixtures")

VOCAB = [
    "<unk>",
    "<|vision_start|>",
    "<|image_pad|>",
    "<|vision_end|>",
    "hello",
    "<|video_pad|>",
    "<|mimo_audio_start|>",
    "<|audio_pad|>",
    "<|mimo_audio_end|>",
    "<|video_start|>",
    "<|video_end|>",
    "<pad>",
]

# MiMo-V2.5's real image geometry (config.json processor_config/vision_config).
MIMO_PROCESSOR_CONFIG = dict(
    image_token_id=VOCAB.index("<|image_pad|>"),
    vision_start_token_id=VOCAB.index("<|vision_start|>"),
    vision_end_token_id=VOCAB.index("<|vision_end|>"),
    video_token_id=VOCAB.index("<|video_pad|>"),
    video_start_token_id=VOCAB.index("<|video_start|>"),
    video_end_token_id=VOCAB.index("<|video_end|>"),
    audio_token_id=VOCAB.index("<|audio_pad|>"),
    audio_start_token_id=VOCAB.index("<|mimo_audio_start|>"),
    audio_end_token_id=VOCAB.index("<|mimo_audio_end|>"),
    audio_sampling_rate=24000,
    image_min_pixels=8192,
    image_max_pixels=8388608,
    use_video_timestamps=True,
)
MIMO_VISION_CONFIG = dict(
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    tokens_per_second=2,
)


def make_processor():
    backend = Tokenizer(
        WordLevel(
            {token: index for index, token in enumerate(VOCAB)}, unk_token=VOCAB[0]
        )
    )
    backend.pre_tokenizer, backend.decoder = WhitespaceSplit(), decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=VOCAB[0],
        pad_token=VOCAB[-1],
        additional_special_tokens=[t for t in VOCAB if t.startswith("<|")],
    )
    hf_config = SimpleNamespace(
        model_type="mimo_v2",
        architectures=["MiMoV2ForCausalLM"],
        rope_scaling=None,
        vision_config=dict(MIMO_VISION_CONFIG),
        processor_config=dict(MIMO_PROCESSOR_CONFIG),
    )
    server_args = SimpleNamespace(
        keep_mm_feature_on_device=False,
        mm_feature_transport="cpu",
        disable_fast_image_processor=True,
        skip_tokenizer_init=False,
        # Read by NativeMmHost._use_feature_shm (single-rank fixture → the
        # inline zero-copy transport, like the 1-GPU e2e).
        tp_size=1,
        dist_init_addr=None,
        mm_process_config={},
        mm_io_worker_num=1,
        mm_processor_worker_num=1,
        tokenizer_worker_num=1,
        base_gpu_id=0,
        device="cpu",
    )
    return MiMoV2Processor(
        hf_config,
        server_args,
        SimpleNamespace(tokenizer=tokenizer),
        None,
        skip_mm_pool=True,
    )


def snapshot(input_ids, output):
    """Scheduler-boundary view, normalized across item packaging: the Python
    MiMo path batches every image into one ``MultimodalDataItem`` (stacked
    grids, one offset list) while the native path emits one item per image, so
    grids/offsets/features are flattened before comparing."""
    return {
        "input_ids": tuple(input_ids),
        "grids": tuple(
            tuple(grid)
            for item in output.mm_items
            for grid in item.image_grid_thw.reshape(-1, 3).tolist()
        ),
        "offsets": tuple(
            tuple(offset) for item in output.mm_items for offset in item.offsets
        ),
        "features": np.concatenate(
            [item.feature.detach().cpu().numpy() for item in output.mm_items]
        ),
        "mrope": output.mrope_positions.detach().cpu().numpy(),
        "delta": int(output.mrope_position_delta.item()),
        "tokens": (output.im_start_id, output.im_token_id, output.im_end_id),
    }
