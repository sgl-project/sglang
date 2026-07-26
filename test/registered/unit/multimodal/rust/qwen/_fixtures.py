from types import SimpleNamespace

import numpy as np
from tokenizers import Tokenizer, decoders
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor as HfQwenImageProcessor,
)

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.multimodal.processors.qwen_vl import (  # noqa: E402
    QwenVLImageProcessor,
)

register_cpu_ci(est_time=0, suite="base-a-test-cpu", disabled="Qwen test fixtures")


def make_processor(config):
    vocab = [
        "<unk>",
        "<|vision_start|>",
        "<|image_pad|>",
        "<|vision_end|>",
        "hello",
        "<|video_pad|>",
        "<pad>",
    ]
    backend = Tokenizer(
        WordLevel(
            {token: index for index, token in enumerate(vocab)}, unk_token=vocab[0]
        )
    )
    backend.pre_tokenizer, backend.decoder = WhitespaceSplit(), decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=vocab[0],
        pad_token=vocab[-1],
        additional_special_tokens=vocab[1:4] + [vocab[5]],
    )
    processor = Qwen2VLProcessor(
        image_processor=HfQwenImageProcessor(**config),
        video_processor=Qwen2VLVideoProcessor(),
        tokenizer=tokenizer,
    )
    hf_config = SimpleNamespace(
        model_type="qwen2_5_vl",
        vision_start_token_id=1,
        image_token_id=2,
        vision_end_token_id=3,
        video_token_id=5,
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=2),
    )
    server_args = SimpleNamespace(
        keep_mm_feature_on_device=False,
        mm_feature_transport="cpu",
        disable_fast_image_processor=True,
        skip_tokenizer_init=False,
        mm_process_config={},
        mm_io_worker_num=1,
        mm_processor_worker_num=1,
        tokenizer_worker_num=1,
        base_gpu_id=0,
    )
    return QwenVLImageProcessor(
        hf_config, server_args, processor, None, skip_mm_pool=True
    )


def snapshot(input_ids, output):
    return {
        "input_ids": tuple(input_ids),
        "grids": tuple(
            tuple(item.image_grid_thw.flatten().tolist()) for item in output.mm_items
        ),
        "offsets": tuple(item.offsets[0] for item in output.mm_items),
        "features": np.concatenate(
            [item.feature.detach().cpu().numpy() for item in output.mm_items]
        ),
        "mrope": output.mrope_positions.detach().cpu().numpy(),
        "delta": int(output.mrope_position_delta.item()),
        "tokens": (output.im_start_id, output.im_token_id, output.im_end_id),
    }
