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
from sglang.srt.runtime_context import publish, reset_context  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

register_cpu_ci(est_time=0, suite="base-a-test-cpu", disabled="Qwen test fixtures")


def make_processor(case, config, image_processor_cls=None):
    """A ``QwenVLImageProcessor`` over a tiny hand-built tokenizer.
    ``image_processor_cls`` picks the HF backend; they resample differently."""
    image_processor_cls = image_processor_cls or HfQwenImageProcessor
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
        image_processor=image_processor_cls(**config),
        video_processor=Qwen2VLVideoProcessor(),
        tokenizer=tokenizer,
    )
    hf_config = SimpleNamespace(
        model_type="qwen2_5_vl",
        architectures=["Qwen2_5_VLForConditionalGeneration"],
        vision_start_token_id=1,
        image_token_id=2,
        vision_end_token_id=3,
        video_token_id=5,
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=2),
    )
    server_args = SimpleNamespace(
        # Non-auto: get_resolved_model_impl would choke on a SimpleNamespace.
        model_impl="sglang",
        keep_mm_feature_on_device=False,
        mm_feature_transport="cpu",
        image_processor_backend="auto",
        disable_fast_image_processor=True,
        skip_tokenizer_init=False,
        mm_preprocess_cache_size_mb=0,
        trust_mm_content_hashes=False,
        # Read by NativeMmHost._use_feature_shm (single-rank fixture → the
        # inline zero-copy transport, like the 1-GPU e2e).
        tp_size=1,
        dist_init_addr=None,
        mm_process_config={},
        mm_io_worker_num=1,
        mm_processor_worker_num=1,
        tokenizer_worker_num=1,
        base_gpu_id=0,
        rl_on_policy_target=None,
        allowed_media_domains=[],
        media_url_max_file_size_mb=64,
    )
    # Left at the default backend, the fast image processor sends the tensor to
    # `cuda:<base_gpu_id>`, which a CPU-only host cannot do.
    publish(
        ServerArgs(
            model_path="dummy",
            mm_feature_transport=server_args.mm_feature_transport,
            mm_process_config=server_args.mm_process_config,
            allowed_media_domains=server_args.allowed_media_domains,
            disable_fast_image_processor=server_args.disable_fast_image_processor,
        ),
        role="tokenizer",
    )
    case.addCleanup(reset_context)
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
