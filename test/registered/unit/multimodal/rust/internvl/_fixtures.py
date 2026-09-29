import os

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

from types import SimpleNamespace

from tokenizers import Tokenizer, Regex, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.multimodal.processors.internvl import InternVLProcessor
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=0, suite="base-a-test-cpu", disabled="InternVL test fixtures")

# Token ids in the tiny tokenizer `make_tokenizer` builds.
IMG_START_ID = 1
IMG_CONTEXT_ID = 2
IMG_END_ID = 3
IMAGE_TOKEN_ID = 4  # "<image>", consumed by prompt placeholders only
TEXT_ID = 5  # "hello"

IMAGE_SIZE = 448
PATCH_SIZE = 14
DOWNSAMPLE_RATIO = 2
NUM_IMAGE_TOKEN = (IMAGE_SIZE // PATCH_SIZE) ** 2 * (DOWNSAMPLE_RATIO**2)

VOCAB = ["<unk>", "<img>", "<IMG_CONTEXT>", "</img>", "<image>", "hello", "<pad>"]


def make_tokenizer():
    """A WordLevel tokenizer that keeps InternVL wrapper tokens atomic even
    when the processor concatenates them without whitespace between them."""
    vocab = {token: index for index, token in enumerate(VOCAB)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token=VOCAB[0]))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Split(
                Regex(r"(<img>|<IMG_CONTEXT>|</img>|<image>)"), behavior="isolated"
            ),
        ]
    )
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token=VOCAB[0], pad_token=VOCAB[-1]
    )


def make_processor(case):
    """An ``InternVLProcessor`` over ``make_tokenizer``, running on CPU."""
    tokenizer = make_tokenizer()
    hf_config = SimpleNamespace(
        model_type="internvl_chat",
        architectures=["InternVLChatModel"],
        vision_config=SimpleNamespace(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE),
        downsample_ratio=DOWNSAMPLE_RATIO,
    )
    server_args = SimpleNamespace(
        model_impl="sglang",
        mm_feature_transport="cpu",
        mm_enable_dp_encoder=False,
        image_processor_backend="auto",
        disable_fast_image_processor=True,
        skip_tokenizer_init=False,
        mm_preprocess_cache_size_mb=0,
        trust_mm_content_hashes=False,
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
    publish(
        ServerArgs(
            model_path="dummy",
            model_impl=server_args.model_impl,
            mm_feature_transport=server_args.mm_feature_transport,
            mm_process_config=server_args.mm_process_config,
            allowed_media_domains=server_args.allowed_media_domains,
            disable_fast_image_processor=server_args.disable_fast_image_processor,
        ),
        role="tokenizer",
    )
    case.addCleanup(reset_context)
    return InternVLProcessor(hf_config, server_args, tokenizer, None, skip_mm_pool=True)


def snapshot(input_ids, output):
    """The scheduler-input fields the two paths must agree on. The hash is
    deliberately excluded: native and Python hash different things."""
    import numpy as np

    return {
        "input_ids": tuple(input_ids),
        "offsets": tuple(item.offsets[0] for item in output.mm_items),
        "features": np.concatenate(
            [item.feature.detach().float().cpu().numpy() for item in output.mm_items]
        ),
        "tokens": (output.im_start_id, output.im_token_id, output.im_end_id),
    }
