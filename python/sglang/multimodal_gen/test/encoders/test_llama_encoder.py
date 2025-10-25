# SPDX-License-Identifier: Apache-2.0
import gc
import os

import numpy as np
import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.testing import assert_close
from transformers import AutoConfig, AutoTokenizer, LlamaModel

from sgl_diffusion.api.configs.models.encoders import LlamaConfig
from sgl_diffusion.api.configs.pipelines import PipelineConfig
from sgl_diffusion.runtime.loader.component_loader import TextEncoderLoader
from sgl_diffusion.runtime.managers.forward_context import set_forward_context
from sgl_diffusion.runtime.server_args import ServerArgs
from sgl_diffusion.runtime.utils.logging_utils import init_logger
from sgl_diffusion.utils import maybe_download_model

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "hunyuanvideo-community/HunyuanVideo"
MODEL_PATH = maybe_download_model(
    BASE_MODEL_PATH, local_dir=os.path.join("data", BASE_MODEL_PATH)
)
TEXT_ENCODER_PATH = os.path.join(MODEL_PATH, "text_encoder")
TOKENIZER_PATH = os.path.join(MODEL_PATH, "tokenizer")


@pytest.mark.usefixtures("distributed_setup")
def test_llama_encoder():
    """
    Tests compatibility between two different implementations for loading text encoders:
    1. load_text_encoder from sgl_diffusion.runtime.models.hunyuan.text_encoder
    2. TextEncoderLoader from sgl_diffusion.runtime.loader

    The test verifies that both implementations:
    - Load models with the same weights and parameters
    - Produce nearly identical outputs for the same input prompts
    """
    args = ServerArgs(
        model_path="meta-llama/Llama-2-7b-hf",
        pipeline_config=PipelineConfig(
            text_encoder_configs=(LlamaConfig(),), text_encoder_precisions=("fp16",)
        ),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the two model implementations
    logger.info("Loading models from %s", args.model_path)
    hf_config = AutoConfig.from_pretrained(TEXT_ENCODER_PATH)
    print(hf_config)

    # Load HuggingFace implementation
    model1 = (
        LlamaModel.from_pretrained(TEXT_ENCODER_PATH)
        .to(torch.float16)
        .to(device)
        .eval()
    )
    loader = TextEncoderLoader()
    device = torch.device("cuda:0")
    model2 = loader.load(TEXT_ENCODER_PATH, args)

    # Convert to float16 and move to device
    # model2 = model2.to(torch.float16)
    model2.eval()

    # Sanity check weights between the two models
    logger.info("Comparing model weights for sanity check...")
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    # Check number of parameters
    logger.info("Model1 has %d parameters", len(params1))
    logger.info("Model2 has %d parameters", len(params2))

    # Compare a few key parameters
    weight_diffs = []
    # check if embed_tokens are the same
    device = model1.embed_tokens.weight.device
    assert torch.allclose(
        model1.embed_tokens.weight,
        (
            model2.embed_tokens.weight.to_local().to(device)
            if isinstance(model2.embed_tokens.weight, DTensor)
            else model2.embed_tokens.weight.to(device)
        ),
    )
    weights = [
        "layers.{}.input_layernorm.weight",
        "layers.{}.post_attention_layernorm.weight",
    ]

    for name1, param1 in sorted(params1.items()):
        name2 = name1
        skip = False
        for (
            param_name,
            weight_name,
            shard_id,
        ) in model2.config.arch_config.stacked_params_mapping:
            if weight_name not in name1:
                skip = True
        # stacked params are more troublesome
        if skip:
            continue
        param2 = params2[name2]
        param2 = (
            param2.to_local().to(device)
            if isinstance(param2, DTensor)
            else param2.to(device)
        )
        assert_close(param1, param2, atol=1e-4, rtol=1e-4)
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Test with some sample prompts
    prompts = [
        "Once upon a time",
        # "The quick brown fox jumps over",
        # "In a galaxy far, far away"
    ]

    logger.info("Testing LLaMA encoder with sample prompts")

    with torch.no_grad():
        for prompt in prompts:
            logger.info("Testing prompt: '%s'", prompt)

            # Tokenize the prompt
            tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # Get outputs from our implementation
            # filter out padding input_ids
            # tokens.input_ids = tokens.input_ids[tokens.attention_mask==1]
            # tokens.attention_mask = tokens.attention_mask[tokens.attention_mask==1]
            outputs1 = model1(input_ids=tokens.input_ids, output_hidden_states=True)
            print("--------------------------------")
            logger.info("Testing model2")

            # Get outputs from HuggingFace implementation
            with set_forward_context(current_timestep=0, attn_metadata=None):
                outputs2 = model2(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    output_hidden_states=True,
                )

            # Compare last hidden states
            last_hidden_state1 = outputs1.last_hidden_state[tokens.attention_mask == 1]
            last_hidden_state2 = outputs2.last_hidden_state[tokens.attention_mask == 1]

            assert (
                last_hidden_state1.shape == last_hidden_state2.shape
            ), f"Hidden state shapes don't match: {last_hidden_state1.shape} vs {last_hidden_state2.shape}"

            max_diff_hidden = torch.max(
                torch.abs(last_hidden_state1 - last_hidden_state2)
            )
            mean_diff_hidden = torch.mean(
                torch.abs(last_hidden_state1 - last_hidden_state2)
            )

            logger.info(
                "Maximum difference in last hidden states: %f", max_diff_hidden.item()
            )
            logger.info(
                "Mean difference in last hidden states: %f", mean_diff_hidden.item()
            )

            # Check if outputs are similar (allowing for small numerical differences)
            assert (
                mean_diff_hidden < 1e-2
            ), f"Hidden states differ significantly: mean diff = {mean_diff_hidden.item()}"
            assert (
                max_diff_hidden < 1e-1
            ), f"Hidden states differ significantly: max diff = {max_diff_hidden.item()}"
