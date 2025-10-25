# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.testing import assert_close
from transformers import AutoConfig, AutoTokenizer, UMT5EncoderModel

from sgl_diffusion.api.configs.models.encoders import T5Config
from sgl_diffusion.api.configs.pipelines import PipelineConfig
from sgl_diffusion.runtime.loader.component_loader import TextEncoderLoader
from sgl_diffusion.runtime.managers.forward_context import set_forward_context
from sgl_diffusion.runtime.server_args import ServerArgs
from sgl_diffusion.runtime.utils.logging_utils import init_logger
from sgl_diffusion.utils import PRECISION_TO_TYPE, maybe_download_model

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_PATH = maybe_download_model(
    BASE_MODEL_PATH, local_dir=os.path.join("data", BASE_MODEL_PATH)
)
TEXT_ENCODER_PATH = os.path.join(MODEL_PATH, "text_encoder")
TOKENIZER_PATH = os.path.join(MODEL_PATH, "tokenizer")


@pytest.mark.usefixtures("distributed_setup")
def test_t5_encoder():
    # Initialize the two model implementations
    hf_config = AutoConfig.from_pretrained(TEXT_ENCODER_PATH)
    print(hf_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision_str = "fp32"
    precision = PRECISION_TO_TYPE[precision_str]
    model1 = (
        UMT5EncoderModel.from_pretrained(TEXT_ENCODER_PATH)
        .to(precision)
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    args = ServerArgs(
        model_path=TEXT_ENCODER_PATH,
        pipeline_config=PipelineConfig(
            text_encoder_configs=(T5Config(),), text_encoder_precisions=(precision_str,)
        ),
        pin_cpu_memory=False,
    )
    loader = TextEncoderLoader()
    model2 = loader.load(TEXT_ENCODER_PATH, args)
    model2 = model2.to(precision)
    model2.eval()

    # Sanity check weights between the two models
    logger.info("Comparing model weights for sanity check...")
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    # Check number of parameters
    logger.info("Model1 has %s parameters", len(params1))
    logger.info("Model2 has %s parameters", len(params2))

    weight_diffs = []
    # check if embed_tokens are the same
    weights = [
        "encoder.block.{}.layer.0.layer_norm.weight",
        "encoder.block.{}.layer.0.SelfAttention.relative_attention_bias.weight",
        "encoder.block.{}.layer.0.SelfAttention.o.weight",
        "encoder.block.{}.layer.1.DenseReluDense.wi_0.weight",
        "encoder.block.{}.layer.1.DenseReluDense.wi_1.weight",
        "encoder.block.{}.layer.1.DenseReluDense.wo.weight",
        "encoder.block.{}.layer.1.layer_norm.weight",
        "encoder.final_layer_norm.weight",
    ]

    for idx in range(hf_config.num_hidden_layers):
        for w in weights:
            name1 = w.format(idx)
            name2 = w.format(idx)
            p1 = params1[name1]
            p2 = params2[name2]
            p2 = (p2.to_local() if isinstance(p2, DTensor) else p2).to(p1)
            assert_close(p1, p2, atol=1e-4, rtol=1e-4)

    # Test with some sample prompts
    prompts = [
        "Once upon a time",
        "The quick brown fox jumps over",
        "In a galaxy far, far away",
    ]

    logger.info("Testing T5 encoder with sample prompts")

    with torch.no_grad():
        for prompt in prompts:
            logger.info("Testing prompt: %s", prompt)

            # Tokenize the prompt
            tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # Get outputs from HuggingFace implementation
            # filter out padding input_ids
            # tokens.input_ids = tokens.input_ids[tokens.attention_mask==1]
            # tokens.attention_mask = tokens.attention_mask[tokens.attention_mask==1]
            outputs1 = model1(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            ).last_hidden_state
            print("--------------------------------")
            logger.info("Testing model2")

            # Get outputs from our implementation
            with set_forward_context(current_timestep=0, attn_metadata=None):
                outputs2 = model2(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                ).last_hidden_state

            # Compare last hidden states
            last_hidden_state1 = outputs1[tokens.attention_mask == 1]
            last_hidden_state2 = outputs2[tokens.attention_mask == 1]

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
                "Maximum difference in last hidden states: %s", max_diff_hidden.item()
            )
            logger.info(
                "Mean difference in last hidden states: %s", mean_diff_hidden.item()
            )
            logger.info(
                "Max memory allocated: %s GB",
                torch.cuda.max_memory_allocated() / 1024**3,
            )
            # Check if outputs are similar (allowing for small numerical differences)
            assert (
                mean_diff_hidden < 1e-4
            ), f"Hidden states differ significantly: mean diff = {mean_diff_hidden.item()}"
            assert (
                max_diff_hidden < 1e-4
            ), f"Hidden states differ significantly: max diff = {max_diff_hidden.item()}"
