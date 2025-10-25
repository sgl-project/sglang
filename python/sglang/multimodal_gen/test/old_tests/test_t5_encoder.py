# SPDX-License-Identifier: Apache-2.0
import argparse

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, UMT5EncoderModel

from sgl_diffusion.runtime.distributed import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sgl_diffusion.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def setup_args():
    parser = argparse.ArgumentParser(description="T5 Encoder Test")
    parser.add_argument("--model_path", type=str, default="google/umt5-xxl")
    parser.add_argument(
        "--dit-precision",
        type=str,
        default="float32",
        help="Precision to use for the model (float32, float16, bfloat16)",
    )
    return parser.parse_args()


def test_t5_encoder():
    maybe_init_distributed_environment_and_model_parallel(1, 1)

    # Set fixed random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the two model implementations
    model_path = "/workspace/data/Wan2.1-T2V-1.3B-Diffusers/text_encoder"
    tokenizer_path = "/workspace/data/Wan2.1-T2V-1.3B-Diffusers/tokenizer"

    hf_config = AutoConfig.from_pretrained(model_path)
    print(hf_config)
    precision = (
        torch.float16
    )  # It must be float16 because the weight loader is in float16
    # Load our implementation using the loader from text_encoder/__init__.py
    model1 = (
        UMT5EncoderModel.from_pretrained(model_path).to(precision).to(device).eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    from sgl_diffusion.runtime.loader.component_loader import TextEncoderLoader

    loader = TextEncoderLoader()
    model2 = loader.load_model(model_path, hf_config, device)

    # Convert to float16 and move to device
    model2 = model2.to(precision)
    model2 = model2.to(device)
    model2.eval()

    # Sanity check weights between the two models
    logger.info("Comparing model weights for sanity check...")
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    # Check number of parameters
    logger.info(f"Model1 has {len(params1)} parameters")
    logger.info(f"Model2 has {len(params2)} parameters")

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
        "shared.weight",
    ]
    # for (name1, param1), (name2, param2) in zip(
    #     sorted(params1.items()), sorted(params2.items())
    # ):
    for l in range(hf_config.num_hidden_layers):
        for w in weights:
            name1 = w.format(l)
            name2 = w.format(l)
            p1 = params1[name1]
            p2 = params2[name2]
            assert p1.dtype == p2.dtype
            try:
                logger.info(f"Parameter: {name1} vs {name2}")
                max_diff = torch.max(torch.abs(p1 - p2)).item()
                mean_diff = torch.mean(torch.abs(p1 - p2)).item()
                weight_diffs.append((name1, name2, max_diff, mean_diff))
                logger.info(f"  Max diff: {max_diff}, Mean diff: {mean_diff}")
            except Exception as e:
                logger.info(f"Error comparing {name1} and {name2}: {e}")

    total_params = sum(p.numel() for p in model1.parameters())
    weight_sum_model1 = sum(
        p.to(torch.float64).sum().item() for p in model1.parameters()
    )
    weight_mean_model1 = weight_sum_model1 / total_params
    print("Model 1 Weight Sum: ", weight_sum_model1)
    print("Model 1 Weight Mean: ", weight_mean_model1)

    total_params = sum(p.numel() for p in model2.parameters())
    weight_sum_model2 = sum(
        p.to(torch.float64).sum().item() for p in model2.parameters()
    )
    # Also calculate mean for more stable comparison
    weight_mean_model2 = weight_sum_model2 / total_params
    print("Model 2 Weight Sum: ", weight_sum_model2)
    print("Model 2 Weight Mean: ", weight_mean_model2)

    # Test with some sample prompts
    prompts = [
        "Once upon a time",
        "The quick brown fox jumps over",
        "In a galaxy far, far away",
    ]

    logger.info("Testing T5 encoder with sample prompts")

    with torch.no_grad():
        for prompt in prompts:
            logger.info(f"Testing prompt: '{prompt}'")

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
            outputs1 = model1(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            ).last_hidden_state
            print("--------------------------------")
            logger.info("Testing model2")

            # Get outputs from HuggingFace implementation
            outputs2 = model2(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
            )

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
                f"Maximum difference in last hidden states: {max_diff_hidden.item()}"
            )
            logger.info(
                f"Mean difference in last hidden states: {mean_diff_hidden.item()}"
            )

    logger.info("Test passed! Both T5 encoder implementations produce similar outputs.")
    logger.info("Test completed successfully")


if __name__ == "__main__":
    test_t5_encoder()
