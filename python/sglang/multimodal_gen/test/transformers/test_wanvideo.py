# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
from diffusers import WanTransformer3DModel

from sgl_diffusion.api.configs.models.dits import WanVideoConfig
from sgl_diffusion.api.configs.pipelines import PipelineConfig
from sgl_diffusion.runtime.loader.component_loader import TransformerLoader
from sgl_diffusion.runtime.managers.forward_context import set_forward_context
from sgl_diffusion.runtime.pipelines.pipeline_batch_info import Req
from sgl_diffusion.runtime.server_args import ServerArgs
from sgl_diffusion.runtime.utils.hf_diffusers_utils import maybe_download_model
from sgl_diffusion.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_PATH = maybe_download_model(
    BASE_MODEL_PATH, local_dir=os.path.join("data", BASE_MODEL_PATH)
)
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")


@pytest.mark.usefixtures("distributed_setup")
def test_wan_transformer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = ServerArgs(
        model_path=TRANSFORMER_PATH,
        dit_cpu_offload=True,
        pipeline_config=PipelineConfig(
            dit_config=WanVideoConfig(), dit_precision=precision_str
        ),
    )
    args.device = device

    loader = TransformerLoader()
    model2 = loader.load(TRANSFORMER_PATH, args).to(dtype=precision)

    model1 = (
        WanTransformer3DModel.from_pretrained(
            TRANSFORMER_PATH, device=device, torch_dtype=precision
        )
        .to(device, dtype=precision)
        .requires_grad_(False)
    )

    total_params = sum(p.numel() for p in model1.parameters())
    # Calculate weight sum for model1 (converting to float64 to avoid overflow)
    weight_sum_model1 = sum(
        p.to(torch.float64).sum().item() for p in model1.parameters()
    )
    # Also calculate mean for more stable comparison
    weight_mean_model1 = weight_sum_model1 / total_params
    logger.info("Model 1 weight sum: %s", weight_sum_model1)
    logger.info("Model 1 weight mean: %s", weight_mean_model1)

    # Calculate weight sum for model2 (converting to float64 to avoid overflow)
    total_params_model2 = sum(p.numel() for p in model2.parameters())
    weight_sum_model2 = sum(
        p.to(torch.float64).sum().item() for p in model2.parameters()
    )
    # Also calculate mean for more stable comparison
    weight_mean_model2 = weight_sum_model2 / total_params_model2
    logger.info("Model 2 weight sum: %s", weight_sum_model2)
    logger.info("Model 2 weight mean: %s", weight_mean_model2)

    weight_sum_diff = abs(weight_sum_model1 - weight_sum_model2)
    logger.info("Weight sum difference: %s", weight_sum_diff)
    weight_mean_diff = abs(weight_mean_model1 - weight_mean_model2)
    logger.info("Weight mean difference: %s", weight_mean_diff)

    # Set both models to eval mode
    model1 = model1.eval()
    model2 = model2.eval()

    # Create identical inputs for both models
    batch_size = 1
    seq_len = 30

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(
        batch_size, 16, 21, 160, 90, device=device, dtype=precision
    )

    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(
        batch_size, seq_len + 1, 4096, device=device, dtype=precision
    )

    # Timestep
    timestep = torch.tensor([500], device=device, dtype=precision)

    forward_batch = Req(
        data_type="dummy",
    )

    with torch.amp.autocast("cuda", dtype=precision):
        output1 = model1(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            return_dict=False,
        )[0]
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=forward_batch,
        ):
            output2 = model2(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )

    # Check if outputs have the same shape
    assert (
        output1.shape == output2.shape
    ), f"Output shapes don't match: {output1.shape} vs {output2.shape}"
    assert (
        output1.dtype == output2.dtype
    ), f"Output dtype don't match: {output1.dtype} vs {output2.dtype}"

    # Check if outputs are similar (allowing for small numerical differences)
    max_diff = torch.max(torch.abs(output1 - output2))
    mean_diff = torch.mean(torch.abs(output1 - output2))
    logger.info("Max Diff: %s", max_diff.item())
    logger.info("Mean Diff: %s", mean_diff.item())
    assert max_diff < 1e-1, f"Maximum difference between outputs: {max_diff.item()}"
    # mean diff
    assert mean_diff < 1e-2, f"Mean difference between outputs: {mean_diff.item()}"
