# SPDX-License-Identifier: Apache-2.0

import os
from itertools import chain

import pytest
import torch
import torch.nn as nn

from sgl_diffusion.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sgl_diffusion.runtime.loader.fsdp_load import shard_model
from sgl_diffusion.runtime.managers.forward_context import set_forward_context
from sgl_diffusion.runtime.models.dits.hunyuanvideo import (
    HunyuanVideoTransformer3DModel as HunyuanVideoDit,
)
from sgl_diffusion.runtime.pipelines.pipeline_batch_info import Req
from sgl_diffusion.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

# Latent generated on commit c021e8a27cf437ac22827f2bc58b7f006561317f with 1 x L40S
REFERENCE_LATENT = 1472.079828262329


def initialize_identical_weights(model, seed=42):
    """Initialize both models with identical weights using a fixed seed for reproducibility."""
    # Get all parameters from both models
    params1 = dict(model.named_parameters())

    # Initialize each layer with identical values
    with torch.no_grad():
        # Initialize weights
        for name1, param1 in params1.items():
            if "weight" in name1:
                # Set seed before each weight initialization
                torch.manual_seed(seed)
                nn.init.normal_(param1, mean=0.0, std=0.05)

        # Initialize biases
        for name1, param1 in params1.items():
            if "bias" in name1:
                torch.manual_seed(seed)
                nn.init.normal_(param1, mean=0.0, std=0.05)
                param1.data = param1.data.to(torch.bfloat16)

    logger.info("Model initialized with identical weights in bfloat16")
    return model


@pytest.mark.skip(reason="Incompatible with the new config")
def test_hunyuanvideo_distributed():
    # Get tensor parallel info
    sp_rank = get_sp_parallel_rank()
    sp_world_size = get_sp_world_size()

    # Small model parameters for testing
    hidden_size = 128
    heads_num = 4
    mm_double_blocks_depth = 2
    mm_single_blocks_depth = 2
    torch.cuda.set_device("cuda:0")
    # Initialize the two model implementations
    model = HunyuanVideoDit(
        patch_size=2,
        patch_size_t=1,
        in_channels=4,
        out_channels=4,
        attention_head_dim=hidden_size // heads_num,
        num_attention_heads=heads_num,
        num_layers=mm_double_blocks_depth,
        num_single_layers=mm_single_blocks_depth,
        rope_axes_dim=[8, 16, 8],  # sum = hidden_size // heads_num = 32
        dtype=torch.bfloat16,
    ).to(torch.bfloat16)

    # Initialize with identical weights
    model = initialize_identical_weights(model, seed=42)
    shard_model(
        model,
        cpu_offload=True,
        reshard_after_forward=True,
        fsdp_shard_conditions=model._fsdp_shard_conditions,
    )
    for n, p in chain(model.named_parameters(), model.named_buffers()):
        if p.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")
    for p in model.parameters():
        p.requires_grad = False

    model.eval()

    # Move to GPU based on local rank (0 or 1 for 2 GPUs)
    device = torch.device(f"cuda:0")
    model = model

    batch_size = 1
    seq_len = 3

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(
        batch_size, 4, 8, 16, 16, device=device, dtype=torch.bfloat16
    )
    chunk_per_rank = hidden_states.shape[2] // sp_world_size
    hidden_states = hidden_states[
        :, :, sp_rank * chunk_per_rank : (sp_rank + 1) * chunk_per_rank
    ]

    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(
        batch_size, seq_len + 1, 4096, device=device, dtype=torch.bfloat16
    )

    # Timestep
    timestep = torch.tensor([500], device=device, dtype=torch.bfloat16)

    forward_batch = Req(
        data_type="dummy",
    )

    # Disable gradients for inference
    with torch.no_grad():
        with set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=forward_batch
        ):
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )

    latent = output.double().sum().item()

    # Check if latents are similar
    diff_output_latents = abs(REFERENCE_LATENT - latent)
    logger.info(f"Reference latent: {REFERENCE_LATENT}, Current latent: {latent}")
    assert (
        diff_output_latents < 1e-4
    ), f"Output latents differ significantly: max diff = {diff_output_latents}"
