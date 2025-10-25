# SPDX-License-Identifier: Apache-2.0

import glob
import json
import os

import pytest
import torch

from sgl_diffusion.api.configs.models.dits import HunyuanVideoConfig
from sgl_diffusion.api.configs.pipelines.base import PipelineConfig
from sgl_diffusion.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sgl_diffusion.runtime.loader.component_loader import TransformerLoader
from sgl_diffusion.runtime.managers.forward_context import set_forward_context
from sgl_diffusion.runtime.pipelines.pipeline_batch_info import Req
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
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")
CONFIG_PATH = os.path.join(TRANSFORMER_PATH, "config.json")

LOCAL_RANK = 0
RANK = 0
WORLD_SIZE = 1

# Latent generated on commit c021e8a27cf437ac22827f2bc58b7f006561317f with 1 x L40S
REFERENCE_LATENT = 89.7002067565918


@pytest.mark.usefixtures("distributed_setup")
def test_hunyuanvideo_distributed():
    logger.info(
        f"Initializing process: rank={RANK}, local_rank={LOCAL_RANK}, world_size={WORLD_SIZE}"
    )

    torch.cuda.set_device(f"cuda:{LOCAL_RANK}")

    # Get tensor parallel info
    sp_rank = get_sp_parallel_rank()
    sp_world_size = get_sp_world_size()

    logger.info(
        f"Process rank {RANK} initialized with SP rank {sp_rank} in SP world size {sp_world_size}"
    )

    config = json.load(open(CONFIG_PATH))
    # remove   "_class_name": "HunyuanVideoTransformer3DModel",   "_diffusers_version": "0.32.0.dev0",
    # TODO: write normalize config function
    config.pop("_class_name")
    config.pop("_diffusers_version")

    precision_str = "bf16"
    args = ServerArgs(
        model_path=TRANSFORMER_PATH,
        dit_cpu_offload=True,
        pipeline_config=PipelineConfig(
            dit_config=HunyuanVideoConfig(), dit_precision=precision_str
        ),
    )
    args.device = torch.device(f"cuda:{LOCAL_RANK}")

    loader = TransformerLoader()
    model = loader.load(TRANSFORMER_PATH, args)

    model.eval()

    # Create random inputs for testing
    batch_size = 1
    seq_len = 3
    device = torch.device(f"cuda:{LOCAL_RANK}")

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(
        batch_size, 16, 8, 16, 16, device=device, dtype=torch.bfloat16
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
        # Run inference on model
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
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
