"""WAN TI2V-specific helpers shared by the generic denoising stage."""

import math

import torch
from einops import rearrange

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    Wan2_2_TI2V_5B_Config,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import masks_like


def should_apply_wan_ti2v(batch: Req, server_args: ServerArgs) -> bool:
    """Return whether the request should use the Wan2.2 TI2V latent path."""

    return bool(
        server_args.pipeline_config.task_type == ModelTaskType.TI2V
        and batch.condition_image is not None
        and type(server_args.pipeline_config) is Wan2_2_TI2V_5B_Config
    )


def prepare_wan_ti2v_latents(
    vae: object,
    latents: torch.Tensor,
    target_dtype: torch.dtype,
    batch: Req,
    server_args: ServerArgs,
) -> tuple[int, torch.Tensor, list[torch.Tensor]]:
    """Encode the conditioning image and splice it into Wan TI2V latents."""

    # Wan2.2 TI2V directly replaces the first frame of the latent with
    # the image latent instead of appending along the channel dim.
    assert batch.image_latent is None, "TI2V task should not have image latents"
    assert vae is not None, "VAE is not provided for TI2V task"

    vae = vae.to(batch.condition_image.device)
    z = vae.encode(batch.condition_image).mean.float()
    if getattr(vae, "device", None) != "cpu" and server_args.vae_cpu_offload:
        vae = vae.to("cpu")

    if hasattr(vae, "shift_factor") and vae.shift_factor is not None:
        if isinstance(vae.shift_factor, torch.Tensor):
            z -= vae.shift_factor.to(z.device, z.dtype)
        else:
            z -= vae.shift_factor

    if isinstance(vae.scaling_factor, torch.Tensor):
        z = z * vae.scaling_factor.to(z.device, z.dtype)
    else:
        z = z * vae.scaling_factor

    latent_model_input = latents.to(target_dtype)
    assert latent_model_input.ndim == 5

    latent_for_mask = latent_model_input.squeeze(0)
    _, reserved_frames_masks = masks_like([latent_for_mask], zero=True)
    reserved_frames_mask = reserved_frames_masks[0].unsqueeze(0)

    latents = (
        1.0 - reserved_frames_mask
    ) * z + reserved_frames_mask * latent_model_input
    assert latents.ndim == 5
    batch.latents = latents.to(get_local_torch_device())

    num_frames = batch.num_frames
    temporal_scale = (
        server_args.pipeline_config.vae_config.arch_config.scale_factor_temporal
    )
    spatial_scale = (
        server_args.pipeline_config.vae_config.arch_config.scale_factor_spatial
    )
    patch_size = server_args.pipeline_config.dit_config.arch_config.patch_size
    seq_len = (
        ((num_frames - 1) // temporal_scale + 1)
        * (batch.height // spatial_scale)
        * (batch.width // spatial_scale)
        // (patch_size[1] * patch_size[2])
    )
    seq_len = int(math.ceil(seq_len / get_sp_world_size())) * get_sp_world_size()
    return seq_len, z, reserved_frames_masks


def prepare_wan_ti2v_sp_inputs(
    z: torch.Tensor | None,
    reserved_frames_masks: list[torch.Tensor] | None,
    batch: Req,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Shard Wan TI2V image-conditioning state to match SP-sharded video latents."""

    rank_in_sp_group = get_sp_parallel_rank()
    sp_world_size = get_sp_world_size()

    if getattr(batch, "did_sp_shard_latents", False):
        if z is not None and z.shape[2] == 1:
            z_sp = z if rank_in_sp_group == 0 else None
        else:
            z_sp = z

        if reserved_frames_masks is not None:
            reserved_frames_mask = reserved_frames_masks[0]
            time_dim = reserved_frames_mask.shape[1]
            if time_dim > 0 and time_dim % sp_world_size == 0:
                reserved_frames_mask_sp_tensor = rearrange(
                    reserved_frames_mask,
                    "c (n t) h w -> c n t h w",
                    n=sp_world_size,
                ).contiguous()
                reserved_frames_mask_sp = reserved_frames_mask_sp_tensor[
                    :, rank_in_sp_group, :, :, :
                ]
            else:
                reserved_frames_mask_sp = reserved_frames_mask
        else:
            reserved_frames_mask_sp = None
    else:
        z_sp = z
        reserved_frames_mask_sp = (
            reserved_frames_masks[0] if reserved_frames_masks is not None else None
        )

    return reserved_frames_mask_sp, z_sp


def expand_wan_ti2v_timestep(
    batch: Req,
    t_device: torch.Tensor,
    target_dtype: torch.dtype,
    seq_len: int,
    reserved_frames_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Expand the timestep tensor for Wan TI2V's first-frame masking semantics."""

    batch_size = batch.raw_latent_shape[0]
    t_device_rounded = t_device.to(target_dtype)

    local_seq_len = seq_len
    if get_sp_world_size() > 1 and getattr(batch, "did_sp_shard_latents", False):
        local_seq_len = seq_len // get_sp_world_size()

    if get_sp_parallel_rank() == 0 and reserved_frames_mask is not None:
        temp_ts = (reserved_frames_mask[0][:, ::2, ::2] * t_device_rounded).flatten()
        temp_ts = torch.cat(
            [
                temp_ts,
                temp_ts.new_ones(local_seq_len - temp_ts.size(0)) * t_device_rounded,
            ]
        )
        return temp_ts.unsqueeze(0).repeat(batch_size, 1)

    return t_device.repeat(batch_size, local_seq_len)


def blend_wan_ti2v_latents(
    latents: torch.Tensor,
    reserved_frames_mask: torch.Tensor | None,
    z: torch.Tensor | None,
) -> torch.Tensor:
    """Restore Wan TI2V's conditioned first frame after each denoising step."""

    if z is None or reserved_frames_mask is None:
        return latents
    return (
        1.0 - reserved_frames_mask.unsqueeze(0)
    ) * z + reserved_frames_mask.unsqueeze(0) * latents
