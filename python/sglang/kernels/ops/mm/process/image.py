import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_MAX_TRITON_ELEMENTS = 2**31 - 1


@triton.jit
def _normalize_and_patchify_kernel(
    input_ptr,
    scale_ptr,
    bias_ptr,
    output_ptr,
    channels: tl.constexpr,
    input_height,
    input_width,
    grid_height,
    grid_width,
    patch_size: tl.constexpr,
    element_count,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count

    patch_area = patch_size * patch_size
    patch_offset = offsets % patch_area
    patch_x = patch_offset % patch_size
    patch_y = patch_offset // patch_size
    remaining = offsets // patch_area
    channel = remaining % channels
    remaining = remaining // channels
    grid_x = remaining % grid_width
    remaining = remaining // grid_width
    grid_y = remaining % grid_height
    batch = remaining // grid_height

    input_y = grid_y * patch_size + patch_y
    input_x = grid_x * patch_size + patch_x
    input_mask = mask & (input_y < input_height) & (input_x < input_width)
    input_offset = (
        (batch * channels + channel) * input_height + input_y
    ) * input_width + input_x
    value = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0)
    scale = tl.load(scale_ptr + channel, mask=mask)
    bias = tl.load(bias_ptr + channel, mask=mask)
    tl.store(output_ptr + offsets, value * scale + bias, mask=mask)


def _normalize_and_patchify_torch(
    image: torch.Tensor,
    image_scale: torch.Tensor,
    image_bias: torch.Tensor,
    patch_size: int,
    padded_height: int,
    padded_width: int,
) -> torch.Tensor:
    pad_height = padded_height - image.shape[-2]
    pad_width = padded_width - image.shape[-1]
    if pad_height > 0 or pad_width > 0:
        image = F.pad(image, (0, pad_width, 0, pad_height), value=0.0)
    image = torch.addcmul(image_bias, image, image_scale)
    batch, channels, height, width = image.shape
    grid_height = height // patch_size
    grid_width = width // patch_size
    image = image.view(batch, channels, grid_height, patch_size, grid_width, patch_size)
    return image.permute(0, 2, 4, 1, 3, 5).reshape(
        batch, -1, channels, patch_size, patch_size
    )


def normalize_and_patchify(
    image: torch.Tensor,
    image_scale: torch.Tensor,
    image_bias: torch.Tensor,
    patch_size: int,
    padded_height: int,
    padded_width: int,
) -> torch.Tensor:
    batch, channels, _, _ = image.shape
    grid_height = padded_height // patch_size
    grid_width = padded_width // patch_size
    element_count = (
        batch * grid_height * grid_width * channels * patch_size * patch_size
    )
    if (
        not image.is_cuda
        or torch.version.hip is not None
        or not image.is_contiguous()
        or not image_scale.is_contiguous()
        or not image_bias.is_contiguous()
        or element_count > _MAX_TRITON_ELEMENTS
    ):
        return _normalize_and_patchify_torch(
            image,
            image_scale,
            image_bias,
            patch_size,
            padded_height,
            padded_width,
        )

    output = torch.empty(
        (batch, grid_height * grid_width, channels, patch_size, patch_size),
        dtype=image.dtype,
        device=image.device,
    )
    block_size = 256
    _normalize_and_patchify_kernel[(triton.cdiv(element_count, block_size),)](
        image,
        image_scale,
        image_bias,
        output,
        channels,
        image.shape[-2],
        image.shape[-1],
        grid_height,
        grid_width,
        patch_size,
        element_count,
        block_size,
    )
    return output
