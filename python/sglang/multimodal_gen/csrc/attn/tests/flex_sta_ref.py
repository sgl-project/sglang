from typing import Tuple

import torch
from torch import BoolTensor, IntTensor
from torch.nn.attention.flex_attention import create_block_mask

# Peiyuan: This is neccesay. Dont know why. see https://github.com/pytorch/pytorch/issues/135028
torch._inductor.config.realize_opcount_threshold = 100


def generate_sta_mask(canvas_twh, kernel_twh, tile_twh, text_length):
    """Generates a 3D NATTEN attention mask with a given kernel size.

    Args:
        canvas_t: The time dimension of the canvas.
        canvas_h: The height of the canvas.
        canvas_w: The width of the canvas.
        kernel_t: The time dimension of the kernel.
        kernel_h: The height of the kernel.
        kernel_w: The width of the kernel.
    """
    canvas_t, canvas_h, canvas_w = canvas_twh
    kernel_t, kernel_h, kernel_w = kernel_twh
    tile_t_size, tile_h_size, tile_w_size = tile_twh
    total_tile_size = tile_t_size * tile_h_size * tile_w_size
    canvas_tile_t, canvas_tile_h, canvas_tile_w = (
        canvas_t // tile_t_size,
        canvas_h // tile_h_size,
        canvas_w // tile_w_size,
    )
    img_seq_len = canvas_t * canvas_h * canvas_w

    def get_tile_t_x_y(idx: IntTensor) -> Tuple[IntTensor, IntTensor, IntTensor]:
        tile_id = idx // total_tile_size
        tile_t = tile_id // (canvas_tile_h * canvas_tile_w)
        tile_h = (tile_id % (canvas_tile_h * canvas_tile_w)) // canvas_tile_w
        tile_w = tile_id % canvas_tile_w
        return tile_t, tile_h, tile_w

    def sta_mask_3d(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        q_t_tile, q_x_tile, q_y_tile = get_tile_t_x_y(q_idx)
        kv_t_tile, kv_x_tile, kv_y_tile = get_tile_t_x_y(kv_idx)
        # kernel nominally attempts to center itself on the query, but kernel center
        # is clamped to a fixed distance (kernel half-length) from the canvas edge
        kernel_center_t = q_t_tile.clamp(
            kernel_t // 2, (canvas_tile_t - 1) - kernel_t // 2
        )
        kernel_center_x = q_x_tile.clamp(
            kernel_h // 2, (canvas_tile_h - 1) - kernel_h // 2
        )
        kernel_center_y = q_y_tile.clamp(
            kernel_w // 2, (canvas_tile_w - 1) - kernel_w // 2
        )
        time_mask = (kernel_center_t - kv_t_tile).abs() <= kernel_t // 2
        hori_mask = (kernel_center_x - kv_x_tile).abs() <= kernel_h // 2
        vert_mask = (kernel_center_y - kv_y_tile).abs() <= kernel_w // 2
        image_mask = (q_idx < img_seq_len) & (kv_idx < img_seq_len)
        image_to_text_mask = (
            (q_idx < img_seq_len)
            & (kv_idx >= img_seq_len)
            & (kv_idx < img_seq_len + text_length)
        )
        text_to_all_mask = (q_idx >= img_seq_len) & (kv_idx < img_seq_len + text_length)
        return (
            (image_mask & time_mask & hori_mask & vert_mask)
            | image_to_text_mask
            | text_to_all_mask
        )

    sta_mask_3d.__name__ = (
        f"natten_3d_c{canvas_t}x{canvas_w}x{canvas_h}_k{kernel_t}x{kernel_w}x{kernel_h}"
    )
    return sta_mask_3d


def get_sliding_tile_attention_mask(
    kernel_size, tile_size, img_size, text_length, device, text_max_len=256
):
    img_seq_len = img_size[0] * img_size[1] * img_size[2]
    image_mask = generate_sta_mask(img_size, kernel_size, tile_size, text_length)
    mask = create_block_mask(
        image_mask,
        B=None,
        H=None,
        Q_LEN=img_seq_len + text_max_len,
        KV_LEN=img_seq_len + text_max_len,
        device=device,
        _compile=True,
    )
    return mask
