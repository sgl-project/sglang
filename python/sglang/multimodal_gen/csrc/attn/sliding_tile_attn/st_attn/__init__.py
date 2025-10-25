import math

import torch
from torch.utils.checkpoint import detach_variable

try:
    from st_attn_cuda import sta_fwd
except ImportError:
    sta_fwd = None


def sliding_tile_attention(
    q_all,
    k_all,
    v_all,
    window_size,
    text_length,
    has_text=True,
    dit_seq_shape="30x48x80",
):
    seq_length = q_all.shape[2]
    dit_seq_shape_mapping = {
        "30x48x80": 1,
        "36x48x48": 2,
        "18x48x80": 3,
    }
    if has_text:
        assert (
            q_all.shape[2] >= 115200 and q_all.shape[2] <= 115456
        ), f"Unsupported {dit_seq_shape}, current shape is {q_all.shape}, only support '30x48x80' for HunyuanVideo"
        assert q_all.shape[1] == len(
            window_size
        ), "Number of heads must match the number of window sizes"
        target_size = math.ceil(seq_length / 384) * 384
        pad_size = target_size - seq_length
        if pad_size > 0:
            q_all = torch.cat([q_all, q_all[:, :, -pad_size:]], dim=2)
            k_all = torch.cat([k_all, k_all[:, :, -pad_size:]], dim=2)
            v_all = torch.cat([v_all, v_all[:, :, -pad_size:]], dim=2)
    else:
        if dit_seq_shape == "36x48x48":  # Stepvideo 204x768x68
            assert q_all.shape[2] == 82944
        elif dit_seq_shape == "18x48x80":  # Wan 69x768x1280
            assert q_all.shape[2] == 69120
        else:
            raise ValueError(
                f"Unsupported {dit_seq_shape}, current shape is {q_all.shape}, only support '36x48x48' for Stepvideo and '18x48x80' for Wan"
            )

    kernel_aspect_ratio_flag = dit_seq_shape_mapping[dit_seq_shape]
    hidden_states = torch.empty_like(q_all)
    # This for loop is ugly. but it is actually quite efficient. The sequence dimension alone can already oversubscribe SMs
    for head_index, (t_kernel, h_kernel, w_kernel) in enumerate(window_size):
        for batch in range(q_all.shape[0]):
            q_head, k_head, v_head, o_head = (
                q_all[batch : batch + 1, head_index : head_index + 1],
                k_all[batch : batch + 1, head_index : head_index + 1],
                v_all[batch : batch + 1, head_index : head_index + 1],
                hidden_states[batch : batch + 1, head_index : head_index + 1],
            )

            _ = sta_fwd(
                q_head,
                k_head,
                v_head,
                o_head,
                t_kernel,
                h_kernel,
                w_kernel,
                text_length,
                False,
                has_text,
                kernel_aspect_ratio_flag,
            )
    if has_text:
        _ = sta_fwd(
            q_all,
            k_all,
            v_all,
            hidden_states,
            3,
            3,
            3,
            text_length,
            True,
            True,
            kernel_aspect_ratio_flag,
        )
    return hidden_states[:, :, :seq_length]
