import sgl_kernel.ops._kernels
import torch


def lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv):
    torch.ops.sgl_kernels.lightning_attention_decode(
        q, k, v, past_kv, slope, output, new_kv
    )
