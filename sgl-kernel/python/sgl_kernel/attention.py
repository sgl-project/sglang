import torch


"""TODO: Add docstring."""
def lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv):
    torch.ops.sgl_kernel.lightning_attention_decode(
        q, k, v, past_kv, slope, output, new_kv
    )
