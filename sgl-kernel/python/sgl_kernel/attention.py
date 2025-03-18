import torch


def lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv):
    torch.ops.sgl_kernel.lightning_attention_decode.default(
        q, k, v, past_kv, slope, output, new_kv
    )

def cutlass_mla(q_absorbed, ckv_kpe_cache, seq_lens, page_table):
    return torch.ops.sgl_kernel.cutlass_mla(
        q_absorbed, ckv_kpe_cache, seq_lens, page_table
    )
