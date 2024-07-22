import random
from typing import Optional

import numpy as np
import torch
from transformers import AutoConfig
from vllm.model_executor.layers.rotary_embedding import get_rope


def gen_rope_model(config):
    hidden_size = config.hidden_size
    rope_theta = getattr(config, "rope_theta", 10000)
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None and getattr(
        config, "original_max_position_embeddings", None
    ):
        rope_scaling["original_max_position_embeddings"] = (
            config.original_max_position_embeddings
        )
    rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
    max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
    num_head = getattr(config, "num_attention_heads")
    head_dim = hidden_size // num_head

    print(
        head_dim, max_position_embeddings, rope_theta, rope_scaling, rope_is_neox_style
    )
    rope = get_rope(
        head_dim,  # 128
        rotary_dim=head_dim,  # 128
        max_position=max_position_embeddings,  # 8192
        base=rope_theta,  # 5e5
        rope_scaling=rope_scaling,  # None
        is_neox_style=rope_is_neox_style,  # True
    )
    return rope


def get_metadata(config):
    hidden_size = config.hidden_size
    num_head = getattr(config, "num_attention_heads")
    num_kv_head = 8
    head_dim = hidden_size // num_head
    return num_head, num_kv_head, head_dim


def gen_data(config, tp_size, sp_size, num_tokens):
    num_head, num_kv_head, head_dim = get_metadata(config)
    num_head_tp = num_head // tp_size
    num_kv_head_tp = num_kv_head // tp_size
    torch.manual_seed(42)
    random.seed(42)
    full_q = torch.rand((num_tokens, num_head_tp, head_dim), dtype=torch.bfloat16)
    full_k = torch.rand((num_tokens, num_kv_head_tp, head_dim), dtype=torch.bfloat16)
    full_pos = []
    extend_seq_lens = []
    extend_start_loc = []
    # req_offset = []
    while len(full_pos) < num_tokens:
        length = random.randint(sp_size, num_tokens // 2)
        length = min(length, num_tokens - len(full_pos))
        len_offset = random.randint(0, num_tokens)

        extend_seq_lens.append(length)
        extend_start_loc.append(len(full_pos))

        full_pos.extend(range(len_offset, length + len_offset))
        # req_offset.append(len_offset)
    print(f"extend_seq_lens:{extend_seq_lens}")
    full_pos = torch.from_numpy(np.fromiter(full_pos, dtype=np.int32)).to(torch.long)
    extend_seq_lens = torch.from_numpy(np.fromiter(extend_seq_lens, dtype=np.int32)).to(
        torch.long
    )
    extend_start_loc = torch.from_numpy(
        np.fromiter(extend_start_loc, dtype=np.int32)
    ).to(torch.long)
    return full_q, full_k, full_pos, extend_seq_lens, extend_start_loc


def gen_sp_indices(
    sp_size, extend_seq_lens, extend_start_loc, num_tokens, test: bool = False
):
    all_indices = set()
    indices = []
    for sp_rank in range(sp_size):
        sp_indices = get_prefill_indices(
            sp_rank, sp_size, extend_seq_lens, extend_start_loc
        )
        indices.append(sp_indices)
        if test:
            num_sp_indices = set([i for i in sp_indices])
            assert all_indices.isdisjoint(
                set(num_sp_indices)
            ), all_indices.intersection(num_sp_indices)
            all_indices.update(set(num_sp_indices))
    if test:
        assert all_indices == set(range(0, num_tokens))
    return indices


######## Function to be test
def get_prefill_indices(sp_rank, sp_size, extend_seq_lens, extend_start_loc):
    sp_req_len = extend_seq_lens // sp_size + (
        (extend_seq_lens % sp_size) > sp_rank
    ).to(torch.long)
    # the offset of each request in the batch. Only the first few ranks may get 1 more token (for each).
    # for sp_rank=r, there are r ranks ahread (since 0-based), each may get one token
    sp_in_req_offset = extend_seq_lens // sp_size * sp_rank + torch.clamp(
        extend_seq_lens % sp_size, max=sp_rank
    )
    sp_req_start = extend_start_loc + sp_in_req_offset
    sp_indices = torch.concat(
        [torch.arange(s, s + l) for s, l in zip(sp_req_start, sp_req_len)]
    )
    return sp_indices.cpu().numpy()


def get_decode_mask(sp_rank, sp_size, seq_lens):
    # True means the corresponding token is located on this device. Otherwise False.
    return seq_lens % sp_size == sp_rank


@torch.no_grad()
def main():
    config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tp_size = 4
    sp_size = 4
    num_tokens = 1280
    (full_q, full_k, full_pos, extend_seq_lens, extend_start_loc) = gen_data(
        config, tp_size, sp_size, num_tokens
    )
    sp_indices = gen_sp_indices(
        sp_size, extend_seq_lens, extend_start_loc, num_tokens, test=True
    )
    # correct result
    rope = gen_rope_model(config)
    full_pos = full_pos.cuda()
    full_q = full_q.cuda().reshape(num_tokens, -1)
    full_k = full_k.cuda().reshape(num_tokens, -1)

    cor_q, cor_k = rope(full_pos, torch.clone(full_q), torch.clone(full_k))

    # simulated result
    test_q = torch.zeros_like(cor_q).squeeze(0)
    test_k = torch.zeros_like(cor_k).squeeze(0)
    for sp_idx in sp_indices:
        q, k = rope(
            full_pos[sp_idx], torch.clone(full_q[sp_idx]), torch.clone(full_k[sp_idx])
        )
        test_q[sp_idx] = q
        test_k[sp_idx] = k
    test_q = test_q.reshape(cor_q.shape)
    test_k = test_k.reshape(cor_k.shape)

    torch.testing.assert_close(cor_q, test_q)
    torch.testing.assert_close(cor_k, test_k)


main()
