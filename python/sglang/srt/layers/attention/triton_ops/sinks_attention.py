import torch
import triton
import triton.language as tl


@triton.jit
def attention_sinks_kernel(
    query,
    k_cache,
    v_cache,
    sinks,
    attn_out,
    block_tables,
    kv_seq_lens,
    scale,
    sliding_window_size,
    q_head_num: tl.constexpr,
    k_head_num: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    sync_space,
):
    i_s, i_qh = tl.program_id(0), tl.program_id(1)
    i_kvh = i_qh // (q_head_num // k_head_num)

    kv_seq_len = tl.load(kv_seq_lens + i_s)
    page_num = tl.cdiv(kv_seq_len, PAGE_SIZE)
    start_page_num = 0
    start_kv_len = 0
    if sliding_window_size != -1 and kv_seq_len > sliding_window_size:
        start_kv_len = (kv_seq_len - sliding_window_size).to(tl.int32)
        start_page_num = start_kv_len // PAGE_SIZE

    cur_page_start = i_s * MAX_BLOCKS
    offset_page = tl.arange(0, PAGE_SIZE)
    offset_d = tl.arange(0, D)
    Br: tl.constexpr = 1

    sink = tl.load(sinks + i_qh)
    history_max = tl.zeros([Br], dtype=tl.float32) + sink
    l = tl.zeros([Br], dtype=tl.float32)
    acc = tl.zeros([Br, D], dtype=tl.float32)

    offset_q = i_qh * D + offset_d
    offset_seq = (tl.arange(0, Br) + i_s) * D * q_head_num
    q = tl.load(query + offset_seq[:, None] + offset_q[None, :]).to(tl.float32)

    for page_idx in range(start_page_num, page_num):
        block_idx = tl.load(block_tables + cur_page_start + page_idx)
        mask_page = ((page_idx * PAGE_SIZE + offset_page) < kv_seq_len) & ((page_idx * PAGE_SIZE + offset_page) >= start_kv_len)

        offset_k = (
            block_idx * PAGE_SIZE * k_head_num * D
            + offset_page[:, None] * k_head_num * D
            + i_kvh * D
            + offset_d[None, :]
        )
        k = tl.load(k_cache + offset_k, mask=mask_page[:, None]).to(tl.float32)
        v = tl.load(v_cache + offset_k, mask=mask_page[:, None]).to(tl.float32)

        k = tl.trans(k, (1, 0))
        qk = tl.dot(q, k)
        qk = qk * scale
        qk = tl.where(mask_page[None, :], qk, float("-inf"))

        new_e_max = tl.maximum(tl.max(qk, 1), history_max)
        re_scale = tl.exp(history_max - new_e_max)
        p_exp = tl.exp(qk - new_e_max[:, None])

        # Online softmax update
        l = l * re_scale + tl.sum(p_exp, 1)
        acc = acc * re_scale[:, None] + tl.dot(p_exp.to(v.dtype), v)
        tl.store(sync_space + tl.arange(0, Br), new_e_max)
        history_max = new_e_max

    sink = tl.math.exp(sink - history_max)
    l = l + sink
    acc = acc / l[:, None]
    tl.store(attn_out + offset_seq[:, None] + offset_q[None, :], acc.to(attn_out.type.element_ty))


def attention_sinks_triton(
    query,
    k_cache,
    v_cache,
    sinks,
    block_tables,
    context_lens,
    scale,
    sliding_window_size,
    q_head_num,
    k_head_num,
):
    S = query.shape[0]
    D = query.shape[-1] // q_head_num
    PAGE_SIZE = k_cache.shape[1]
    v_head_dim = v_cache.shape[-1]
    attn_output = torch.zeros(
        (S, q_head_num, v_head_dim),
        dtype=query.dtype,
        device=query.device,
    )
    sync_space = torch.empty(
        (PAGE_SIZE,),
        dtype=torch.float32,
        device=query.device,
    )

    if isinstance(context_lens, list):
        context_lens = torch.tensor(context_lens, device=query.device)
    else:
        context_lens = context_lens.to(query.device)
    
    grid = [S, q_head_num]
    attention_sinks_kernel[grid](
        query,
        k_cache,
        v_cache,
        sinks,
        attn_output,
        block_tables,
        context_lens,
        scale,
        sliding_window_size,
        q_head_num,
        k_head_num,
        D,
        PAGE_SIZE,
        block_tables.stride(0),
        sync_space,
    )

    return attn_output.reshape(-1, q_head_num * v_head_dim)


@triton.jit
def attention_sinks_prefill_kernel(
    query,
    k_cache,
    v_cache,
    sinks,
    attn_out,
    block_tables,
    kv_seq_lens,
    scale,
    sliding_window_size,
    q_head_num: tl.constexpr,
    k_head_num: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    B: tl.constexpr,
    BS: tl.constexpr,
    sync_space,
):
    i_ns, i_qh = tl.program_id(0), tl.program_id(1)
    i_kvh = i_qh // (q_head_num // k_head_num)
    
    for i_bs in range(BS):
        i_s = i_ns * BS + i_bs

        i_pos = -1
        kv_seq_len = i_s

        for i in range(B):
            tmp_seq_len = tl.load(kv_seq_lens + i)
            if kv_seq_len >= tmp_seq_len and i_pos == -1:
                kv_seq_len -= tmp_seq_len
            elif i_pos == -1:
                i_pos = i

        if i_pos != -1:
            kv_seq_len += 1

            page_num = tl.cdiv(kv_seq_len, PAGE_SIZE)
            start_page_num = 0
            start_kv_len = 0
            if sliding_window_size != -1 and kv_seq_len > sliding_window_size:
                start_kv_len = (kv_seq_len - sliding_window_size).to(tl.int32)
                start_page_num = start_kv_len // PAGE_SIZE
            
            cur_page_start = i_pos * MAX_BLOCKS
            offset_page = tl.arange(0, PAGE_SIZE)
            offset_d = tl.arange(0, D)
            Br: tl.constexpr = 1

            sink = tl.load(sinks + i_qh)
            history_max = tl.zeros([Br], dtype=tl.float32) + sink
            l = tl.zeros([Br], dtype=tl.float32)
            acc = tl.zeros([Br, D], dtype=tl.float32)

            offset_q = i_qh * D + offset_d
            offset_seq = (tl.arange(0, Br) + i_s) * D * q_head_num
            q = tl.load(query + offset_seq[:, None] + offset_q[None, :]).to(tl.float32)

            for page_idx in range(start_page_num, page_num):
                block_idx = tl.load(block_tables + cur_page_start + page_idx)
                mask_page = ((page_idx * PAGE_SIZE + offset_page) < kv_seq_len) & ((page_idx * PAGE_SIZE + offset_page) >= start_kv_len)

                offset_k = (
                    block_idx * PAGE_SIZE * k_head_num * D
                    + offset_page[:, None] * k_head_num * D
                    + i_kvh * D
                    + offset_d[None, :]
                )
                k = tl.load(k_cache + offset_k, mask=mask_page[:, None]).to(tl.float32)
                v = tl.load(v_cache + offset_k, mask=mask_page[:, None]).to(tl.float32)

                k = tl.trans(k, (1, 0))
                qk = tl.dot(q, k)
                qk = qk * scale
                qk = tl.where(mask_page[None, :], qk, float("-inf"))

                new_e_max = tl.maximum(tl.max(qk, 1), history_max)
                re_scale = tl.exp(history_max - new_e_max)
                p_exp = tl.exp(qk - new_e_max[:, None])

                # Online softmax update
                l = l * re_scale + tl.sum(p_exp, 1)
                acc = acc * re_scale[:, None] + tl.dot(p_exp.to(v.dtype), v)
                tl.store(sync_space + tl.arange(0, Br), new_e_max)
                history_max = new_e_max

            sink = tl.math.exp(sink - history_max)
            l = l + sink
            acc = acc / l[:, None]
            tl.store(attn_out + offset_seq[:, None] + offset_q[None, :], acc.to(attn_out.type.element_ty))



def attention_sinks_prefill_triton(
    query,
    k_cache,
    v_cache,
    sinks,
    block_tables,
    context_lens,
    scale,
    sliding_window_size,
    q_head_num,
    k_head_num,
):
    S = query.shape[0]
    kernel_num = 40
    BS = triton.cdiv(S, kernel_num)
    NS = triton.cdiv(S, BS)

    D = query.shape[-1] // q_head_num
    PAGE_SIZE = k_cache.shape[1]
    v_head_dim = v_cache.shape[-1]
    attn_output = torch.zeros(
        (S, q_head_num, v_head_dim),
        dtype=query.dtype,
        device=query.device,
    )
    sync_space = torch.empty(
        (PAGE_SIZE,),
        dtype=torch.float32,
        device=query.device,
    )

    if isinstance(context_lens, list):
        context_lens = torch.tensor(context_lens, device=query.device)
    else:
        context_lens = context_lens.to(query.device)
    B = context_lens.shape[0]

    grid = [NS, q_head_num]
    attention_sinks_prefill_kernel[grid](
        query,
        k_cache,
        v_cache,
        sinks,
        attn_output,
        block_tables,
        context_lens,
        scale,
        sliding_window_size,
        q_head_num,
        k_head_num,
        D,
        PAGE_SIZE,
        block_tables.stride(0),
        B,
        BS,
        sync_space,
    )

    return attn_output.reshape(-1, q_head_num * v_head_dim)



def native_attn_sinks(
    q,
    k,
    v,
    scale,
    sinks,
):
    seq_len_q, q_head, group, _ = q.shape
    qk = torch.einsum("qhgd,khd->hgqk", q, k)
    qk = qk * scale
    sinks = sinks.reshape(q_head, group, 1, 1).expand(q_head, group, seq_len_q, 1)
    qk = torch.cat([qk, sinks], dim=-1)
    w = torch.nn.functional.softmax(qk, dim=-1)
    w = w[..., :-1]
    out = torch.einsum("hgqk,khd->qhgd", w, v)
    return out


def native_gqa_sinks(
    query,
    k_cache,
    v_cache,
    sinks,
    block_tables,
    context_lens,
    scale,
    sliding_window_size,
    q_head_num,
    k_head_num,
    is_extend,
):
    group_size = q_head_num // k_head_num
    q_dim = query.shape[-1] // q_head_num
    k_dim = k_cache.shape[-1]
    v_dim = v_cache.shape[-1]
    page_size = k_cache.shape[1]
    out = []
    last = 0
    for i in range(len(context_lens)):
        k = []
        v = []
        seq_len = context_lens[i].item()
        for page in block_tables[i]:
            idx = min(seq_len, page_size)
            k.append(k_cache[page][:idx])
            v.append(v_cache[page][:idx])
            if seq_len <= page_size:
                break
            seq_len -= page_size

        k = torch.cat(k, dim=0)
        v = torch.cat(v, dim=0)

        if is_extend:
            q = query[last : last + context_lens[i]]
            last += context_lens[i]
        else:
            q = query[last : last + 1]
            last += 1

        o = []
        for idx in range(q.shape[0]):
            q_ = q[idx : idx + 1]
            k_ = k[: idx + 1] if is_extend else k
            v_ = v[: idx + 1] if is_extend else v
            if sliding_window_size != -1 and sliding_window_size < k_.shape[0]:
                k_ = k_[-sliding_window_size:]
                v_ = v_[-sliding_window_size:]
            q_ = q_.view(-1, k_head_num, group_size, q_dim)
            k_ = k_.view(-1, k_head_num, k_dim)
            v_ = v_.view(-1, k_head_num, v_dim)
            o_ = native_attn_sinks(q_, k_, v_, scale, sinks)
            o.append(o_)

        o = torch.cat(o, dim=0)
        out.append(o)

    out = torch.cat(out, dim=0)
    return out.reshape(-1, q_head_num * v_dim)


