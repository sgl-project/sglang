"""Independent paged-attention fixtures and FP64 oracle for gfx1250 tests."""

import torch


def make_paged_case(
    dim=64,
    dtype=torch.bfloat16,
    fp8=False,
    page=16,
    group=8,
    qlens=(7, 1, 17),
    klens=(31, 513, 2063),
    window=-1,
    causal=True,
):
    torch.manual_seed(42)
    kv_heads = 2
    pages_per_seq = [(n + page - 1) // page for n in klens]
    num_pages = sum(pages_per_seq) + 3
    k = torch.randn((num_pages, page, kv_heads, dim), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    k_scale = torch.tensor(0.37, device="cuda") if fp8 else None
    v_scale = torch.tensor(0.23, device="cuda") if fp8 else None
    if fp8:
        k = (k.float() / k_scale).to(torch.float8_e4m3fn)
        v = (v.float() / v_scale).to(torch.float8_e4m3fn)
    allocation = torch.randperm(num_pages - 3, device="cuda").to(torch.int32)
    table = torch.full(
        (len(qlens), max(pages_per_seq) + 2), -1, device="cuda", dtype=torch.int32
    )
    start = 0
    for i, n in enumerate(pages_per_seq):
        table[i, :n] = allocation[start : start + n]
        start += n
    q = torch.randn((sum(qlens), kv_heads * group, dim), device="cuda", dtype=dtype)
    qptr = torch.tensor(
        [0, *torch.tensor(qlens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor(klens, device="cuda", dtype=torch.int64)
    out = torch.full_like(q, float("nan"))
    args = dict(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=qptr,
        seqused_k=lengths,
        max_seqlen_q=max(qlens),
        max_seqlen_k=max(klens),
        softmax_scale=0.19,
        causal=causal,
        window_size=(window, 0 if causal else -1),
        block_table=table,
        k_descale=k_scale,
        v_descale=v_scale,
    )
    reference = paged_reference(args, qlens, klens)
    return args, reference


def paged_reference(args, qlens, klens):
    outputs = []
    begin = 0
    q, k, v = args["q"], args["k"], args["v"]
    page = k.shape[1]
    group = q.shape[1] // k.shape[2]
    for seq, (qlen, klen) in enumerate(zip(qlens, klens)):
        if not qlen:
            continue
        if not klen:
            outputs.append(
                torch.zeros_like(q[begin : begin + qlen], dtype=torch.float64)
            )
            begin += qlen
            continue
        ids = args["block_table"][seq, : (klen + page - 1) // page].long()
        keys = k.double()[ids].flatten(0, 1)[:klen]
        values = v.double()[ids].flatten(0, 1)[:klen]
        if args["k_descale"] is not None:
            keys *= args["k_descale"].double()
            values *= args["v_descale"].double()
        keys = keys.repeat_interleave(group, dim=1).transpose(0, 1)
        values = values.repeat_interleave(group, dim=1).transpose(0, 1)
        query = q[begin : begin + qlen].double().transpose(0, 1)
        if args.get("q_descale") is not None:
            query *= args["q_descale"].double()
        score = query @ keys.transpose(-1, -2) * args["softmax_scale"]
        key_pos = torch.arange(klen, device="cuda")[None, :]
        query_pos = torch.arange(qlen, device="cuda")[:, None] + klen - qlen
        valid = torch.ones((qlen, klen), device="cuda", dtype=torch.bool)
        if args["causal"]:
            valid &= key_pos <= query_pos
        if args["window_size"][0] >= 0:
            valid &= key_pos >= query_pos - args["window_size"][0]
        score.masked_fill_(~valid, float("-inf"))
        probability = torch.softmax(score, -1).nan_to_num(0.0)
        outputs.append((probability @ values).transpose(0, 1))
        begin += qlen
    return torch.cat(outputs)
