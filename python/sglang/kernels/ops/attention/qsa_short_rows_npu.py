"""Select QSA blocks in an ordinary packed prefill with 4 tokens per block and K=512."""
from functools import lru_cache

import torch


@lru_cache(maxsize=64)
def packed_short_ranges(prefixes, sizes, rows):
    if len(prefixes) != len(sizes) or sum(sizes) != rows:
        return ()
    spans = []
    offset = 0
    for prefix, count in zip(prefixes, sizes):
        if min(prefix,count) < 0:
            return ()
        short = max(0, min(count, (512 + 1)*4 - 1 - prefix))
        if short:
            if spans and spans[-1][1] == offset:
                spans[-1] = (spans[-1][0], offset+short)
            else:
                spans.append((offset,offset+short))
        offset += count
    return tuple((a,b) for a,b in spans if b-a >= 128)


def select_packed_groups(q, keys, starts, ends, spans, chunk, *, row_offset=0, total_rows=None):
    from sgl_kernel_npu.implementation.triton.indexer import qsa_mqa, qsa_topk

    rows, width = q.size(0), keys.size(0)
    out = torch.empty((rows,512), dtype=torch.int32, device=q.device)
    lengths = ends-starts
    total_rows = rows if total_rows is None else total_rows
    for base in range(row_offset // chunk * chunk, row_offset + rows, chunk):
        stop = min(base + chunk, total_rows)
        begin, end = max(0, base - row_offset), min(rows, stop - row_offset)
        implementation = qsa_topk.select_implementation(stop-base,width,512)

        def score(lo,hi):
            if lo == hi:
                return
            s = slice(lo,hi)
            logits = qsa_mqa.packed(q[s],keys,starts[s],ends[s])
            if implementation == "hybrid":
                groups = qsa_topk.hybrid_topk(logits,lengths[s],512,starts[s])
            else:
                groups = qsa_topk.fast_topk(logits,lengths[s],512,starts[s])
            out[s].copy_(groups)

        cursor = begin
        for a,b in spans:
            lo,hi = max(a-row_offset,begin),min(b-row_offset,end)
            if hi <= lo:
                continue
            score(cursor,lo)
            n = hi-lo
            cores = min(qsa_topk._vector_cores(q.device.index),(n+3)//4)
            qsa_topk._short[(cores,)](lengths[lo:hi],out[lo:hi],n,512,4,cores)
            cursor = hi
        score(cursor,end)
    return out
