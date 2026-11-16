"""Does sgl-kernel's top-k v2 return the same selection, and the same output order, on
repeated calls? Run once per SGLANG_TOPK_V2_MODE:

    SGLANG_TOPK_V2_MODE=<1|2> PYTHONPATH=<tree>/python python3 topk_order_check.py \
        <tree>/test/registered/kernels/ops/attention/test_dsv41_topk_selector.py

Uses the parity test's tied rows, page tables and shapes; three calls per selection on the
same inputs, exact int32 outputs (order) and per-row sorted outputs (set) compared.
"""

import runpy
import sys
from types import SimpleNamespace

import torch

g = runpy.run_path(sys.argv[1], run_name="order_check")
from sglang.srt.environ import envs  # noqa: E402

print("SGLANG_TOPK_V2_MODE", envs.SGLANG_TOPK_V2_MODE.get(), flush=True)
with torch.inference_mode():
    for row_lens, width, k in (
        ([0, 1, 511, 512, 513, 3000, 8192], 8192, 512),
        ([65536, 40000], 65536, 2048),
        ([131072] * 34, 131072, 512),
    ):
        lens = torch.tensor(row_lens, dtype=torch.int32, device="cuda")
        rows = len(row_lens)

        def scores():
            return g["tied_scores"](lens, width, torch.float32, seed=width, stride=width + 64)

        page_table = g["random_page_table"](rows, width // 64, seed=width + 1)
        metadata = SimpleNamespace(
            compressed_seq_lens=lens,
            page_table=page_table,
            compressed_page_size=64,
            use_topk_v2=True,
            topk_metadata=g["plan_topk_v2"](lens),
        )
        result = {}
        with g["make_selector"](*g["SGL_KERNEL"]) as selector:
            for name in ("paged", "raw", "ragged", "blocks"):
                runs = []
                for _ in range(3):
                    out = torch.full((rows, k), -2, dtype=torch.int32, device="cuda")
                    if name == "paged":
                        selector.paged_from_metadata(scores(), metadata, out)
                    elif name == "raw":
                        selector.raw(scores(), lens, out=out, plan=g["plan_topk_v2"](lens))
                    elif name == "ragged":
                        selector.ragged(scores(), lens, out=out, offsets=torch.zeros_like(lens))
                    else:
                        logits = g["tied_scores"](lens, width, torch.bfloat16, seed=width + 2)
                        out = selector.amax_blocks(logits, lens, (lens + 7) // 8, min(2048, width // 16))
                    torch.cuda.synchronize()
                    runs.append(out.clone())
                order = all(torch.equal(runs[0], r) for r in runs[1:])
                picks = all(
                    torch.equal(runs[0].sort(dim=1).values, r.sort(dim=1).values) for r in runs[1:]
                )
                result[name] = (order, picks)
        print(
            f"width {width} k {k}: "
            + ", ".join(
                f"{n}: set {'same' if s else 'differs'}, order {'same' if o else 'differs'}"
                for n, (o, s) in result.items()
            ),
            flush=True,
        )
