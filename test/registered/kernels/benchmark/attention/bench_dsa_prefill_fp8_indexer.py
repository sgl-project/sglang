"""FP8 DSA prefill indexer candidate filter vs DeepGEMM `fp8_mqa_logits`
(the dense logits producer it replaces, which does strictly less work)."""

import sys
from pathlib import Path

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsa.litetopk_prefill.fp8_indexer import (
    fp8_indexer_candidates,
)
from sglang.test.ci.ci_register import register_cuda_ci

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ops" / "attention"))
from test_dsa_prefill_fp8_indexer import (  # noqa: E402
    CAP,
    TOPK,
    gen_inputs,
    make_ragged,
)

register_cuda_ci(est_time=120, stage="nightly", runner_config="4-gpu-b200")

K = 1024
CELLS = {
    "a64k": lambda rows: [(64 * K, rows, "last")],
    "a256k": lambda rows: [(256 * K, rows, "last")],
    "a1m": lambda rows: [(1024 * K, rows, "last")],
    "c_mixed": lambda rows: [
        (1024 * K, rows // 4, "last"),
        (256 * K, rows // 4, "last"),
        (128 * K, rows // 4, "last"),
        (64 * K, rows // 4, "last"),
    ],
    "d_short": lambda rows: [((16 + 3 * i) * K, rows // 16, "last") for i in range(16)],
}


@marker.parametrize("cell", list(CELLS), ["a64k", "d_short"])
@marker.parametrize("rows", [1024, 4096, 8192], [1024])
@marker.parametrize("heads", [32, 64], [32, 64])
@marker.benchmark("impl", ["ours", "deepgemm"], unit="us")
def benchmark(cell: str, rows: int, heads: int, impl: str):
    if torch.cuda.get_device_capability()[0] != 10:
        return marker.skip("SM100 only")
    ks, ke, keys = make_ragged(CELLS[cell](rows))
    q, w, kv, sf = gen_inputs(rows, keys, heads=heads)
    if impl == "ours":
        out = fp8_indexer_candidates(q, w, kv, sf, ks, ke, TOPK, cap=CAP)
        fn = lambda: fp8_indexer_candidates(
            q, w, kv, sf, ks, ke, TOPK, cap=CAP, out=out
        )
    else:
        import deep_gemm

        max_len = int((ke - ks).max())
        fn = lambda: deep_gemm.fp8_mqa_logits(
            q, (kv, sf), w, ks, ke, clean_logits=False, max_seqlen_k=max_len
        )
    # The KV stream dominates and is far larger than L2 at long contexts; the
    # kernels run >= 100 us, so no graph is needed and no clone either.
    return marker.do_bench(
        fn,
        use_cuda_graph=False,
        graph_clone_args=(),
        memory_args=(kv,),
        memory_output=None,
    )


if __name__ == "__main__":
    benchmark.run()
