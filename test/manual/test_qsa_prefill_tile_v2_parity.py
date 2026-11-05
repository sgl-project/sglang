"""Source-bound CUDA parity; refuses CPU and never relaxes ordered-ID checks."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--widths",
        nargs="+",
        type=int,
        default=[511, 512, 513, 4096, 22000, 32768, 65536],
    )
    parser.add_argument(
        "--rows", nargs="+", type=int, default=[1, 31, 32, 33, 128, 1024, 1504]
    )
    args = parser.parse_args()
    if not args.out.is_absolute() or min(args.widths + args.rows) <= 0:
        parser.error("absolute own output and positive geometries required")
    root = args.source_root.resolve()
    sys.path.insert(0, str(root / "python"))
    result = {
        "status": "NOT_RUN",
        "cases": [],
        "geometry_scope": "synthetic ABI, not observed serving operands",
    }
    rc = 1
    try:
        import torch

        from sglang.srt.layers.attention.qsa import mqa
        from sglang.srt.layers.attention.qsa.kernel import (
            expand_qsa_block_indices,
            qsa_fast_topk,
        )
        from sglang.srt.layers.attention.qsa.qsa_indexer import (
            _qsa_prefill_row_chunk_size,
        )

        if not torch.cuda.is_available():
            result["status"] = "REFUSED_CPU_NOT_CUDA_PARITY"
            return 2
        if not mqa.HAS_TILELANG:
            raise RuntimeError("TileLang required; no scorer substitution")
        result["environment"] = dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            capability=torch.cuda.get_device_capability(),
            device=torch.cuda.get_device_name(),
        )
        result["source_leaves"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "python/sglang/srt/environ.py",
                "python/sglang/srt/layers/attention/qsa/mqa.py",
                "test/manual/test_qsa_prefill_tile_v2_parity.py",
            )
        }
        original_tile, original_fold = (
            mqa._QSA_PREFILL_TILE_V2,
            mqa._QSA_PREFILL_SCALE_FOLD,
        )
        generator = torch.Generator(device="cuda").manual_seed(9352)
        try:
            for width in args.widths:
                request_lengths = [width, max(width // 2, 1), max(width // 4, 1)]
                request_starts = [0, request_lengths[0], sum(request_lengths[:2])]
                for requested_rows in args.rows:
                    rows = _qsa_prefill_row_chunk_size(requested_rows, width, 4)
                    rows = min(rows, requested_rows)
                    lengths_host = [
                        max(
                            1,
                            request_lengths[i % 3]
                            - i % min(request_lengths[i % 3], 17),
                        )
                        for i in range(rows)
                    ]
                    starts_host = [request_starts[i % 3] for i in range(rows)]
                    starts = torch.tensor(starts_host, dtype=torch.int32, device="cuda")
                    lengths = torch.tensor(
                        lengths_host, dtype=torch.int32, device="cuda"
                    )
                    ends, zeros = starts + lengths, torch.zeros_like(starts)
                    q = torch.randn(
                        rows,
                        4,
                        128,
                        dtype=torch.bfloat16,
                        device="cuda",
                        generator=generator,
                    )
                    q[0].zero_()  # All cutoff ties: IDs must retain the native order.
                    k = torch.randn(
                        sum(request_lengths),
                        1,
                        128,
                        dtype=torch.bfloat16,
                        device="cuda",
                        generator=generator,
                    )
                    k[::11] = k[0].clone()  # Repeated keys and large tie buckets.
                    positions = lengths * 4 - 1
                    sequence_lengths = lengths * 4
                    mask = (
                        torch.arange(width, device="cuda")[None, :] < lengths[:, None]
                    )

                    def launch(enabled):
                        mqa._QSA_PREFILL_TILE_V2 = enabled
                        logits = mqa.tilelang_qsa_mqa_prefill_relative(
                            q, k, starts, ends, width
                        )
                        ids = qsa_fast_topk(logits, zeros, lengths, topk=512)
                        tokens = expand_qsa_block_indices(
                            ids,
                            positions,
                            sequence_lengths,
                            compress_ratio=4,
                            token_topk=2048,
                        )
                        return logits, ids, tokens

                    def compare(a, b):
                        bits = int(
                            (
                                a[0][mask].view(torch.int32)
                                != b[0][mask].view(torch.int32)
                            ).sum()
                        )
                        finite = bool(
                            torch.isfinite(a[0][mask]).all()
                            and torch.isfinite(b[0][mask]).all()
                        )
                        id_mismatches = int((a[1] != b[1]).sum())
                        token_mismatches = int((a[2] != b[2]).sum())
                        assert (
                            bits == id_mismatches == token_mismatches == 0 and finite
                        ), (bits, id_mismatches, token_mismatches, finite)
                        return dict(
                            score_bit_mismatches=bits,
                            ordered_id_mismatches=id_mismatches,
                            expanded_token_mismatches=token_mismatches,
                            finite=finite,
                        )

                    for folded in (False, True):
                        mqa._QSA_PREFILL_SCALE_FOLD = folded
                        # Compile/warm before capture. Undefined columns are never compared.
                        a, b = launch(False), launch(True)
                        record = dict(
                            rows=rows,
                            requested_rows=requested_rows,
                            width=width,
                            keys=k.shape[0],
                            heads=4,
                            head_dim=128,
                            dtype=str(q.dtype),
                            actual_lengths=lengths_host,
                            actual_starts=starts_host,
                            topk=512,
                            folded=folded,
                            eager=compare(a, b),
                        )
                        for operand in (b[0].view(torch.int32), b[1]):
                            operand[0, 0].bitwise_xor_(1)
                            try:
                                try:
                                    compare(a, b)
                                except AssertionError:
                                    pass
                                else:
                                    raise RuntimeError(
                                        "parity comparator accepted a mutation"
                                    )
                            finally:
                                operand[0, 0].bitwise_xor_(1)
                        record["score_and_ordered_id_mutations_rejected"] = True
                        graphs, outputs = [], []
                        for enabled in (False, True):
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                captured = launch(enabled)
                            graphs.append(graph)
                            outputs.append(captured)
                        for graph in graphs:
                            graph.replay()
                        record["capture_replay"] = compare(*outputs)
                        q.add_(torch.tensor(0.125, dtype=q.dtype, device=q.device))
                        q[0].zero_()
                        for graph in graphs:
                            graph.replay()
                        expected = launch(False)
                        record["changed_input_replay"] = compare(*outputs)
                        compare(expected, outputs[1])
                        torch.cuda.synchronize()
                        result["cases"].append(record)
                        del a, b, graphs, outputs, expected, captured
            result["status"] = "PASS_REAL_CUDA_SCORE_BITS_ORDERED_IDS_TOKENS_AND_REPLAY"
            rc = 0
        finally:
            mqa._QSA_PREFILL_TILE_V2, mqa._QSA_PREFILL_SCALE_FOLD = (
                original_tile,
                original_fold,
            )
    except Exception:
        result["status"] = "FAILED_NO_RETRY_NO_TOLERANCE_RELAXATION"
        result["error"] = traceback.format_exc()
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
