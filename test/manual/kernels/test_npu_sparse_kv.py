"""NPU numerical checks for sparse KV publication, including graph replay."""

import pytest
import torch

pytest.importorskip("torch_npu")

from sglang.kernels.ops.kvcache import prepare_sparse_kv_publication

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")


@pytest.mark.parametrize(
    "mode,q",
    [("decode", 1), ("verify", 4), ("verify", 14), ("compact", 1), ("padded", 1)],
)
@pytest.mark.parametrize("real_shape", [False, True])
def test_publication(mode, q, real_shape):
    prefill = mode in ("compact", "padded")
    lengths = [2, 0, 3] if mode == "compact" else [2, 0, 1]
    if real_shape:
        lengths = lengths + [1] * 13
    batch = len(lengths)
    tokens = (
        sum(lengths) if mode == "compact" else (batch * 3 if prefill else batch * q + 2)
    )
    device = "npu"
    if real_shape:
        k = torch.randn(tokens, 1, 512, device=device, dtype=torch.bfloat16)
        rope = torch.randn(tokens, 1, 64, device=device, dtype=torch.bfloat16)
    else:
        k = torch.randn(tokens, 1, 16, device=device, dtype=torch.bfloat16)[:, :, ::2]
        rope = k[:, :, ::2]
    req = torch.tensor(
        [1, 0, 5] + list(range(6, batch + 3)),
        device=device,
        dtype=torch.int64 if real_shape else torch.int32,
    )
    seq = torch.tensor([10, 5, 8] + [9] * (batch - 3), device=device, dtype=torch.int32)
    positions = (
        torch.arange(
            tokens, device=device, dtype=torch.int64 if real_shape else torch.int32
        )
        + 5
    )
    loc = (
        torch.arange(
            tokens, device=device, dtype=torch.int64 if real_shape else torch.int32
        )
        - 1
    )
    if tokens > 3:
        loc[3] = 100
    ext = torch.tensor(lengths, device=device, dtype=torch.int32)
    prefix = torch.tensor([3, 4], device=device, dtype=torch.int32)

    def call():
        return prepare_sparse_kv_publication(
            k,
            rope,
            req,
            seq,
            positions,
            loc,
            request_rows=20,
            context_len=20,
            host_capacity=50,
            query_width=q,
            extend_seq_lens=ext if prefill else None,
            extend_prefix_lens=prefix if prefill else None,
            compact_prefill=mode == "compact",
        )

    def check(result):
        src, dst, pos, valid, current = result
        req_cpu, seq_cpu, loc_cpu = req.cpu(), seq.cpu(), loc.cpu()
        lengths = ext.cpu().tolist()
        expected_pos, expected_valid = [], []
        for t in range(tokens):
            if prefill:
                if mode == "compact":
                    row = next(i for i in range(batch) if t < sum(lengths[: i + 1]))
                    offset = t - sum(lengths[:row])
                else:
                    row, offset = divmod(t, tokens // batch)
                p = (int(prefix[row]) if row < prefix.numel() else 0) + offset
                live = req_cpu[row] > 0 and offset < lengths[row] and 0 <= p < 20
            else:
                row = t // q
                p = int(positions[t]) if row < batch else -1
                live = (
                    row < batch
                    and 0 < req_cpu[row] < 20
                    and seq_cpu[row] > 0
                    and 0 <= p < 20
                )
            expected_pos.append(p)
            expected_valid.append(bool(live and 0 <= loc_cpu[t] < 50))
        torch.testing.assert_close(src.cpu(), torch.arange(tokens), rtol=0, atol=0)
        torch.testing.assert_close(dst.cpu(), loc_cpu.to(torch.int64), rtol=0, atol=0)
        torch.testing.assert_close(
            pos.cpu(), torch.tensor(expected_pos), rtol=0, atol=0
        )
        torch.testing.assert_close(
            valid.cpu(), torch.tensor(expected_valid), rtol=0, atol=0
        )
        if prefill:
            assert current[0] is k and current[1] is rope
        else:
            torch.testing.assert_close(
                current, torch.cat((k, rope), dim=-1), rtol=0, atol=0
            )

    check(call())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = call()
    positions.add_(2)
    loc.add_(1)
    req[0] = 0
    seq[2] = 0
    if mode == "compact":
        ext[0], ext[2] = 3, 2
    else:
        ext[0] = 1
    prefix[0] += 1
    graph.replay()
    torch.npu.synchronize()
    check(result)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
