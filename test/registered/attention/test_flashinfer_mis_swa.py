"""Logical-window MIS kernel against an independent FP64 dense oracle."""

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashinfer_mis_swa import MultiItemSWAWrapper
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-small")


def reference_mask(prefix_len, item_lens, window_left):
    # Labels describe semantic tokens, not offsets reconstructed by the kernel.
    tokens = [("query", -1, p) for p in range(prefix_len)]
    for item, length in enumerate(item_lens):
        tokens.append(("delimiter", item, prefix_len - 1))
        tokens.extend(("item", item, prefix_len + p) for p in range(length))
    tokens.append(("delimiter", len(item_lens), prefix_len - 1))
    mask = torch.zeros((len(tokens), len(tokens)), dtype=torch.bool)
    for row, (qkind, qitem, qpos) in enumerate(tokens):
        for col, (kkind, kitem, kpos) in enumerate(tokens):
            visible = kkind == "query" or (
                qkind == "item" and kkind == "item" and qitem == kitem
            )
            mask[row, col] = (
                visible
                and kpos <= qpos
                and (window_left < 0 or qpos - kpos <= window_left)
            )
    return mask


def metadata(prefix_len, item_lens):
    positions = []
    for length in item_lens:
        positions.extend(range(length + 1))
    positions.append(0)
    return positions


def _check_reference():
    mask = reference_mask(3, [2, 1], 2)
    # Packed: Q0 Q1 Q2 D A0 A1 D B0 D; logical B0=3.
    assert mask[7].nonzero().flatten().tolist() == [1, 2, 7]
    assert mask[5].nonzero().flatten().tolist() == [2, 4, 5]
    assert mask[2].nonzero().flatten().tolist() == [0, 1, 2]
    assert reference_mask(3, [2, 1], 0)[7].nonzero().flatten().tolist() == [7]
    assert reference_mask(0, [1, 2], 1)[4].nonzero().flatten().tolist() == [3, 4]
    assert not reference_mask(0, [1], 1)[0].any()
    print("PASS independent reference mask and exact boundaries")


def _check_batched_variant(dtype=torch.float16):
    _check_reference()
    torch.manual_seed(50896)
    cases = [
        (12, [4, 4]),
        (16, [8] * 6),
        (64, [8] * 4),
        (1, [1, 31, 32, 33]),
        (0, [1, 7]),
        (80, [1, 70, 3]),
    ]
    lengths = [p + sum(n + 1 for n in its) + 1 for p, its in cases]
    n = sum(lengths)
    ptr = (
        torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).to("cuda", torch.int32)
    )
    q = torch.randn(n, 4, 64, device="cuda", dtype=dtype)
    k = torch.randn(n, 1, 2, 64, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    stride = max(len(metadata(p, its)) for p, its in cases)
    positions = [metadata(p, its) for p, its in cases]
    params = SimpleNamespace(
        prefix_len_ptr=torch.tensor(
            [p for p, _ in cases], device="cuda", dtype=torch.uint32
        ),
        token_pos_in_items_ptr=torch.tensor(
            [s + [0] * (stride - len(s)) for s in positions],
            device="cuda",
            dtype=torch.uint16,
        ).flatten(),
        token_pos_in_items_len=stride,
    )
    wrapper = MultiItemSWAWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        dtype,
        dtype,
        64,
        64,
    )
    wrapper.multi_item_params = params
    wrapper.plan(
        ptr,
        ptr,
        torch.arange(n, device="cuda", dtype=torch.int32),
        torch.ones(len(cases), device="cuda", dtype=torch.int32),
        4,
        2,
        64,
        1,
        q_data_type=dtype,
        kv_data_type=dtype,
        window_left=-1,
    )
    for window in [-1, 0, 1, 31, 32, 63, 127]:
        for cap in [0.0, 50.0]:
            out = wrapper.forward(
                q,
                (k, v),
                causal=True,
                window_left=window,
                sm_scale=1 / math.sqrt(64),
                logits_soft_cap=cap,
            )
            offset = 0
            for (p, its), length in zip(cases, lengths):
                qq = q[offset : offset + length].double().transpose(0, 1)
                kk = (
                    k[offset : offset + length, 0]
                    .double()
                    .repeat_interleave(2, 1)
                    .transpose(0, 1)
                )
                vv = (
                    v[offset : offset + length, 0]
                    .double()
                    .repeat_interleave(2, 1)
                    .transpose(0, 1)
                )
                logits = (qq @ kk.transpose(-1, -2)) / math.sqrt(64)
                if cap:
                    logits = cap * torch.tanh(logits / cap)
                mask = reference_mask(p, its, window).cuda()
                prob = torch.softmax(
                    logits.masked_fill(~mask, -torch.inf), -1
                ).nan_to_num()
                expected = (prob @ vv).transpose(0, 1).float()
                torch.testing.assert_close(
                    out[offset : offset + length].float(),
                    expected,
                    atol=0.003 if dtype == torch.float16 else 0.02,
                    rtol=0.003 if dtype == torch.float16 else 0.02,
                )
                offset += length
            print(
                f"PASS {dtype} batched kernel window={window} softcap={cap}", flush=True
            )


class TestFlashInferMISSWA(unittest.TestCase):
    def test_reference_and_batched_kernel(self):
        _check_batched_variant()

    def test_bfloat16(self):
        _check_batched_variant(torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
