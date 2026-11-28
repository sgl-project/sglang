"""gfx942 Triton FP4 MQA logits for the DeepSeek V4.1 low-ratio indexer.

* the full-context scorer matches a torch reference of
  ``sum_h relu(q . k) * w`` over the swizzled FP4 index-K pool;
* the candidate-only scorer is bitwise identical to scoring every position and
  gathering the candidate blocks (``gather_candidate_blocks``), including the
  ``-inf`` of unreachable columns, so the level-two top-k is unchanged.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, stage="base-b", runner_config="1-gpu-small-amd")

PAGE = 64
D = 128
CBS = 8
TOPK_BLOCKS = 2048


def _is_gfx942() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is not None
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx942")
    )


def _make_case(lens, heads, seed, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = len(lens)
    width = (max(lens) + PAGE - 1) // PAGE + 1
    n_pages = sum((n + PAGE - 1) // PAGE for n in lens) + 8
    payload = torch.randint(
        0, 256, (n_pages, 1, 4, 64, 16), generator=g, dtype=torch.uint8
    )
    scale = torch.randint(118, 136, (n_pages, 1, 4, 64), generator=g, dtype=torch.uint8)
    perm = torch.randperm(n_pages, generator=g).to(torch.int32)
    page_table = torch.zeros((rows, width), dtype=torch.int32)
    off = 0
    for r, n in enumerate(lens):
        k = (n + PAGE - 1) // PAGE
        page_table[r, :k] = perm[off : off + k]
        off += k
    q = torch.randn((rows, heads, D), generator=g).to(torch.bfloat16)
    w = (torch.randn((rows, heads), generator=g) * 0.1).to(torch.bfloat16)
    ids = torch.full((rows, TOPK_BLOCKS), -1, dtype=torch.int32)
    for r, n in enumerate(lens):
        nb = (n + CBS - 1) // CBS
        k = min(TOPK_BLOCKS, nb)
        pick = torch.cat(
            [torch.randperm(nb - 1, generator=g)[: k - 1], torch.tensor([nb - 1])]
        )
        ids[r, :k] = pick[torch.randperm(k, generator=g)].to(torch.int32)
    to = lambda t: t.to(device)
    return dict(
        q=to(q),
        weights=to(w),
        k_payload=to(payload),
        k_scale=to(scale),
        page_table=to(page_table),
        c4_seq_lens=to(torch.tensor(lens, dtype=torch.int32)),
        ids=to(ids),
        max_seq_len=width * PAGE,
    )


def _reference_logits(c):
    """Dequantize the swizzled FP4 pool in torch and score every position."""
    mags = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device="cuda"
    )
    out = torch.full((c["q"].shape[0], c["max_seq_len"]), float("nan"), device="cuda")
    d = torch.arange(D, device="cuda")
    seg, within = d // 32, (d // 2) % 16
    for t, n in enumerate(c["c4_seq_lens"].tolist()):
        pos = torch.arange(n, device="cuda")
        page = c["page_table"][t, pos // PAGE].long()
        slot = pos % PAGE
        byte = c["k_payload"][
            page[:, None], 0, seg[None, :], slot[:, None], within[None, :]
        ].int()
        nib = torch.where(d[None, :] % 2 == 1, (byte >> 4) & 0xF, byte & 0xF)
        k = mags[nib & 7] * torch.where((nib >> 3) & 1 == 1, -1.0, 1.0)
        exp = c["k_scale"][page[:, None], 0, seg[None, :], slot[:, None]].float()
        k = k * torch.exp2(exp - 127)
        s = torch.relu(k @ c["q"][t].float().T) @ c["weights"][t].float()
        out[t, :n] = s
    return out


def _gather(logits, seq_lens, ids):
    width = logits.shape[1]
    j = torch.arange(ids.shape[1] * CBS, device=logits.device)
    blk = ids[:, j // CBS].long()
    pos = blk * CBS + j % CBS
    ok = (blk >= 0) & (pos < seq_lens[:, None].long()) & (pos < width)
    vals = torch.gather(logits, 1, pos.clamp(0, width - 1))
    return torch.where(ok, vals, torch.full_like(vals, float("-inf")))


@unittest.skipUnless(_is_gfx942(), "gfx942 only")
class TestTritonMqaLogitsGfx942(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.attention.dsv4 import triton_mqa_logits_gfx942 as m

        self.m = m

    def _full(self, c):
        return self.m.triton_paged_mqa_logits(
            q=c["q"],
            weights=c["weights"],
            k_payload=c["k_payload"],
            k_scale=c["k_scale"],
            page_table=c["page_table"],
            c4_seq_lens=c["c4_seq_lens"],
            max_seq_len=c["max_seq_len"],
            page_size=PAGE,
        )

    def test_full_matches_reference(self):
        c = _make_case([1, 63, 64, 1000, 4097], heads=64, seed=0)
        got, ref = self._full(c), _reference_logits(c)
        for t, n in enumerate(c["c4_seq_lens"].tolist()):
            torch.testing.assert_close(got[t, :n], ref[t, :n], rtol=2e-2, atol=2e-2)

    def test_candidate_only_is_bitwise_gather(self):
        cases = [
            ([16385, 20000, 70000, 131072], 64, 1),
            ([131072] * 8, 64, 2),
            ([300001, 524288, 1048000], 64, 3),
            ([17000 + 37 * i for i in range(32)], 64, 4),
            ([65536, 99999], 32, 5),
            ([100, 16384, 16391], 64, 6),
        ]
        for lens, heads, seed in cases:
            with self.subTest(lens=lens[:4], rows=len(lens), heads=heads):
                c = _make_case(lens, heads, seed)
                ref = _gather(self._full(c), c["c4_seq_lens"], c["ids"])
                got = self.m.triton_candidate_mqa_logits(
                    q=c["q"],
                    weights=c["weights"],
                    k_payload=c["k_payload"],
                    k_scale=c["k_scale"],
                    page_table=c["page_table"],
                    c4_seq_lens=c["c4_seq_lens"],
                    ids=c["ids"],
                    block_size=CBS,
                    page_size=PAGE,
                )
                self.assertTrue(
                    torch.equal(ref.view(torch.int32), got.view(torch.int32))
                )


if __name__ == "__main__":
    unittest.main()
