"""AITER MLA decode in deterministic mode on gfx950.

The aiter backend runs MLA decode in deterministic mode on the split-KV kernel
with a fixed KV split count, at 16 heads per GPU (after padding). A request's
output must then not depend on its batchmates: its row is bitwise equal alone,
in batches of any size and in a repeat of the same batch.
"""

import unittest

import torch

from sglang.srt.utils import is_hip
from sglang.srt.utils.device import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

_RUNNABLE = is_hip() and is_gfx95_supported()
if _RUNNABLE:
    from aiter.mla import mla_decode_fwd

    from sglang.srt.layers.attention.aiter_backend import (
        _DETERMINISTIC_MLA_KV_SPLITS,
    )

_DEVICE = "cuda"
_NUM_HEADS = 16
_KV_LORA_RANK = 512
_QK_ROPE_DIM = 64
_QK_HEAD_DIM = _KV_LORA_RANK + _QK_ROPE_DIM
# Every length up to 200 (splits shorter than the kernel's 16- and 64-token
# blocks, and empty splits), then lengths around larger powers of two.
_KV_LENS = list(range(1, 201)) + [255, 256, 257, 511, 512, 513, 1023, 1024, 1025]
_KV_LENS += [2047, 2048, 2049, 4096, 4097, 8191, 8192, 8193]


def _make_requests(seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    num_slots = sum(_KV_LENS)
    kv_buffer = torch.randn(
        (num_slots, _QK_HEAD_DIM), generator=gen, dtype=torch.float32
    ).to(device=_DEVICE, dtype=torch.bfloat16)
    slots = torch.randperm(num_slots, generator=gen).to(torch.int32)
    indices = [idx.to(_DEVICE) for idx in torch.split(slots, _KV_LENS)]
    q = torch.randn(
        (len(_KV_LENS), _NUM_HEADS, _QK_HEAD_DIM), generator=gen, dtype=torch.float32
    ).to(device=_DEVICE, dtype=torch.bfloat16)
    return q, kv_buffer, indices


def _decode(q, kv_buffer, indices, rows):
    bs = len(rows)
    lens = torch.tensor([indices[r].numel() for r in rows], dtype=torch.int32)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(lens, dim=0)
    o = torch.empty(
        (bs, _NUM_HEADS, _KV_LORA_RANK), dtype=torch.bfloat16, device=_DEVICE
    )
    mla_decode_fwd(
        q[rows].contiguous(),
        kv_buffer.view(-1, 1, 1, _QK_HEAD_DIM),
        o,
        torch.arange(bs + 1, dtype=torch.int32, device=_DEVICE),
        kv_indptr.to(_DEVICE),
        torch.cat([indices[r] for r in rows]),
        torch.ones(bs, dtype=torch.int32, device=_DEVICE),
        1,
        sm_scale=_QK_HEAD_DIM**-0.5,
        num_kv_splits=_DETERMINISTIC_MLA_KV_SPLITS,
    )
    return o


@unittest.skipUnless(_RUNNABLE, "AITER MLA decode on gfx950 only")
class TestAiterMlaDecodeDeterministic(CustomTestCase):
    def test_rows_do_not_depend_on_the_batch(self):
        q, kv_buffer, indices = _make_requests(seed=0)
        num_reqs = len(_KV_LENS)
        full = _decode(q, kv_buffer, indices, list(range(num_reqs)))
        gen = torch.Generator(device="cpu").manual_seed(1)
        batches = [list(range(num_reqs))] + [[r] for r in range(num_reqs)]
        for size in (2, 3, 8, 17, 31, 64, 128):
            batches.append(torch.randperm(num_reqs, generator=gen)[:size].tolist())
        differing = set()
        for rows in batches:
            out = _decode(q, kv_buffer, indices, rows)
            differing.update(
                _KV_LENS[r]
                for i, r in enumerate(rows)
                if not torch.equal(out[i], full[r])
            )
        self.assertEqual(sorted(differing), [], "KV lengths whose rows changed")

    def test_fixed_splits_match_reference(self):
        """Empty and short splits must not drop or double-count KV."""
        q, kv_buffer, indices = _make_requests(seed=2)
        rows = [0, 6, 15, 62, 63, 64, 199, len(_KV_LENS) - 1]
        out = _decode(q, kv_buffer, indices, rows)
        for i, r in enumerate(rows):
            kv = kv_buffer[indices[r].long()].float()
            scores = (q[r].float() @ kv.T) * _QK_HEAD_DIM**-0.5
            ref = torch.softmax(scores, dim=-1) @ kv[:, :_KV_LORA_RANK]
            torch.testing.assert_close(out[i].float(), ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
