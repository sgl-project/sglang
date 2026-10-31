# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DCP verify fp8 query gather against the bf16 gather + cat + cast it must match.

python -m pytest test/registered/unit/layers/test_dcp_fp8_q_gather.py -v
"""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import torch

from sglang.srt.layers.dcp import comm
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_WORLD = 8
_HEADS = 12
_NOPE = 512
_ROPE = 64


class _ThreadGroup:
    """Byte-copy all_gather along dim 0 across one thread per rank."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self._barrier = threading.Barrier(world_size)
        self._parts = [None] * world_size
        self._local = threading.local()

    def bind(self, rank: int) -> None:
        self._local.rank = rank

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == 0
        self._parts[self._local.rank] = input_.clone()
        self._barrier.wait()
        out = torch.cat(self._parts, dim=0)
        self._barrier.wait()
        return out


def _rank_queries(*, rank: int, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(rank)
    # The absorb BMM writes [H, B, nope]; the rope half is a slice of q_b_proj.
    q_nope_out = torch.randn(_HEADS, num_tokens, _NOPE, generator=gen) * 64
    q = torch.randn(num_tokens, _HEADS, 128 + _ROPE, generator=gen) * 64
    # Past the e4m3 range and below its subnormals.
    q_nope_out[0, 0, :4] = torch.tensor([1e4, -1e4, 1e-9, -1e-9])
    return (
        q_nope_out.to(torch.bfloat16).transpose(0, 1),
        q.to(torch.bfloat16)[..., 128:],
    )


def _run_ranks(fn, *, num_tokens: int):
    group = _ThreadGroup(_WORLD)

    def run(rank: int):
        group.bind(rank)
        q_nope_out, q_pe = _rank_queries(rank=rank, num_tokens=num_tokens)
        return fn(q_nope_out=q_nope_out, q_pe=q_pe)

    with get_parallel().override(dcp_group=group):
        with ThreadPoolExecutor(max_workers=_WORLD) as pool:
            return list(pool.map(run, range(_WORLD)))


class TestDcpFp8QueryGather(CustomTestCase):
    def test_matches_cast_after_bf16_gather(self):
        # The cprr verify kernel casts its query at unit scale;
        # the gather copies bytes, so casting first must give the same bytes per rank.
        for fp8_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            for num_tokens in (5, 260):
                with self.subTest(dtype=fp8_dtype, num_tokens=num_tokens):

                    def bf16_then_cast(*, q_nope_out, q_pe):
                        q_nope_out, q_pe = comm.all_gather_q_for_mla_decode(
                            q_nope_out=q_nope_out, q_pe=q_pe
                        )
                        q = torch.cat([q_nope_out, q_pe], dim=-1)
                        rows = q.shape[0]
                        return q.view(rows, -1).to(fp8_dtype).view(q.shape)

                    def fp8_gather(*, q_nope_out, q_pe):
                        return comm.all_gather_q_fp8_for_mla_verify(
                            q_nope_out=q_nope_out, q_pe=q_pe, fp8_dtype=fp8_dtype
                        )

                    want = _run_ranks(bf16_then_cast, num_tokens=num_tokens)
                    got = _run_ranks(fp8_gather, num_tokens=num_tokens)
                    for rank in range(_WORLD):
                        self.assertEqual(
                            got[rank].shape,
                            (num_tokens, _HEADS * _WORLD, _NOPE + _ROPE),
                        )
                        self.assertTrue(got[rank].is_contiguous())
                        self.assertTrue(
                            torch.equal(
                                got[rank].view(torch.uint8),
                                want[rank].view(torch.uint8),
                            )
                        )


if __name__ == "__main__":
    unittest.main()
