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

"""Unit tests for the GLM-5.2 DSA prefill context-parallel helpers.

Covers the pure-python layout / indexing / gating logic exercised by the NPU
DSA prefill CP flow (interleave & zigzag strategies) without an NPU or server:

  * ``ZigzagCPStrategy.build_metadata`` block layout and per-rank token counts
  * ``ZigzagCPStrategy.local_q_indices`` front/back block local row selection
  * ``InterleaveCPStrategy`` metadata counts and strided local row selection
  * ``pad_logical_token_to_physical`` / ``pad_local_rows`` alignment math
  * ``is_cp_active`` gating (forward mode, strategy, input_ids)
  * GLM DSA layer-split helpers in ``layers/cp/utils.py``

The strategies only read ``get_parallel()`` / ``get_device()`` through lazy
lookups, so every test runs on CPU with plain mocks.
"""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that transitively pull sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


class _FakeParallel:
    def __init__(self, cp_rank: int = 0, cp_size: int = 4):
        self.attn_cp_rank = cp_rank
        self.attn_cp_size = cp_size


class _FakeStrategy:
    """Minimal strategy stand-in for is_cp_active tests."""

    def __init__(self, cp_size: int = 4):
        self.cp_size = cp_size

    def can_apply(self, num_tokens: int, forward_batch) -> bool:
        return num_tokens >= self.cp_size * 2


class _FakeForwardMode:
    def __init__(self, is_cp_extend: bool = True):
        self._is_cp_extend = is_cp_extend

    def is_context_parallel_extend(self) -> bool:
        return self._is_cp_extend


def _fake_batch(num_tokens: int, is_cp_extend: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=_FakeForwardMode(is_cp_extend),
        input_ids=torch.arange(num_tokens),
        extend_seq_lens_cpu=[num_tokens],
        seq_lens_cpu=[num_tokens],
        attn_cp_metadata=None,
    )


class TestZigzagBuildMetadata(CustomTestCase):
    """Zigzag splits each sequence into 2 * cp_size blocks; rank r owns block
    r (front) and block 2 * cp_size - 1 - r (back)."""

    def _build(self, seqs, cp_size=4, cp_rank=0):
        from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy

        strategy = ZigzagCPStrategy(cp_size=cp_size)
        with patch(
            "sglang.srt.layers.cp.base.get_parallel",
            return_value=_FakeParallel(cp_rank, cp_size),
        ), patch(
            "sglang.srt.layers.cp.zigzag.get_device",
            return_value=SimpleNamespace(device="cpu"),
        ):
            meta = strategy.build_metadata(
                num_tokens=sum(seqs), seqs_len=list(seqs), extend_seqs_len=list(seqs)
            )
        return strategy, meta

    def _local_q_indices(self, seqs, cp_size=4, cp_rank=0):
        strategy, meta = self._build(seqs, cp_size=cp_size, cp_rank=cp_rank)
        batch = _fake_batch(sum(seqs))
        batch.attn_cp_metadata = meta
        return strategy.local_q_indices(sum(seqs), batch).tolist()

    def test_two_requests_block_layout(self):
        # 2 requests x 1024 tokens, cp_size=4 -> 2 * 4 = 8 blocks of 128 per
        # request, 16 blocks in total.
        _, meta = self._build([1024, 1024], cp_size=4, cp_rank=0)

        self.assertEqual(meta.bs, 2)
        self.assertEqual(len(meta.split_list), 2 * 2 * 4)
        self.assertEqual(meta.split_list, [128] * 16)
        self.assertEqual(sum(meta.split_list), 2048)
        # Each rank owns 2 blocks per request -> 512 tokens per request.
        self.assertEqual(meta.per_rank_actual_token, [512] * 4)
        # rank 0's gather order: front blocks of both requests, then the back
        # blocks of both requests.
        self.assertEqual(meta.zigzag_index, [0, 8, 7, 15])
        self.assertEqual(meta.total_seq_lens, 2048)

    def test_uneven_request_length(self):
        # 1010 / 8 blocks = 126 rem 2: first 2 blocks get 127, the rest 126.
        _, meta = self._build([1010], cp_size=4, cp_rank=0)

        self.assertEqual(len(meta.split_list), 8)
        self.assertEqual(meta.split_list, [127, 127, 126, 126, 126, 126, 126, 126])
        # rank 0 owns blocks 0 and 7 -> 127 + 126 = 253 tokens.
        self.assertEqual(meta.per_rank_actual_token[0], 253)
        self.assertEqual(sum(meta.per_rank_actual_token), 1010)

    def test_local_q_indices_single_request_front_and_back_blocks(self):
        # rank 0: block 0 = [0, 256) and block 7 = [1792, 2048).
        self.assertEqual(
            self._local_q_indices([2048], cp_rank=0),
            list(range(0, 256)) + list(range(1792, 2048)),
        )
        # rank 3: block 3 = [768, 1024) and block 4 = [1024, 1280).
        self.assertEqual(
            self._local_q_indices([2048], cp_rank=3),
            list(range(768, 1024)) + list(range(1024, 1280)),
        )

    def test_local_q_indices_two_requests(self):
        # rank 0 gathers the front block of each request followed by the back
        # block of each request; 4 pieces x 128 = 512 tokens in total.
        self.assertEqual(
            self._local_q_indices([1024, 1024], cp_rank=0),
            list(range(0, 128))
            + list(range(1024, 1152))
            + list(range(896, 1024))
            + list(range(1920, 2048)),
        )


class TestInterleaveStrategy(CustomTestCase):
    def _build(self, seqs, cp_size=4, cp_rank=0):
        from sglang.srt.layers.cp.interleave import InterleaveCPStrategy

        strategy = InterleaveCPStrategy(cp_size=cp_size)
        with patch(
            "sglang.srt.layers.cp.base.get_parallel",
            return_value=_FakeParallel(cp_rank, cp_size),
        ):
            meta = strategy.build_metadata(
                num_tokens=sum(seqs), seqs_len=list(seqs), extend_seqs_len=list(seqs)
            )
        return strategy, meta

    def _local_q_indices(self, num_tokens, cp_size=4, cp_rank=0):
        strategy, _ = self._build([num_tokens], cp_size=cp_size, cp_rank=cp_rank)
        # local_q_indices reads self.cp_rank lazily, so keep the parallel
        # context mock active for the call itself.
        with patch(
            "sglang.srt.layers.cp.base.get_parallel",
            return_value=_FakeParallel(cp_rank, cp_size),
        ):
            return strategy.local_q_indices(
                num_tokens, _fake_batch(num_tokens)
            ).tolist()

    def test_build_metadata_counts(self):
        # 1010 tokens over 4 ranks: [253, 253, 252, 252].
        _, meta = self._build([1010], cp_size=4)

        self.assertEqual(meta.bs, 1)
        self.assertEqual(meta.total_seq_lens, 1010)
        self.assertEqual(meta.per_rank_actual_token, [253, 253, 252, 252])
        self.assertEqual(meta.max_rank_len, [253] * 4)

    def test_local_q_indices_strided_selection(self):
        # rank r owns tokens r, r + cp_size, r + 2 * cp_size, ...
        self.assertEqual(self._local_q_indices(16, cp_rank=0), [0, 4, 8, 12])
        self.assertEqual(self._local_q_indices(16, cp_rank=3), [3, 7, 11, 15])

    def test_local_q_indices_uneven_token_count(self):
        # 10 tokens over 4 ranks: ranks 0/1 get 3, ranks 2/3 get 2.
        self.assertEqual(self._local_q_indices(10, cp_rank=0), [0, 4, 8])
        self.assertEqual(self._local_q_indices(10, cp_rank=1), [1, 5, 9])
        self.assertEqual(self._local_q_indices(10, cp_rank=2), [2, 6])

    def test_interleave_rows_per_request(self):
        from sglang.srt.layers.cp.interleave import interleave_rows_per_request

        # rank 0 of [5, 5]: tokens 0, 4 of request 0 and token 8 of request 1.
        self.assertEqual(interleave_rows_per_request([5, 5], 0, 4), [2, 1])
        # rank 3 of [5, 5]: token 3 of request 0 and token 7 of request 1.
        self.assertEqual(interleave_rows_per_request([5, 5], 3, 4), [1, 1])


class TestCPPaddingHelpers(CustomTestCase):
    @staticmethod
    def _meta(counts):
        return SimpleNamespace(
            per_rank_actual_token=list(counts),
            per_rank_logical_token=None,
            max_rank_len=None,
        )

    def test_pad_logical_token_to_physical_zigzag_alignment(self):
        # zigzag aligns to cp_size * 2 = 8; 1010 -> 1016.
        from sglang.srt.layers.cp.padding import pad_logical_token_to_physical

        meta = self._meta([1009, 1010, 1009, 1010])
        with patch(
            "sglang.srt.layers.cp.padding.get_parallel",
            return_value=_FakeParallel(0, 4),
        ), patch("sglang.srt.layers.cp.base.is_zigzag", return_value=True):
            pad_logical_token_to_physical(meta)

        self.assertEqual(meta.per_rank_logical_token, [1009, 1010, 1009, 1010])
        self.assertEqual(meta.per_rank_actual_token, [1016] * 4)
        self.assertEqual(meta.max_rank_len, [1016] * 4)

    def test_pad_logical_token_to_physical_interleave_alignment(self):
        # interleave aligns to cp_size = 4; 1010 -> 1012.
        from sglang.srt.layers.cp.padding import pad_logical_token_to_physical

        meta = self._meta([1010] * 4)
        with patch(
            "sglang.srt.layers.cp.padding.get_parallel",
            return_value=_FakeParallel(0, 4),
        ), patch("sglang.srt.layers.cp.base.is_zigzag", return_value=False):
            pad_logical_token_to_physical(meta)

        self.assertEqual(meta.per_rank_actual_token, [1012] * 4)

    def test_pad_local_rows(self):
        from sglang.srt.layers.cp.padding import pad_local_rows

        meta = SimpleNamespace(
            per_rank_logical_token=[10], per_rank_actual_token=[16]
        )
        x = torch.arange(10)
        out = pad_local_rows(x, meta, dim=0)
        self.assertEqual(out.shape, (16,))
        self.assertTrue(torch.equal(out[:10], x))
        self.assertTrue(torch.equal(out[10:], torch.zeros(6, dtype=torch.long)))

    def test_pad_local_rows_noop_when_already_aligned(self):
        from sglang.srt.layers.cp.padding import pad_local_rows

        meta = SimpleNamespace(
            per_rank_logical_token=[10], per_rank_actual_token=[10]
        )
        x = torch.arange(10)
        self.assertIs(pad_local_rows(x, meta, dim=0), x)


class TestIsCpActive(CustomTestCase):
    def _run(self, batch):
        from sglang.srt.layers.cp import utils as cp_utils

        with patch(
            "sglang.srt.layers.cp.utils.get_cp_strategy",
            return_value=_FakeStrategy(),
        ):
            return cp_utils.is_cp_active(batch)

    def test_active_with_input_ids(self):
        self.assertTrue(self._run(_fake_batch(8192)))

    def test_inactive_when_strategy_cannot_apply(self):
        from sglang.srt.layers.cp import utils as cp_utils

        with patch(
            "sglang.srt.layers.cp.utils.get_cp_strategy",
            return_value=_FakeStrategy(cp_size=5000),
        ):
            self.assertFalse(cp_utils.is_cp_active(_fake_batch(8192)))

    def test_inactive_without_input_ids(self):
        # Stages without input_ids (e.g. PP ranks beyond the first) must not
        # engage CP: the gate requires the input tensor.
        batch = _fake_batch(8192)
        batch.input_ids = None
        self.assertFalse(self._run(batch))

    def test_inactive_for_non_cp_forward_mode(self):
        self.assertFalse(self._run(_fake_batch(8192, is_cp_extend=False)))

    def test_inactive_when_strategy_missing(self):
        from sglang.srt.layers.cp import utils as cp_utils

        with patch("sglang.srt.layers.cp.utils.get_cp_strategy", return_value=None):
            self.assertFalse(cp_utils.is_cp_active(_fake_batch(8192)))


class TestDsaLayerSplitHelpers(CustomTestCase):
    """GLM DSA KV/indexer cache layer sharding helpers in layers/cp/utils.py."""

    def test_layer_shard_range_even_split(self):
        from sglang.srt.layers.cp.utils import get_layer_shard_range

        self.assertEqual(get_layer_shard_range(0, 4, 16), (0, 4))
        self.assertEqual(get_layer_shard_range(1, 4, 16), (4, 8))
        self.assertEqual(get_layer_shard_range(2, 4, 16), (8, 12))
        self.assertEqual(get_layer_shard_range(3, 4, 16), (12, 16))

    def test_layer_shard_range_uneven_split(self):
        from sglang.srt.layers.cp.utils import get_layer_shard_range

        # 10 layers over 4 ranks: 3, 3, 2, 2.
        self.assertEqual(get_layer_shard_range(0, 4, 10), (0, 3))
        self.assertEqual(get_layer_shard_range(1, 4, 10), (3, 6))
        self.assertEqual(get_layer_shard_range(2, 4, 10), (6, 8))
        self.assertEqual(get_layer_shard_range(3, 4, 10), (8, 10))

    def test_layer_owner(self):
        from sglang.srt.layers.cp.utils import get_layer_owner

        self.assertEqual(get_layer_owner(2, 4, 10), 0)
        self.assertEqual(get_layer_owner(5, 4, 10), 1)
        self.assertEqual(get_layer_owner(7, 4, 10), 2)
        self.assertEqual(get_layer_owner(9, 4, 10), 3)

    def test_layer_owner_out_of_range_raises(self):
        from sglang.srt.layers.cp.utils import get_layer_owner

        with self.assertRaises(ValueError):
            get_layer_owner(17, 4, 10)


if __name__ == "__main__":
    unittest.main()
