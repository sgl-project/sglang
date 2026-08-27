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
"""The gfx950 MLA decode tuning has to stay numerically equivalent to the stock path.

SGLANG_MLA_DECODE_TUNE replaces the scheduler's per-sequence num_kv_splits with one
batch-wide count. Its failure mode is silent: stage-2 derives where split `i` starts
from the same count stage-1 used, so if only one of the two launches gets the count,
the merge reads partials that were never written and the output is quietly wrong
rather than an error. Only running both stages for real catches that.

    python -m pytest test/registered/unit/layers/attention/test_mla_decode_forced_splits.py -v
"""

import unittest

import torch

from sglang.kernels.ops.attention import decode_attention as da
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# The gfx95 runner: stage-b-test-1-gpu-large-amd is MI300, where every case below skips.
register_amd_ci(est_time=6, suite="stage-b-test-1-gpu-small-amd-mi35x")

LK, LV = 576, 512

# A different split count reassociates the fp32 softmax reduction, so the result can
# land a rounding step away once it is stored back to bf16. Measured at exactly one
# bf16 ULP on every case below, hence rtol of two. The atol floor is for outputs small
# enough that the error is set by the terms being summed rather than by the result:
# 1.34e-3 was the most any case needed, on elements around 0.04.
BF16_ULP = 2**-8
ATOL = 3e-3

# Batches straddling every bucket edge, plus the head counts that give head_tiles 1
# and 8, plus the sequence shapes a batch-wide split count has to survive.
#
# The two head_tiles=8 cases (h128) are the only ones that catch stage-1 being handed
# the wrong count while stage-2 keeps the right one; do not drop both.
CASES = (
    ("b1_h16", [4096], 16, 1),
    ("b5_h128", [4096] * 5, 128, 1),
    ("b6_h16", [4096] * 6, 16, 1),
    ("b24_h16", [4096] * 24, 16, 1),
    ("b25_h16", [4096] * 25, 16, 1),
    ("b136_h16", [1024] * 136, 16, 1),
    ("mixed_lengths", [1, 31, 33, 257, 1024, 4095, 4096, 16384], 16, 1),
    ("mixed_skew", [16384, 1, 1, 1, 1, 1, 1, 1], 16, 1),
    ("all_length_1", [1, 1, 1, 1], 16, 1),
    ("mixed_h128", [7, 512, 1023, 1025, 2048, 4095], 128, 1),
    ("page64", [4096] * 4, 16, 64),
    ("page64_mixed", [1 + (257 * i) % 2048 for i in range(32)], 16, 64),
)


def _inputs(seq_lens, head_num, page_size, max_kv_splits, seed):
    """Stock decode inputs, including the num_kv_splits the scheduler would write."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev, batch = "cuda", len(seq_lens)
    total = sum(seq_lens)
    n_slots = total + 64

    if page_size == 1:
        pool = torch.randn(
            n_slots, 1, LK, dtype=torch.bfloat16, device=dev, generator=gen
        )
    else:
        n_pages = (n_slots + page_size - 1) // page_size
        pool = torch.randn(
            n_pages, page_size, 1, LK, dtype=torch.bfloat16, device=dev, generator=gen
        )
        n_slots = n_pages * page_size

    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=dev)
    kv_indptr[1:] = torch.cumsum(
        torch.tensor(seq_lens, dtype=torch.int32, device=dev), dim=0
    )
    # scattered slots, like a pool that has been recycled
    kv_indices = torch.randperm(n_slots, device=dev, generator=gen)[:total].to(
        torch.int32
    )

    lens = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    num_kv_splits = torch.clamp(
        torch.div(lens, 256, rounding_mode="floor") + 1, 1, max_kv_splits
    ).to(torch.int32)

    return {
        "q": torch.randn(
            batch, head_num, LK, dtype=torch.bfloat16, device=dev, generator=gen
        ),
        "k_buffer": pool,
        "v_buffer": pool[..., :LV],
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "num_kv_splits": num_kv_splits,
        "max_kv_splits": max_kv_splits,
        "page_size": page_size,
    }


def _forced(inp):
    return da._mla_launch_plan(inp["q"], inp["k_buffer"], inp["max_kv_splits"], True)[1]


def _run(inp):
    batch, head_num = inp["q"].shape[0], inp["q"].shape[1]
    dev, mks = "cuda", inp["max_kv_splits"]
    o = torch.zeros(batch, head_num, LV, dtype=torch.bfloat16, device=dev)
    logits = torch.empty(batch, head_num, mks, LV, dtype=torch.float32, device=dev)
    # the DCP path relies on untouched entries staying -inf
    lse = torch.full(
        (batch, head_num, mks), -float("inf"), dtype=torch.float32, device=dev
    )
    da.decode_attention_fwd_grouped(
        inp["q"],
        inp["k_buffer"],
        inp["v_buffer"],
        o,
        inp["kv_indptr"],
        inp["kv_indices"],
        logits,
        lse,
        inp["num_kv_splits"],
        mks,
        1.0 / LK**0.5,
        1.0,
        has_mla=True,
        page_size=inp["page_size"],
    )
    torch.cuda.synchronize()
    return o


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "the tuning only engages on gfx95"
)
class TestMlaDecodeForcedSplits(CustomTestCase):
    def setUp(self):
        # a plain config, so the count comes out batch-wide whatever ran before this
        self._publish()
        da._LOGGED_TUNE = False

    def tearDown(self):
        # None, not False: the real resolution has to run again for anything later in
        # this process.
        da._KEEP_SCHEDULER_SPLITS = None

    def _publish(self, **fields):
        override = get_context().override_server_args(**fields)
        override.install()
        self.addCleanup(override.restore)
        da._KEEP_SCHEDULER_SPLITS = None  # resolve it from what was just published

    def test_tuned_matches_stock(self):
        # Not bit-for-bit: a different split count reassociates the fp32 softmax
        # reduction. Bounded at two bf16 ULP, which is one rounding step of headroom
        # over what this actually measures.
        for seed, (name, seq_lens, head_num, page_size) in enumerate(CASES):
            with self.subTest(case=name):
                inp = _inputs(seq_lens, head_num, page_size, 256, seed=seed)
                with envs.SGLANG_MLA_DECODE_TUNE.override(False):
                    stock = _run(inp)
                    self.assertEqual(_forced(inp), 0)
                with envs.SGLANG_MLA_DECODE_TUNE.override(True):
                    # without this the comparison passes by comparing stock to stock
                    self.assertGreater(_forced(inp), 0, "tuning did not engage")
                    tuned = _run(inp)
                self.assertFalse(torch.isnan(tuned).any())
                torch.testing.assert_close(
                    tuned.float(),
                    stock.float(),
                    rtol=2 * BF16_ULP,
                    atol=ATOL,
                )

    def test_both_stages_get_the_same_count(self):
        # The mismatch the tolerance above can only catch indirectly: assert the entry
        # point hands one count to both launches instead of letting them disagree.
        # Dropping tune_mla is invisible to the numerics, it only costs the geometry.
        seen = {}

        def record(key, real):
            def wrapper(*args, forced_kv_splits=0, **kwargs):
                seen[key] = (forced_kv_splits, kwargs.get("tune_mla"))
                return real(*args, forced_kv_splits=forced_kv_splits, **kwargs)

            return wrapper

        stage1, stage2 = da._decode_grouped_att_m_fwd, da._decode_softmax_reducev_fwd
        da._decode_grouped_att_m_fwd = record("stage1", stage1)
        da._decode_softmax_reducev_fwd = record("stage2", stage2)
        try:
            with envs.SGLANG_MLA_DECODE_TUNE.override(True):
                _run(_inputs([4096] * 24, 16, 1, 256, seed=0))
        finally:
            da._decode_grouped_att_m_fwd = stage1
            da._decode_softmax_reducev_fwd = stage2

        self.assertEqual(seen["stage1"][0], seen["stage2"][0])
        self.assertGreater(
            seen["stage1"][0], 0, "tuning did not engage, nothing tested"
        )
        self.assertTrue(seen["stage1"][1], "stage-1 was left on the stock geometry")

    def test_scheduler_splits_are_kept_when_asked(self):
        # the geometry still changes under --enable-deterministic-inference, but it is
        # batch-independent (see the next test); only the count has to survive verbatim
        inp = _inputs([1, 4095, 4096, 16384], 16, 1, 256, seed=7)
        with envs.SGLANG_MLA_DECODE_TUNE.override(False):
            stock = _run(inp)
        self._publish(enable_deterministic_inference=True)
        with envs.SGLANG_MLA_DECODE_TUNE.override(True):
            self.assertEqual(_forced(inp), 0)
            kept = _run(inp)
        torch.testing.assert_close(
            kept.float(), stock.float(), rtol=2 * BF16_ULP, atol=ATOL
        )

    def test_deterministic_mode_is_batch_invariant(self):
        # The point of keeping the scheduler's count: a request's reduction tree must
        # not depend on who it shares the batch with.
        self._publish(enable_deterministic_inference=True)
        lens = [4096, 1, 777, 16384]
        with envs.SGLANG_MLA_DECODE_TUNE.override(True):
            batched = _run(_inputs(lens, 16, 1, 256, seed=3))
            for i in range(len(lens)):
                inp = _inputs(lens, 16, 1, 256, seed=3)
                # same q and same KV slots for row i, on its own
                start, end = inp["kv_indptr"][i].item(), inp["kv_indptr"][i + 1].item()
                alone = dict(inp)
                alone["q"] = inp["q"][i : i + 1].clone()
                alone["kv_indices"] = inp["kv_indices"][start:end].clone()
                alone["kv_indptr"] = torch.tensor(
                    [0, end - start], dtype=torch.int32, device="cuda"
                )
                alone["num_kv_splits"] = inp["num_kv_splits"][i : i + 1].clone()
                with self.subTest(row=i, seq_len=lens[i]):
                    torch.testing.assert_close(
                        _run(alone)[0], batched[i], rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
