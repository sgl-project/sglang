# Copyright 2023-2025 SGLang Team
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

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import os
import tempfile
import unittest

import sglang.srt.multiplex.pdmux_context as pdmux
from sglang.srt.multiplex.pdmux_context import (
    divide_sm,
    get_arch_constraints,
    load_pdmux_config,
)


class TestLoadPDMuxConfig(unittest.TestCase):
    def _write_config(self, content: str) -> str:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "pdmux.yaml")
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_empty_path_returns_defaults(self):
        cfg = load_pdmux_config("")
        self.assertEqual(cfg.sm_group_num, 8)
        self.assertEqual(cfg.manual_divisions, [])
        self.assertEqual(cfg.split_forward_token_budget, 65536)
        self.assertEqual(cfg.decode_bs_divisor, 36)

    def test_missing_sm_group_num_raises(self):
        path = self._write_config("split_forward_token_budget: 100\n")
        with self.assertRaises(ValueError) as ctx:
            load_pdmux_config(path)
        self.assertIn("sm_group_num", str(ctx.exception))

    def test_sm_group_num_below_3_raises(self):
        path = self._write_config("sm_group_num: 2\n")
        with self.assertRaises(ValueError):
            load_pdmux_config(path)

    def test_sm_group_num_3_is_valid(self):
        cfg = load_pdmux_config(self._write_config("sm_group_num: 3\n"))
        self.assertEqual(cfg.sm_group_num, 3)
        self.assertEqual(cfg.manual_divisions, [])

    def test_manual_divisions_wrong_length_raises(self):
        # sm_group_num=8 expects 8-2=6 division entries.
        divisions = "\n".join(
            f"  - [{64 - i * 8}, {32 + i * 8}, {8 * (i + 1)}]" for i in range(3)
        )
        path = self._write_config(f"sm_group_num: 8\nmanual_divisions:\n{divisions}\n")
        with self.assertRaises(ValueError) as ctx:
            load_pdmux_config(path)
        self.assertIn("must have 6 entries", str(ctx.exception))

    def test_full_config(self):
        divisions = "\n".join(
            f"  - [{112 - i * 16}, {20 + i * 16}, {10 * (i + 1)}]" for i in range(6)
        )
        content = (
            f"sm_group_num: 8\n"
            f"manual_divisions:\n{divisions}\n"
            f"split_forward_token_budget: 4096\n"
            f"decode_bs_divisor: 24\n"
        )
        cfg = load_pdmux_config(self._write_config(content))
        self.assertEqual(cfg.sm_group_num, 8)
        self.assertEqual(len(cfg.manual_divisions), 6)
        self.assertEqual(cfg.manual_divisions[0], [112, 20, 10])
        self.assertEqual(cfg.split_forward_token_budget, 4096)
        self.assertEqual(cfg.decode_bs_divisor, 24)

    def test_optional_fields_default_when_only_sm_group_num(self):
        cfg = load_pdmux_config(self._write_config("sm_group_num: 5\n"))
        self.assertEqual(cfg.split_forward_token_budget, 65536)
        self.assertEqual(cfg.decode_bs_divisor, 36)
        self.assertEqual(cfg.manual_divisions, [])


class TestGetArchConstraints(unittest.TestCase):
    def test_pascal(self):
        self.assertEqual(get_arch_constraints((6, 1)), (1, 1))
        self.assertEqual(get_arch_constraints((6, 9)), (1, 1))

    def test_volta_and_turing(self):
        self.assertEqual(get_arch_constraints((7, 0)), (2, 2))
        self.assertEqual(get_arch_constraints((7, 5)), (2, 2))

    def test_ampere_and_ada(self):
        self.assertEqual(get_arch_constraints((8, 0)), (4, 2))
        self.assertEqual(get_arch_constraints((8, 9)), (4, 2))

    def test_hopper_and_blackwell(self):
        self.assertEqual(get_arch_constraints((9, 0)), (8, 8))
        self.assertEqual(get_arch_constraints((9, 90)), (8, 8))

    def test_unsupported_architectures_raise(self):
        for compute_capability in [(5, 0), (10, 0), (11, 8)]:
            with self.assertRaises(ValueError):
                get_arch_constraints(compute_capability)


class TestDivideSM(unittest.TestCase):
    def test_h100_like_132_sms(self):
        # Arch 9: min per part 8, multiple 8. Candidates are multiples of 8
        # with x >= 132 - x and 132 - x >= 16, i.e. x in {72, ..., 112}.
        # groups=6 picks all six candidates; reversing puts larger prefill
        # partitions first.
        self.assertEqual(
            divide_sm(132, (9, 0), 6),
            [(112, 20), (104, 28), (96, 36), (88, 44), (80, 52), (72, 60)],
        )

    def test_a100_like_108_sms_stride_sampling(self):
        # Arch 8: candidates are even values in [54, 92] (20 values).
        # step = 20 // 6 = 3 -> picks 54, 60, 66, 72, 78, 84.
        self.assertEqual(
            divide_sm(108, (8, 0), 6),
            [(84, 24), (78, 30), (72, 36), (66, 42), (60, 48), (54, 54)],
        )

    def test_fewer_candidates_than_groups_returns_all(self):
        # Arch 9 candidates in [24, 32]: only 24 and 32, fewer than 5 groups.
        self.assertEqual(divide_sm(48, (9, 0), 5), [(32, 16), (24, 24)])

    def test_too_few_sms_raises(self):
        with self.assertRaises(ValueError):
            divide_sm(20, (9, 0), 4)

    def test_min_decode_floor_raises(self):
        # Candidate 16 passes the prefill>=decode check but violates the
        # 16-SM decode floor (24 - 16 = 8 < 16), so no partition exists.
        with self.assertRaises(ValueError):
            divide_sm(24, (9, 0), 2)

    def test_partition_invariants(self):
        for total_sms, compute_capability in [(132, (9, 0)), (108, (8, 0))]:
            _, multiple = get_arch_constraints(compute_capability)
            divisions = divide_sm(total_sms, compute_capability, 6)
            for prefill, decode in divisions:
                self.assertEqual(prefill + decode, total_sms)
                self.assertGreaterEqual(prefill, decode)
                self.assertGreaterEqual(decode, 16)
                self.assertEqual(prefill % multiple, 0)
            self.assertEqual(
                divisions, sorted(divisions, key=lambda d: d[0], reverse=True)
            )


class TestStreamIndexState(unittest.TestCase):
    """Tests for the module-level stream index bookkeeping.

    ``set_current_stream_idx`` and the getters operate on module globals, so
    each test swaps in fake stream groups and restores the originals on
    cleanup. No CUDA streams are created.
    """

    def setUp(self):
        self._saved = {
            name: getattr(pdmux, name)
            for name in (
                "STREAM_GROUPS",
                "SM_COUNTS",
                "CURRENT_STREAM_IDX",
                "CURRENT_STREAM_GROUP",
            )
        }
        self.groups = [(f"prefill{i}", f"decode{i}") for i in range(3)]
        pdmux.STREAM_GROUPS = self.groups
        pdmux.SM_COUNTS = [(48, 0), (32, 16), (0, 48)]
        pdmux.CURRENT_STREAM_IDX = 0
        pdmux.CURRENT_STREAM_GROUP = self.groups[0]

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(pdmux, name, value)

    def test_set_current_stream_idx_valid(self):
        pdmux.set_current_stream_idx(2)
        self.assertEqual(pdmux.CURRENT_STREAM_IDX, 2)
        self.assertIs(pdmux.CURRENT_STREAM_GROUP, self.groups[2])
        self.assertEqual(pdmux.get_current_stream_idx(), 2)

    def test_set_current_stream_idx_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            pdmux.set_current_stream_idx(3)

    def test_set_current_stream_idx_negative_raises(self):
        with self.assertRaises(ValueError):
            pdmux.set_current_stream_idx(-1)

    def test_getters_reflect_globals(self):
        self.assertEqual(pdmux.get_stream_groups(), self.groups)
        self.assertEqual(pdmux.get_sm_counts(), [(48, 0), (32, 16), (0, 48)])
        self.assertEqual(pdmux.get_current_stream_idx(), 0)


if __name__ == "__main__":
    unittest.main()
