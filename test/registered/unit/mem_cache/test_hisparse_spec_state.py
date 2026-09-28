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
"""CPU tagged-copy oracle against the actual candidate production module."""

import importlib.util
import pathlib
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


ROOT = pathlib.Path(__file__).resolve().parents[4]
PATH = ROOT / "python/sglang/srt/mem_cache/hisparse_spec_state.py"
SPEC = importlib.util.spec_from_file_location("hisparse_spec_state", PATH)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


class UnionTests(unittest.TestCase):
    def inputs(self, selections=((0, 1, -1), (2, 3, -1)), old=12, key=None, layer=0):
        key = key or m.SpecTxnKey(2, 3, 4)
        return dict(
            key=key,
            layer_id=layer,
            hot_key=key,
            hot_layer_id=layer,
            old_kv_len=old,
            rows=[
                m.VerifyRow(key, old + i, tuple(row))
                for i, row in enumerate(selections)
            ],
            host_slots={p: 1000 + 7 * p for p in range(old)},
            hot_tokens=[8, 9, 10, 11],
            hot_slots=[65, 131, 260, 390],
            victim_order=[0, 1, 2, 3],
            arena=m.ProvisionalArena(
                key,
                len(selections),
                (20,),
                tuple((old + i, 1280 + i) for i in range(len(selections))),
            ),
        )

    def check_copies(self, args):
        plan = m.plan_union(**args)

        def tag(p):
            return (args["key"], args["layer_id"], p)

        device = {
            slot: tag(p) for p, slot in zip(args["hot_tokens"], args["hot_slots"])
        }
        device.update({slot: tag(p) for p, slot in args["arena"].position_slots})
        host = {slot: tag(p) for p, slot in args["host_slots"].items()}
        for source, destination in zip(plan.miss_src, plan.miss_dst):
            device[destination] = host[source]
        for row, table in zip(args["rows"], plan.row_device_tables):
            for position, slot in zip(row.selections, table):
                if position == -1:
                    self.assertEqual(slot, -1)
                else:
                    self.assertEqual(device[slot], tag(position))
        return plan

    def test_sequential_counterexample_and_union(self):
        args = self.inputs()
        # Old row-local replacement can reuse the same physical destinations.
        memory, tables = {}, []
        for row in args["rows"]:
            table = []
            for position, slot in zip(row.selections[:2], args["hot_slots"][:2]):
                memory[slot] = position
                table.append(slot)
            tables.append(table)
        self.assertNotEqual([memory[s] for s in tables[0]], [0, 1])
        plan = self.check_copies(args)
        self.assertEqual(plan.miss_positions, (0, 1, 2, 3))
        self.assertIsNot(plan.row_device_tables[0], plan.row_device_tables[1])

    def test_pin_later_row_hit_before_earlier_miss(self):
        args = self.inputs(((0, 0, -1), (8, 1, 1)))
        plan = self.check_copies(args)
        self.assertEqual(plan.pinned_hot_indices, (0,))
        self.assertEqual(plan.miss_positions, (0, 1))
        self.assertNotIn(65, plan.miss_dst)

    def test_overflow_atomic(self):
        args = self.inputs(((0, 1, 2), (3, 4, -1)))
        before = repr(args)
        with self.assertRaises(m.UnionCapacityError):
            m.plan_union(**args)
        self.assertEqual(repr(args), before)

    def test_invalid_and_noncausal(self):
        for rows in (((-2,), (0,)), ((13,), (0,)), ((True,), (0,))):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                m.plan_union(**self.inputs(rows))
        args = self.inputs()
        args["hot_slots"][0] = 0
        with self.assertRaises(ValueError):
            m.plan_union(**args)
        args = self.inputs()
        args["host_slots"][0] = -1
        with self.assertRaises(ValueError):
            m.plan_union(**args)

    def test_provisional_and_page_boundaries(self):
        for prefix in (63, 64, 65, 127):
            for width in (2, 4):
                rows = tuple((0, prefix + i, -1) for i in range(width))
                args = self.inputs(rows, old=prefix)
                plan = self.check_copies(args)
                self.assertEqual(len(plan.miss_positions), 1)
                self.assertEqual(args["arena"].page_ids, (20,))

    def test_identity_and_layer_isolation(self):
        for key, layer in ((m.SpecTxnKey(2, 3, 4), 0), (m.SpecTxnKey(7, 8, 9), 2)):
            self.check_copies(self.inputs(key=key, layer=layer))
        for field, value in (
            ("hot_key", m.SpecTxnKey(2, 4, 4)),
            ("hot_key", m.SpecTxnKey(2, 3, 5)),
            ("hot_layer_id", 1),
        ):
            args = self.inputs()
            args[field] = value
            with self.assertRaises(ValueError):
                m.plan_union(**args)
        args = self.inputs()
        args["rows"][0] = m.VerifyRow(m.SpecTxnKey(8, 3, 4), 12, (0,))
        with self.assertRaises(ValueError):
            m.plan_union(**args)

    def test_sentinel_and_host_zero(self):
        plan = self.check_copies(self.inputs(((-1, -1), (-1, -1))))
        self.assertEqual(plan.miss_src, ())
        args = self.inputs()
        args["host_slots"][0] = 0
        self.check_copies(args)

    def test_page_ownership_padding_and_rounding(self):
        key = m.SpecTxnKey(2, 3, 4)
        for count, rounded in ((1, 64), (63, 64), (64, 64), (65, 128), (129, 192)):
            self.assertEqual(m.rounded_rows(count), rounded)
            pages = (20, 31, 44)[: rounded // 64]
            arena = m.ProvisionalArena(key, count, pages, ((12, 1280),))
            self.assertEqual(arena.page_ids, pages)
        args = self.inputs()
        args["hot_slots"][0] = 1343  # unused arena padding still belongs to arena
        with self.assertRaises(ValueError):
            m.plan_union(**args)
        for pages, positions in (
            ((20, 20), ((12, 1280),)),
            ((20,), ((12, 1280), (13, 1280))),
            ((20,), ((12, 64),)),
        ):
            with self.assertRaises(ValueError):
                m.ProvisionalArena(key, 2, pages, positions)

    def test_page_zero_reserved(self):
        args = self.inputs()
        args["hot_slots"][0] = 63
        with self.assertRaises(ValueError):
            m.plan_union(**args)
        args["hot_slots"][0] = 64
        self.check_copies(args)

    def test_host_mapping_missing_or_aliased(self):
        args = self.inputs()
        del args["host_slots"][0]
        with self.assertRaises(ValueError):
            m.plan_union(**args)
        args = self.inputs()
        args["host_slots"][1] = args["host_slots"][0]
        with self.assertRaises(ValueError):
            m.plan_union(**args)

    def test_result_does_not_alias_caller_lists(self):
        args = self.inputs()
        selections = [0, 1, -1]
        args["rows"][0] = m.VerifyRow(args["key"], 12, selections)
        plan = m.plan_union(**args)
        expected_tables = plan.row_device_tables
        expected_metadata = plan.pending_hot_tokens
        selections[:] = [7, 8, 9]
        args["rows"].clear()
        args["hot_tokens"][:] = [-1] * 4
        args["hot_slots"][:] = [64] * 4
        args["victim_order"].reverse()
        args["host_slots"].clear()
        self.assertEqual(plan.row_device_tables, expected_tables)
        self.assertEqual(plan.pending_hot_tokens, expected_metadata)
        self.assertEqual(plan.row_device_tables[0], (65, 131, -1))
        self.assertEqual(plan.pending_hot_tokens, (0, 1, 2, 3))
        with self.assertRaises(TypeError):
            plan.row_device_tables[0][0] = 99
        with self.assertRaises(TypeError):
            plan.pending_hot_tokens[0] = 99

    def test_arena_requires_transaction_key(self):
        for key in (None, (2, 3, 4), "2:3:4"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.ProvisionalArena(key, 1, (20,), ((12, 1280),))

    def test_unmapped_and_duplicate_metadata(self):
        args = self.inputs(((12,), (13,)))
        args["arena"] = m.ProvisionalArena(args["key"], 2, (20,), ((12, 1280),))
        with self.assertRaises(ValueError):
            m.plan_union(**args)
        for field, value in (
            ("hot_tokens", [8, 8, 10, 11]),
            ("hot_slots", [65, 65, 260, 390]),
            ("victim_order", [0, 0, 2, 3]),
        ):
            args = self.inputs()
            args[field] = value
            with self.assertRaises(ValueError):
                m.plan_union(**args)


if __name__ == "__main__":
    unittest.main()
