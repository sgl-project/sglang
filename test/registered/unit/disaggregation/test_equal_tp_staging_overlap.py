import ast
import pathlib
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = pathlib.Path(__file__).resolve().parents[4] / "python/sglang/srt/disaggregation"


def method(file, name):
    tree = ast.parse((ROOT / file).read_text())
    matches = [
        x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == name
    ]
    assert matches, f"Missing staging behavior: {name}"
    n = matches[0]
    n.decorator_list = []
    n.returns = None
    for a in n.args.args:
        a.annotation = None
    env = {"envs": NS(SGLANG_DISAGG_STAGING_ALWAYS=NS(get=lambda: False))}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[n], type_ignores=[])),
            str(file),
            "exec",
        ),
        env,
    )
    return env[name]


class Contract(CustomTestCase):
    def test_equal_tp_preserves_partial_early_send(self):
        f = method("mooncake/conn.py", "requires_staging_grid_alignment")
        mgr = NS(
            attn_tp_size=1,
            pp_size=1,
            transfer_infos={7: {"s": NS(is_dummy=False, mooncake_session_id="s")}},
            decode_kv_args_table={"s": NS(dst_attn_tp_size=1)},
        )
        sender = NS(kv_mgr=mgr, bootstrap_room=7)
        self.assertFalse(f(sender))
        mgr.decode_kv_args_table["s"].dst_attn_tp_size = 2
        self.assertTrue(f(sender))
        mgr.decode_kv_args_table.clear()
        self.assertTrue(f(sender))

    def test_partial_slots_never_reuse_staging_allocation(self):
        f = method("mooncake/conn.py", "_is_first_staging_piece")
        s = NS(full_chunk_pages=16384)
        for start, size, expected in [
            (0, 16384, True),
            (32768, 16320, True),
            (32768, 0, False),
            (32768, 16385, False),
            (49088, 64, False),
            (16384, 16384, True),
        ]:
            self.assertEqual(f(s, start, size), expected)

    def test_monotonic_partial_sends_have_unique_staging_ids(self):
        f = method("mooncake/conn.py", "_is_first_staging_piece")
        strategy = NS(full_chunk_pages=16)
        for first_length in range(1, 17):
            pieces = [(0, first_length)]
            pieces += [(i, 1) for i in range(first_length, 32)]
            staged = [start // 16 for start, size in pieces if f(strategy, start, size)]
            self.assertEqual(staged, [0, 1])


class DirectPath(CustomTestCase):
    def test_contiguous_and_partial_sends_stay_unsplit(self):
        f = method("mooncake/conn.py", "should_split_staging")
        info = NS(is_dummy=False, dst_kv_indices=list(range(32)))
        mgr = NS(
            transfer_infos={7: {"s": info}},
            _fragmented_groups=lambda src, dst: (
                1
                + sum(
                    src[i] != src[i - 1] + 1 or dst[i] != dst[i - 1] + 1
                    for i in range(1, len(src))
                )
            ),
        )
        sender = NS(
            kv_mgr=mgr,
            bootstrap_room=7,
            curr_idx=0,
            requires_staging_grid_alignment=lambda: False,
        )
        self.assertFalse(f(sender, list(range(16)), 8, 4))
        self.assertTrue(f(sender, list(range(0, 32, 2)), 8, 4))
        self.assertTrue(f(sender, [0, 2, 4, 6], 8, 4))
        sender.curr_idx = 2
        self.assertFalse(f(sender, [0, 2, 4, 6], 8, 4))


class SchedulerBoundary(CustomTestCase):
    def test_actual_scheduler_keeps_cached_tail(self):
        tree = ast.parse((ROOT / "prefill.py").read_text())
        n = next(
            x
            for x in ast.walk(tree)
            if isinstance(x, ast.FunctionDef) and x.name == "_send_kv_chunk"
        )
        cut = next(
            i
            for i, x in enumerate(n.body)
            if isinstance(x, ast.AnnAssign) and x.target.id == "state_indices"
        )
        n.body = n.body[:cut] + [
            ast.Return(value=ast.Name(id="end_idx", ctx=ast.Load()))
        ]
        n.decorator_list = []
        n.returns = None
        for a in n.args.args:
            a.annotation = None
        env = {
            "get_schedule": lambda: NS(chunked_prefill_size=16384),
            "staging_grid_tokens": lambda c, p: c,
            "logger": NS(debug=lambda *a: None),
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[n], type_ignores=[])),
                "boundary",
                "exec",
            ),
            env,
        )
        s = NS(enable_staging=True, token_to_kv_pool_allocator=NS(page_size=1))
        for aligned, expected in [(False, 49088), (True, 32768)]:
            req = NS(
                start_send_idx=0,
                origin_input_ids=[0] * 49152,
                disagg_decode_prefix_len=0,
                disagg_kv_sender=NS(requires_staging_grid_alignment=lambda: aligned),
            )
            self.assertEqual(env["_send_kv_chunk"](s, req, end_idx=49088), expected)


class TestForceStaging(CustomTestCase):
    def test_force_staging_splits_contiguous_ranges(self):
        f = method("mooncake/conn.py", "should_split_staging")
        f.__globals__["envs"].SGLANG_DISAGG_STAGING_ALWAYS.get = lambda: True
        sender = NS(
            requires_staging_grid_alignment=lambda: False,
            kv_mgr=NS(
                transfer_infos={
                    7: {"s": NS(is_dummy=False, dst_kv_indices=list(range(16)))}
                },
                _fragmented_groups=lambda *args: 1,
            ),
            bootstrap_room=7,
            curr_idx=0,
        )
        self.assertTrue(f(sender, list(range(16)), 8, 8))

    def test_multi_pp_keeps_original_grid_alignment(self):
        f = method("mooncake/conn.py", "requires_staging_grid_alignment")
        manager = NS(
            pp_size=2,
            attn_tp_size=1,
            transfer_infos={7: {"s": NS(is_dummy=False, mooncake_session_id="s")}},
            decode_kv_args_table={"s": NS(dst_attn_tp_size=1)},
        )
        self.assertTrue(f(NS(kv_mgr=manager, bootstrap_room=7)))


class TestSchedulerSendRange(CustomTestCase):
    def test_partial_staging_and_direct_continuation(self):
        # Execute the entire production sender with fake allocation/transport;
        # assert emitted page ranges and progress, not the source structure.
        tree = ast.parse((ROOT / "prefill.py").read_text())
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_send_kv_chunk"
        )
        node.decorator_list = []
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        grid_tree = ast.parse((ROOT / "common/staging_buffer.py").read_text())
        grid_nodes = [
            n
            for n in grid_tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("compute_grid_segments", "staging_grid_tokens")
        ]
        env = dict(
            Optional=__import__("typing").Optional,
            List=list,
            Tuple=__import__("typing").Tuple,
            get_schedule=lambda: NS(chunked_prefill_size=16),
            kv_to_page_indices=lambda indices, page_size: (
                indices[::page_size] // page_size
            ),
            logger=NS(debug=lambda *args: None),
            envs=NS(SGLANG_DISAGG_STAGING_MIN_GROUPS=NS(get=lambda: 4)),
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=grid_nodes + [node], type_ignores=[])
                ),
                "scheduler_send",
                "exec",
            ),
            env,
        )
        split = method("mooncake/conn.py", "should_split_staging")
        gate = method("mooncake/conn.py", "_is_first_staging_piece")
        groups = method("mooncake/conn.py", "_fragmented_groups")
        groups.__globals__["np"] = np
        mgr = NS(
            transfer_infos={
                7: {"s": NS(is_dummy=False, dst_kv_indices=np.arange(80)[::2])}
            },
            _fragmented_groups=lambda src, dst: groups(None, src, dst),
        )
        sends = []
        sender = NS(
            curr_idx=0,
            kv_mgr=mgr,
            bootstrap_room=7,
            requires_staging_grid_alignment=lambda: False,
            should_send_kv_chunk=lambda *args: True,
            get_max_transfer_tokens=lambda: None,
        )
        sender.should_split_staging = lambda *args: split(sender, *args)

        def send(pages, state, **kwargs):
            sends.append((sender.curr_idx, pages.copy(), state, kwargs))
            sender.curr_idx += len(pages)

        sender.send = send
        translate = Mock(side_effect=lambda x: x)
        req = NS(
            kv=NS(req_pool_idx=0),
            start_send_idx=0,
            origin_input_ids=list(range(32)),
            disagg_decode_prefix_len=0,
            disagg_kv_sender=sender,
            rid="r",
        )
        scheduler = NS(
            enable_staging=True,
            token_to_kv_pool_allocator=NS(
                page_size=1, translate_kv_indices_for_transfer=translate
            ),
            req_to_token_pool=NS(req_to_token=np.arange(32).reshape(1, 32)),
            disagg_prefill_pending_chunk_rids=set(),
        )
        env["_send_kv_chunk"](scheduler, req, end_idx=31)
        self.assertEqual([len(x[1]) for x in sends], [16, 15])
        self.assertEqual(translate.call_count, 1)
        self.assertEqual(req.start_send_idx, 31)
        env["_send_kv_chunk"](scheduler, req, end_idx=32)
        self.assertEqual([len(x[1]) for x in sends], [16, 15, 1])
        self.assertEqual(req.start_send_idx, 32)
        self.assertEqual(
            [
                gate(NS(full_chunk_pages=16), start, len(pages))
                for start, pages, *_ in sends
            ],
            [True, True, False],
        )
        self.assertEqual(
            np.concatenate([x[1] for x in sends]).tolist(), list(range(32))
        )

    def test_finite_dcp_pack_limit_is_retained_when_splitting(self):
        # A DCP pack-buffer cap (sender.get_max_transfer_tokens) smaller than
        # the staging grid must survive equal-TP splitting: segments stay within
        # the cap instead of growing back to one full grid slot.
        tree = ast.parse((ROOT / "prefill.py").read_text())
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_send_kv_chunk"
        )
        node.decorator_list = []
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        grid_tree = ast.parse((ROOT / "common/staging_buffer.py").read_text())
        grid_nodes = [
            n
            for n in grid_tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("compute_grid_segments", "staging_grid_tokens")
        ]
        env = dict(
            Optional=__import__("typing").Optional,
            List=list,
            Tuple=__import__("typing").Tuple,
            get_schedule=lambda: NS(chunked_prefill_size=16),
            kv_to_page_indices=lambda indices, page_size: (
                indices[::page_size] // page_size
            ),
            logger=NS(debug=lambda *args: None),
            envs=NS(SGLANG_DISAGG_STAGING_MIN_GROUPS=NS(get=lambda: 4)),
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=grid_nodes + [node], type_ignores=[])
                ),
                "scheduler_send",
                "exec",
            ),
            env,
        )
        split = method("mooncake/conn.py", "should_split_staging")
        gate = method("mooncake/conn.py", "_is_first_staging_piece")
        groups = method("mooncake/conn.py", "_fragmented_groups")
        groups.__globals__["np"] = np
        mgr = NS(
            transfer_infos={
                7: {"s": NS(is_dummy=False, dst_kv_indices=np.arange(80)[::2])}
            },
            _fragmented_groups=lambda src, dst: groups(None, src, dst),
        )
        sends = []
        sender = NS(
            curr_idx=0,
            kv_mgr=mgr,
            bootstrap_room=7,
            requires_staging_grid_alignment=lambda: False,
            should_send_kv_chunk=lambda *args: True,
            # DCP pack buffer bound, smaller than the 16-token staging grid.
            get_max_transfer_tokens=lambda: 8,
        )
        sender.should_split_staging = lambda *args: split(sender, *args)

        def send(pages, state, **kwargs):
            sends.append((sender.curr_idx, pages.copy(), state, kwargs))
            sender.curr_idx += len(pages)

        sender.send = send
        translate = Mock(side_effect=lambda x: x)
        req = NS(
            kv=NS(req_pool_idx=0),
            start_send_idx=0,
            origin_input_ids=list(range(32)),
            disagg_decode_prefix_len=0,
            disagg_kv_sender=sender,
            rid="r",
        )
        scheduler = NS(
            enable_staging=True,
            token_to_kv_pool_allocator=NS(
                page_size=1, translate_kv_indices_for_transfer=translate
            ),
            req_to_token_pool=NS(req_to_token=np.arange(32).reshape(1, 32)),
            disagg_prefill_pending_chunk_rids=set(),
        )
        env["_send_kv_chunk"](scheduler, req, end_idx=31)
        # cap=8 -> four pieces; the last one is the partial tail (7 pages).
        self.assertEqual([len(x[1]) for x in sends], [8, 8, 8, 7])
        self.assertEqual(translate.call_count, 1)
        self.assertEqual(req.start_send_idx, 31)
        env["_send_kv_chunk"](scheduler, req, end_idx=32)
        # Only the 1-page tail was left after the first call (start_send_idx=31).
        self.assertEqual([len(x[1]) for x in sends], [8, 8, 8, 7, 1])
        self.assertEqual(req.start_send_idx, 32)
        self.assertEqual(
            [
                gate(NS(full_chunk_pages=16), start, len(pages))
                for start, pages, *_ in sends
            ],
            [True, False, True, False, False],
        )
        self.assertEqual(
            np.concatenate([x[1] for x in sends]).tolist(), list(range(32))
        )


if __name__ == "__main__":
    unittest.main()
