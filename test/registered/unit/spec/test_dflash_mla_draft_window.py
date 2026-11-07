"""--speculative-draft-window-size on the MLA-only DFLASH draft backends.

trtllm_mla / cutedsl_mla read the draft's req_to_token rows by absolute
position (sliding-window index lists, the DCP page table), so the compact
draft table must be dropped where it changes nothing and rejected where it
would. The expected verdicts come from a brute-force coverage check over
request lengths, not from the resolver's own inequality.
"""

import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import _resolve_dflash_mla_draft_window
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_utils import resolve_mla_draft_compact_window
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

HOOK_MODULE = "sglang.srt.arg_groups.speculative_hook"
BLOCK = 8


def _mla_draft(layer_types, sliding_window=None):
    return {
        "num_hidden_layers": len(layer_types),
        "dflash_config": {"attention_mode": "mla"},
        "layer_types": list(layer_types),
        "sliding_window": sliding_window,
    }


def _compact_rows(prefix_len: int, window: int, page_size: int) -> range:
    # The compact table's committed suffix (dflash_worker_v2
    # _compute_compact_draft_seq_lens): the last min(prefix, window) positions,
    # extended left to a page boundary, then the BLOCK draft positions.
    visible = min(prefix_len, window)
    start = prefix_len - visible
    start -= start % page_size
    return range(start, prefix_len + BLOCK)


def _compact_changes_nothing(hf_window: int, window: int, page_size: int) -> bool:
    """Brute force: for every committed length, each draft row's own window
    [p - (hf_window - 1), p] lies inside the compact rows."""
    window_left = hf_window - 1
    for prefix_len in range(0, 3 * max(window, hf_window) + 2 * page_size):
        rows = _compact_rows(prefix_len, window, page_size)
        for p in range(prefix_len, prefix_len + BLOCK):
            if max(p - window_left, 0) < rows.start:
                return False
    return True


class TestResolveMlaDraftCompactWindow(CustomTestCase):
    def test_all_sliding_drops_exactly_when_the_compact_window_covers(self):
        window = 64
        for hf_window in (16, 63, 64, 65, 66, 80):
            for page_size in (1, 16):
                cfg = _mla_draft(["sliding_attention"] * 5, hf_window)
                with self.subTest(hf_window=hf_window, page_size=page_size):
                    if _compact_changes_nothing(hf_window, window, page_size):
                        self.assertIsNone(
                            resolve_mla_draft_compact_window(
                                cfg, window_size=window, dcp_size=16
                            )
                        )
                    else:
                        with self.assertRaises(ValueError):
                            resolve_mla_draft_compact_window(
                                cfg, window_size=window, dcp_size=16
                            )

    def test_five_layer_swa_drafter_shape_is_dropped(self):
        # A five-layer sliding-window drafter (window 1024) with
        # --speculative-draft-window-size 1024, with and without DCP.
        cfg = _mla_draft(["sliding_attention"] * 5, 1024)
        for dcp_size in (1, 8, 16):
            with self.subTest(dcp_size=dcp_size):
                self.assertIsNone(
                    resolve_mla_draft_compact_window(
                        cfg, window_size=1024, dcp_size=dcp_size
                    )
                )

    def test_mixed_layers_are_rejected(self):
        cfg = _mla_draft(["full_attention"] + ["sliding_attention"] * 4, 1024)
        for dcp_size in (1, 16):
            with self.subTest(dcp_size=dcp_size):
                with self.assertRaises(ValueError):
                    resolve_mla_draft_compact_window(
                        cfg, window_size=1024, dcp_size=dcp_size
                    )

    def test_full_attention_keeps_the_table_only_without_dcp(self):
        cfg = _mla_draft(["full_attention"] * 7)
        self.assertEqual(
            resolve_mla_draft_compact_window(cfg, window_size=1024, dcp_size=1), 1024
        )
        for dcp_size in (2, 8, 16):
            with self.subTest(dcp_size=dcp_size):
                with self.assertRaises(ValueError):
                    resolve_mla_draft_compact_window(
                        cfg, window_size=1024, dcp_size=dcp_size
                    )
        no_types = {"num_hidden_layers": 7, "dflash_config": {"attention_mode": "mla"}}
        with self.assertRaises(ValueError):
            resolve_mla_draft_compact_window(no_types, window_size=1024, dcp_size=16)

    def test_no_window_flag_is_a_no_op(self):
        cfg = _mla_draft(["full_attention"] * 7)
        self.assertIsNone(
            resolve_mla_draft_compact_window(cfg, window_size=None, dcp_size=16)
        )


def _args(draft_backend: str, dcp_size: int) -> ServerArgs:
    return ServerArgs(
        model_path="dummy",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path="draft",
        speculative_num_draft_tokens=8,
        speculative_draft_window_size=1024,
        speculative_draft_attention_backend=draft_backend,
        dcp_size=dcp_size,
    )


class TestResolveDflashMlaDraftWindowHook(CustomTestCase):
    def _run(self, draft_backend: str, dcp_size: int, draft_cfg) -> ServerArgs:
        args = _args(draft_backend, dcp_size)
        with patch(f"{HOOK_MODULE}._load_draft_hf_config", return_value=draft_cfg):
            _resolve_dflash_mla_draft_window(args)
        return args

    def test_swa_mla_draft_on_mla_backends_drops_the_window(self):
        draft_cfg = _mla_draft(["sliding_attention"] * 5, 1024)
        for backend in ("cutedsl_mla", "trtllm_mla"):
            with self.subTest(backend=backend):
                args = self._run(backend, 16, draft_cfg)
                self.assertIsNone(
                    resolution_result(args, "speculative_draft_window_size")
                )

    def test_other_backends_keep_the_window(self):
        draft_cfg = _mla_draft(["sliding_attention"] * 5, 1024)
        args = self._run("triton", 1, draft_cfg)
        self.assertEqual(resolution_result(args, "speculative_draft_window_size"), 1024)

    def test_non_mla_draft_keeps_the_window(self):
        gqa = {
            "num_hidden_layers": 6,
            "dflash_config": {"attention_mode": "gqa"},
            "layer_types": ["sliding_attention"] * 6,
            "sliding_window": 4096,
        }
        args = self._run("cutedsl_mla", 16, gqa)
        self.assertEqual(resolution_result(args, "speculative_draft_window_size"), 1024)

    def test_full_mla_draft_under_dcp_is_rejected(self):
        with self.assertRaises(ValueError):
            self._run("cutedsl_mla", 16, _mla_draft(["full_attention"] * 7))


if __name__ == "__main__":
    unittest.main()
