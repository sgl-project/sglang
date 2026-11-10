"""--speculative-draft-window-size with a sliding-window MLA draft on the
MLA-only DFLASH draft backends, which apply the draft's own window."""

import unittest
from typing import Optional
from unittest.mock import patch

from sglang.srt.arg_groups.speculative_hook import (
    _reject_dflash_sliding_mla_draft_window,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

HOOK_MODULE = "sglang.srt.arg_groups.speculative_hook"


def _mla_draft(layer_types, sliding_window=None):
    return {
        "num_hidden_layers": len(layer_types),
        "dflash_config": {"attention_mode": "mla"},
        "layer_types": list(layer_types),
        "sliding_window": sliding_window,
    }


def _check(draft_backend: str, window_size: Optional[int], draft_cfg) -> None:
    args = ServerArgs(
        model_path="dummy",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path="draft",
        speculative_num_draft_tokens=8,
        speculative_draft_window_size=window_size,
        speculative_draft_attention_backend=draft_backend,
    )
    with patch(f"{HOOK_MODULE}._load_draft_hf_config", return_value=draft_cfg):
        _reject_dflash_sliding_mla_draft_window(args)


class TestRejectDflashSlidingMlaDraftWindow(CustomTestCase):
    def test_sliding_mla_draft_on_mla_backends_is_rejected(self):
        draft_cfg = _mla_draft(["sliding_attention"] * 5, 1024)
        for backend in ("cutedsl_mla", "trtllm_mla"):
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError):
                    _check(backend, 1024, draft_cfg)

    def test_full_attention_mla_draft_keeps_the_flag(self):
        _check("cutedsl_mla", 1024, _mla_draft(["full_attention"] * 7))

    def test_no_flag_is_accepted(self):
        _check("cutedsl_mla", None, _mla_draft(["sliding_attention"] * 5, 1024))


if __name__ == "__main__":
    unittest.main()
