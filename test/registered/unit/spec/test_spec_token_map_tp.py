import unittest
from types import SimpleNamespace

import torch

from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_utils import validate_hot_token_ids_fit
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_spec_args(algorithm: str, tp_size: int, token_map, **overrides) -> ServerArgs:
    # model_path="dummy" short-circuits ServerArgs.__post_init__; invoke the
    # speculative hook directly (same pattern as the unit/server_args tests).
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = algorithm
    args.tp_size = tp_size
    args.speculative_token_map = token_map
    # Fully specify the chain config so the hook doesn't auto-choose params.
    args.speculative_num_steps = 3
    args.speculative_eagle_topk = 1
    args.speculative_num_draft_tokens = 4
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            get_text_config=lambda: SimpleNamespace(),
        )
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestSpecTokenMapTP(CustomTestCase):
    """--speculative-token-map must not reach the draft worker and fault the
    device: global hot token ids cannot index a vocab-sharded target lm_head,
    and the standalone draft never reduces a head at all.
    """

    def test_token_map_with_tp_gt_1_is_rejected(self):
        args = _make_spec_args("EAGLE", tp_size=2, token_map="/path/to/map.pt")

        with self.assertRaisesRegex(ValueError, "tp_size > 1"):
            handle_speculative_decoding(args)

    def test_token_map_with_tp_1_is_allowed(self):
        args = _make_spec_args("EAGLE", tp_size=1, token_map="/path/to/map.pt")

        handle_speculative_decoding(args)

    def test_eagle3_ignores_token_map_so_tp_gt_1_is_allowed(self):
        # EAGLE3 drops the flag and uses the checkpoint's own reduced head.
        args = _make_spec_args("EAGLE3", tp_size=2, token_map="/path/to/map.pt")

        handle_speculative_decoding(args)

    def test_standalone_rejects_token_map_even_at_tp_1(self):
        # StandaloneDraftWorker.init_lm_head is a no-op, so the map reduces no
        # head and the full-vocab proposal would index it out of bounds.
        args = _make_spec_args("STANDALONE", tp_size=1, token_map="/path/to/map.pt")

        with self.assertRaisesRegex(ValueError, "STANDALONE"):
            handle_speculative_decoding(args)

    def test_attention_dp_defers_to_the_runtime_row_check(self):
        # Under attention DP the head may be replicated per rank, which only the
        # model class decides, so arg validation must not pre-emptively reject.
        args = _make_spec_args(
            "EAGLE",
            tp_size=2,
            token_map="/path/to/map.pt",
            attn_dp_size=2,
            enable_dp_lm_head=True,
        )

        handle_speculative_decoding(args)


class TestHotTokenIdsFitHead(CustomTestCase):
    """The exact row check behind the arg guard: a hot id at or past the head's
    row count would gather out of bounds.
    """

    def test_ids_past_the_last_row_are_rejected(self):
        # tp=2 over vocab 248077 leaves 124064 rows per rank (issue #42397).
        hot_token_id = torch.tensor([0, 1000, 248076], dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "248076"):
            validate_hot_token_ids_fit(hot_token_id, 124064)

    def test_ids_inside_a_replicated_head_are_accepted(self):
        hot_token_id = torch.tensor([0, 1000, 248076], dtype=torch.int64)

        validate_hot_token_ids_fit(hot_token_id, 248077)

    def test_the_last_row_is_in_bounds(self):
        validate_hot_token_ids_fit(torch.tensor([7], dtype=torch.int64), 8)


if __name__ == "__main__":
    unittest.main()
