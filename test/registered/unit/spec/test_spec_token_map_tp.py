import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_spec_args(algorithm: str, tp_size: int, token_map) -> ServerArgs:
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
    return args


class TestSpecTokenMapTP(CustomTestCase):
    """--speculative-token-map at tp_size > 1 must be rejected at arg
    validation, not reach the draft worker and fault the device (a vocab-sharded
    target lm_head gathered by global hot token ids)."""

    def test_token_map_with_tp_gt_1_is_rejected(self):
        args = _make_spec_args("EAGLE", tp_size=2, token_map="/path/to/map.pt")

        with self.assertRaisesRegex(ValueError, "speculative-token-map"):
            handle_speculative_decoding(args)

    def test_token_map_with_tp_1_is_allowed(self):
        args = _make_spec_args("EAGLE", tp_size=1, token_map="/path/to/map.pt")

        handle_speculative_decoding(args)

    def test_eagle3_ignores_token_map_so_tp_gt_1_is_allowed(self):
        # EAGLE3 drops the flag and uses the checkpoint's own reduced head.
        args = _make_spec_args("EAGLE3", tp_size=2, token_map="/path/to/map.pt")

        handle_speculative_decoding(args)


if __name__ == "__main__":
    unittest.main()
