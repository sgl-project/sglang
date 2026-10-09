import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.layout import TokenAxis
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeModelRunner:
    max_decode_logits_rows = ModelRunner.max_decode_logits_rows

    def __init__(self, *, initial_width: int, cuda_graph_bs: list[int]):
        self.initial_width = initial_width
        self.cuda_graph_bs = cuda_graph_bs

    def decode_num_tokens_per_req(self, *, num_draft_tokens=None):
        return self.initial_width if num_draft_tokens is None else num_draft_tokens


def _alignment_8_capture_bs(runner, width):
    return ([bs for bs in runner.cuda_graph_bs if bs * width % 8 == 0], [])


class TestModelRunnerDecodeRows(unittest.TestCase):
    def test_adaptive_sizing_covers_a_wider_candidate_width(self):
        """The shared logits buffer is sized for the widest adaptive candidate
        width: bs 12 at width 6 needs 72 rows."""
        runner = _FakeModelRunner(initial_width=4, cuda_graph_bs=[4, 8, 12])
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            f.write('{"1":{"candidate_steps":[3,5]}}')
            f.flush()
            spec = SimpleNamespace(
                speculative_adaptive=True, speculative_adaptive_config=f.name
            )
            with (
                patch(
                    "sglang.srt.model_executor.model_runner.get_spec", return_value=spec
                ),
                patch(
                    "sglang.srt.model_executor.model_runner.max_speculative_num_draft_tokens",
                    return_value=6,
                ),
                patch(
                    "sglang.srt.model_executor.model_runner.get_batch_sizes_to_capture",
                    side_effect=_alignment_8_capture_bs,
                ),
            ):
                self.assertEqual(runner.max_decode_logits_rows(), 72)


class _FakePPRunner:
    is_pp_proxy_input_scattered = ModelRunner.is_pp_proxy_input_scattered
    is_pp_proxy_output_scattered = ModelRunner.is_pp_proxy_output_scattered

    def __init__(self, first_attn, last_ffn):
        self.first_attn = first_attn
        self.last_ffn = last_ffn

    def _pp_boundary_stages(self):
        return self.first_attn, self.last_ffn


class TestModelRunnerPPProxyRows(CustomTestCase):
    def test_prefill_pp_input_uses_stage_boundary_row_layout(self):
        """A PP receiver must recognize attention-TP-sharded input rows."""
        runner = _FakePPRunner(
            first_attn=SimpleNamespace(input_on_attn_tp_slices=True),
            last_ffn=None,
        )
        with patch(
            "sglang.srt.model_executor.model_runner.get_parallel",
            return_value=SimpleNamespace(pp_size=4, pp_rank=1),
        ):
            self.assertTrue(runner.is_pp_proxy_input_scattered())

    def test_prefill_pp_output_uses_ordinary_ffn_exit_layout(self):
        """A PP sender must preserve attention-TP-sharded output rows."""
        produced = SimpleNamespace(layout=SimpleNamespace(sharded={TokenAxis.ATTN_TP}))
        edges = {
            BatchVariant.ORDINARY: SimpleNamespace(
                outgoing=SimpleNamespace(produced=produced)
            )
        }
        runner = _FakePPRunner(
            first_attn=None,
            last_ffn=SimpleNamespace(plan=SimpleNamespace(edges=edges)),
        )
        with patch(
            "sglang.srt.model_executor.model_runner.get_parallel",
            return_value=SimpleNamespace(pp_size=4, pp_rank=0),
        ):
            self.assertTrue(runner.is_pp_proxy_output_scattered())


if __name__ == "__main__":
    unittest.main()
