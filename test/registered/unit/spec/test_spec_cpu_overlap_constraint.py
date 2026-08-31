import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _make_spec_args(device: str, algorithm: str = "EAGLE", **overrides) -> ServerArgs:
    # model_path="dummy" short-circuits ServerArgs.__post_init__; invoke the
    # speculative hook directly (same pattern as the unit/server_args tests).
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = algorithm
    args.device = device
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


class TestSpecCPUOverlapConstraint(CustomTestCase):
    def test_cpu_eagle_forces_disable_overlap_schedule(self):
        args = _make_spec_args(device="cpu")
        self.assertFalse(resolution_result(args, "disable_overlap_schedule"))

        handle_speculative_decoding(args)

        self.assertTrue(resolution_result(args, "disable_overlap_schedule"))

    def test_cpu_eagle3_forces_disable_overlap_schedule(self):
        args = _make_spec_args(device="cpu", algorithm="EAGLE3")

        handle_speculative_decoding(args)

        self.assertTrue(resolution_result(args, "disable_overlap_schedule"))

    def test_cpu_explicit_disable_overlap_is_preserved(self):
        args = _make_spec_args(device="cpu", disable_overlap_schedule=True)

        # Already disabled: the hook must not flip the flag, and (unlike the
        # forced-disable cases) must not warn about overriding it.
        with self.assertLogs(
            "sglang.srt.arg_groups.speculative_hook", "WARNING"
        ) as logs:
            handle_speculative_decoding(args)

        self.assertTrue(resolution_result(args, "disable_overlap_schedule"))
        self.assertFalse(
            any("Overlap schedule" in message for message in logs.output),
            f"hook warned about overriding an already-disabled overlap: {logs.output}",
        )

    def test_cuda_eagle_keeps_overlap_schedule(self):
        # Guard the constraint's scope: the hook must not touch non-CPU devices.
        args = _make_spec_args(device="cuda")

        handle_speculative_decoding(args)

        self.assertFalse(resolution_result(args, "disable_overlap_schedule"))

    def test_hybrid_kda_allows_plan_stream_graph_load(self):
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
        from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend

        full_backend = Mock(
            token_to_kv_pool=object(),
            req_to_token_pool=object(),
            max_context_len=4096,
            needs_cpu_seq_lens=False,
        )
        linear_backend = Mock(
            spec=KDAAttnBackend,
            needs_cpu_seq_lens=False,
            supports_overlap_plan_stream_graph_load=(
                KDAAttnBackend.supports_overlap_plan_stream_graph_load
            ),
        )

        backend = HybridLinearAttnBackend(full_backend, linear_backend, [])

        self.assertTrue(backend.supports_overlap_plan_stream_graph_load)
        self.assertFalse(backend.needs_cpu_seq_lens)

    def test_unaudited_linear_backend_defers_plan_stream_graph_load(self):
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
            MambaAttnBackendBase,
        )

        full_backend = Mock(
            token_to_kv_pool=object(),
            req_to_token_pool=object(),
            max_context_len=4096,
            needs_cpu_seq_lens=False,
        )
        linear_backend = Mock(
            spec=MambaAttnBackendBase,
            needs_cpu_seq_lens=False,
            supports_overlap_plan_stream_graph_load=(
                MambaAttnBackendBase.supports_overlap_plan_stream_graph_load
            ),
        )

        backend = HybridLinearAttnBackend(full_backend, linear_backend, [])

        self.assertFalse(backend.supports_overlap_plan_stream_graph_load)


if __name__ == "__main__":
    unittest.main()
