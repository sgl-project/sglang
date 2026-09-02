import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

QWEN3_NEXT_ARCH = "Qwen3NextForCausalLM"


def _make_spec_args(
    device: str, arch: str, topk: int, algorithm: str = "EAGLE"
) -> ServerArgs:
    # model_path="dummy" short-circuits ServerArgs.__post_init__; invoke the
    # speculative hook directly (same pattern as the unit/server_args tests).
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = algorithm
    args.device = device
    # Resolved during __post_init__ on real args; the topk > 1 paths compare it.
    args.page_size = 1
    args.speculative_num_steps = 3
    args.speculative_eagle_topk = topk
    args.speculative_num_draft_tokens = topk * 3 + 1
    # CPU speculative decoding needs the radix cache off; the guard tests below
    # re-enable it to exercise the rejection.
    args.disable_radix_cache = True
    # A configuration a fixture supplies carries no build key, so model_config_of
    # hands it back instead of resolving model_path.
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[arch],
            get_text_config=lambda: SimpleNamespace(),
        )
    )
    return args


class TestSpecMambaRadixCacheGuardOnCPU(CustomTestCase):
    def test_spec_rejects_radix_cache_on_cpu(self):
        """CPU resolves the strategy to no_buffer, which has no spec-aware
        state tracking, so the combination must be rejected rather than
        resolved into a spec-incompatible setup."""
        args = _make_spec_args(device="cpu", arch=QWEN3_NEXT_ARCH, topk=1)
        args.disable_radix_cache = False
        with self.assertRaisesRegex(ValueError, "radix cache"):
            handle_speculative_decoding(args)

    def test_ngram_rejects_radix_cache_on_cpu(self):
        # the guard sits in the NGRAM handler as well as the EAGLE family one
        args = _make_spec_args(
            device="cpu", arch=QWEN3_NEXT_ARCH, topk=1, algorithm="NGRAM"
        )
        args.disable_radix_cache = False
        with self.assertRaisesRegex(ValueError, "radix cache"):
            handle_speculative_decoding(args)

    def test_spec_allows_radix_cache_on_cuda(self):
        args = _make_spec_args(device="cuda", arch=QWEN3_NEXT_ARCH, topk=1)
        args.disable_radix_cache = False
        handle_speculative_decoding(args)

    def test_spec_allows_radix_cache_for_non_mamba_model(self):
        args = _make_spec_args(device="cpu", arch="LlamaForCausalLM", topk=1)
        args.disable_radix_cache = False
        handle_speculative_decoding(args)


class TestSpecCPUTreeVerifyAllowed(CustomTestCase):
    """The CPU causal-conv and gated-delta-rule kernels are tree-aware, so
    every EAGLE topk combination must pass argument resolution on CPU."""

    def test_cpu_topk_gt1_allowed_for_hybrid_gdn(self):
        args = _make_spec_args(device="cpu", arch=QWEN3_NEXT_ARCH, topk=2)
        handle_speculative_decoding(args)

    def test_cpu_topk1_allowed_for_hybrid_gdn(self):
        args = _make_spec_args(device="cpu", arch=QWEN3_NEXT_ARCH, topk=1)
        handle_speculative_decoding(args)

    def test_cpu_topk_gt1_allowed_for_non_mamba_model(self):
        args = _make_spec_args(device="cpu", arch="LlamaForCausalLM", topk=2)
        handle_speculative_decoding(args)

    def test_cuda_topk_gt1_allowed_for_hybrid_gdn(self):
        args = _make_spec_args(device="cuda", arch=QWEN3_NEXT_ARCH, topk=2)
        handle_speculative_decoding(args)


if __name__ == "__main__":
    unittest.main()
