"""Structural tests for the per-config `capture_routed_experts` opt-out.

These tests verify the static contract of the flag without requiring a CUDA
runtime: the dataclass default, the constructor kwarg, and the propagation
through TopK.__init__ -> TopKConfig storage. End-to-end runtime behavior
(draft model not polluting target R3 buffer) is exercised by the integration
test in test/registered/rl/.
"""

import inspect
import unittest

from sglang.srt.layers.moe.topk import TopK, TopKConfig, _post_process_topk_ids


class TopKConfigCaptureRoutedExpertsFlagTest(unittest.TestCase):
    """AC-1: TopKConfig field + TopK.__init__ kwarg + capture-site gate."""

    def test_dataclass_default_is_true(self):
        cfg = TopKConfig(top_k=8)
        self.assertTrue(
            cfg.capture_routed_experts,
            "default must remain True so target-only R3 behavior is preserved",
        )

    def test_dataclass_accepts_explicit_false(self):
        cfg = TopKConfig(top_k=8, capture_routed_experts=False)
        self.assertFalse(cfg.capture_routed_experts)

    def test_topk_init_propagates_default(self):
        topk = TopK(top_k=8)
        self.assertTrue(topk.topk_config.capture_routed_experts)

    def test_topk_init_propagates_false(self):
        topk = TopK(top_k=8, capture_routed_experts=False)
        self.assertFalse(topk.topk_config.capture_routed_experts)

    def test_post_process_capture_site_reads_flag(self):
        # Source-level guard: the capture site must read
        # `topk_config.capture_routed_experts` BEFORE calling
        # `cap.capture(...)`. A future refactor that drops the gate
        # would silently re-introduce draft pollution; this assertion
        # is the regression tripwire.
        source = inspect.getsource(_post_process_topk_ids)
        self.assertIn("topk_config.capture_routed_experts", source)
        # Verify ordering: the flag check appears before cap.capture(.
        flag_idx = source.find("topk_config.capture_routed_experts")
        capture_idx = source.find("cap.capture(")
        self.assertGreater(capture_idx, flag_idx)
        self.assertGreater(flag_idx, 0)


class TopKConfigBypassPathPropagationTest(unittest.TestCase):
    """AC-2: BYPASS piecewise-CUDA-graph path carries the flag.

    The path decomposes `topk_output.topk_config` into scalar args at the
    caller, reconstructs `TopKConfig(...)` inside the registered custom op,
    and feeds the rebuilt config into `forward_impl`. If any of the three
    sites loses the flag, the rebuilt `TopKConfig` defaults back to True
    and draft pollution re-emerges silently. We assert all three sites
    mention `capture_routed_experts` at the source level.
    """

    def test_caller_decomposes_capture_flag(self):
        from sglang.srt.layers.moe.fused_moe_triton import layer as fmt_layer

        source = inspect.getsource(fmt_layer)
        # The dispatch call site must pass capture_routed_experts as one of
        # the scalar args. The exact call appears once for the BYPASS branch.
        self.assertIn(
            "topk_output.topk_config.capture_routed_experts",
            source,
            "caller dispatch must propagate capture_routed_experts into the "
            "custom-op call (otherwise the rebuilt TopKConfig defaults to True)",
        )

    def test_custom_op_signature_accepts_flag(self):
        # The @register_custom_op decorator wraps the underlying function
        # with a generic (*args, **kwargs) signature, so we cannot use
        # inspect.signature on the public callable. Verify the source of
        # the original definition instead.
        from sglang.srt.layers.moe.fused_moe_triton import layer as fmt_layer

        source = inspect.getsource(fmt_layer)
        # The original `def fused_moe_bypassed_piecewise_cuda_graph_impl(...)`
        # block must declare `capture_routed_experts: bool` as a parameter.
        self.assertRegex(
            source,
            r"def fused_moe_bypassed_piecewise_cuda_graph_impl\([^)]*"
            r"capture_routed_experts:\s*bool",
            "the @register_custom_op-decorated function definition must "
            "declare capture_routed_experts: bool; otherwise the caller's "
            "value is dropped at the op boundary",
        )

    def test_rebuilt_topkconfig_carries_flag(self):
        # `fused_moe_bypassed_piecewise_cuda_graph_impl` is wrapped by
        # @register_custom_op into a torch OpOverloadPacket, so we cannot
        # inspect.getsource() the function-level symbol. Inspect the module
        # source and assert the fresh TopKConfig(...) reconstruction inside
        # the impl carries the flag forward.
        from sglang.srt.layers.moe.fused_moe_triton import layer as fmt_layer

        source = inspect.getsource(fmt_layer)
        self.assertIn("capture_routed_experts=capture_routed_experts", source)


if __name__ == "__main__":
    unittest.main()
