import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import (
    _mamba_radix_cache_resolution,
    supports_mamba_cache_extra_buffer,
)
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

QWEN3_NEXT_ARCH = "Qwen3NextForCausalLM"


def _make_view(arch: str = QWEN3_NEXT_ARCH, **overrides) -> SimpleNamespace:
    view = SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[arch])
        ),
        disable_radix_cache=False,
        speculative_algorithm=None,
        mamba_radix_cache_strategy="auto",
        disable_overlap_schedule=False,
        page_size=1,
        linear_attn_backend="triton",
    )
    for key, value in overrides.items():
        setattr(view, key, value)
    return view


class _patch_device:
    """Patch the physical-device probes in both namespaces that read them."""

    def __init__(self, cuda: bool):
        self._patchers = [
            patch.multiple(
                module,
                is_cuda=lambda: cuda,
                is_musa=lambda: False,
                is_npu=lambda: False,
            )
            for module in (
                "sglang.srt.arg_groups.overrides",
                "sglang.srt.server_args",
            )
        ]

    def __enter__(self):
        for patcher in self._patchers:
            patcher.start()
        return self

    def __exit__(self, *exc):
        for patcher in self._patchers:
            patcher.stop()
        return False


def _make_spec_args(device: str, arch: str, topk: int) -> ServerArgs:
    # model_path="dummy" short-circuits ServerArgs.__post_init__; invoke the
    # speculative hook directly (same pattern as the unit/server_args tests).
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = "EAGLE"
    args.device = device
    # Resolved during __post_init__ on real args; the topk > 1 paths compare it.
    args.page_size = 1
    args.speculative_num_steps = 3
    args.speculative_eagle_topk = topk
    args.speculative_num_draft_tokens = topk * 3 + 1
    args.get_model_config = lambda: SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[arch],
            get_text_config=lambda: SimpleNamespace(),
        )
    )
    return args


def _make_handler_args(
    arch: str = QWEN3_NEXT_ARCH, speculative_algorithm=None
) -> ServerArgs:
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = speculative_algorithm
    args.get_model_config = lambda: SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[arch])
    )
    return args


class TestMambaRadixCacheResolutionOffCuda(CustomTestCase):
    def test_extra_buffer_not_picked_without_cuda(self):
        """Bug regression: on CPU the auto strategy used to resolve to
        extra_buffer (overlap wanted + triton linear backend), which then
        failed _validate_mamba_extra_buffer's CUDA/MUSA/NPU assert at startup
        for any hybrid GDN model, speculative or not."""
        with _patch_device(cuda=False):
            self.assertFalse(
                supports_mamba_cache_extra_buffer(_make_view(), QWEN3_NEXT_ARCH)
            )
            declared = _mamba_radix_cache_resolution(_make_view())

        self.assertEqual(
            declared,
            {
                "uses_mamba_radix_cache": True,
                "mamba_radix_cache_strategy": "no_buffer",
                "disable_overlap_schedule": True,
            },
        )

    def test_extra_buffer_still_picked_on_cuda(self):
        with _patch_device(cuda=True):
            declared = _mamba_radix_cache_resolution(_make_view())

        self.assertEqual(
            declared,
            {
                "uses_mamba_radix_cache": True,
                "mamba_radix_cache_strategy": "extra_buffer",
            },
        )

    def test_spec_disables_radix_cache_without_cuda(self):
        """Speculative decoding + mamba radix cache needs extra_buffer
        (CUDA/MUSA/NPU-only); off those devices the handler must drop the
        radix cache instead of resolving a spec-incompatible no_buffer
        setup."""
        args = _make_handler_args(speculative_algorithm="EAGLE")
        with _patch_device(cuda=False):
            args._handle_mamba_radix_cache(model_arch=QWEN3_NEXT_ARCH)

        self.assertTrue(args.disable_radix_cache)

    def test_spec_keeps_radix_cache_on_cuda(self):
        args = _make_handler_args(speculative_algorithm="EAGLE")
        with _patch_device(cuda=True):
            args._handle_mamba_radix_cache(model_arch=QWEN3_NEXT_ARCH)

        self.assertFalse(args.disable_radix_cache)

    def test_spec_keeps_radix_cache_for_non_mamba_model(self):
        args = _make_handler_args(
            arch="LlamaForCausalLM", speculative_algorithm="EAGLE"
        )
        with _patch_device(cuda=False):
            args._handle_mamba_radix_cache(model_arch="LlamaForCausalLM")

        self.assertFalse(args.disable_radix_cache)


class TestSpecCPUTreeVerifyConstraint(CustomTestCase):
    def test_cpu_topk_gt1_rejected_for_hybrid_gdn(self):
        """The CPU causal-conv / gated-delta-rule kernels are not tree-aware;
        without this startup guard, EAGLE topk > 1 on a hybrid GDN model would
        reach the kernels and die on a mid-inference TORCH_CHECK instead."""
        args = _make_spec_args(device="cpu", arch=QWEN3_NEXT_ARCH, topk=2)
        with self.assertRaisesRegex(ValueError, "hybrid"):
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
