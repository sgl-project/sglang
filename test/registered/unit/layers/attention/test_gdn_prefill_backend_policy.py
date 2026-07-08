import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import torch

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    GDNKernelDispatcher,
    _build_flashinfer_checkpoint_plan,
    maybe_set_default_flashinfer_gdn_prefill,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_runner(
    *,
    state_dtype=torch.bfloat16,
    key_dim=128,
    value_dim=128,
    multimodal=False,
    model_impl=ModelImpl.SGLANG,
    **arg_overrides,
):
    args = SimpleNamespace(
        linear_attn_backend="triton",
        linear_attn_prefill_backend=None,
        uses_mamba_radix_cache=False,
        enable_page_major_kv_layout=False,
        mamba_radix_cache_strategy="no_buffer",
        enable_dynamic_chunking=False,
        chunked_prefill_size=8192,
    )
    for name, value in arg_overrides.items():
        setattr(args, name, value)

    return SimpleNamespace(
        server_args=args,
        hybrid_gdn_config=SimpleNamespace(
            linear_key_head_dim=key_dim,
            linear_value_head_dim=value_dim,
        ),
        model_config=SimpleNamespace(
            is_multimodal=multimodal,
            _resolved_model_impl=model_impl,
        ),
        req_to_token_pool=SimpleNamespace(
            mamba_pool=SimpleNamespace(
                mamba_cache=SimpleNamespace(temporal=SimpleNamespace(dtype=state_dtype))
            )
        ),
    )


class TestFlashInferGDNPrefillBackendPolicy(unittest.TestCase):
    def apply_policy(
        self,
        runner,
        *,
        cuda=True,
        capability=(10, 0),
        cuda_version="13.0",
        flashinfer_available=True,
    ):
        with (
            patch.object(gdn_backend, "is_cuda", return_value=cuda),
            patch.object(torch.cuda, "get_device_capability", return_value=capability),
            patch.object(torch.version, "cuda", cuda_version),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                "is_flashinfer_gdn_prefill_available",
                return_value=flashinfer_available,
            ),
        ):
            maybe_set_default_flashinfer_gdn_prefill(runner)
        return runner.server_args.linear_attn_prefill_backend

    def test_selects_flashinfer_for_supported_sm100_gdn(self):
        self.assertEqual(self.apply_policy(make_runner()), "flashinfer")

    def test_selects_flashinfer_for_radix_cache_strategies(self):
        for strategy in ("no_buffer", "extra_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                runner = make_runner(
                    uses_mamba_radix_cache=True,
                    mamba_radix_cache_strategy=strategy,
                )
                self.assertEqual(self.apply_policy(runner), "flashinfer")

    def test_selects_flashinfer_for_native_multimodal_model(self):
        self.assertEqual(self.apply_policy(make_runner(multimodal=True)), "flashinfer")

    def test_preserves_explicit_prefill_override(self):
        for backend in ("triton", "flashinfer", "cutedsl"):
            with self.subTest(backend=backend):
                runner = make_runner(linear_attn_prefill_backend=backend)
                self.assertEqual(self.apply_policy(runner), backend)

    def test_rejects_unsupported_capability(self):
        cases = (
            ("non_cuda", {}, {"cuda": False}),
            ("hopper", {}, {"capability": (9, 0)}),
            ("future_sm", {}, {"capability": (12, 0)}),
            ("cuda_12", {}, {"cuda_version": "12.9"}),
            ("fp32_state", {"state_dtype": torch.float32}, {}),
            ("key_dim", {"key_dim": 64}, {}),
            ("value_dim", {"value_dim": 64}, {}),
            ("missing_api", {}, {"flashinfer_available": False}),
        )
        for name, runner_args, hardware in cases:
            with self.subTest(name=name):
                self.assertIsNone(
                    self.apply_policy(make_runner(**runner_args), **hardware)
                )

    def test_rejects_gdn_config_without_qwen_head_dims(self):
        runner = make_runner()
        runner.hybrid_gdn_config = SimpleNamespace()
        self.assertIsNone(self.apply_policy(runner))

    def test_rejects_unvalidated_runtime_modes(self):
        cases = (
            ("non_triton_base", {"linear_attn_backend": "cutedsl"}),
            ("page_major_kv", {"enable_page_major_kv_layout": True}),
            ("dynamic_chunk", {"enable_dynamic_chunking": True}),
            ("unchunked", {"chunked_prefill_size": -1}),
            ("unknown_chunk", {"chunked_prefill_size": None}),
            ("large_chunk", {"chunked_prefill_size": 8193}),
            (
                "transformers_multimodal",
                {"multimodal": True, "model_impl": ModelImpl.TRANSFORMERS},
            ),
        )
        for name, runner_args in cases:
            with self.subTest(name=name):
                self.assertIsNone(self.apply_policy(make_runner(**runner_args)))

    def test_builds_compact_checkpoint_plan_for_packed_sequences(self):
        checkpoint_cu_starts, track_checkpoint_src = _build_flashinfer_checkpoint_plan(
            extend_seq_lens=torch.tensor([63, 64, 65, 127, 128, 129]),
            mamba_track_mask=torch.tensor([False, True, True, True, True, True]),
            # 65 on the 128-token sequence represents an interior S64
            # boundary encoded as S64 + 1 by the scheduler.
            mamba_track_seqlens=torch.tensor([63, 64, 65, 127, 65, 129]),
            prefix_lens=torch.zeros(6, dtype=torch.int64),
            checkpoint_every_n_tokens=64,
        )

        torch.testing.assert_close(
            checkpoint_cu_starts, torch.tensor([0, 0, 1, 2, 3, 5, 7])
        )
        torch.testing.assert_close(track_checkpoint_src, torch.tensor([1, 2, 3, 6]))

    def test_decode_tracking_without_h_source_skips_checkpoint_plan(self):
        backend = object.__new__(GDNAttnBackend)
        backend.kernel_dispatcher = SimpleNamespace(extend_uses_state_checkpoints=True)
        metadata = SimpleNamespace(has_mamba_track_mask=True, track_ssm_h_src=None)
        forward_batch = SimpleNamespace(
            mamba_track_mask=torch.tensor([True]),
            mamba_track_indices=torch.tensor([7]),
        )

        def init_base(instance, _forward_batch):
            instance.forward_metadata = metadata

        with patch.object(MambaAttnBackendBase, "init_forward_metadata", init_base):
            backend.init_forward_metadata(forward_batch)

        torch.testing.assert_close(metadata.conv_states_mask_indices, torch.tensor([7]))

    def test_tree_verify_uses_triton_kernel(self):
        flashinfer_kernel = MagicMock(supports_target_verify=True)
        with (
            patch.object(gdn_backend, "is_cuda", return_value=True),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                "FlashInferGDNKernel",
                return_value=flashinfer_kernel,
            ),
        ):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.TRITON,
                LinearAttnKernelBackend.FLASHINFER,
            )

        self.assertIsInstance(dispatcher.tree_verify_kernel, TritonGDNKernel)

        tensor = sentinel.tensor
        with patch.object(
            dispatcher.tree_verify_kernel, "target_verify"
        ) as tree_verify:
            dispatcher.target_verify(
                *([tensor] * 7),
                ssm_states=tensor,
                cache_indices=tensor,
                query_start_loc=tensor,
                retrieve_parent_token=sentinel.parent_token,
            )

        tree_verify.assert_called_once()
        flashinfer_kernel.target_verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
