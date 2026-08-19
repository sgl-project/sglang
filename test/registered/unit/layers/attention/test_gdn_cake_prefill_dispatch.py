import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _UnsupportedError(NotImplementedError):
    pass


class _CakeAPI:
    CakeGDNUnsupportedError = _UnsupportedError

    def __init__(self):
        self.select_cake_gdn_prefill_variant = MagicMock(
            return_value=SimpleNamespace(
                route_id="cake.gdn_prefill.noncp.full_dv",
                variant_name="prefill_bf16_indexed",
            )
        )
        self.load_cake_gdn_kernel = MagicMock()


def _kernel_and_inputs():
    api = _CakeAPI()
    entry = MagicMock()
    kernel = object.__new__(FlashInferGDNKernel)
    kernel._cake_gdn_api = api
    kernel._cake_gdn_arch = "sm_100a"
    kernel._cake_gdn_entries = {"prefill_bf16_indexed": entry}
    kernel._cake_gdn_logged_routes = set()
    kernel._flashinfer_gdn_should_use_cp_host = MagicMock(return_value=False)
    kernel._flashinfer_gdn_num_sms = 148
    kernel._flashinfer_gdn_device_name = "NVIDIA B200"
    kernel._flashinfer_gdn_device_capability = (10, 0)

    total_tokens = 320
    q = torch.empty(total_tokens, 4, 128, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty(total_tokens, 8, 128, dtype=torch.bfloat16)
    state = torch.empty(7, 8, 128, 128, dtype=torch.bfloat16)
    state_indices = torch.tensor([5, 3, 1, 6, 2], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 64, 128, 192, 256, 320], dtype=torch.int32)
    alpha = torch.empty(total_tokens, 8, dtype=torch.float32)
    beta = torch.empty_like(alpha)
    output = torch.empty_like(v)
    workspace = torch.empty(32 * 4 * 128, dtype=torch.uint8)
    empty_state = torch.empty(1, dtype=torch.bfloat16)
    empty_i32 = torch.empty(1, dtype=torch.int32)

    kernel._cake_prefill_output_buffer = MagicMock(return_value=output)
    kernel._cake_prefill_workspace = MagicMock(return_value=workspace)
    kernel._cake_prefill_dummy_buffers = MagicMock(
        return_value=(empty_state, empty_i32)
    )
    inputs = dict(
        q=q,
        k=k,
        v=v,
        alpha=alpha,
        beta=beta,
        state=state,
        state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        seq_lens_cpu=[64] * 5,
        layer_id=7,
        num_state_checkpoints=0,
        state_checkpoint_every_n_tokens=0,
    )
    return kernel, api, entry, inputs, output, workspace, empty_state, empty_i32


class TestCakeGDNPrefillDispatch(unittest.TestCase):
    def test_exact_bf16_indexed_row_uses_in_place_state_and_frozen_grid(self):
        kernel, api, entry, inputs, output, workspace, empty_state, empty_i32 = (
            _kernel_and_inputs()
        )

        result = kernel._try_cake_prefill(**inputs)

        self.assertEqual(tuple(result.shape), (1, 320, 8, 128))
        api.select_cake_gdn_prefill_variant.assert_called_once_with(
            arch="sm_100a",
            io_dtype="bfloat16",
            state_dtype="bfloat16",
            num_seqs=5,
            total_seq_len=320,
            max_seq_len=64,
            num_q_heads=4,
            num_k_heads=4,
            num_v_heads=8,
            use_initial_state=True,
            store_final_state=True,
            checkpoint_every_n_tokens=0,
            use_state_indices=True,
        )
        entry.assert_called_once()
        args = entry.call_args.args
        self.assertIs(args[3], output)
        self.assertIs(args[7], inputs["state_indices"])
        self.assertIs(args[8], inputs["state"])
        self.assertIs(args[9], inputs["state"])
        self.assertIs(args[10], empty_state)
        self.assertIs(args[11], empty_i32)
        self.assertIs(args[12], workspace)
        self.assertEqual(args[13], inputs["state"].stride(0))
        self.assertEqual(args[14], inputs["state"].stride(0))
        self.assertEqual(args[20:24], (40, 40, 1, 1))

    def test_public_auto_cp_route_is_not_intercepted(self):
        kernel, api, entry, inputs, *_ = _kernel_and_inputs()
        kernel._flashinfer_gdn_should_use_cp_host.return_value = True

        self.assertIsNone(kernel._try_cake_prefill(**inputs))
        api.select_cake_gdn_prefill_variant.assert_not_called()
        entry.assert_not_called()

    def test_checkpoint_and_missing_cpu_metadata_fail_closed(self):
        for override in (
            {"num_state_checkpoints": 1},
            {"state_checkpoint_every_n_tokens": 64},
            {"seq_lens_cpu": None},
        ):
            with self.subTest(override=override):
                kernel, api, entry, inputs, *_ = _kernel_and_inputs()
                inputs.update(override)

                self.assertIsNone(kernel._try_cake_prefill(**inputs))
                api.select_cake_gdn_prefill_variant.assert_not_called()
                entry.assert_not_called()

    def test_persistent_grid_matches_frozen_prefill_launcher(self):
        cases = (
            ("cake.gdn_prefill.noncp.dvsplit", 4, 8192, 8, 148, (64, 64)),
            ("cake.gdn_prefill.noncp.full_dv", 16, 512, 16, 160, (256, 128)),
            ("cake.gdn_prefill.noncp.full_dv", 20, 64, 8, 160, (160, 128)),
            ("cake.gdn_prefill.noncp.full_dv", 20, 4096, 8, 160, (160, 160)),
        )
        for route, num_seqs, max_seq_len, heads, clusters, expected in cases:
            with self.subTest(route=route, num_seqs=num_seqs):
                self.assertEqual(
                    FlashInferGDNKernel._cake_prefill_grid_x(
                        route_id=route,
                        num_seqs=num_seqs,
                        max_seq_len=max_seq_len,
                        num_v_heads=heads,
                        max_active_clusters=clusters,
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
