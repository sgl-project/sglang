"""UT for compact verify layout resolution (shared helper) and DSA graph keying.

Dormant until a model runs compact verify on DSA, so pin the gating logic:
mode gate, eager-vs-graph pad semantics, CP guard, token-keyed graph selection.
"""

import types
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    compute_target_verify_graph_key,
    resolve_compact_verify_layout,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")
GRID = [8, 16, 32]
NUM_DRAFT_TOKENS = 6


def _make_layout(verify_lens_cpu=(2, 3, 1)):
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=list(verify_lens_cpu), device=DEVICE, grid=GRID
    )


def _spec_info(layout):
    return types.SimpleNamespace(ragged_verify_layout=layout)


def _parallel(attn_cp_size=1):
    return mock.patch(
        "sglang.srt.runtime_context.get_parallel",
        return_value=types.SimpleNamespace(attn_cp_size=attn_cp_size),
    )


def _resolve(spec_info, padded_bs):
    return resolve_compact_verify_layout(
        spec_info, padded_bs=padded_bs, backend_name="DSA"
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestResolveCompactVerifyLayout(CustomTestCase):
    def test_no_layout_returns_none(self):
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"), _parallel():
            self.assertIsNone(_resolve(_spec_info(None), padded_bs=4))
            self.assertIsNone(_resolve(None, padded_bs=4))

    def test_non_compact_mode_returns_none(self):
        layout = _make_layout()
        for mode in ("static", "cap-accept"):
            with envs.SGLANG_RAGGED_VERIFY_MODE.override(mode), _parallel():
                self.assertIsNone(_resolve(_spec_info(layout), padded_bs=4))

    def test_eager_returns_raw_layout(self):
        layout = _make_layout()
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"), _parallel():
            self.assertIs(_resolve(_spec_info(layout), padded_bs=None), layout)

    def test_graph_pads_to_slots(self):
        layout = _make_layout((2, 3, 1))  # total 6 -> bucket 8
        padded_bs = 5
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"), _parallel():
            resolved = _resolve(_spec_info(layout), padded_bs=padded_bs)
        self.assertEqual(resolved.verify_lens.numel(), padded_bs)
        # Padding rows absorb the tier's slack: total == graph_num_tokens.
        self.assertEqual(int(resolved.verify_lens.sum().item()), 8)
        self.assertEqual(resolved.graph_num_tokens, 8)
        self.assertEqual(resolved.total_verify_tokens, 8)

    def test_cp_raises(self):
        layout = _make_layout()
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"), _parallel(2):
            with self.assertRaises(NotImplementedError):
                _resolve(_spec_info(layout), padded_bs=4)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDsaTargetVerifyGraphKey(CustomTestCase):
    def _backend(self):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        backend = types.SimpleNamespace(speculative_num_draft_tokens=NUM_DRAFT_TOKENS)
        backend._target_verify_graph_key = types.MethodType(
            DeepseekSparseAttnBackend._target_verify_graph_key, backend
        )
        return backend

    def test_no_layout_keys_by_bs(self):
        self.assertEqual(self._backend()._target_verify_graph_key(4, None), 4)

    def test_ragged_layout_keys_by_tokens(self):
        layout = _make_layout((2, 3, 1))  # bucket 8 <= 6*4
        key = self._backend()._target_verify_graph_key(4, layout)
        self.assertEqual(key, (8, 8))
        self.assertEqual(
            key,
            compute_target_verify_graph_key(
                bs=4, num_draft_tokens=NUM_DRAFT_TOKENS, ragged_layout=layout
            ),
        )


if __name__ == "__main__":
    unittest.main()
