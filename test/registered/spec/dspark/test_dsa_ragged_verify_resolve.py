"""UT for the DSA backend's ragged-verify layout resolution and graph keying.

These paths are dormant until a model runs compact verify on DSA, so pin the
gating logic directly: mode gate, eager-vs-graph pad semantics, the CP guard,
and the token-keyed graph selection.
"""

import types
import unittest
from unittest import mock

import torch

from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    compute_target_verify_graph_key,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")
GRID = [8, 16, 32]
NUM_DRAFT_TOKENS = 6


def _make_backend(num_draft_tokens=NUM_DRAFT_TOKENS):
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    backend = types.SimpleNamespace(speculative_num_draft_tokens=num_draft_tokens)
    for name in ("_resolve_verify_layout", "_target_verify_graph_key"):
        setattr(
            backend,
            name,
            types.MethodType(getattr(DeepseekSparseAttnBackend, name), backend),
        )
    return backend


def _make_layout(verify_lens_cpu=(2, 3, 1)):
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=list(verify_lens_cpu), device=DEVICE, grid=GRID
    )


def _spec_info(layout):
    return types.SimpleNamespace(ragged_verify_layout=layout)


def _parallel(attn_cp_size=1):
    return mock.patch(
        "sglang.srt.layers.attention.dsa_backend.get_parallel",
        return_value=types.SimpleNamespace(attn_cp_size=attn_cp_size),
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDsaResolveVerifyLayout(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.srt.environ import envs

        self.envs = envs
        self.backend = _make_backend()

    def test_no_layout_returns_none(self):
        with self.envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"), _parallel():
            self.assertIsNone(
                self.backend._resolve_verify_layout(_spec_info(None), bs=4)
            )

    def test_non_compact_mode_returns_none(self):
        layout = _make_layout()
        for mode in ("static", "cap-accept"):
            with self.envs.SGLANG_RAGGED_VERIFY_MODE.override(mode), _parallel():
                self.assertIsNone(
                    self.backend._resolve_verify_layout(_spec_info(layout), bs=4)
                )

    def test_eager_returns_raw_layout(self):
        layout = _make_layout()
        with self.envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"), _parallel():
            resolved = self.backend._resolve_verify_layout(
                _spec_info(layout), bs=4, pad_to_slots=False
            )
        self.assertIs(resolved, layout)

    def test_graph_pads_to_slots(self):
        layout = _make_layout((2, 3, 1))  # total 6 -> bucket 8
        padded_bs = 5
        with self.envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"), _parallel():
            resolved = self.backend._resolve_verify_layout(
                _spec_info(layout), bs=padded_bs
            )
        self.assertEqual(resolved.verify_lens.numel(), padded_bs)
        # Padding rows absorb the tier's slack: total == graph_num_tokens.
        self.assertEqual(int(resolved.verify_lens.sum().item()), 8)
        self.assertEqual(resolved.graph_num_tokens, 8)
        self.assertEqual(resolved.total_verify_tokens, 8)

    def test_cp_raises(self):
        layout = _make_layout()
        with self.envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"), _parallel(2):
            with self.assertRaises(NotImplementedError):
                self.backend._resolve_verify_layout(_spec_info(layout), bs=4)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDsaTargetVerifyGraphKey(CustomTestCase):
    def test_no_layout_keys_by_bs(self):
        backend = _make_backend()
        self.assertEqual(backend._target_verify_graph_key(4, None), 4)

    def test_ragged_layout_keys_by_tokens(self):
        backend = _make_backend()
        layout = _make_layout((2, 3, 1))  # bucket 8 <= 6*4
        key = backend._target_verify_graph_key(4, layout)
        self.assertEqual(key, (8, 8))
        self.assertEqual(
            key,
            compute_target_verify_graph_key(
                bs=4, num_draft_tokens=NUM_DRAFT_TOKENS, ragged_layout=layout
            ),
        )


if __name__ == "__main__":
    unittest.main()
