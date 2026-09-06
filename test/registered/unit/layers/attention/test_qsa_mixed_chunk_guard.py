"""Guard for the compressed-QSA mixed-chunk crash (Bug B).

``--enable-mixed-chunk`` produces ``ForwardMode.MIXED`` batches in which the
scheduler injects running decode rows into an EXTEND batch, each with
``extend_len == 1`` and therefore ``prefix_len == seq_len - 1``. That prefix is
almost never a multiple of ``indexer_compress_ratio``, so it violates the
ratio-aligned prefix invariant the compressed write plan relies on and the
``torch._assert_async((prefix_lens % ratio == 0).all())`` in
``_qsa_build_write_plan`` fires device-side (kills every TP rank).

The fix is a fail-fast ``NotImplementedError`` guard in
``QwenSparseAttnBackend._metadata_from_forward_batch`` so a MIXED batch never
reaches the write plan. This is a pure-CPU regression test: it drives the
metadata entry with a MIXED forward batch and asserts the guard fires with a
disable-MIXED-chunk directive.
"""

import unittest
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_backend() -> QwenSparseAttnBackend:
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    # _metadata_from_forward_batch reads device from seq_lens.device and
    # max_context_len from req_to_token.shape[1] when unset.
    backend.device = None
    backend.max_context_len = None
    backend.req_to_token = torch.zeros((1, 64), dtype=torch.int64)
    backend.token_to_kv_pool = SimpleNamespace(qsa_compress_ratio=4)
    return backend


def _mixed_batch(n_rows: int):
    # MIXED batch: 1 genuine extend row + decode rows injected by
    # mix_with_running(). is_mixed() must be True.
    return SimpleNamespace(
        forward_mode=ForwardMode.MIXED,
        seq_lens=torch.ones(n_rows, dtype=torch.int32),
        spec_info=None,
    )


class TestQsaMixedChunkGuard(CustomTestCase):
    def test_mixed_mode_raises_not_implemented(self):
        """A ForwardMode.MIXED batch must fail fast with a directive, not reach
        the ratio assert in _qsa_build_write_plan."""
        backend = _make_backend()
        fb = _mixed_batch(2)
        with pytest.raises(NotImplementedError) as exc:
            backend._metadata_from_forward_batch(fb)
        msg = str(exc.value)
        self.assertIn("Disable --enable-mixed-chunk", msg)
        self.assertIn("MIXED", msg)

    def test_mixed_guard_fires_before_empty_metadata(self):
        """The guard is a hard gate even on an otherwise empty-ish batch
        (seq_lens non-empty here); it must fire regardless of the extend/decode
        content of the mixed rows."""
        backend = _make_backend()
        fb = _mixed_batch(1)
        with self.assertRaises(NotImplementedError):
            backend._metadata_from_forward_batch(fb)

    def test_decode_mode_is_not_guarded(self):
        """Sanity: a plain DECODE batch (the common path) is not rejected by the
        mixed guard. We only assert the guard itself is skipped here, since the
        full DECODE metadata path needs more pool fixtures."""
        backend = _make_backend()
        fb = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens=torch.zeros(0, dtype=torch.int32),
            spec_info=None,
        )
        # Empty decode batch takes the _empty_metadata path; reaching it means
        # the mixed guard did not fire.
        try:
            backend._metadata_from_forward_batch(fb)
        except (NotImplementedError, ValueError) as e:
            # Allowed: some later path may raise; what must NOT happen is the
            # MIXED guard firing with its MIXED message.
            self.assertNotIn("MIXED", str(e))
        except Exception:
            # Reaching any non-MIXED outcome is fine for this sanity check.
            pass


if __name__ == "__main__":
    unittest.main()
