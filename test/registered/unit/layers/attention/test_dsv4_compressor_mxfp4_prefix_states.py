"""MXFP4 compressor store must keep writing online-C128 prefix states.

Regression: ``CompressorBackendMixin.forward_unified`` returned early after
``_forward_mxfp4``, so the ``online_c128_mtp.write_prefix_states`` call that
closes every other store path never ran for MXFP4 caches.  With
``SGLANG_OPT_USE_ONLINE_COMPRESS=1`` +
``SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1`` + speculative decoding, the
target-verify step silently skipped its C128 prefix-state write and the next
commit continued from stale state.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Mode:
    def is_idle(self) -> bool:
        return False

    def is_target_verify(self) -> bool:
        return True


class _RecordingBackend:
    """Minimal host for the mixin: only the MXFP4 branch's collaborators."""

    def __init__(self) -> None:
        self.token_to_kv_pool = SimpleNamespace(dsv4_kv_cache_store_mxfp4=True)
        self.mxfp4_store_calls: list[int] = []
        self.prefix_state_calls: list[tuple[int, object]] = []
        self.online_c128_mtp = SimpleNamespace(
            write_prefix_states=self._write_prefix_states
        )

    def _forward_mxfp4(self, *, layer_id, **_) -> None:
        self.mxfp4_store_calls.append(layer_id)

    def _write_prefix_states(
        self, *, layer_id, compressor, kv_score_input, logical_forward_mode
    ) -> None:
        self.prefix_state_calls.append((layer_id, logical_forward_mode))


@pytest.fixture()
def mixin_backend():
    from sglang.srt.layers.attention.dsv4.compressor_v2 import (
        CompressorBackendMixin,
    )

    class _Backend(_RecordingBackend, CompressorBackendMixin):
        pass

    return _Backend()


def test_mxfp4_store_writes_online_c128_prefix_states(mixin_backend) -> None:
    """The MXFP4 early-return branch must still route through the shared
    prefix-state write (guards like enabled()/target-verify live inside the
    controller; the store path only guarantees the call happens)."""
    verify_mode = _Mode()
    forward_batch = SimpleNamespace(
        forward_mode=_Mode(), _original_forward_mode=verify_mode
    )
    compressor = SimpleNamespace(
        is_in_indexer=False,
        ratio=128,
        compute_kv_score=lambda x, fb: torch.empty(8),
        get_state_pool=lambda backend: object(),
    )

    mixin_backend.forward_unified(
        torch.empty(4), forward_batch, layer_id=3, compressor=compressor
    )

    assert mixin_backend.mxfp4_store_calls == [3]
    assert mixin_backend.prefix_state_calls == [(3, verify_mode)]


def test_mxfp4_store_prefix_states_use_logical_mode_fallback(mixin_backend) -> None:
    """Without an _original_forward_mode the write falls back to the batch's
    own mode, matching the non-MXFP4 tail's semantics."""
    decode_mode = _Mode()
    forward_batch = SimpleNamespace(forward_mode=decode_mode)
    compressor = SimpleNamespace(
        is_in_indexer=False,
        ratio=128,
        compute_kv_score=lambda x, fb: torch.empty(8),
        get_state_pool=lambda backend: object(),
    )

    mixin_backend.forward_unified(
        torch.empty(4), forward_batch, layer_id=7, compressor=compressor
    )

    assert mixin_backend.prefix_state_calls == [(7, decode_mode)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
