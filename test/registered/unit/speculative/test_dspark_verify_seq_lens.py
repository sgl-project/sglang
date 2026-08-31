from types import SimpleNamespace

from sglang.srt.speculative.dspark_components.dspark_verify import (
    TargetVerifyExecutor,
)


def _executor(backend):
    executor = TargetVerifyExecutor.__new__(TargetVerifyExecutor)
    executor.target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=backend)
    )
    executor._verify_backend_self_adds_seq_lens_cache = None
    return executor


def test_explicit_backend_capability_avoids_double_counting_verify_width():
    backend = SimpleNamespace(target_verify_self_adds_seq_lens=True)
    assert _executor(backend)._verify_backend_self_adds_seq_lens()


def test_ordinary_backend_keeps_preextended_lengths():
    assert not _executor(SimpleNamespace())._verify_backend_self_adds_seq_lens()


def test_raw_verify_backend_also_owns_length_adjustment():
    backend = SimpleNamespace(make_forward_metadata_from_raw_verify=lambda: None)
    assert _executor(backend)._verify_backend_self_adds_seq_lens()
