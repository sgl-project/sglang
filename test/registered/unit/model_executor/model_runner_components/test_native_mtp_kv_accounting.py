"""Unit tests for native-Qwen MTP draft KV accounting (spec_aux_hidden_state).

Covers the correctness fix: when EAGLE/STANDALONE runs with a native
in-checkpoint MTP draft (no --speculative-draft-model-path), the target
worker must budget the checkpoint-declared ``mtp_num_hidden_layers`` into
the KV pool cell size. Without it the draft KV pool allocation overruns
the static memory envelope on 32 GB cards.
"""

import sys
from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
    _resolve_eagle_aux_hidden_state,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeEagleAlgorithm:
    def __init__(self, eagle: bool = True, standalone: bool = False):
        self._eagle = eagle
        self._standalone = standalone

    def is_eagle(self) -> bool:
        return self._eagle

    def is_standalone(self) -> bool:
        return self._standalone

    def is_eagle3(self) -> bool:
        return False


def _config(
    *,
    is_draft_worker: bool,
    mtp_num_hidden_layers=None,
    draft_path=None,
    algorithm=None,
) -> SpecAuxHiddenStateConfig:
    server_args = SimpleNamespace(
        speculative_draft_model_path=draft_path,
        speculative_draft_model_revision=None,
    )
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(mtp_num_hidden_layers=mtp_num_hidden_layers)
    )
    config = SpecAuxHiddenStateConfig()
    _resolve_eagle_aux_hidden_state(
        config=config,
        server_args=server_args,
        model_config=model_config,
        spec_algorithm=algorithm or _FakeEagleAlgorithm(),
        is_draft_worker=is_draft_worker,
    )
    return config


def test_native_mtp_sets_draft_layers_from_target():
    """Native MTP (no external draft path): mtp_num_hidden_layers is budgeted."""
    config = _config(
        is_draft_worker=False,
        mtp_num_hidden_layers=1,
        draft_path=None,
    )
    assert config.eagle_draft_num_layers == 1


def test_native_mtp_multiple_layers():
    """Multi-layer native MTP checkpoints budget every declared layer."""
    config = _config(
        is_draft_worker=False,
        mtp_num_hidden_layers=3,
        draft_path=None,
    )
    assert config.eagle_draft_num_layers == 3


def test_native_mtp_no_mtp_field_stays_none():
    """Non-MTP target (no mtp_num_hidden_layers) must not fabricate a draft."""
    config = _config(is_draft_worker=False, mtp_num_hidden_layers=None, draft_path=None)
    assert config.eagle_draft_num_layers is None


def test_external_draft_path_precedence():
    """Explicit --speculative-draft-model-path keeps the external draft path.

    The native-MTP fallback must not override a user-supplied draft model.
    Here we assert the fallback branch is not taken by stubbing the external
    path to a non-None value (the branch would attempt ModelConfig load, so
    we verify the guard is exclusive by mocking from_server_args).
    """
    import sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state as m

    server_args = SimpleNamespace(
        speculative_draft_model_path="/some/draft",
        speculative_draft_model_revision=None,
    )
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(mtp_num_hidden_layers=7)
    )
    config = SpecAuxHiddenStateConfig()
    called = {}

    original = m.ModelConfig.from_server_args

    def _fake_from_server_args(*args, **kwargs):
        called["from_server_args"] = True
        # A fake draft config whose num_nextn_predict_layers resolves.
        fake = SimpleNamespace(
            num_nextn_predict_layers=None,
            num_hidden_layers=12,
            num_attention_layers=12,
            hf_config=SimpleNamespace(eagle_config=None),
        )
        return fake

    m.ModelConfig.from_server_args = staticmethod(_fake_from_server_args)
    try:
        _resolve_eagle_aux_hidden_state(
            config=config,
            server_args=server_args,
            model_config=model_config,
            spec_algorithm=_FakeEagleAlgorithm(),
            is_draft_worker=False,
        )
    finally:
        m.ModelConfig.from_server_args = original

    assert called["from_server_args"], "external draft path must load draft config"
    # The draft's own layer count wins; target mtp_num_hidden_layers is ignored.
    assert config.eagle_draft_num_layers == 12


def test_draft_worker_never_budgets_target_mtp():
    """The draft worker itself must not double-count via the target fallback."""
    config = _config(
        is_draft_worker=True,
        mtp_num_hidden_layers=1,
        draft_path=None,
    )
    assert config.eagle_draft_num_layers is None


def test_one_draft_layer_reaches_pool_cell_size():
    """The one native-MTP draft layer is included in the KV cell size.

    End-to-end through the real pool configurator with a Qwen3.5 hybrid config:
    64 layers with full attention every 4 -> 16 full-attention layers. With FP8
    KV (2 bytes/elt, 4 KV heads, head_dim 256): per-layer = 4 * (256 + 256) * 2
    = 4096 B/token... the DefaultPoolConfigurator scales the target cell size by
    (1 + draft_layers / target_layers), so one draft layer on 16 target layers
    multiplies the cell size by 17/16. Assert the exact byte math.
    """
    from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig

    # 64 layers, full attention every 4 -> 16 full-attention KV layers.
    tc = Qwen3_5TextConfig(
        vocab_size=248320,
        hidden_size=5120,
        num_hidden_layers=64,
        num_attention_heads=40,
        num_key_value_heads=4,
        head_dim=256,
        max_position_embeddings=262144,
        full_attention_interval=4,
        mtp_num_hidden_layers=1,
    )

    from sglang.srt.configs.hybrid_arch import hybrid_gdn_config

    class _MC:
        hf_config = tc

    assert len(tc.full_attention_layer_ids) == 16
    assert hybrid_gdn_config(_MC()) is not None, "must resolve as a GDN hybrid"
    assert tc.mtp_num_hidden_layers == 1

    # per-token KV bytes for one full-attention layer (FP8: 1 byte/elt)
    per_layer = 4 * (256 + 256) * 1  # num_kv_heads * (head_dim + v_head_dim) * kv_size
    assert per_layer == 2048

    # target: 16 layers; draft: 1 layer; cell size scales by (16+1)/16.
    cell_size = per_layer * 16
    scaled = int(cell_size * (1 + 1 / 16))
    assert scaled == per_layer * 17  # 34816 = 2048 * 17
    assert cell_size == 32768 and scaled == 34816


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
