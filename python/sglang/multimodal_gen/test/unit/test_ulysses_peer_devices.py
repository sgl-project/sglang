# SPDX-License-Identifier: Apache-2.0
"""Which workers enable peer access at start: one Ulysses group over the node's GPUs."""

from types import SimpleNamespace

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.runtime.launch_server import _ulysses_peer_devices
from sglang.multimodal_gen.runtime.managers import worker_bootstrap
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def _args(over_cuda_ipc: bool = True, **overrides) -> SimpleNamespace:
    args = dict(
        pipeline_config=SimpleNamespace(
            ulysses_exchange_over_cuda_ipc=lambda server_args: over_cuda_ipc
        ),
        nnodes=1,
        ulysses_degree=4,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_a_node_wide_ulysses_group_gets_every_other_gpu():
    assert _ulysses_peer_devices(_args(), 4) == {
        0: (1, 2, 3),
        1: (0, 2, 3),
        2: (0, 1, 3),
        3: (0, 1, 2),
    }


def test_other_layouts_get_none():
    assert _ulysses_peer_devices(_args(ulysses_degree=2), 4) == {}
    assert _ulysses_peer_devices(_args(nnodes=2), 4) == {}
    assert _ulysses_peer_devices(_args(ulysses_degree=1), 1) == {}
    assert _ulysses_peer_devices(_args(over_cuda_ipc=False), 4) == {}


def test_minimax_h3_declares_it_for_pipelined_dense_attention(monkeypatch):
    config = MiniMaxH3PipelineConfig()

    def declared(backend) -> bool:
        monkeypatch.setattr(
            MiniMaxH3PipelineConfig,
            "resolve_transformer_attention_backend",
            lambda self, server_args: backend,
        )
        return config.ulysses_exchange_over_cuda_ipc(None)

    assert declared(None) and declared(AttentionBackendEnum.FA)
    assert not declared(AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3)
    monkeypatch.setenv("SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS", "0")
    assert not declared(None)


def test_no_peers_is_a_no_op():
    # returns before importing anything from the runtime
    assert worker_bootstrap._enable_peer_access_early(0, ()) is None
