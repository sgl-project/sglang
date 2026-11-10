from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    _sm70_dense_prefill_max_tokens,
    use_sm70_qsa_dense_prefill,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _FakeTensor:
    def __init__(self, shape, dtype, *, is_cuda=True):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.is_cuda = is_cuda
        self.device = torch.device("cuda:0")

    @property
    def ndim(self):
        return len(self.shape)


@pytest.mark.parametrize(
    "forward_mode",
    [
        ForwardMode.DECODE,
        ForwardMode.TARGET_VERIFY,
        ForwardMode.DRAFT_EXTEND_V2,
    ],
)
def test_sm70_qsa_decode_accepts_paged_decode_modes(monkeypatch, forward_mode):
    rows = 4
    q = _FakeTensor((rows, 6, 256), torch.float16)
    k = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    v = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    topk_indices = _FakeTensor((rows, 2048), torch.int32)
    metadata = SimpleNamespace(
        sequence_lengths=SimpleNamespace(numel=lambda: rows),
    )
    forward_batch = SimpleNamespace(forward_mode=forward_mode)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))

    assert QwenSparseAttnBackend._can_use_sm70_sparse_decode(
        q,
        k,
        v,
        forward_batch,
        metadata,
        topk_indices,
    )


def test_sm70_qsa_decode_rejects_ordinary_extend(monkeypatch):
    rows = 4
    q = _FakeTensor((rows, 6, 256), torch.float16)
    k = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    v = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    topk_indices = _FakeTensor((rows, 2048), torch.int32)
    metadata = SimpleNamespace(
        sequence_lengths=SimpleNamespace(numel=lambda: rows),
    )
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))

    assert not QwenSparseAttnBackend._can_use_sm70_sparse_decode(
        q,
        k,
        v,
        forward_batch,
        metadata,
        topk_indices,
    )


def test_sm70_qsa_dense_prefill_uses_packaged_kernel_per_sequence(monkeypatch):
    from sglang.srt.layers.attention.tilelang_fa_v100 import _kernels_dense_d256

    calls = []

    def fake_get_kernel(heads, heads_kv):
        assert (heads, heads_kv) == (6, 1)

        def fake_kernel(q, k, v, prefix_len, softmax_scale):
            calls.append((q.shape[0], k.shape[0], v.shape[0], prefix_len))
            assert softmax_scale == pytest.approx(0.0625)
            return q + len(calls)

        return fake_kernel

    monkeypatch.setattr(
        _kernels_dense_d256, "get_dense_prefix_d256_kernel", fake_get_kernel
    )
    q = torch.zeros((5, 6, 256), dtype=torch.float16)
    k = torch.zeros((5, 1, 256), dtype=torch.float16)
    v = torch.zeros_like(k)

    output = QwenSparseAttnBackend._forward_sm70_dense_prefill(
        q, k, v, [2, 3], 0.0625
    )

    assert calls == [(2, 2, 2, 0), (3, 3, 3, 0)]
    torch.testing.assert_close(output[:2], torch.ones_like(output[:2]))
    torch.testing.assert_close(output[2:], torch.full_like(output[2:], 2))


def test_sm70_qsa_dense_prefill_rejects_mismatched_packing():
    q = torch.zeros((5, 6, 256), dtype=torch.float16)
    k = torch.zeros((5, 1, 256), dtype=torch.float16)
    v = torch.zeros_like(k)

    with pytest.raises(ValueError, match="packed rows"):
        QwenSparseAttnBackend._forward_sm70_dense_prefill(
            q, k, v, [2, 2], 0.0625
        )


@pytest.mark.parametrize(
    "lengths,extend_lengths,mode,expected",
    [
        ([1000], [1000], ForwardMode.EXTEND, True),
        ([8192, 1000], [8192, 1000], ForwardMode.EXTEND, True),
        ([8193], [8193], ForwardMode.EXTEND, False),
        ([8192], [4096], ForwardMode.EXTEND, False),
        ([1000], [1], ForwardMode.DECODE, False),
        ([4], [4], ForwardMode.TARGET_VERIFY, False),
        ([4], [4], ForwardMode.DRAFT_EXTEND_V2, False),
    ],
)
def test_sm70_qsa_dense_selection_matches_attention_route(
    monkeypatch, lengths, extend_lengths, mode, expected
):
    monkeypatch.setenv("SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS", "8192")
    _sm70_dense_prefill_max_tokens.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))
    batch = SimpleNamespace(
        seq_lens_cpu=lengths, extend_seq_lens_cpu=extend_lengths, forward_mode=mode
    )
    assert use_sm70_qsa_dense_prefill(batch, torch.device("cuda")) is expected
    batch._original_forward_mode = ForwardMode.TARGET_VERIFY
    assert not use_sm70_qsa_dense_prefill(batch, torch.device("cuda"))
    _sm70_dense_prefill_max_tokens.cache_clear()


def test_sm70_qsa_dense_selection_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS", "0")
    _sm70_dense_prefill_max_tokens.cache_clear()
    batch = SimpleNamespace(
        seq_lens_cpu=[1000], extend_seq_lens_cpu=[1000], forward_mode=ForwardMode.EXTEND
    )
    assert not use_sm70_qsa_dense_prefill(batch, torch.device("cuda"))
    _sm70_dense_prefill_max_tokens.cache_clear()


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_dense_prefill_retains_mtp_seed_indices(monkeypatch, capture, compressed):
    from sglang.srt.layers.attention import qwen_sparse_attn_backend
    from sglang.srt.layers.attention.qsa import glue
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
    from sglang.srt.models import qwen4_exp

    calls = []

    class RecordingIndexer(QSAIndexer if compressed else torch.nn.Module):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def forward(self, hidden, positions, batch, metadata, **kwargs):
            skip = kwargs.get("skip_prefill_selection", False)
            calls.append(skip)
            return torch.zeros((3, 0 if skip else 2051), dtype=torch.int32)

    captured = []
    backend = SimpleNamespace(
        should_capture_mtp_sparse_indices=lambda batch: capture,
        capture_mtp_sparse_indices=lambda indices, *args, **kwargs: captured.append(
            indices
        ),
    )
    monkeypatch.setattr(qwen4_exp, "get_attn_backend", lambda: backend)
    monkeypatch.setattr(glue, "resolve_qsa_sparse_backend", lambda backend: backend)
    monkeypatch.setattr(glue, "get_qsa_indexer_metadata", lambda *args: object())
    monkeypatch.setattr(
        qwen_sparse_attn_backend, "use_sm70_qsa_dense_prefill", lambda *args: True
    )
    layer = SimpleNamespace(indexer=RecordingIndexer(), layer_id=2)
    output = qwen4_exp.Qwen4ExpAttentionDecoderLayer._compute_qsa_topk_indices(
        layer, torch.zeros((3, 2560)), torch.arange(3), SimpleNamespace()
    )
    assert calls == [compressed and not capture]
    if capture:
        assert captured == [output]
        assert output.shape == (3, 2051)
    else:
        assert not captured


def test_dense_prefill_selection_skip_still_updates_keys():
    from unittest.mock import Mock

    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    positions = torch.arange(5)
    hidden = torch.zeros((5, 2560))
    token_k = torch.randn((3, 1, 128))
    slots = torch.arange(3)
    metadata = SimpleNamespace(
        get_token_to_batch_idx=lambda: torch.zeros(3, dtype=torch.int32),
        pending_ring_slots=slots,
        token_to_kv_pool=object(),
    )
    indexer = SimpleNamespace(
        index_n_heads=4,
        project_qk=Mock(return_value=(torch.zeros((3, 4, 128)), token_k, True)),
        update_key_state_and_compress=Mock(),
    )
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, positions=positions)
    output = QSAIndexer._forward_impl(
        indexer, hidden, positions, batch, metadata, skip_prefill_selection=True
    )
    assert output.shape == (3, 0)  # Physical DP padding does not create semantic rows.
    assert output.dtype == torch.int32
    projected = indexer.project_qk.call_args.args
    assert projected[0].shape == (3, 2560)
    args = indexer.update_key_state_and_compress.call_args
    assert args.args[0] is token_k
    torch.testing.assert_close(args.args[1], positions[:3])
    assert args.kwargs == {"state_slots": slots, "state_stored": True}


@pytest.mark.parametrize(
    "mode", [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2]
)
def test_selection_skip_rejects_decode_and_speculation(mode):
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    with pytest.raises(ValueError, match="ordinary prefill"):
        QSAIndexer._forward_impl(
            object(),
            None,
            None,
            SimpleNamespace(forward_mode=mode),
            None,
            skip_prefill_selection=True,
        )
