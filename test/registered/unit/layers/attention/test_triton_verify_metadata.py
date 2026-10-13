import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.triton_backend import (
    ForwardMetadata,
    TritonAttnBackend,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _BonusTokenVerifyInput(SpecInput):
    def __init__(self):
        super().__init__(SpecInputType.EAGLE_VERIFY)
        self.draft_token_num = 6
        self.num_tokens_per_req = 7


class _KVIndexTranslator:
    is_translating = False

    def fill_packed_read_stream(
        self,
        *,
        req_pool_indices,
        seq_lens,
        indptr,
        total_tokens,
        out,
    ):
        out.zero_()


class _RecordingTritonBackend(TritonAttnBackend):
    def build_unified_kv_indices(
        self,
        _prefix_kv_indptr,
        _prefix_kv_indices,
        extend_start_loc,
        extend_seq_lens,
        _extend_kv_indices,
        batch_size,
    ):
        self.recorded_extend_start_loc = extend_start_loc.clone()
        self.recorded_extend_seq_lens = extend_seq_lens.clone()
        return (
            torch.zeros(batch_size + 1, dtype=torch.int32),
            torch.zeros(batch_size * 7, dtype=torch.int64),
            torch.zeros(batch_size, dtype=torch.int32),
        )

    def extend_attention_fwd_unified(self, *_args, **_kwargs):
        self.recorded_window_start_pos = _kwargs["window_start_pos"].clone()
        return None


def _make_backend(batch_size, capture_width=7):
    backend = TritonAttnBackend.__new__(TritonAttnBackend)
    backend.device = torch.device("cpu")
    backend.num_draft_tokens = 9
    backend.target_verify_num_tokens_per_req = capture_width
    backend.max_context_len = 128
    backend.qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    backend.kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    backend.window_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    backend.mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int64)
    backend.cuda_graph_kv_indices = torch.zeros(256, dtype=torch.int64)
    backend.kv_index_translator = _KVIndexTranslator()
    backend.sliding_window_size = None
    backend.use_sliding_window_kv_pool = False
    backend._verify_mask = None
    return backend


def _make_forward_batch(batch_size, spec_info):
    seq_lens = torch.arange(32, 32 + batch_size, dtype=torch.int64)
    return SimpleNamespace(
        batch_size=batch_size,
        input_ids=torch.zeros(batch_size * 7, dtype=torch.int64),
        req_pool_indices=torch.arange(batch_size, dtype=torch.int64),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        seq_lens_sum=int(seq_lens.sum()),
        encoder_lens=None,
        spec_info=spec_info,
        forward_mode=ForwardMode.TARGET_VERIFY,
        out_cache_loc=torch.zeros(batch_size * 7, dtype=torch.int64),
    )


def _assert_verify_width(backend, batch_size):
    assert torch.equal(
        backend.forward_metadata.qo_indptr,
        torch.arange(0, (batch_size + 1) * 7, 7, dtype=torch.int32),
    )
    assert backend.forward_metadata.max_extend_len == 7


def test_eager_target_verify_uses_bonus_token_width():
    batch_size = 2
    backend = _make_backend(batch_size, capture_width=9)
    spec_info = _BonusTokenVerifyInput()

    backend.init_forward_metadata(_make_forward_batch(batch_size, spec_info))

    assert spec_info.draft_token_num == 6
    assert spec_info.num_tokens_per_req == 7
    _assert_verify_width(backend, batch_size)
    assert torch.equal(
        backend.forward_metadata.mask_indptr,
        torch.tensor([0, 273, 553], dtype=torch.int64),
    )


@pytest.mark.parametrize(
    ("batch_size", "raw_batch_size"),
    ((2, 2), (4, 3)),
)
@pytest.mark.parametrize("unset_width", (None, -1, 0))
def test_graph_capture_and_padded_replay_use_bonus_token_width(
    batch_size, raw_batch_size, unset_width
):
    backend = _make_backend(batch_size)
    capture_spec = None
    if unset_width is not None:
        capture_spec = SpecInput(SpecInputType.EAGLE_VERIFY)
        capture_spec.num_tokens_per_req = unset_width
    capture_batch = _make_forward_batch(batch_size, capture_spec)

    backend.init_forward_metadata_out_graph(
        capture_batch,
        in_capture=True,
    )
    _assert_verify_width(backend, batch_size)

    spec_info = _BonusTokenVerifyInput()
    replay_batch = _make_forward_batch(batch_size, spec_info)
    replay_batch.seq_lens[raw_batch_size:] = 1

    backend.init_forward_metadata_out_graph(
        replay_batch,
        in_capture=False,
    )

    _assert_verify_width(backend, batch_size)


@pytest.mark.parametrize("with_spec_info", (False, True))
def test_unified_target_verify_uses_resolved_metadata(with_spec_info):
    batch_size = 2
    backend = _RecordingTritonBackend.__new__(_RecordingTritonBackend)
    backend.device = torch.device("cpu")
    backend.dcp_size = 1
    backend.enable_deterministic = True
    backend.use_dense_fp8_chunked_prefill = False
    backend.allow_bidirectional_attention_in_extend = False
    backend.page_size = 1
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda _layer_id: torch.zeros((1, 1, 4)),
        get_value_buffer=lambda _layer_id: torch.zeros((1, 1, 4)),
    )
    backend.forward_metadata = ForwardMetadata(
        attn_logits=None,
        attn_lse=None,
        max_extend_len=7,
        num_kv_splits=None,
        kv_indptr=torch.zeros(batch_size + 1, dtype=torch.int32),
        kv_indices=torch.zeros(1, dtype=torch.int64),
        qo_indptr=torch.tensor([0, 7, 14], dtype=torch.int32),
        custom_mask=None,
        mask_indptr=None,
        window_kv_indptr=torch.tensor([0, 8, 20], dtype=torch.int32),
        window_kv_indices=torch.zeros(20, dtype=torch.int64),
        window_num_kv_splits=None,
        window_kv_offsets=None,
    )
    layer = SimpleNamespace(
        layer_id=0,
        qk_head_dim=4,
        v_head_dim=4,
        tp_q_head_num=1,
        k_scale=None,
        v_scale=None,
        logit_capping_method="tanh",
        logit_cap=0.0,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        sliding_window_size=16,
        scaling=0.5,
        xai_temperature_len=None,
    )
    forward_batch = SimpleNamespace(
        batch_size=batch_size,
        forward_mode=ForwardMode.TARGET_VERIFY,
        mha_one_shot=False,
        out_cache_loc=torch.zeros(batch_size * 7, dtype=torch.int64),
        extend_seq_lens=None,
        extend_start_loc=None,
        extend_prefix_lens=None,
        seq_lens=torch.tensor([32, 64], dtype=torch.int64),
        spec_info=_BonusTokenVerifyInput() if with_spec_info else None,
    )
    q = torch.zeros((batch_size * 7, 4))

    output = backend.forward_extend(
        q,
        q,
        q,
        layer,
        forward_batch,
        save_kv_cache=False,
    )

    assert output.shape == q.shape
    assert torch.equal(
        backend.recorded_extend_seq_lens,
        torch.tensor([7, 7], dtype=torch.int32),
    )
    assert torch.equal(
        backend.recorded_extend_start_loc,
        torch.tensor([0, 7], dtype=torch.int32),
    )
    assert torch.equal(
        backend.recorded_window_start_pos,
        torch.tensor([24, 52], dtype=torch.int64),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
