from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.beam_search.batch_tail import append_beam_tail
from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.shared_context_attention import build_metadata
from sglang.srt.beam_search.trie_output_head import build_trie_row_levels
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

BEAM = 8
PROMPT_LEN = 5
NUM_GROUPS = 2
DECODE_LEN = 2


class _FakeRunner(SimpleNamespace):
    """Weak-referenceable: the shared-context support verdict is cached per runner."""

    __hash__ = object.__hash__


def _decode_batch():
    """A decode batch of NUM_GROUPS beam leaders, before its member tail."""
    reqs = []
    for g in range(NUM_GROUPS):
        group = BeamGroup(
            beam_width=BEAM, max_new_tokens=3, device="cuda", device_history=True
        )
        group.prompt_len = PROMPT_LEN
        group.num_generated = DECODE_LEN
        rows = list(
            range(NUM_GROUPS + g * (BEAM - 1), NUM_GROUPS + (g + 1) * (BEAM - 1))
        )
        group.member_rows_cpu = torch.tensor(rows, dtype=torch.int64)
        group.member_rows = group.member_rows_cpu.cuda()
        reqs.append(SimpleNamespace(beam_group=group))
    seq_lens = [PROMPT_LEN + DECODE_LEN] * NUM_GROUPS
    return SimpleNamespace(
        reqs=reqs,
        forward_mode=ForwardMode.DECODE,
        device=torch.device("cuda"),
        beam_tail=None,
        req_pool_indices=torch.arange(NUM_GROUPS, device="cuda"),
        req_pool_indices_cpu=torch.arange(NUM_GROUPS),
        seq_lens=torch.tensor(seq_lens, device="cuda"),
        seq_lens_cpu=torch.tensor(seq_lens),
        orig_seq_lens=torch.tensor(seq_lens, device="cuda"),
        seq_lens_sum=None,
    )


def _prepare(batch, model_runner):
    append_beam_tail(batch)
    levels = build_trie_row_levels(batch, model_runner)
    metadata = build_metadata(batch, model_runner)
    return levels, metadata


def test_beam_decode_batch_prep_never_syncs_the_host():
    """Overlap: building a beam decode batch (member tail, per-row trie levels,
    shared-context metadata) issues no host sync, so the overlap scheduler
    prepares and launches the next step while the current forward still runs."""
    model_runner = _FakeRunner(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(64 * 16, dtype=torch.int32, device="cuda").view(
                64, 16
            )
        ),
        beam_trie_compact_enabled=True,
        model_config=SimpleNamespace(
            beam_trie_config=SimpleNamespace(num_codebooks=3),
            is_encoder_decoder=False,
            attention_arch=AttentionArch.MHA,
        ),
        decode_attention_backend_str="flashinfer",
        sliding_window_size=None,
        kv_cache_dtype=torch.bfloat16,
        model=torch.nn.Sequential(),
    )
    no_two_batch_overlap = SimpleNamespace(
        overlap=SimpleNamespace(enable_two_batch_overlap=False)
    )
    with (
        envs.SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION.override(True),
        patch("sglang.srt.runtime_context.get_exec", lambda: no_two_batch_overlap),
    ):
        # Warm-up: first allocations and kernel loads.
        _prepare(_decode_batch(), model_runner)
        torch.cuda.synchronize()
        batch = _decode_batch()
        torch.cuda.set_sync_debug_mode("error")
        try:
            levels, metadata = _prepare(batch, model_runner)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    rows = NUM_GROUPS * BEAM
    assert batch.seq_lens.tolist() == [PROMPT_LEN + DECODE_LEN] * rows
    assert levels.tolist() == [DECODE_LEN] * rows
    assert metadata is not None and len(metadata.buckets) == 1
    bucket = metadata.buckets[0]
    assert bucket.decode_lens.tolist() == [DECODE_LEN] * NUM_GROUPS
    assert bucket.context_cu_seqlens.tolist() == [0, PROMPT_LEN, 2 * PROMPT_LEN]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
