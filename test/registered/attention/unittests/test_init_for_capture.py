"""Parity tests for `ForwardBatch.init_for_capture`.

The factory replaces the hand-coded `ForwardBatch(...)` constructors at two
graph-capture entry points (`_dummy_run` and the full-graph decode capture in
`DecodeCudaGraphRunner.capture_prepare`). This test mirrors each original
constructor verbatim against a `init_for_capture(...)` call and verifies
field-by-field equality.

Pure CPU test — no kernels involved; the factory only assigns dataclass
fields. CUDA-only attribute (tensor `.device`) is exercised against the CPU
device for determinism.

Parity is the primary acceptance criterion for this change: it mirrors each
original constructor verbatim, so once the call sites switch to the factory
this suite acts as a regression net against silent drift between the factory
and any future ad-hoc construction at a capture entry point.
"""

import dataclasses
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    CaptureKind,
    ForwardBatch,
    ForwardMode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-b-test-cpu")


class _FakeSlot:
    """Stand-in for GraphSlot: returns its tensor verbatim from slice_for."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def slice_for(self, padded_bs: int, padded_num_tokens: int) -> torch.Tensor:
        return self._tensor


class _FakeRegistry:
    """Minimal CudaGraphBufferRegistry stand-in for the full-graph case.

    Holds a fixed set of named slots; the factory's full-graph path slices
    each via get_slot(name).slice_for(...) and probes optional ones with
    has_slot(name) (slots that are absent resolve to None, as in production).
    """

    def __init__(self, **slots: torch.Tensor):
        self._slots = slots

    def has_slot(self, name: str) -> bool:
        return name in self._slots

    def get_slot(self, name: str) -> _FakeSlot:
        return _FakeSlot(self._slots[name])


def _assert_fb_equal(a: ForwardBatch, b: ForwardBatch) -> None:
    """Field-by-field equality on two ForwardBatch instances.

    Compares every dataclass field; tensors compared via `torch.equal` (must
    share dtype + shape + values); other values via `==`.
    """
    fields = dataclasses.fields(ForwardBatch)
    mismatches = []
    for f in fields:
        va = getattr(a, f.name)
        vb = getattr(b, f.name)
        if isinstance(va, torch.Tensor) or isinstance(vb, torch.Tensor):
            if va is None or vb is None:
                if va is not vb:
                    mismatches.append((f.name, va, vb))
                continue
            if va.shape != vb.shape or va.dtype != vb.dtype or not torch.equal(va, vb):
                mismatches.append((f.name, va, vb))
        else:
            if va != vb:
                mismatches.append((f.name, va, vb))
    assert not mismatches, "field mismatches:\n" + "\n".join(
        f"  {name}: factory={fv!r} reference={rv!r}" for name, fv, rv in mismatches
    )


class TestInitForCaptureParity(CustomTestCase):
    """Verify `init_for_capture(...)` matches each historical hand-coded ctor."""

    DEVICE = torch.device("cpu")

    def test_full_graph_minimal(self) -> None:
        """Mirror `CudaGraphRunner.capture_one_batch_size` (decode, no spec, no DP, no LoRA, no mamba)."""
        bs = 4
        num_tokens = 4
        input_ids = torch.zeros(num_tokens, dtype=torch.int32)
        req_pool_indices = torch.arange(bs, dtype=torch.int32)
        seq_lens = torch.full((bs,), 32, dtype=torch.int32)
        seq_lens_cpu = seq_lens.clone()
        out_cache_loc = torch.zeros(num_tokens, dtype=torch.int64)
        positions = torch.zeros(num_tokens, dtype=torch.int64)
        mrope_positions = torch.zeros(3, num_tokens, dtype=torch.int64)
        next_token_logits_buffer = torch.zeros(num_tokens, 256, dtype=torch.float32)
        num_token_non_padded = torch.tensor(num_tokens, dtype=torch.int32)

        reference = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(seq_lens.sum().item()),
            mamba_track_indices=None,
            mamba_track_mask=None,
            mamba_track_seqlens=None,
            encoder_lens=None,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_gpu=None,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=None,
            mrope_positions=mrope_positions,
            spec_algorithm=None,
            spec_info=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=num_token_non_padded,
            global_forward_mode=ForwardMode.DECODE,
            lora_ids=None,
        )

        factory = ForwardBatch.init_for_capture(
            capture_kind=CaptureKind.FULL_GRAPH,
            registry=_FakeRegistry(
                input_ids=input_ids,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=out_cache_loc,
                positions=positions,
                mrope_positions=mrope_positions,
            ),
            buffers=SimpleNamespace(
                next_token_logits_buffer=next_token_logits_buffer,
                num_token_non_padded=num_token_non_padded,
                global_num_tokens_gpu=None,
                global_num_tokens_for_logprob_gpu=None,
                rids_int=None,
                bootstrap_room_ids_int=None,
            ),
            bs=bs,
            num_tokens=num_tokens,
            forward_mode=ForwardMode.DECODE,
        )

        _assert_fb_equal(factory, reference)

    def test_full_graph_with_canary(self) -> None:
        """Mirror full-graph capture with the KV-canary token oracle enabled.

        Guards the factory's forwarding of the per-request identity tensors
        `rids_int` / `bootstrap_room_ids_int` (sliced from the canary buffers
        when the token-oracle env flag is set). A regression that drops either
        field from the factory would surface here.
        """
        bs = 4
        num_tokens = 4
        input_ids = torch.zeros(num_tokens, dtype=torch.int32)
        req_pool_indices = torch.arange(bs, dtype=torch.int32)
        seq_lens = torch.full((bs,), 32, dtype=torch.int32)
        seq_lens_cpu = seq_lens.clone()
        out_cache_loc = torch.zeros(num_tokens, dtype=torch.int64)
        positions = torch.zeros(num_tokens, dtype=torch.int64)
        mrope_positions = torch.zeros(3, num_tokens, dtype=torch.int64)
        next_token_logits_buffer = torch.zeros(num_tokens, 256, dtype=torch.float32)
        num_token_non_padded = torch.tensor(num_tokens, dtype=torch.int32)
        # Length-bs identity tensors mirror `buffers.rids_int[:bs]` etc.; distinct
        # values so a dropped/zeroed field is caught by the value comparison.
        rids_int = torch.arange(1, bs + 1, dtype=torch.int64)
        bootstrap_room_ids_int = torch.arange(1, bs + 1, dtype=torch.int64) + 100
        # Non-None DP token counts exercise the factory's global_num_tokens_cpu
        # forwarding (the decode capture site passes it; the factory must too).
        global_num_tokens_cpu = [num_tokens, num_tokens]

        reference = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(seq_lens.sum().item()),
            mamba_track_indices=None,
            mamba_track_mask=None,
            mamba_track_seqlens=None,
            encoder_lens=None,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_gpu=None,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=None,
            mrope_positions=mrope_positions,
            spec_algorithm=None,
            spec_info=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=num_token_non_padded,
            global_num_tokens_cpu=global_num_tokens_cpu,
            global_forward_mode=ForwardMode.DECODE,
            lora_ids=None,
            rids_int=rids_int,
            bootstrap_room_ids_int=bootstrap_room_ids_int,
        )

        factory = ForwardBatch.init_for_capture(
            capture_kind=CaptureKind.FULL_GRAPH,
            registry=_FakeRegistry(
                input_ids=input_ids,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=out_cache_loc,
                positions=positions,
                mrope_positions=mrope_positions,
            ),
            buffers=SimpleNamespace(
                next_token_logits_buffer=next_token_logits_buffer,
                num_token_non_padded=num_token_non_padded,
                global_num_tokens_gpu=None,
                global_num_tokens_for_logprob_gpu=None,
                rids_int=rids_int,
                bootstrap_room_ids_int=bootstrap_room_ids_int,
            ),
            bs=bs,
            num_tokens=num_tokens,
            forward_mode=ForwardMode.DECODE,
            global_num_tokens_cpu=global_num_tokens_cpu,
        )

        # Explicit intent guard: the factory must not drop these to None.
        self.assertIsNotNone(factory.rids_int)
        self.assertIsNotNone(factory.bootstrap_room_ids_int)
        self.assertEqual(factory.global_num_tokens_cpu, global_num_tokens_cpu)
        _assert_fb_equal(factory, reference)

    def test_dummy_run_decode_minimal(self) -> None:
        """Mirror `ModelRunner._dummy_run` (decode path: extend_* fields all None)."""
        bs = 2
        num_tokens = 2
        input_ids = torch.zeros(num_tokens, dtype=torch.int32)
        req_pool_indices = torch.arange(bs, dtype=torch.int32)
        seq_lens = torch.full((bs,), 16, dtype=torch.int32)
        seq_lens_cpu = seq_lens.clone()
        out_cache_loc = torch.zeros(num_tokens, dtype=torch.int64)
        positions = torch.zeros(num_tokens, dtype=torch.int64)
        next_token_logits_buffer = torch.zeros(num_tokens, 128, dtype=torch.float32)
        encoder_lens = None
        mrope_positions = torch.zeros(3, num_tokens, dtype=torch.int64)
        num_token_non_padded = torch.tensor(num_tokens, dtype=torch.int32)

        reference = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(seq_lens.sum().item()),
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            extend_num_tokens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            extend_start_loc=None,
            extend_prefix_lens_cpu=None,
            extend_seq_lens_cpu=None,
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_gpu=None,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=None,
            mrope_positions=mrope_positions,
            spec_algorithm=None,
            spec_info=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=num_token_non_padded,
            global_forward_mode=ForwardMode.DECODE,
            lora_ids=None,
        )

        factory = ForwardBatch.init_for_capture(
            capture_kind=CaptureKind.DUMMY_RUN,
            buffers=SimpleNamespace(
                input_ids=input_ids,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                next_token_logits_buffer=next_token_logits_buffer,
                out_cache_loc=out_cache_loc,
                encoder_lens=encoder_lens,
                positions=positions,
                mrope_positions=mrope_positions,
                num_token_non_padded=num_token_non_padded,
                global_num_tokens_gpu=None,
                global_num_tokens_for_logprob_gpu=None,
            ),
            bs=bs,
            num_tokens=num_tokens,
            forward_mode=ForwardMode.DECODE,
        )

        _assert_fb_equal(factory, reference)


if __name__ == "__main__":
    unittest.main()
