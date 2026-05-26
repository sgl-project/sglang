"""LoRA overlap loading × chunked: naive ScriptedRuntime smoke.

Adapter H2D copy runs concurrently with the chunked prefill loop. We
want the chunked-resume path to make progress even while LoRA H2D is
in flight, and not double-fire when an abort hits during H2D (issue
#25413).

The naive smoke just verifies the engine starts and completes a
chunked LoRA request with overlap loading enabled.

In addition to the baseline smoke, this file pins:

* H2D copy in progress at chunk-admission boundary
* abort hitting the LoRA H2D + chunked sync window (#25413)
* back-to-back distinct adapters all chunking
"""

import unittest

from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)

_LORA_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
_LORA_ADAPTER = "philschmid/llama-3-2-1b-instruct-finetuning-lora-cookbook-test"
# Second adapter follows the placeholder convention from
# test_scripted_lora.py; the harness wishlist will pin a real second
# adapter when it lands.
_LORA_ADAPTER_B = "philschmid/llama-3-2-1b-instruct-finetuning-lora-cookbook-test-b"


class TestLoRAOverlapSingleAdapter(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER],
        enable_lora_overlap_loading=True,
    )

    def test_naive_lora_overlap_chunked(self):
        """LoRA overlap loading × chunked: naive ScriptedRuntime smoke."""
        self.runtime.run(self._script_naive_lora_overlap_chunked)

    @staticmethod
    def _script_naive_lora_overlap_chunked(t: ScriptedRuntime):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            lora_path=_LORA_ADAPTER,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2


class TestLoRAOverlapH2dDuringChunk(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER, _LORA_ADAPTER_B],
        enable_lora_overlap_loading=True,
        max_loras_per_batch=2,
        max_loaded_loras=1,
    )

    def test_lora_overlap_h2d_during_chunk_admit(self):
        """LoRA H2D copy in flight when chunked admission for a different adapter fires."""
        self.runtime.run(self._script_lora_overlap_h2d_during_chunk_admit)

    # H2D copy of adapter B is overlapping the scheduler
    # loop when r_b's first chunk wants to admit. Pre-fix, the chunked
    # admission would race the not-yet-resident weight pointer; the
    # admission must wait for H2D to settle. r_a continues chunking on
    # adapter A meanwhile and both reqs finish on their pinned adapter.
    @staticmethod
    def _script_lora_overlap_h2d_during_chunk_admit(t: ScriptedRuntime):
        r_a = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER,
        )
        # Park r_a mid-chunked-prefill so adapter A is the resident weight.
        yield from run_until(r_a, lambda h: h.is_chunking and h.chunks_done >= 1)

        # max_loaded_loras=1 forces an H2D copy of adapter B as soon as
        # r_b is admitted — overlap loading lets that happen concurrently
        # with r_a continuing to chunk on adapter A.
        r_b = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER_B,
        )
        yield from run_until_all_finished(handles=[r_a, r_b], max_steps=1200)
        assert r_a.finished and r_b.finished
        assert r_a.chunks_done >= 2 and r_b.chunks_done >= 2
        # Adapter pin survives the overlapped H2D; no cross-contamination.
        assert r_a.lora_path == _LORA_ADAPTER
        assert r_b.lora_path == _LORA_ADAPTER_B


class TestLoRAOverlapAdapterRotation(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER, _LORA_ADAPTER_B],
        enable_lora_overlap_loading=True,
        max_loras_per_batch=1,
        max_loaded_loras=1,
    )

    def test_lora_overlap_abort_during_h2d(self):
        """Abort hitting the LoRA H2D + chunked sync window cancels cleanly without double-fire."""
        self.runtime.run(self._script_lora_overlap_abort_during_h2d)

    # r_a chunks on adapter A. r_b on adapter
    # B is submitted, forcing an overlapped H2D of adapter B and a
    # chunked admission event for r_b. Aborting r_b mid-H2D must cancel
    # cleanly: no double-fire on the completion event, no orphaned
    # adapter ref, no KV leak. r_a's adapter pin must survive.
    @staticmethod
    def _script_lora_overlap_abort_during_h2d(t: ScriptedRuntime):
        r_a = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER,
        )
        yield from run_until(r_a, lambda h: h.is_chunking and h.chunks_done >= 1)

        r_b = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER_B,
        )
        yield  # let the overlapped H2D kick off

        t.abort(r_b)
        yield

        assert r_b.kv_pages == 0
        assert r_b.lock_refs == 0
        assert r_b.finish_event_count <= 1, (
            f"abort during overlapped H2D must not double-fire; "
            f"got finish_event_count={r_b.finish_event_count}"
        )
        # r_a's adapter must not have been corrupted by the cancelled H2D.
        yield from run_until_finished(r_a, max_steps=800)
        assert r_a.finished
        assert r_a.lora_path == _LORA_ADAPTER

    def test_lora_overlap_back_to_back_adapters_chunked(self):
        """Back-to-back distinct adapter loads with chunked reqs all finish on their own adapter."""
        self.runtime.run(self._script_lora_overlap_back_to_back_adapters_chunked)

    # Several chunked reqs submitted back-to-back, each on
    # a distinct adapter, with overlap loading on and only one adapter
    # resident at a time. Each adapter must H2D-load to completion
    # before its req's first chunk runs, and every req finishes on its
    # pinned adapter without cross-contamination.
    @staticmethod
    def _script_lora_overlap_back_to_back_adapters_chunked(t: ScriptedRuntime):
        adapters = [_LORA_ADAPTER, _LORA_ADAPTER_B, _LORA_ADAPTER, _LORA_ADAPTER_B]
        reqs = [
            t.start_req(
                prompt_len=VERY_LONG_PROMPT_LEN,
                max_new_tokens=2,
                lora_path=adapter,
            )
            for adapter in adapters
        ]
        yield from run_until_all_finished(handles=reqs, max_steps=2400)
        assert all(r.finished for r in reqs)
        assert all(r.chunks_done >= 2 for r in reqs)
        for r, adapter in zip(reqs, adapters):
            assert r.lora_path == adapter


if __name__ == "__main__":
    unittest.main()
