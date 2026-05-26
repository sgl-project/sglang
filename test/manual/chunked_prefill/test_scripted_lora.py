"""LoRA × chunked: ScriptedRuntime tests.

Reproduces the path that 5ed4faf0ab "Bypass LoRA scheduling gate for
chunked-resume reqs" had to fix: a LoRA request must be able to
chunk-prefill across multiple iterations without the LoRA drainer
deadlock that previously kept it stuck in waiting_queue.

In addition to the baseline smoke, this file pins:

* drainer bypass on chunked-resume
* multi-adapter / mid-chunk adapter switching
* adapter eviction interleaved with chunked-resume
* abort during adapter eviction
* return_logprob + chunked + LoRA lm-head pass index

Many tests reference forward-pointing harness APIs (``t.abort``,
``r.lora_path``, ``r.is_chunking``, etc.); they will start exercising
real behavior as the wishlist lands.
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
# Second adapter used by adapter-switch / multi-adapter tests.
# Placeholder name follows the existing convention; the harness
# wishlist (§4 LoRA) will pin a real second adapter when it lands.
_LORA_ADAPTER_B = "philschmid/llama-3-2-1b-instruct-finetuning-lora-cookbook-test-b"


class TestLoRASingleAdapter(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER],
    )

    def test_naive_lora_chunked(self):
        """LoRA chunked prefill: req finishes on the requested adapter and releases resources cleanly."""
        self.runtime.run(self._script_naive_lora_chunked)

    @staticmethod
    def _script_naive_lora_chunked(t: ScriptedRuntime):
        # When ``start_req`` learns ``lora_path=`` (wishlist §4 P2 (10)
        # adjacent — same kwarg-routing pattern as ``priority``), this
        # script will route to the adapter; until then it exercises the
        # chunked path with LoRA enabled at engine level.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            lora_path=_LORA_ADAPTER,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        # Adapter pin must hold across chunk boundaries.
        assert r.lora_path == _LORA_ADAPTER
        # Resource release sanity for LoRA + chunked.
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert len(r.output_tokens) == 4

    def test_lora_logprob_chunked_pass_idx(self):
        """LoRA + return_logprob + chunked: lm_head_pass_idx accumulates across chunks."""
        self.runtime.run(self._script_lora_logprob_chunked_pass_idx)

    # return_logprob + LoRA + chunked exercises the lm_head pass
    # index code path on the LoRA wrapper. Forward-pointing pin: once
    # ``start_req`` learns ``return_logprob=`` (wishlist §4 P2 (12)) and
    # ``ReqHandle`` exposes ``num_input_logprobs``, this asserts that
    # logprobs accumulate across every chunk for a LoRA-attached req.
    @staticmethod
    def _script_lora_logprob_chunked_pass_idx(t: ScriptedRuntime):
        prompt_len: int = VERY_LONG_PROMPT_LEN
        r = t.start_req(
            prompt_len=prompt_len,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER,
            return_logprob=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert r.num_input_logprobs == prompt_len


class TestLoRADrainerBypass(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER, _LORA_ADAPTER_B],
        max_loras_per_batch=1,
    )

    def test_lora_drainer_does_not_block_chunked_resume(self):
        """Chunked-resume req bypasses the LoRA drainer gate and does not deadlock."""
        self.runtime.run(self._script_lora_drainer_does_not_block_chunked_resume)

    # LoRA drainer must not block a chunked-resume
    # req that's parked in waiting_queue between chunks. Before the fix,
    # the drainer would refuse to admit further work while it tried to
    # evict adapter A, leaving the chunked-resume req stuck mid-prefill.
    # With ``max_loras_per_batch=1``, admitting r_b forces a drain of
    # adapter A even while r_a still has chunks pending.
    @staticmethod
    def _script_lora_drainer_does_not_block_chunked_resume(t: ScriptedRuntime):
        r_a = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER,
        )
        # Park r_a in chunked-resume state.
        yield from run_until(r_a, lambda h: h.is_chunking and h.chunks_done >= 1)
        chunks_before = r_a.chunks_done

        # Submit a req on a different adapter — triggers a drainer cycle
        # while r_a is still mid-chunk.
        r_b = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE // 2,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER_B,
        )

        # r_a must keep making forward progress despite the drainer.
        for _ in range(200):
            if r_a.chunks_done > chunks_before:
                break
            yield
        else:
            raise AssertionError(
                f"chunked-resume r_a starved by LoRA drainer; "
                f"chunks_done stuck at {chunks_before}"
            )

        yield from run_until_all_finished(handles=[r_a, r_b])
        assert r_a.finished and r_b.finished


class TestLoRAAdapterSwitch(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER, _LORA_ADAPTER_B],
        max_loras_per_batch=2,
    )

    def test_lora_adapter_switch_mid_chunk(self):
        """Two chunked reqs on distinct adapters interleave without cross-contamination."""
        self.runtime.run(self._script_lora_adapter_switch_mid_chunk)

    # R1 chunks with adapter A and R2 chunks with adapter B,
    # both in flight simultaneously. Each req's lora_path stays pinned
    # to its own adapter across chunk boundaries — no cross-contamination.
    @staticmethod
    def _script_lora_adapter_switch_mid_chunk(t: ScriptedRuntime):
        r_a = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER,
        )
        r_b = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER_B,
        )
        yield from run_until_all_finished(handles=[r_a, r_b])
        assert r_a.finished and r_b.finished
        assert r_a.chunks_done >= 2 and r_b.chunks_done >= 2
        # Forward-pointing pin: each req's adapter id is preserved.
        assert r_a.lora_path == _LORA_ADAPTER
        assert r_b.lora_path == _LORA_ADAPTER_B


class TestLoRAAllDistinctAdapters(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER, _LORA_ADAPTER_B],
        max_loras_per_batch=2,
        max_loaded_loras=4,
    )

    def test_lora_all_distinct_adapters_chunked(self):
        """N chunked reqs each on its own adapter do not deadlock the adapter pool."""
        self.runtime.run(self._script_lora_all_distinct_adapters_chunked)

    # N chunked reqs, N adapters. With max_loras_per_batch < N
    # the pool must rotate adapters cleanly; no deadlock, every req
    # completes. With only two distinct adapter slots configured here,
    # the rotation is forced by ``max_loras_per_batch=2`` vs 4 reqs.
    @staticmethod
    def _script_lora_all_distinct_adapters_chunked(t: ScriptedRuntime):
        adapters = [_LORA_ADAPTER, _LORA_ADAPTER_B, _LORA_ADAPTER, _LORA_ADAPTER_B]
        reqs = [
            t.start_req(
                prompt_len=VERY_LONG_PROMPT_LEN,
                max_new_tokens=2,
                lora_path=adapter,
            )
            for adapter in adapters
        ]
        yield from run_until_all_finished(handles=reqs, max_steps=2000)
        assert all(r.finished for r in reqs)
        # Each req's adapter pin must be preserved across the rotation
        # — that's the actual deadlock-avoidance invariant the pool
        # protects: no req can drift onto a different adapter mid-prefill.
        for r, adapter in zip(reqs, adapters):
            assert r.lora_path == adapter, (
                f"adapter drift under rotation; expected {adapter}, got "
                f"{r.lora_path}"
            )
            assert r.kv_pages == 0
            assert r.lock_refs == 0
            assert r.chunks_done >= 2


class TestLoRAAdapterEviction(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER, _LORA_ADAPTER_B],
        max_loras_per_batch=1,
        max_loaded_loras=1,
    )

    def test_lora_adapter_eviction_between_chunks(self):
        """Adapter eviction triggered between two chunks of a chunked-resume req does not strand it."""
        self.runtime.run(self._script_lora_adapter_eviction_between_chunks)

    # r_a is parked between chunks (chunked-resume). r_b on
    # a different adapter is submitted; with max_loaded_loras=1 it forces
    # an adapter eviction. Per b-5ed4faf0ab the chunked-resume must
    # bypass the drainer gate, so r_a's adapter is preserved (re-loaded)
    # rather than dropped mid-prefill.
    @staticmethod
    def _script_lora_adapter_eviction_between_chunks(t: ScriptedRuntime):
        r_a = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER,
        )
        yield from run_until(r_a, lambda h: h.is_chunking and h.chunks_done >= 1)

        r_b = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE // 2,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER_B,
        )
        yield from run_until_all_finished(handles=[r_a, r_b], max_steps=800)
        assert r_a.finished and r_b.finished
        # r_a finishing on the same adapter it started with verifies
        # eviction did not corrupt its mid-prefill adapter pin.
        assert r_a.lora_path == _LORA_ADAPTER

    def test_lora_chunked_abort_during_eviction(self):
        """Aborting a chunked LoRA req mid-eviction cleans up adapter refs and KV without leaking."""
        self.runtime.run(self._script_lora_chunked_abort_during_eviction)

    # Triple-race: LoRA + chunked + abort. r_a is mid-chunk on
    # adapter A; r_b on adapter B forces an eviction; then abort r_a
    # while the eviction is in flight. Expected: no orphaned adapter
    # ref, no leaked KV pages.
    @staticmethod
    def _script_lora_chunked_abort_during_eviction(t: ScriptedRuntime):
        r_a = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER,
        )
        yield from run_until(r_a, lambda h: h.is_chunking and h.chunks_done >= 1)

        r_b = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE // 2,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER_B,
        )
        yield  # let the eviction kick off

        t.abort(r_a)
        yield

        assert r_a.status in ("finished", "unknown")
        assert r_a.kv_pages == 0
        assert r_a.lock_refs == 0
        # W3: triple-race abort across the LoRA drainer must not double-
        # release; aborting a chunked-resume in flight is the exact path
        # commit de3859646b protects.
        assert r_a.abort_double_release_count == 0, (
            f"W3 violation in LoRA + chunked + abort triple-race; got "
            f"{r_a.abort_double_release_count}"
        )
        yield from run_until_finished(r_b)
        assert r_b.finished
        # r_b stays pinned to adapter B — eviction churn must not strand
        # it on the wrong adapter mid-prefill.
        assert r_b.lora_path == _LORA_ADAPTER_B
        assert r_b.kv_pages == 0
        assert r_b.lock_refs == 0


if __name__ == "__main__":
    unittest.main()
