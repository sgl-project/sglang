import unittest

from sglang.srt.lora.lora_registry import LoRARef
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)

_LORA_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
_LORA_ADAPTER = "codelion/Llama-3.2-1B-Instruct-tool-calling-lora"
_LORA_ADAPTER_B = "nicoboss/Llama-3.2-1B-Instruct-Uncensored-Lora"


def _expected_lora_id(adapter_path: str) -> str:
    # Bare-string --lora-paths entries derive their id from (path, path); see
    # ServerArgs LoRA parsing. The per-request lora_id is this uuid5 hex, not the path.
    return LoRARef.deterministic_id(adapter_path, adapter_path)


class TestLoRAOverlapSingleAdapter(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER],
        enable_lora_overlap_loading=True,
        max_loras_per_batch=1,
        max_loaded_loras=1,
    )

    def test_naive_lora_overlap_chunked(self):
        self.server.execute_script(self._script_naive_lora_overlap_chunked)

    @staticmethod
    def _script_naive_lora_overlap_chunked(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            lora_path=_LORA_ADAPTER,
        )
        yield from run_until_finished(r)
        assert r.finished
        # Sole req in this class: no prior same-namespace prefix exists, so it
        # does a full fresh prefill of 2048 tokens at chunk size 256 -> exactly
        # 8 chunks.
        assert r.chunks_done == VERY_LONG_PROMPT_LEN // DEFAULT_CHUNK_SIZE


class TestLoRAOverlapH2dDuringChunk(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER, _LORA_ADAPTER_B],
        enable_lora_overlap_loading=True,
        max_loras_per_batch=2,
        max_loaded_loras=2,
    )

    def test_lora_overlap_h2d_during_chunk_admit(self):
        self.server.execute_script(self._script_lora_overlap_h2d_during_chunk_admit)

    @staticmethod
    def _script_lora_overlap_h2d_during_chunk_admit(t: ScriptedContext):
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
        yield from run_until_all_finished(handles=[r_a, r_b], max_steps=1200)
        assert r_a.finished and r_b.finished
        # r_a (A) and r_b (B) use distinct adapters, so their radix namespaces
        # are disjoint and neither can prefix-hit the other; each is the only req
        # of its adapter in this class, so both do a full fresh prefill of 2048
        # tokens at chunk size 256 -> exactly 8 chunks each (adapter-eviction
        # parks while overlap-loading B are not counted as chunks).
        expected_chunks = VERY_LONG_PROMPT_LEN // DEFAULT_CHUNK_SIZE
        assert r_a.chunks_done == expected_chunks
        assert r_b.chunks_done == expected_chunks
        assert r_a.req.lora_id == _expected_lora_id(_LORA_ADAPTER)
        assert r_b.req.lora_id == _expected_lora_id(_LORA_ADAPTER_B)


class TestLoRAOverlapAbortDuringH2d(ScriptedTestCase):
    # max_loras_per_batch=2 gives the LoRA memory pool two slots, so while r_a
    # (adapter A) is chunking, validate_lora_batch({A, B}) passes and the overlap
    # loader genuinely starts an async H2D transfer of adapter B. With
    # max_loras_per_batch=1 the pool has a single slot, validate_lora_batch({A, B})
    # fails, and r_b's adapter never enters H2D at all -- the abort then merely
    # pops a never-loadable queued req and the test verifies nothing about H2D.
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER, _LORA_ADAPTER_B],
        enable_lora_overlap_loading=True,
        max_loras_per_batch=2,
        max_loaded_loras=2,
    )

    def test_lora_overlap_abort_during_h2d(self):
        self.server.execute_script(self._script_lora_overlap_abort_during_h2d)

    @staticmethod
    def _script_lora_overlap_abort_during_h2d(t: ScriptedContext):
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
        # One step: the overlap loader observes B as NOT_LOADED, starts its async
        # H2D load, and returns False -- so r_b stays in the waiting queue with its
        # adapter mid-load. We abort precisely in that H2D window.
        yield
        b_loader = t.scheduler.lora_overlap_loader
        b_lora_id = _expected_lora_id(_LORA_ADAPTER_B)
        assert (
            b_lora_id in b_loader.lora_to_overlap_load_event
            or b_lora_id in b_loader.lora_manager.memory_pool.uid_to_buffer_id
        ), "adapter B never entered H2D; the abort would not exercise the H2D path"

        t.abort(r_b)
        yield

        # r_b was aborted while still queued (H2D in flight), so it was popped from
        # the waiting queue: it never ran a batch and never allocated KV or locks.
        # Assert this unconditionally -- req is None is the expected cleaned-up
        # outcome here, not a reason to skip the resource checks.
        assert r_b.chunks_done == 0
        assert r_b.req is None
        assert r_b.status in ("finished", "unknown")
        # The in-flight H2D abort must not corrupt the loader: r_a continues and
        # finishes with its own adapter id (at-most-one finish is enforced by the
        # engine -- output_streamer: assert not req.finished_output).
        yield from run_until_finished(r_a, max_steps=800)
        assert r_a.finished
        assert r_a.req.lora_id == _expected_lora_id(_LORA_ADAPTER)


class TestLoRAOverlapAdapterRotation(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER, _LORA_ADAPTER_B],
        enable_lora_overlap_loading=True,
        max_loras_per_batch=1,
        max_loaded_loras=2,
    )

    def test_lora_overlap_back_to_back_adapters_chunked(self):
        self.server.execute_script(
            self._script_lora_overlap_back_to_back_adapters_chunked
        )

    @staticmethod
    def _script_lora_overlap_back_to_back_adapters_chunked(t: ScriptedContext):
        adapters = [_LORA_ADAPTER, _LORA_ADAPTER_B, _LORA_ADAPTER, _LORA_ADAPTER_B]
        reqs = [
            t.start_req(
                prompt_len=VERY_LONG_PROMPT_LEN,
                max_new_tokens=2,
                lora_path=adapter,
            )
            for adapter in adapters
        ]
        expected_ids = [_expected_lora_id(adapter) for adapter in adapters]
        # Capture each req's lora_id live (a finished req is removed from the
        # scheduler, so r.req becomes None) while probing every handle every
        # step so the faster reqs stay registered.
        lora_id_by_rid: dict = {}
        done = [False] * len(reqs)
        for _ in range(2400):
            for i, r in enumerate(reqs):
                req = r.req
                if req is not None:
                    lora_id_by_rid[r.rid] = req.lora_id
                done[i] = done[i] or r.finished
            if all(done):
                break
            yield
        assert all(done)
        # Identical all-ones prompts + repeated A/B adapters under
        # max_loras_per_batch=1: adapters run one at a time, so the second A and
        # second B each run only after their twin commits the full 2048-token
        # prefix to radix. Each twin then prefix-hits all 2048 tokens and finishes
        # its prefill in a single extend batch -- it is never held as chunked_req,
        # so chunks_done == 0. The first req of each adapter cannot prefix-hit and
        # processes all 2048 tokens in 256-chunks -> exactly 8 chunks.
        # LPM scheduling does not guarantee which req of a same-adapter pair
        # runs cold first (observed [8, 0, 0, 8]: B's twin ran before B's first).
        # The invariant is per-adapter: exactly one req of each pair does the
        # full cold 8-chunk prefill and its twin full-prefix-hits (0 chunks).
        expected_first_chunks = VERY_LONG_PROMPT_LEN // DEFAULT_CHUNK_SIZE
        chunk_counts = [r.chunks_done for r in reqs]
        a_counts = sorted([reqs[0].chunks_done, reqs[2].chunks_done])
        b_counts = sorted([reqs[1].chunks_done, reqs[3].chunks_done])
        assert a_counts == [0, expected_first_chunks], (
            f"adapter A pair must be one cold full prefill + one full prefix "
            f"hit; per-req chunks_done={chunk_counts}"
        )
        assert b_counts == [0, expected_first_chunks], (
            f"adapter B pair must be one cold full prefill + one full prefix "
            f"hit; per-req chunks_done={chunk_counts}"
        )
        for r, expected_id in zip(reqs, expected_ids):
            assert lora_id_by_rid.get(r.rid) == expected_id


if __name__ == "__main__":
    unittest.main()
