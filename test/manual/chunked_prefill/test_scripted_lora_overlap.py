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
        expected_chunks = VERY_LONG_PROMPT_LEN // DEFAULT_CHUNK_SIZE
        assert r_a.chunks_done == expected_chunks
        assert r_b.chunks_done == expected_chunks
        assert r_a.req.lora_id == _expected_lora_id(_LORA_ADAPTER)
        assert r_b.req.lora_id == _expected_lora_id(_LORA_ADAPTER_B)


class TestLoRAOverlapAbortDuringH2d(ScriptedTestCase):
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
        yield
        b_loader = t.scheduler.lora_overlap_loader
        b_lora_id = _expected_lora_id(_LORA_ADAPTER_B)
        assert (
            b_lora_id in b_loader.lora_to_overlap_load_event
            or b_lora_id in b_loader.lora_manager.memory_pool.uid_to_buffer_id
        ), "adapter B never entered H2D; the abort would not exercise the H2D path"

        t.abort(r_b)
        yield

        assert r_b.chunks_done == 0
        assert r_b.req is None
        assert r_b.status in ("finished", "unknown")
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
