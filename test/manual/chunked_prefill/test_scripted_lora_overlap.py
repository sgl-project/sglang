import unittest

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
_LORA_ADAPTER = "philschmid/llama-3-2-1b-instruct-finetuning-lora-cookbook-test"
_LORA_ADAPTER_B = "philschmid/llama-3-2-1b-instruct-finetuning-lora-cookbook-test-b"


class TestLoRAOverlapSingleAdapter(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER],
        enable_lora_overlap_loading=True,
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
        assert r.chunks_done >= 2


class TestLoRAOverlapH2dDuringChunk(ScriptedTestCase):
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
        assert r_a.chunks_done >= 2 and r_b.chunks_done >= 2
        assert r_a.lora_path == _LORA_ADAPTER
        assert r_b.lora_path == _LORA_ADAPTER_B


class TestLoRAOverlapAdapterRotation(ScriptedTestCase):
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

        t.abort(r_b)
        yield

        assert r_b.kv_pages == 0
        assert r_b.lock_refs == 0
        assert r_b.finish_event_count <= 1, (
            f"abort during overlapped H2D must not double-fire; "
            f"got finish_event_count={r_b.finish_event_count}"
        )
        yield from run_until_finished(r_a, max_steps=800)
        assert r_a.finished
        assert r_a.lora_path == _LORA_ADAPTER

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
        yield from run_until_all_finished(handles=reqs, max_steps=2400)
        assert all(r.finished for r in reqs)
        assert all(r.chunks_done >= 2 for r in reqs)
        for r, adapter in zip(reqs, adapters):
            assert r.lora_path == adapter


if __name__ == "__main__":
    unittest.main()
