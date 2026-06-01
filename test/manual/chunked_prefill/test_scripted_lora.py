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


class TestLoRASingleAdapter(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER],
    )

    def test_naive_lora_chunked(self):
        self.server.execute_script(self._script_naive_lora_chunked)

    @staticmethod
    def _script_naive_lora_chunked(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            lora_path=_LORA_ADAPTER,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert r.req.lora_id == _LORA_ADAPTER
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert len(r.req.output_ids) == 4

    def test_lora_logprob_chunked_pass_idx(self):
        self.server.execute_script(self._script_lora_logprob_chunked_pass_idx)

    @staticmethod
    def _script_lora_logprob_chunked_pass_idx(t: ScriptedContext):
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
        assert len(r.req.logprob.input_token_logprobs_val) == prompt_len


class TestLoRADrainerBypass(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER, _LORA_ADAPTER_B],
        max_loras_per_batch=1,
    )

    def test_lora_drainer_does_not_block_chunked_resume(self):
        self.server.execute_script(
            self._script_lora_drainer_does_not_block_chunked_resume
        )

    @staticmethod
    def _script_lora_drainer_does_not_block_chunked_resume(t: ScriptedContext):
        r_a = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER,
        )
        yield from run_until(r_a, lambda h: h.is_chunking and h.chunks_done >= 1)
        chunks_before = r_a.chunks_done

        r_b = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE // 2,
            max_new_tokens=2,
            lora_path=_LORA_ADAPTER_B,
        )

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


class TestLoRAAdapterSwitch(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER, _LORA_ADAPTER_B],
        max_loras_per_batch=2,
    )

    def test_lora_adapter_switch_mid_chunk(self):
        self.server.execute_script(self._script_lora_adapter_switch_mid_chunk)

    @staticmethod
    def _script_lora_adapter_switch_mid_chunk(t: ScriptedContext):
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
        assert r_a.req.lora_id == _LORA_ADAPTER
        assert r_b.req.lora_id == _LORA_ADAPTER_B


class TestLoRAAllDistinctAdapters(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER, _LORA_ADAPTER_B],
        max_loras_per_batch=2,
        max_loaded_loras=4,
    )

    def test_lora_all_distinct_adapters_chunked(self):
        self.server.execute_script(self._script_lora_all_distinct_adapters_chunked)

    @staticmethod
    def _script_lora_all_distinct_adapters_chunked(t: ScriptedContext):
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
        for r, adapter in zip(reqs, adapters):
            assert r.req.lora_id == adapter, (
                f"adapter drift under rotation; expected {adapter}, got "
                f"{r.req.lora_id}"
            )
            assert r.kv_pages == 0
            assert r.lock_refs == 0
            assert r.chunks_done >= 2


class TestLoRAAdapterEviction(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_LORA_BASE_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_lora=True,
        lora_paths=[_LORA_ADAPTER, _LORA_ADAPTER_B],
        max_loras_per_batch=1,
        max_loaded_loras=1,
    )

    def test_lora_adapter_eviction_between_chunks(self):
        self.server.execute_script(self._script_lora_adapter_eviction_between_chunks)

    @staticmethod
    def _script_lora_adapter_eviction_between_chunks(t: ScriptedContext):
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
        assert r_a.req.lora_id == _LORA_ADAPTER

    def test_lora_chunked_abort_during_eviction(self):
        self.server.execute_script(self._script_lora_chunked_abort_during_eviction)

    @staticmethod
    def _script_lora_chunked_abort_during_eviction(t: ScriptedContext):
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
        yield

        t.abort(r_a)
        yield

        assert r_a.status in ("finished", "unknown")
        assert r_a.kv_pages == 0
        assert r_a.lock_refs == 0
        yield from run_until_finished(r_b)
        assert r_b.finished
        assert r_b.req.lora_id == _LORA_ADAPTER_B
        assert r_b.kv_pages == 0
        assert r_b.lock_refs == 0


if __name__ == "__main__":
    unittest.main()
