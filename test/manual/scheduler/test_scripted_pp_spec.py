"""PP2 + EAGLE on the PP ring (2 GPUs). The scripted hook freezes every slot
step from finalize to commit, so a write into a launched step fails the run."""

import unittest

from sglang.srt.environ import envs
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    base_engine_kwargs,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
)


class TestScriptedPPSpec(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=DEFAULT_TARGET_MODEL_EAGLE,
        pp_size=2,
        disable_overlap_schedule=True,
        speculative_algorithm="EAGLE",
        speculative_draft_model_path=DEFAULT_DRAFT_MODEL_EAGLE,
        speculative_num_steps=2,
        speculative_eagle_topk=2,
        speculative_num_draft_tokens=4,
        mem_fraction_static=0.7,
        # The KV canary fails EAGLE draft CUDA-graph capture here. This test
        # checks that no launched step is written, not KV contents.
        kv_canary="none",
        kv_canary_real_data="none",
        kv_canary_sweep_interval=0,
    )

    @classmethod
    def setUpClass(cls):
        cls._pp_spec = envs.SGLANG_ENABLE_PP_SPEC.override(True)
        cls._pp_spec.__enter__()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            cls._pp_spec.__exit__(None, None, None)

    def test_chunked_requests_decode_in_both_slots(self):
        self.server.execute_script(self._script_chunked_requests_decode_in_both_slots)

    @staticmethod
    def _script_chunked_requests_decode_in_both_slots(t: ScriptedContext):
        prompt_lens = (16, 3 * DEFAULT_CHUNK_SIZE - 5, 40, 2 * DEFAULT_CHUNK_SIZE + 7)
        reqs = [
            t.start_req(
                prompt_len=prompt_len,
                max_new_tokens=64,
                ignore_eos=True,
                prompt_token=310 + i,
            )
            for i, prompt_len in enumerate(prompt_lens)
        ]
        slots = set()
        for _ in range(800):
            slots |= {req.ring_slot for req in t.scheduler.lists.decoding}
            if all(r.finished for r in reqs):
                break
            yield
        assert all(r.finished for r in reqs), [r.status for r in reqs]
        assert slots == {0, 1}, f"setup invalid: decoding used slots {slots}"


if __name__ == "__main__":
    unittest.main()
