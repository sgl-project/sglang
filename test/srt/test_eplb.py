import tempfile
import unittest
from typing import List

import sglang as sgl
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestEPLB(CustomTestCase):
    def test_eplb_e2e(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine_kwargs = dict(
                model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
                eplb_storage_dir=tmpdir,
            )

            engine = sgl.Engine(**engine_kwargs)
            ref_output = self._engine_generate(engine)
            self._assert_behavior(engine, ref_output)

            engine.eplb_rebalance()
            self._assert_behavior(engine, ref_output)

            engine.shutdown()
            del engine

            engine = sgl.Engine(**engine_kwargs)
            self._assert_behavior(engine, ref_output)

            engine.eplb_rebalance()
            self._assert_behavior(engine, ref_output)

            engine.shutdown()
            del engine

    def _assert_behavior(self, engine: sgl.Engine, ref_output: List[str]):
        ret = engine.flush_cache()
        assert ret.success

        actual_output = self._engine_generate(engine)
        self.assertEqual(actual_output, ref_output)

        physical_to_logical_map = engine.tokenizer_manager.expert_location_metadata.physical_to_logical_map
        TODO

    def _engine_generate(self, engine: sgl.Engine):
        output = engine.generate(prompt=["1+1=2, 2+2=4", "One plus one is two, two plus two is four"],
                                 sampling_params=dict(max_new_tokens=8, temperature=0.0))
        print(f"engine_generate {output=}")
        return [x["text"] for x in output]


if __name__ == "__main__":
    unittest.main()
