import tempfile
import unittest

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

            engine = sgl.Engine( **engine_kwargs)
            self._assert_behavior(engine)

            engine.shutdown()
            del engine

            engine = sgl.Engine( **engine_kwargs)
            self._assert_behavior(engine)

            engine.shutdown()
            del engine

    def _assert_behavior(self, engine: sgl.Engine):
        ret = engine.flush_cache()
        assert ret.success

        TODO


if __name__ == "__main__":
    unittest.main()
