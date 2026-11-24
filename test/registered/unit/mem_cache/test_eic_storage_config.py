"""EIC L3 backend config validation. The eic client library is not installed in
CI, so it is faked at import; the test stops before any client call."""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEICStorageConfig(CustomTestCase):
    def test_config_without_remote_url_names_the_missing_key(self):
        """A config file with no remote_url must fail with an error that names
        remote_url, not with a TypeError from slicing None."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = os.path.join(tmp_dir, "remote-eic.yaml")
            with open(config_file, "w") as fout:
                fout.write("eic_instance_id: test\n")
            with (
                mock.patch.dict(sys.modules, {"eic": types.ModuleType("eic")}),
                temp_set_env(REMOTE_EIC_YAML=config_file),
            ):
                from sglang.srt.mem_cache.storage.eic.eic_storage import EICStorage

                with self.assertRaisesRegex(ValueError, "remote_url"):
                    EICStorage(hicache_config=None, memory_pool_host=None)


if __name__ == "__main__":
    unittest.main()
