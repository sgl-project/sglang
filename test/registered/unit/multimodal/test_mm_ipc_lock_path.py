import os
import tempfile

from sglang.srt.utils.cuda_ipc_transport_utils import SHM_LOCK_FILE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMmIpcLockPathPerUser(CustomTestCase):
    def test_lock_path_is_per_user_and_writable(self):
        # Regression for the shared-host PermissionError: the lock file must not
        # be a single hardcoded path that the first user's umask can lock every
        # other user out of.
        self.assertNotEqual(SHM_LOCK_FILE, "/tmp/shm_wr_lock.lock")
        self.assertTrue(SHM_LOCK_FILE.startswith(tempfile.gettempdir()))
        self.assertIn(str(os.getuid()), SHM_LOCK_FILE)
        # The current user can take the lock (append open is what the code does).
        with open(SHM_LOCK_FILE, "a"):
            pass


if __name__ == "__main__":
    import unittest

    unittest.main()
