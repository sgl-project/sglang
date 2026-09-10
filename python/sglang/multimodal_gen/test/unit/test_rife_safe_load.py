import tempfile
import unittest
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.postprocess.rife_interpolator import (
    _load_rife_state_dict,
)


class _UnsafePayload:
    def __reduce__(self):
        return (eval, ("1 + 1",))


class TestRifeSafeLoad(unittest.TestCase):
    def test_loads_plain_state_dict_with_weights_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flownet.pkl"
            expected = {"module.weight": torch.arange(4, dtype=torch.float32)}
            torch.save(expected, path)

            restored = _load_rife_state_dict(str(path))

            self.assertEqual(restored.keys(), expected.keys())
            torch.testing.assert_close(restored["module.weight"], expected["module.weight"])

    def test_rejects_unsafe_pickle_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flownet.pkl"
            torch.save(_UnsafePayload(), path)

            with self.assertRaises(Exception):
                _load_rife_state_dict(str(path))

    def test_rejects_non_mapping_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flownet.pkl"
            torch.save(torch.arange(4, dtype=torch.float32), path)

            with self.assertRaisesRegex(ValueError, "state_dict mapping"):
                _load_rife_state_dict(str(path))


if __name__ == "__main__":
    unittest.main()
