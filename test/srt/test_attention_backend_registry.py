import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES


class TestAttentionBackendRegistry(unittest.TestCase):
    def test_llada2_cfg_flashinfer_is_registered(self):
        backend_name = "llada2_cfg_flashinfer"

        self.assertIn(backend_name, ATTENTION_BACKENDS)
        self.assertIn(backend_name, ATTENTION_BACKEND_CHOICES)

    def test_llada2_cfg_flashinfer_rejects_mla(self):
        factory = ATTENTION_BACKENDS["llada2_cfg_flashinfer"]
        runner = SimpleNamespace(use_mla_backend=True)

        with self.assertRaisesRegex(ValueError, "does not use an MLA backend"):
            factory(runner)


if __name__ == "__main__":
    unittest.main()
