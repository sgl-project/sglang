"""Phase 3 NPU tests: LoRA adapters delivered as serialized tensors.

An RL trainer holds adapter weights in memory, so the engine must accept them
without a checkpoint. Both payload shapes the trainer can send are covered:
one serialized state dict per rank, and one flattened buffer per rank. The
oracle is the *same* adapter loaded from disk by ``/load_lora_adapter``: the
two paths must produce identical tokens and token logprobs.
"""

import os
import tempfile
import unittest

from sglang.test.ascend.npu_rl_lora_fixture import (
    MODEL,
    PROMPTS,
    SERVER_ARGS,
    adapter_config,
    adapter_tensors,
    write_adapter,
)
from sglang.test.ascend.test_npu_rl_utils import (
    assert_logprob_signature_equal,
    load_lora_adapter,
    load_lora_adapter_from_tensors,
    logprob_signature,
    npu_rl_env,
    npu_server,
    unload_lora_adapter,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=600, suite="extra-b-test-1-npu-a3")
register_npu_ci(est_time=600, suite="nightly-test-1-npu-a3", nightly=True)

HAVE_MODEL = os.path.isfile(os.path.join(MODEL, "config.json"))


@unittest.skipUnless(HAVE_MODEL, f"model not found: {MODEL}")
class TestNpuLoraLoadFromTensor(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tensors = adapter_tensors(MODEL)
        cls.config = adapter_config()
        cls._adapter_dir = tempfile.TemporaryDirectory(prefix="npu-lora-adapter-")
        cls.adapter_dir = write_adapter(
            os.path.join(cls._adapter_dir.name, "adapter"), cls.tensors, cls.config
        )
        cls._server = npu_server(MODEL, extra_args=SERVER_ARGS, env=npu_rl_env())
        cls.base_url = cls._server.__enter__()
        cls.baseline = logprob_signature(cls.base_url, PROMPTS)

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)
        cls._adapter_dir.cleanup()

    def tearDown(self):
        for lora_name in (
            "disk-adapter",
            "tensor-adapter",
            "flat-adapter",
        ):
            try:
                unload_lora_adapter(self.base_url, lora_name)
            except Exception:
                pass

    def _load_disk_adapter(self, lora_name: str) -> None:
        load_lora_adapter(self.base_url, lora_name, self.adapter_dir)

    def _signature(self, lora_name: str):
        return logprob_signature(self.base_url, PROMPTS, lora_path=lora_name)

    def test_01_tensor_load_matches_disk_adapter(self):
        self._load_disk_adapter("disk-adapter")
        from_disk = self._signature("disk-adapter")
        self.assertNotEqual(
            [item[0] for item in from_disk],
            [item[0] for item in self.baseline],
            "the adapter had no effect -- the payload is a no-op",
        )

        load_lora_adapter_from_tensors(
            self.base_url, "tensor-adapter", self.tensors, self.config
        )
        assert_logprob_signature_equal(
            self._signature("tensor-adapter"),
            from_disk,
            "adapter loaded from tensors does not match the same adapter on disk",
        )

        # The adapter just loaded must not be aliased to the disk one: with the
        # disk adapter gone, the tensor-loaded copy still has to produce it.
        unload_lora_adapter(self.base_url, "disk-adapter")
        assert_logprob_signature_equal(
            self._signature("tensor-adapter"),
            from_disk,
            "unloading the disk adapter changed the tensor-loaded adapter",
        )
        assert_logprob_signature_equal(
            logprob_signature(self.base_url, PROMPTS),
            self.baseline,
            "requests without lora_path changed while an adapter was loaded",
        )

    def test_02_flattened_bucket_matches_per_tensor(self):
        self._load_disk_adapter("disk-adapter")
        from_disk = self._signature("disk-adapter")

        load_lora_adapter_from_tensors(
            self.base_url, "tensor-adapter", self.tensors, self.config
        )
        per_tensor = self._signature("tensor-adapter")

        load_lora_adapter_from_tensors(
            self.base_url,
            "flat-adapter",
            self.tensors,
            self.config,
            load_format="flattened_bucket",
        )
        assert_logprob_signature_equal(
            self._signature("flat-adapter"),
            per_tensor,
            "flattened_bucket adapter does not match the per-tensor payload",
        )
        assert_logprob_signature_equal(
            self._signature("flat-adapter"),
            from_disk,
            "flattened_bucket adapter does not match the same adapter on disk",
        )


if __name__ == "__main__":
    unittest.main()
