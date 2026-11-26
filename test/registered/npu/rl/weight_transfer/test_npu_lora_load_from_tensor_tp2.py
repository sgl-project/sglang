"""Verify per-rank LoRA tensor loading on two Ascend NPUs."""

import os
import tempfile
import unittest

from test_npu_lora_load_from_tensor import (
    HAVE_MODEL,
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
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=600, suite="extra-b-test-2-npu-a3")
register_npu_ci(est_time=600, suite="nightly-test-2-npu-a3", nightly=True)

TP_SIZE = 2


@unittest.skipUnless(HAVE_MODEL, f"model not found: {MODEL}")
class TestNpuLoraLoadFromTensorTP2(CustomTestCase):
    def test_tensor_payloads_match_disk_adapter(self):
        tensors = adapter_tensors(MODEL)
        config = adapter_config()
        with (
            tempfile.TemporaryDirectory(prefix="npu-lora-tp2-") as temp_dir,
            npu_server(
                MODEL,
                extra_args=(*SERVER_ARGS, "--tp-size", str(TP_SIZE)),
                env=npu_rl_env(),
            ) as base_url,
        ):
            adapter_dir = write_adapter(
                os.path.join(temp_dir, "adapter"), tensors, config
            )
            baseline = logprob_signature(base_url, PROMPTS)
            load_lora_adapter(base_url, "disk-adapter", adapter_dir)
            from_disk = logprob_signature(base_url, PROMPTS, lora_path="disk-adapter")
            self.assertNotEqual(
                [item[0] for item in from_disk],
                [item[0] for item in baseline],
                "the disk adapter had no effect on TP=2",
            )

            for name, load_format in (
                ("tensor-adapter", None),
                ("flat-adapter", "flattened_bucket"),
            ):
                load_lora_adapter_from_tensors(
                    base_url,
                    name,
                    tensors,
                    config,
                    load_format=load_format,
                    tp_size=TP_SIZE,
                )
                assert_logprob_signature_equal(
                    logprob_signature(base_url, PROMPTS, lora_path=name),
                    from_disk,
                    f"{name} does not match the disk adapter on TP=2",
                )


if __name__ == "__main__":
    unittest.main()
