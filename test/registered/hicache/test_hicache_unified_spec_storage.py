"""
E2E tests for the unified L3 key scheme with speculative decoding.

Each class starts its servers, one after another, on one file-backed L3
store. An algorithm that keeps no draft KV in HiCache must store exactly the
objects of a deployment without speculative decoding and share them with one,
in both directions.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_unified_spec_storage.py -v
"""

import json
import os
import tempfile
import unittest
import uuid

from sglang.srt.utils.device import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.hicache_unified_spec_storage_common import (
    PAGE_SIZE,
    HiCacheUnifiedSpecStorageMixin,
    require_model_snapshot,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    CustomTestCase,
)

# The ring needs a draft attention backend that only Blackwell serves.
register_cuda_ci(est_time=600, stage="nightly", runner_config="4-gpu-b200")


def _ring_eligible_draft_config(config: dict) -> dict:
    """Turn a DFLASH draft config into an MLA draft of sliding layers only.

    That is what the per-request draft KV ring serves. The weights are not
    needed: with a ring no draft KV enters HiCache, so the test loads the
    draft with random weights and judges the target's KV alone.
    """
    config = dict(config)
    config.pop("auto_map", None)
    config["layer_types"] = ["sliding_attention"] * config["num_hidden_layers"]
    config["sliding_window"] = 1024
    config["use_sliding_window"] = True
    config["num_key_value_heads"] = 1
    config.update(
        kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
    )
    config["dflash_config"] = {**config["dflash_config"], "attention_mode": "mla"}
    return config


class TestUnifiedKeysRingDFlash(HiCacheUnifiedSpecStorageMixin, CustomTestCase):
    """DFLASH with its draft KV ring: target-only pages, as without a draft."""

    model = DEFAULT_TARGET_MODEL_DFLASH

    @classmethod
    def setUpClass(cls):
        if not is_sm100_supported():
            raise unittest.SkipTest("the DFLASH draft KV ring needs trtllm_mla")
        require_model_snapshot(cls.model)
        with open(require_model_snapshot(DEFAULT_DRAFT_MODEL_DFLASH)) as f:
            draft_config = _ring_eligible_draft_config(json.load(f))
        super().setUpClass()
        cls._draft_dir = tempfile.TemporaryDirectory(prefix="dflash_ring_draft_")
        with open(os.path.join(cls._draft_dir.name, "config.json"), "w") as f:
            json.dump(draft_config, f)
        cls.ring_dflash_args = [
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            cls._draft_dir.name,
            "--speculative-draft-load-format",
            "dummy",
            "--speculative-draft-attention-backend",
            "trtllm_mla",
        ]

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._draft_dir.cleanup()

    # What a TP1 deployment of this target without speculative decoding
    # stores per page: one chunk, K and V of 32 layers x 8 kv heads x 128 in
    # bfloat16. Literals, so the test does not derive them from the code under
    # test.
    target_object_suffix = "_ukv1-b440160febf92487_L0-32_H0.bin"
    target_object_size = 32 * 2 * PAGE_SIZE * 8 * 128 * 2

    def _assert_only_target_objects(self, objects):
        for name, (size, _) in objects.items():
            self.assertTrue(name.endswith(self.target_object_suffix), name)
            self.assertEqual(size, self.target_object_size, name)

    def _assert_untouched(self, objects):
        now = self.l3_objects()
        self.assertEqual({name: now.get(name) for name in objects}, objects)
        return now

    def test_objects_are_shared_with_a_deployment_without_speculation(self):
        prompt_a = self.random_prompt(seed=20261116)
        prompt_b = self.random_prompt(seed=20261117)
        salt_a, salt_b = uuid.uuid4().hex, uuid.uuid4().hex

        # 1. DFLASH with the ring writes A and reloads it bit for bit.
        self.launch_server(self.ring_dflash_args)
        self.assertTrue(self.server_info()["speculative_dflash_draft_kv_ring"])
        dflash_hit = self.reload_in_process(prompt_a, salt_a, seed=20261116)
        self.assertGreater(dflash_hit["meta_info"]["spec_verify_ct"], 0)
        # The store holds what a deployment without a draft would have
        # written: one target chunk per page, nothing else.
        written = self.l3_objects()
        self.assertGreaterEqual(len(written), 2 * self.prompt_pages)
        self._assert_only_target_objects(written)

        # 2. A deployment without speculative decoding reads A from the
        # store without touching an object, and writes B.
        self.launch_server([])
        storage_hit = self.generate(prompt_a, salt_a)
        self.assert_storage_hit(storage_hit)
        self.assert_close_output(storage_hit, dflash_hit)
        self.assert_kl_against_recompute(storage_hit, prompt_a)
        written = self._assert_untouched(written)
        plain_hit = self.reload_in_process(prompt_b, salt_b, seed=20261117)
        self.assertNotIn("spec_verify_ct", plain_hit["meta_info"])
        written = self._assert_untouched(written)
        self._assert_only_target_objects(written)

        # 3. A restarted DFLASH deployment reads its own pages and the
        # other deployment's.
        self.launch_server(self.ring_dflash_args)
        for prompt, salt, reference in (
            (prompt_a, salt_a, dflash_hit),
            (prompt_b, salt_b, plain_hit),
        ):
            storage_hit = self.generate(prompt, salt)
            self.assert_storage_hit(storage_hit)
            self.assertGreater(storage_hit["meta_info"]["spec_verify_ct"], 0)
            self.assert_close_output(storage_hit, reference)
            self.assert_kl_against_recompute(storage_hit, prompt)
        self._assert_untouched(written)


if __name__ == "__main__":
    unittest.main(verbosity=2)
