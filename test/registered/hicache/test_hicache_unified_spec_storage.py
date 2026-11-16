"""
E2E tests for the unified L3 key scheme with speculative decoding.

Each class starts its servers, one after another, on one file-backed L3
store.

- An algorithm that keeps no draft KV in HiCache must store exactly the
  objects of a deployment without speculative decoding and share them with
  one, in both directions.
- NEXTN-style MTP stores its packed draft layer as one more chunk per page.
  Greedy output is the target's whatever the draft KV holds, so these classes
  also judge the draft: the verify count and accept length of an L3 reload.

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
        dflash_hit, _ = self.reload_in_process(prompt_a, salt_a, seed=20261116)
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
        plain_hit, _ = self.reload_in_process(prompt_b, salt_b, seed=20261117)
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


NOTICE = (
    "The city library opens at nine in the morning and closes at six in the "
    "evening on weekdays. Members may borrow up to five books for three weeks "
    "and renew each loan twice unless another reader has reserved the title. "
    "Returned books are checked by the staff at the front desk, sorted by "
    "subject on the trolley, and shelved again before the end of the day. "
)
SCRUB_NOTICE = (
    "The harbour ferry leaves the north pier every forty minutes from early "
    "morning until midnight. Tickets are sold at the kiosk beside the gate, "
    "and bicycles travel free outside the rush hours. In fog the captain "
    "sounds the horn twice before casting off and once more at the breakwater. "
)


class PackedNextnMixin(HiCacheUnifiedSpecStorageMixin):
    """NEXTN-style MTP: target chunks plus one packed draft chunk per page."""

    prompt_pages = 20
    prompt_extra_tokens = 2
    max_new_tokens = 96
    draft_tokens = 4
    # Per-page object suffixes and sizes of the class's topology: literals.
    target_objects: dict = {}
    draft_objects: dict = {}
    nextn_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]

    def scrub_prompt(self, seed):
        return self.text_prompt(SCRUB_NOTICE)

    def _pages(self, objects):
        """page hash -> {suffix: size}"""
        pages = {}
        for name, (size, _) in objects.items():
            page, suffix = name.split("_", 1)
            pages.setdefault(page, {})[suffix] = size
        return pages

    def _assert_untouched(self, objects):
        now = self.l3_objects()
        self.assertEqual({name: now.get(name) for name in objects}, objects)
        return now

    def _assert_draft_meaningful(self, result):
        """The draft is consulted and is right about some tokens, not all."""
        verify_ct, accept_length = self.spec_stats(result)
        self.assertGreater(verify_ct, 1)
        self.assertGreater(accept_length, 1.2)
        self.assertLess(accept_length, self.draft_tokens)

    def test_draft_kv_follows_its_pages_to_l3_and_back(self):
        prompt = self.text_prompt(NOTICE)
        salt = uuid.uuid4().hex

        # 1. Same process: the L3 hit reproduces the device-cache hit, the
        # draft's work included.
        self.launch_server(self.nextn_args)
        device_hit, storage_hit = self.reload_in_process(prompt, salt, seed=1)
        self._assert_draft_meaningful(device_hit)
        self.assertEqual(self.spec_stats(storage_hit), self.spec_stats(device_hit))

        # Every page holds exactly its target chunks and its draft chunks.
        written = self.l3_objects()
        expected = {**self.target_objects, **self.draft_objects}
        pages = self._pages(written)
        self.assertGreaterEqual(len(pages), 2 * self.prompt_pages)
        for page, chunks in pages.items():
            self.assertEqual(chunks, expected, page)

        # 2. After a restart the pages come back with their draft KV.
        self.launch_server(self.nextn_args)
        restarted = self.generate(prompt, salt)
        self.assert_storage_hit(restarted)
        self.assert_close_output(restarted, device_hit)
        self.assert_kl_against_recompute(restarted, prompt)
        self.assertGreaterEqual(
            self.spec_stats(restarted)[1], 0.9 * self.spec_stats(device_hit)[1]
        )
        # Backups are asynchronous: take stock once the server is down.
        self.stop_server()
        written = self._assert_untouched(written)

        # 3. Another draft: a miss on the draft chunk alone. The reader
        # recomputes, adds its own draft chunks and leaves the target's.
        other_draft = [*self.nextn_args, "--hicache-storage-draft-id", "another-draft"]
        self.launch_server(other_draft)
        missed = self.generate(prompt, salt)
        self.assertEqual(missed["meta_info"]["cached_tokens"], 0)
        self.flush_cache()
        backfilled = self.generate(prompt, salt)
        self.assert_storage_hit(backfilled)
        self.stop_server()
        now = self._assert_untouched(written)
        added = self._pages({name: now[name] for name in now if name not in written})
        draft_sizes = sorted(self.draft_objects.values())
        backfilled_pages = 0
        for page, chunks in added.items():
            drafts = {s: size for s, size in chunks.items() if s.startswith("ukd1-")}
            # Draft chunks of another draft digest, never the first draft's.
            self.assertEqual(sorted(drafts.values()), draft_sizes, page)
            self.assertFalse(set(drafts) & set(self.draft_objects), chunks)
            targets = {s: size for s, size in chunks.items() if s not in drafts}
            if targets:
                # A page this reader was first to compute (the generated
                # tail can differ between a cold and a cached run).
                self.assertEqual(targets, self.target_objects, page)
            else:
                backfilled_pages += 1
        self.assertGreaterEqual(backfilled_pages, self.prompt_pages)

        # 4. Control: with every draft object of the first draft zeroed, the
        # pages still hit and the output holds, and the draft's work shows
        # the loss. Without this, step 1 would not prove that draft KV is
        # what came back.
        for name in written:
            if name.split("_", 1)[1] in self.draft_objects:
                path = os.path.join(self.storage_path, name)
                with open(path, "r+b") as f:
                    f.write(bytes(os.path.getsize(path)))
        self.launch_server(self.nextn_args)
        degraded = self.generate(prompt, salt)
        self.assert_storage_hit(degraded)
        self.assert_close_output(degraded, device_hit)
        self.assertNotEqual(self.spec_stats(degraded), self.spec_stats(device_hit))
        self.assertLess(
            self.spec_stats(degraded)[1], 0.9 * self.spec_stats(device_hit)[1]
        )


class TestUnifiedKeysPackedNextnMHA(PackedNextnMixin, CustomTestCase):
    """MiMo-7B: 36 layers x 8 kv heads x 128, one bundled MTP layer, TP1."""

    model = "XiaomiMiMo/MiMo-7B-RL"
    target_objects = {
        "ukv1-0934a4e4e4ce8776_L0-36_H0.bin": 36 * 2 * PAGE_SIZE * 8 * 128 * 2
    }
    draft_objects = {"ukd1-044423b746c1f19a_D0-1_H0.bin": 2 * PAGE_SIZE * 8 * 128 * 2}

    @classmethod
    def setUpClass(cls):
        require_model_snapshot(cls.model)
        super().setUpClass()


if __name__ == "__main__":
    unittest.main(verbosity=2)
