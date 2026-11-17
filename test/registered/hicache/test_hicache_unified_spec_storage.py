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
- DFLASH without its draft KV ring keeps the draft KV in a host pool of its
  own, whose pages are stored as chunks under the draft digest. A deployment
  without the draft reads the target chunks; one with it backfills.
- For both, the draft KV that other paths write: a batched prefill cut at
  the chunk size, decode, and a reload under eviction pressure.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_unified_spec_storage.py -v
"""

import io
import json
import math
import os
import re
import tempfile
import threading
import time
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
    get_gpu_count,
)

# The ring needs a draft attention backend that only Blackwell serves.
register_cuda_ci(est_time=4800, stage="nightly", runner_config="4-gpu-b200")


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
    # A floor between the accept length the draft reaches on the class's
    # first prompt and the one it falls to when a reload brings no draft KV
    # back; classes with a second prompt give that one its own. The EAGLE3
    # floor of hicache_spec_storage_common.py (7.0 of 8 draft tokens) is out
    # of reach of a one-layer MTP draft with 4. Kernel choices of FlashInfer's
    # autotuner move these accept lengths by a verify step or two, so a floor
    # sits several steps away from both sides.
    min_accept_length = None
    min_accept_length_second = None
    # Chunk coordinates and sizes of one page at the class's topology, and
    # the digests in front of them where they are pinned.
    target_chunks: dict = {}
    draft_chunks: dict = {}
    target_digest = None
    draft_digest = None
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

    @classmethod
    def setUpClass(cls):
        require_model_snapshot(cls.model)
        super().setUpClass()

    def scrub_prompt(self, seed):
        # Never the prompt under test: seeds are odd for NOTICE, even for
        # SCRUB_NOTICE, and the scrub prompt gets the next seed.
        return self.text_prompt(SCRUB_NOTICE if seed % 2 == 0 else NOTICE)

    def _pages(self, objects):
        """page hash -> {digest: {chunk coordinate: size}}"""
        pages = {}
        for name, (size, _) in objects.items():
            page, digest, coordinate = name[: -len(".bin")].split("_", 2)
            pages.setdefault(page, {}).setdefault(digest, {})[coordinate] = size
        return pages

    def _assert_untouched(self, objects, now):
        """Names, sizes and sha256 of earlier objects are what they were."""
        self.assertEqual({name: now.get(name) for name in objects}, objects)
        return now

    def _print_stats(self, result, label):
        verify_ct, accept_length = self.spec_stats(result)
        print(f"{type(self).__name__} {label}: {verify_ct=} {accept_length=:.3f}")
        return verify_ct, accept_length

    def _assert_draft_at_work(self, result, label, floor=None):
        """The draft is consulted and is right about some tokens, not all:
        above the prompt's floor, below the ceiling of a perfect draft."""
        verify_ct, accept_length = self._print_stats(result, label)
        self.assertGreater(verify_ct, 1)
        self.assertGreaterEqual(
            accept_length, self.min_accept_length if floor is None else floor
        )
        # A draft that is always right needs this many steps and no more.
        steps = math.ceil(self.max_new_tokens / self.draft_tokens)
        self.assertLess(accept_length, self.max_new_tokens / steps)

    def _assert_reloaded_elsewhere(self, result, prompt, reference, label, floor=None):
        """A hit in another process or topology: the target's output and the
        draft's work both hold."""
        self.assert_storage_hit(result)
        self.assert_close_output(result, reference)
        self.assert_kl_against_recompute(result, prompt)
        self._assert_draft_at_work(result, label, floor)
        self.assertGreaterEqual(
            self.spec_stats(result)[1], 0.9 * self.spec_stats(reference)[1]
        )

    def _assert_draft_lost(self, result, reference, label, floor=None):
        """A hit on pages whose draft chunks were zeroed: the output holds
        and the draft's work falls below the floor that a hit with draft KV
        has to reach."""
        self.assert_storage_hit(result)
        self.assert_close_output(result, reference)
        _, accept_length = self._print_stats(result, label)
        self.assertLess(
            accept_length, self.min_accept_length if floor is None else floor
        )

    def _audit_pages(self, objects, min_pages):
        """Every page holds exactly its target chunks and its draft chunks,
        each kind under one digest. Returns (target digest, draft digest)."""
        pages = self._pages(objects)
        self.assertGreaterEqual(len(pages), min_pages)
        digests = {digest for chunks in pages.values() for digest in chunks}
        (target,) = [d for d in digests if d.startswith("ukv1-")]
        (draft,) = [d for d in digests if d.startswith("ukd1-")]
        expected = {target: self.target_chunks, draft: self.draft_chunks}
        for page, chunks in pages.items():
            self.assertEqual(chunks, expected, page)
        if self.target_digest is not None:
            self.assertEqual((target, draft), (self.target_digest, self.draft_digest))
        return target, draft

    def _check_draft_kv_moves_between_tp2_and_tp1(self, spec_args, prompt_a, prompt_b):
        salt_a, salt_b = uuid.uuid4().hex, uuid.uuid4().hex
        tp2 = [*spec_args, "--tp-size", "2"]
        on_files = self.storage_backend == "file"
        second = self.min_accept_length_second

        self.launch_server(tp2)
        tp2_hit, _ = self.reload_in_process(prompt_a, salt_a, seed=1)
        self._assert_draft_at_work(tp2_hit, "TP2 device hit")
        if on_files:
            written = self.wait_for_backups()
            self._audit_pages(written, 2 * self.prompt_pages)

        # TP1 assembles its pages from both TP2 ranks' chunks.
        self.launch_server(spec_args)
        tp1_read = self.generate(prompt_a, salt_a)
        self._assert_reloaded_elsewhere(tp1_read, prompt_a, tp2_hit, "TP1 reads TP2")
        tp1_hit, _ = self.reload_in_process(prompt_b, salt_b, seed=2)
        self._assert_draft_at_work(tp1_hit, "TP1 device hit, second prompt", second)
        if on_files:
            written = self._assert_untouched(written, self.wait_for_backups())
            self._audit_pages(written, 3 * self.prompt_pages)

        # And each TP2 rank takes its half of what TP1 wrote.
        self.launch_server(tp2)
        tp2_read = self.generate(prompt_b, salt_b)
        self._assert_reloaded_elsewhere(
            tp2_read, prompt_b, tp1_hit, "TP2 reads TP1", second
        )
        if not on_files:
            return
        written = self._assert_untouched(written, self.wait_for_backups())
        _, draft = self._audit_pages(written, 3 * self.prompt_pages)

        # Control: with every draft object zeroed, each prompt still hits at
        # each TP size and the draft's work shows the loss.
        self.stop_server()
        self.zero_objects([name for name in written if f"_{draft}_" in name])
        self.launch_server(tp2)
        self._assert_draft_lost(
            self.generate(prompt_b, salt_b), tp2_read, "TP2 reads TP1, zeroed", second
        )
        self._assert_draft_lost(
            self.generate(prompt_a, salt_a), tp2_hit, "TP2 reads TP2, zeroed"
        )
        self.launch_server(spec_args)
        self._assert_draft_lost(
            self.generate(prompt_a, salt_a), tp1_read, "TP1 reads TP2, zeroed"
        )
        self._assert_draft_lost(
            self.generate(prompt_b, salt_b), tp1_hit, "TP1 reads TP1, zeroed", second
        )


class SameTopologyPackedNextn(PackedNextnMixin):
    def test_draft_kv_follows_its_pages_to_l3_and_back(self):
        prompt = self.text_prompt(NOTICE)
        salt = uuid.uuid4().hex
        on_files = self.storage_backend == "file"

        # 1. Same process: the L3 hit reproduces the device-cache hit, the
        # draft's work included.
        self.launch_server(self.nextn_args)
        device_hit, storage_hit = self.reload_in_process(prompt, salt, seed=1)
        self._assert_draft_at_work(device_hit, "device hit")
        self.assertEqual(self.spec_stats(storage_hit), self.spec_stats(device_hit))
        if on_files:
            written = self.wait_for_backups()
            target, draft = self._audit_pages(written, 2 * self.prompt_pages)

        # 2. After a restart the pages come back with their draft KV, and
        # reading leaves every object as it was.
        self.launch_server(self.nextn_args)
        self._assert_reloaded_elsewhere(
            self.generate(prompt, salt), prompt, device_hit, "after restart"
        )
        if on_files:
            written = self._assert_untouched(written, self.wait_for_backups())
            self._audit_pages(written, 2 * self.prompt_pages)

        # 3. Another draft: a miss on the draft chunk alone. The reader
        # recomputes, adds its own draft chunks and leaves the target's.
        other_draft = [*self.nextn_args, "--hicache-storage-draft-id", "another-draft"]
        self.launch_server(other_draft)
        missed = self.generate(prompt, salt)
        self.assertEqual(missed["meta_info"]["cached_tokens"], 0)
        self.flush_cache()
        self.assert_storage_hit(self.generate(prompt, salt))
        if not on_files:
            return
        now = self._assert_untouched(written, self.wait_for_backups())
        added = self._pages({name: now[name] for name in now if name not in written})
        backfilled_pages = 0
        for page, chunks in added.items():
            # Draft chunks under another digest, never the first draft's.
            (other,) = [d for d in chunks if d.startswith("ukd1-")]
            self.assertNotEqual(other, draft)
            self.assertEqual(chunks[other], self.draft_chunks, page)
            if len(chunks) == 1:
                backfilled_pages += 1
            else:
                # A page this reader was first to compute (the generated
                # tail can differ between a cold and a cached run).
                self.assertEqual(
                    chunks, {other: self.draft_chunks, target: self.target_chunks}
                )
        self.assertGreaterEqual(backfilled_pages, self.prompt_pages)

        # 4. Control: with every object of the first draft zeroed, the pages
        # still hit and the output holds, and the draft's work shows the
        # loss. Without this, step 1 would not prove that draft KV is what
        # came back.
        self.stop_server()
        for name in written:
            if f"_{draft}_" in name:
                path = os.path.join(self.storage_path, name)
                with open(path, "r+b") as f:
                    f.write(bytes(os.path.getsize(path)))
        self.launch_server(self.nextn_args)
        degraded = self.generate(prompt, salt)
        self.assert_storage_hit(degraded)
        self.assert_close_output(degraded, device_hit)
        verify_ct, accept_length = self.spec_stats(degraded)
        print(f"{type(self).__name__} zeroed drafts: {verify_ct=} {accept_length=:.3f}")
        self.assertNotEqual(self.spec_stats(degraded), self.spec_stats(device_hit))
        self.assertLess(accept_length, 0.9 * self.spec_stats(device_hit)[1])
        self.assertLess(accept_length, self.min_accept_length)


# MiMo-7B: 36 layers x 8 kv heads x 128 in bfloat16, one bundled MTP layer.
# Measured on B200, in verify steps for 96 tokens (accept length): the first
# prompt 45 (2.13), with the draft objects zeroed 51 to 52 (1.85 to 1.88);
# the second prompt 48 to 49 (1.96 to 2.00), zeroed 59 to 60 (1.60 to 1.63).
# The floors sit at 48 and at 54 steps.
MIMO = "XiaomiMiMo/MiMo-7B-RL"
MIMO_MIN_ACCEPT_LENGTH = 2.0
MIMO_MIN_ACCEPT_LENGTH_SECOND = 1.78
MIMO_LAYER_BYTES_PER_HEAD = 2 * PAGE_SIZE * 128 * 2


class TestUnifiedKeysPackedNextnMHA(SameTopologyPackedNextn, CustomTestCase):
    """One rank owns all 8 heads: one target chunk and one draft chunk a page."""

    model = MIMO
    min_accept_length = MIMO_MIN_ACCEPT_LENGTH
    target_chunks = {"L0-36_H0": 36 * 8 * MIMO_LAYER_BYTES_PER_HEAD}
    draft_chunks = {"D0-1_H0": 8 * MIMO_LAYER_BYTES_PER_HEAD}
    # The target digest is the one of a deployment without speculation.
    target_digest = "ukv1-0934a4e4e4ce8776"
    draft_digest = "ukd1-044423b746c1f19a"


class TestUnifiedKeysPackedNextnMooncake(SameTopologyPackedNextn, CustomTestCase):
    """The same flow on Mooncake, where a test cannot list or alter objects."""

    model = MIMO
    min_accept_length = MIMO_MIN_ACCEPT_LENGTH
    storage_backend = "mooncake"
    expected_storage_backend = "MooncakeStore"


class TestUnifiedKeysPackedNextnMLA(SameTopologyPackedNextn, CustomTestCase):
    """GLM-4.7-Flash: 47 layers of 576-wide latent KV, replicated across
    ranks, so the chunks carry no head coordinate. Measured on B200: accept
    length 2.74, and 2.18 with the draft objects zeroed."""

    model = "zai-org/GLM-4.7-Flash"
    min_accept_length = 2.5
    nextn_args = [
        *PackedNextnMixin.nextn_args,
        "--attention-backend",
        "trtllm_mla",
    ]
    target_chunks = {"L0-47": 47 * PAGE_SIZE * 576 * 2}
    draft_chunks = {"D0-1": PAGE_SIZE * 576 * 2}

    @classmethod
    def setUpClass(cls):
        if not is_sm100_supported():
            raise unittest.SkipTest("trtllm_mla needs Blackwell")
        super().setUpClass()


@unittest.skipUnless(get_gpu_count() >= 2, "Requires 2 GPUs for the TP2 server")
class TestUnifiedKeysPackedNextnCrossTP(PackedNextnMixin, CustomTestCase):
    """8 kv heads in 2-head chunks: a TP2 rank owns 2 of them, a TP1 rank 4,
    for the target layers and for the draft layer alike."""

    model = MIMO
    min_accept_length = MIMO_MIN_ACCEPT_LENGTH
    min_accept_length_second = MIMO_MIN_ACCEPT_LENGTH_SECOND
    unified_server_args = ["--hicache-storage-head-partition", "2"]
    target_chunks = {
        f"L0-36_H{g}": 36 * 2 * MIMO_LAYER_BYTES_PER_HEAD for g in range(4)
    }
    draft_chunks = {f"D0-1_H{g}": 2 * MIMO_LAYER_BYTES_PER_HEAD for g in range(4)}

    def test_draft_kv_moves_between_tp2_and_tp1(self):
        self._check_draft_kv_moves_between_tp2_and_tp1(
            self.nextn_args, self.text_prompt(NOTICE), self.text_prompt(SCRUB_NOTICE)
        )


class TestUnifiedKeysPackedNextnCrossTPMooncake(TestUnifiedKeysPackedNextnCrossTP):
    """The same flow on Mooncake: each rank reads and writes its chunks of
    the target layers and of the draft layer through its own client."""

    storage_backend = "mooncake"
    expected_storage_backend = "MooncakeStore"


# Llama-3.1-8B: 32 layers x 8 kv heads x 128 in bfloat16. Its DFLASH draft: 5
# layers of the same row geometry in a KV pool of its own, blocks of 10 tokens.
LLAMA_LAYER_BYTES_PER_HEAD = 2 * PAGE_SIZE * 128 * 2


class DFlashSidecarMixin(PackedNextnMixin):
    """DFLASH without its draft KV ring: target chunks, and the draft KV
    pool's page as chunks of its own."""

    model = DEFAULT_TARGET_MODEL_DFLASH
    prompt_pages = 12
    # The tree keys on single tokens: one token past the last full page.
    prompt_extra_tokens = 1
    draft_tokens = 10
    dflash_args = [
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DEFAULT_DRAFT_MODEL_DFLASH,
    ]

    @classmethod
    def setUpClass(cls):
        require_model_snapshot(DEFAULT_DRAFT_MODEL_DFLASH)
        super().setUpClass()

    def scrub_prompt(self, seed):
        return self.ledger_prompt(seed + 1000)


class SameTopologyDFlashSidecar(DFlashSidecarMixin):
    longer_pages = 16

    def _assert_only_backfilled(self, before, now, draft, min_pages):
        """What a draft reader added: its draft chunks on pages that had the
        target's, and whole pages that it was first to compute."""
        added = self._pages({name: now[name] for name in now if name not in before})
        backfilled_pages = 0
        for page, chunks in added.items():
            if len(chunks) == 1:
                self.assertEqual(chunks, {draft: self.draft_chunks}, page)
                backfilled_pages += 1
            else:
                self.assertEqual(
                    chunks,
                    {draft: self.draft_chunks, self.target_digest: self.target_chunks},
                    page,
                )
        self.assertGreaterEqual(backfilled_pages, min_pages)

    def test_draft_pool_follows_its_pages_to_l3_and_back(self):
        prompt = self.ledger_prompt(seed=1)
        longer = self.ledger_prompt(seed=1, pages=self.longer_pages)
        self.assertEqual(longer[: len(prompt)], prompt)
        salt = uuid.uuid4().hex
        on_files = self.storage_backend == "file"
        continued_pages = self.longer_pages - self.prompt_pages

        # 1. In one process an L3 reload reproduces the device-cache hit, the
        # draft's work included.
        self.launch_server(self.dflash_args)
        self.assertFalse(self.server_info()["speculative_dflash_draft_kv_ring"])
        device_hit, storage_hit = self.reload_in_process(prompt, salt, seed=1)
        self._assert_draft_at_work(device_hit, "device hit")
        self.assertEqual(self.spec_stats(storage_hit), self.spec_stats(device_hit))
        if on_files:
            written = self.wait_for_backups()
            target, draft = self._audit_pages(written, 2 * self.prompt_pages)

        # 2. After a restart the pages come back with their draft KV, and
        # reading leaves every object as it was.
        self.launch_server(self.dflash_args)
        self._assert_reloaded_elsewhere(
            self.generate(prompt, salt), prompt, device_hit, "after restart"
        )
        if on_files:
            written = self._assert_untouched(written, self.wait_for_backups())
            self._audit_pages(written, 2 * self.prompt_pages)

        # 3. A deployment without the draft reads the target chunks of these
        # pages, and continues the prompt with pages of target chunks alone.
        self.launch_server([])
        plain = self.generate(longer, salt)
        self.assert_storage_hit(plain)
        self.assertNotIn("spec_verify_ct", plain["meta_info"])
        self.assert_kl_against_recompute(plain, longer)
        if on_files:
            with_plain = self._assert_untouched(written, self.wait_for_backups())
            plain_pages = self._pages(
                {name: with_plain[name] for name in with_plain if name not in written}
            )
            self.assertGreaterEqual(len(plain_pages), continued_pages)
            for page, chunks in plain_pages.items():
                self.assertEqual(chunks, {target: self.target_chunks}, page)

        # 4. The draft deployment takes from storage exactly the pages that
        # have draft chunks, computes the rest and adds their draft chunks.
        self.launch_server(self.dflash_args)
        partial = self.generate(longer, salt)
        self.assert_storage_hit(partial)
        self.assert_kl_against_recompute(partial, longer)
        if on_files:
            backfilled = self._assert_untouched(with_plain, self.wait_for_backups())
            self._assert_only_backfilled(with_plain, backfilled, draft, continued_pages)
        self.flush_cache()
        self.assert_storage_hit(self.generate(longer, salt), self.longer_pages)
        if on_files:
            written = self._assert_untouched(backfilled, self.wait_for_backups())

        # 5. Another draft: a miss on the draft chunks alone. The reader
        # recomputes, adds its own draft chunks and leaves everything else.
        other_draft = [*self.dflash_args, "--hicache-storage-draft-id", "another"]
        self.launch_server(other_draft)
        missed = self.generate(prompt, salt)
        self.assertEqual(missed["meta_info"]["cached_tokens"], 0)
        self.flush_cache()
        self.assert_storage_hit(self.generate(prompt, salt))
        if not on_files:
            return
        now = self._assert_untouched(written, self.wait_for_backups())
        others = {
            digest
            for name in now
            if name not in written
            for digest in [name.split("_")[1]]
            if digest.startswith("ukd1-")
        }
        self.assertEqual(len(others), 1)
        self.assertNotIn(draft, others)
        self._assert_only_backfilled(written, now, others.pop(), self.prompt_pages)

        # 6. Control: with every object of the first draft zeroed, the pages
        # still hit and the output holds, and the draft's work shows the
        # loss. Without this, step 1 would not prove that draft KV is what
        # came back.
        self.stop_server()
        for name in written:
            if f"_{draft}_" in name:
                path = os.path.join(self.storage_path, name)
                with open(path, "r+b") as f:
                    f.write(bytes(os.path.getsize(path)))
        self.launch_server(self.dflash_args)
        degraded = self.generate(prompt, salt)
        self.assert_storage_hit(degraded)
        self.assert_close_output(degraded, device_hit)
        verify_ct, accept_length = self.spec_stats(degraded)
        print(f"{type(self).__name__} zeroed drafts: {verify_ct=} {accept_length=:.3f}")
        self.assertLess(accept_length, 0.9 * self.spec_stats(device_hit)[1])
        self.assertLess(accept_length, self.min_accept_length)


# Measured on B200, in verify steps for 96 tokens (accept length): the first
# prompt 28 to 32 (3.00 to 3.43), with the draft objects zeroed 53 to 55 (1.75
# to 1.81); the second prompt 21 (4.57), zeroed 32 to 33 (2.91 to 3.00). The
# floors sit at 40 and at 26 steps.
DFLASH_MIN_ACCEPT_LENGTH = 2.4
DFLASH_MIN_ACCEPT_LENGTH_SECOND = 3.7


class TestUnifiedKeysDFlashSidecar(SameTopologyDFlashSidecar, CustomTestCase):
    """One rank owns all 8 heads of the target and of the draft."""

    min_accept_length = DFLASH_MIN_ACCEPT_LENGTH
    target_chunks = {"L0-32_H0": 32 * 8 * LLAMA_LAYER_BYTES_PER_HEAD}
    draft_chunks = {"D0-5_H0": 5 * 8 * LLAMA_LAYER_BYTES_PER_HEAD}
    # The target digest is the one of a deployment without speculation.
    target_digest = "ukv1-b440160febf92487"
    draft_digest = "ukd1-00b6f6f02b29851b"


class TestUnifiedKeysDFlashSidecarMooncake(SameTopologyDFlashSidecar, CustomTestCase):
    """The same flow on Mooncake, where a test cannot list or alter objects."""

    min_accept_length = DFLASH_MIN_ACCEPT_LENGTH
    storage_backend = "mooncake"
    expected_storage_backend = "MooncakeStore"


@unittest.skipUnless(get_gpu_count() >= 2, "Requires 2 GPUs for the TP2 server")
class TestUnifiedKeysDFlashSidecarCrossTP(DFlashSidecarMixin, CustomTestCase):
    """8 kv heads in 2-head chunks on both grids: a TP2 rank owns 2 chunks of
    the target and 2 of the draft pool, a TP1 rank 4 of each. Measured on
    B200: accept length 3.20 at TP2, 3.43 at TP1 on TP2's pages, 4.57 at TP2
    on TP1's pages of the second prompt."""

    min_accept_length = DFLASH_MIN_ACCEPT_LENGTH
    min_accept_length_second = DFLASH_MIN_ACCEPT_LENGTH_SECOND
    unified_server_args = ["--hicache-storage-head-partition", "2"]
    target_chunks = {
        f"L0-32_H{g}": 32 * 2 * LLAMA_LAYER_BYTES_PER_HEAD for g in range(4)
    }
    draft_chunks = {f"D0-5_H{g}": 5 * 2 * LLAMA_LAYER_BYTES_PER_HEAD for g in range(4)}

    def test_draft_pool_moves_between_tp2_and_tp1(self):
        self._check_draft_kv_moves_between_tp2_and_tp1(
            self.dflash_args, self.ledger_prompt(seed=1), self.ledger_prompt(seed=2)
        )


class TestUnifiedKeysDFlashSidecarCrossTPMooncake(TestUnifiedKeysDFlashSidecarCrossTP):
    """The same flow on Mooncake: the draft pool's chunks of each rank go
    through the store's second registered buffer."""

    storage_backend = "mooncake"
    expected_storage_backend = "MooncakeStore"


class DraftKVPathsMixin(PackedNextnMixin):
    """Draft KV that is not written by one whole-prompt prefill: batched and
    chunked prefill, decode, and pages that return under eviction pressure.

    Each test has a file store of its own. Subclasses name the speculative
    arguments, two prompts and the chunk size that cuts their prompts.
    """

    spec_args: list = []
    chunked_prefill_size = None
    # Tokens generated to fill pages with rows written during decode: 16
    # pages, so that the prompt's extra tokens still follow the last page.
    decode_tokens = 16 * PAGE_SIZE
    # A small device pool, so that a few prompts push earlier ones out of
    # the device and host tiers.
    pool_tokens = 8192
    hicache_ratio = 2  # of launch_server

    def setUp(self):
        super().setUp()
        self.use_new_store()

    def first_prompt(self):
        raise NotImplementedError

    def second_prompt(self):
        raise NotImplementedError

    def test_draft_kv_of_a_batched_chunked_prefill(self):
        prompts = [self.ledger_prompt(seed=seed) for seed in (11, 12, 13)]
        salt = uuid.uuid4().hex

        # 1. Reference: one request at a time, each prompt in one forward.
        self.launch_server(self.spec_args)
        reference = [self.compute_and_hit(prompt, salt) for prompt in prompts]
        reference_objects = self.wait_for_backups()

        # 2. In another store the prompts arrive as one batch behind a long
        # prompt that holds the scheduler, and are prefilled together in
        # forwards cut at the chunk size.
        reference_store = self.use_new_store()
        log = io.StringIO()
        self.launch_server(
            self.spec_args,
            ["--chunked-prefill-size", str(self.chunked_prefill_size)],
            log=log,
        )
        blocker = threading.Thread(
            target=self.generate,
            args=(self.ledger_prompt(seed=77, pages=100), uuid.uuid4().hex),
        )
        blocker.start()
        time.sleep(0.05)
        computed = self.generate(prompts, salt)
        blocker.join()
        for result in computed:
            self.assertEqual(result["meta_info"]["cached_tokens"], 0)
        batches = [
            (int(seqs), int(tokens))
            for seqs, tokens in re.findall(
                r"Prefill batch.*?#new-seq: (\d+), #new-token: (\d+)", log.getvalue()
            )
        ]
        print(f"{type(self).__name__} prefill batches: {batches}")
        # At least two of the prompts in one forward, one of them cut: three
        # sequences, of which the blocker's tail can be one, with fewer
        # tokens than two whole prompts.
        self.assertTrue(
            any(
                seqs >= 3 and tokens < (seqs - 1) * len(prompts[0])
                for seqs, tokens in batches
            ),
            batches,
        )

        # 3. What that prefill stored comes back from L3 as it sits on the
        # device, and the draft does the work it does in the reference.
        device_hits = [
            self.assert_device_hit(self.generate(prompt, salt)) for prompt in prompts
        ]
        self.flush_cache()
        self.generate(self.scrub_prompt(8), uuid.uuid4().hex)
        self.flush_cache()
        for i, (prompt, device_hit) in enumerate(zip(prompts, device_hits)):
            storage_hit = self.generate(prompt, salt)
            self.assert_storage_hit(storage_hit)
            self.assert_same_output(storage_hit, device_hit)
            self.assertEqual(self.spec_stats(storage_hit), self.spec_stats(device_hit))
            self._print_stats(reference[i], f"prompt {i}, one request per forward")
            self._print_stats(device_hit, f"prompt {i}, batched and chunked")
            self.assertGreaterEqual(
                self.spec_stats(device_hit)[1], 0.9 * self.spec_stats(reference[i])[1]
            )
        batched_objects = self.wait_for_backups()

        # 4. Bytes: every row that both stores hold was computed twice from
        # the same tokens, so the two agree up to rounding.
        shared = set(reference_objects) & set(batched_objects)
        pages = self._pages({name: batched_objects[name] for name in shared})
        self.assertGreaterEqual(len(pages), len(prompts) * self.prompt_pages)
        for kind in ("ukv1-", "ukd1-"):
            names = [name for name in shared if name.split("_")[1].startswith(kind)]
            share, lowest = self.matching_rows(
                names, reference_store, self.storage_path
            )
            print(f"{type(self).__name__} {kind} rows: {share=:.4f} {lowest=:.4f}")
            self.assertGreaterEqual(share, 0.99, kind)
            self.assertGreaterEqual(lowest, 0.8, kind)

    def test_draft_rows_written_during_decode_come_back(self):
        prompt = self.first_prompt()
        salt = uuid.uuid4().hex

        # 1. The prompt's pages, then a long decode on top of them: its
        # tokens fill pages whose target and draft rows decode wrote.
        self.launch_server(self.spec_args)
        self.generate(prompt, salt, max_new_tokens=1)
        prompt_objects = self.wait_for_backups()
        self.assertEqual(len(self._pages(prompt_objects)), self.prompt_pages)
        decoded = self.assert_device_hit(
            self.generate(prompt, salt, max_new_tokens=self.decode_tokens)
        )
        continued = prompt + decoded["output_ids"]
        pages = len(continued) // PAGE_SIZE
        self.assertEqual(len(continued) % PAGE_SIZE, self.prompt_extra_tokens)
        written = self._assert_untouched(prompt_objects, self.wait_for_backups())
        decode_objects = [name for name in written if name not in prompt_objects]
        self.assertEqual(
            len(self._pages({name: written[name] for name in decode_objects})),
            pages - self.prompt_pages,
        )
        _, draft = self._audit_pages(written, pages)
        device_hit = self.assert_device_hit(self.generate(continued, salt), pages)
        honest = self._print_stats(device_hit, "continued, device hit")
        written = self.wait_for_backups()

        # 2. After a restart those pages come from L3, draft rows included.
        self.launch_server(self.spec_args)
        reloaded = self.generate(continued, salt)
        self.assert_storage_hit(reloaded, pages)
        self.assert_close_output(reloaded, device_hit)
        self.assert_kl_against_recompute(reloaded, continued)
        after_restart = self._print_stats(reloaded, "continued, after restart")
        self.assertGreaterEqual(after_restart[1], 0.9 * honest[1])
        self._assert_untouched(written, self.wait_for_backups())

        # 3. Bytes: the same tokens prefilled as a prompt, in another store.
        # Decode and prefill run different kernels, whose rounding grows
        # with depth and along the sequence, so most rows agree, not all.
        decode_store = self.use_new_store()
        self.launch_server(self.spec_args)
        cold = self.generate(continued, salt)
        self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
        prefilled = self.wait_for_backups()
        self.assertLessEqual(set(decode_objects), set(prefilled))
        share, lowest = self.matching_rows(
            [name for name in decode_objects if f"_{draft}_" in name],
            decode_store,
            self.storage_path,
        )
        print(
            f"{type(self).__name__} decode-written draft rows: {share=:.4f} {lowest=:.4f}"
        )
        self.assertGreaterEqual(share, 0.8)

        # 4. Control on the first store: with the draft chunks of only the
        # decode-written pages zeroed, the pages still hit and the draft's
        # work shows the loss.
        self.stop_server()
        type(self).storage_path = decode_store
        self.zero_objects([name for name in decode_objects if f"_{draft}_" in name])
        self.launch_server(self.spec_args)
        degraded = self.generate(continued, salt)
        self.assert_storage_hit(degraded, pages)
        self.assert_close_output(degraded, device_hit)
        zeroed = self._print_stats(degraded, "continued, decode-page drafts zeroed")
        self.assertLess(zeroed[1], 0.95 * honest[1])

    def test_two_prompts_recover_under_eviction_pressure(self):
        self.launch_server(
            self.spec_args, ["--max-total-tokens", str(self.pool_tokens)]
        )
        hits = []
        for i, prompt in enumerate((self.first_prompt(), self.second_prompt())):
            salt = uuid.uuid4().hex
            hit = self.compute_and_hit(prompt, salt)
            self._print_stats(hit, f"prompt {i}, device hit")
            hits.append((prompt, salt, hit))
        # Each prompt has draft work of its own to recover.
        self.assertNotEqual(self.spec_stats(hits[0][2]), self.spec_stats(hits[1][2]))

        # More prompts than the host tier holds push both out of it.
        host_pages = self.hicache_ratio * self.pool_tokens // PAGE_SIZE
        for seed in range(math.ceil(1.2 * host_pages / (self.prompt_pages + 1))):
            self.generate(self.ledger_prompt(seed=2000 + seed), uuid.uuid4().hex)

        for i, (prompt, salt, hit) in enumerate(hits):
            recovered = self.generate(prompt, salt)
            self.assert_storage_hit(recovered)
            self.assert_same_output(recovered, hit)
            self._print_stats(recovered, f"prompt {i}, after eviction")
            self.assertEqual(self.spec_stats(recovered), self.spec_stats(hit))


class TestUnifiedKeysPackedNextnPaths(DraftKVPathsMixin, CustomTestCase):
    model = MIMO
    min_accept_length = MIMO_MIN_ACCEPT_LENGTH
    spec_args = PackedNextnMixin.nextn_args
    # Prompts of 20 pages and 2 tokens: three of them need two forwards.
    chunked_prefill_size = 2048
    target_chunks = {"L0-36_H0": 36 * 8 * MIMO_LAYER_BYTES_PER_HEAD}
    draft_chunks = {"D0-1_H0": 8 * MIMO_LAYER_BYTES_PER_HEAD}

    def first_prompt(self):
        return self.text_prompt(NOTICE)

    def second_prompt(self):
        return self.text_prompt(SCRUB_NOTICE)


class TestUnifiedKeysDFlashSidecarPaths(
    DraftKVPathsMixin, DFlashSidecarMixin, CustomTestCase
):
    min_accept_length = DFLASH_MIN_ACCEPT_LENGTH
    spec_args = DFlashSidecarMixin.dflash_args
    # Prompts of 12 pages and 1 token: three of them need three forwards.
    chunked_prefill_size = 1024
    target_chunks = {"L0-32_H0": 32 * 8 * LLAMA_LAYER_BYTES_PER_HEAD}
    draft_chunks = {"D0-5_H0": 5 * 8 * LLAMA_LAYER_BYTES_PER_HEAD}

    def first_prompt(self):
        return self.ledger_prompt(seed=1)

    def second_prompt(self):
        return self.ledger_prompt(seed=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
