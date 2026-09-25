"""HiCache round trips for native Beam: a prefix evicted to host (or flushed to
file storage) must reload with the cold run's output, across the sliding window."""

import os
import tempfile
import unittest

import torch

import sglang as sgl
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import (
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
)
from sglang.test.beam_fixture import (
    assert_logprobs_close,
    eager_beam_logits,
    make_beam_fixture,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Cover FMHA-v2/XQA on Hopper and TRT-LLM-GEN on Blackwell; the PP2 class needs
# two GPUs and skips on the single-GPU runner.
# ci-cost-override: the smallest Blackwell runner has four GPUs.
register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")
register_cuda_ci(est_time=300, stage="base-b", runner_config="4-gpu-b200")

# The window is 512 tokens: 510 fits inside it, 513 and 1200 cross it, and
# 1200 leaves most of its sliding-window history outside the restored window.
ROUND_TRIP_LENGTHS = (510, 513, 1200)
LONG_PROMPT = 1200
PARTIAL_PREFIX = 700
FIRST_TURN = 600
FIRST_TURN_DECODE = 128
# Four SWA eviction intervals (128 tokens) past the window.
LONG_DECODE = 600
BATCH_LENGTHS = (513, 900, 1200)
# The device pool holds 4096 tokens; one filler of this size cannot fit beside
# any cached prompt, so it pushes every unlocked prefix to the host tier.
DEVICE_TOKENS = 4096
EVICTION_FILLER = 3800
PAGE_SIZE = 64


def prompt(length, offset=0):
    return [1 + ((i + offset) * 17) % 127 for i in range(length)]


class TestBeamHiCache(CustomTestCase):
    hicache_args = dict(hicache_io_backend="kernel", hicache_mem_layout="page_first")
    extra_args = {}
    min_gpus = 1

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Beam HiCache test requires a CUDA GPU")
        if not is_flashinfer_available() or not (
            is_sm90_supported() or is_sm100_supported() or is_sm120_supported()
        ):
            raise unittest.SkipTest(
                "Beam TRT-LLM MHA test requires FlashInfer and SM90, SM100, or SM120"
            )
        if torch.cuda.device_count() < cls.min_gpus:
            raise unittest.SkipTest(f"requires {cls.min_gpus} GPUs")
        cls.checkpoint = tempfile.TemporaryDirectory(prefix="beam-fixture-")
        cls.config, state = make_beam_fixture(cls.checkpoint.name)
        cls.state = {name: value.cuda() for name, value in state.items()}
        cls.filler_offset = 20000
        cls.engine = sgl.Engine(
            model_path=cls.checkpoint.name,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            tp_size=1,
            attention_backend="trtllm_mha",
            page_size=PAGE_SIZE,
            chunked_prefill_size=256,
            max_total_tokens=DEVICE_TOKENS,
            mem_fraction_static=0.2,
            enable_hierarchical_cache=True,
            hicache_ratio=4,
            hicache_write_policy="write_through",
            **cls.hicache_args,
            **cls.extra_args,
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "engine", None) is not None:
            cls.engine.shutdown()
        if hasattr(cls, "state"):
            del cls.state
        if hasattr(cls, "checkpoint"):
            cls.checkpoint.cleanup()

    def setUp(self):
        self.engine.flush_cache()

    def generate(self, token_ids, count):
        batched = isinstance(token_ids[0], list)
        results = self.engine.generate(
            input_ids=token_ids,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": count,
                "ignore_eos": True,
            },
            return_logprob=True,
            token_ids_logprob=list(range(self.config.vocab_size)),
        )
        if not batched:
            results = [results]
        outputs = []
        for result in results:
            meta = result["meta_info"]
            logprobs = torch.tensor(
                [
                    [entry[0] for entry in row]
                    for row in meta["output_token_ids_logprobs"]
                ],
                device="cuda",
            )
            details = meta.get("cached_tokens_details") or {}
            outputs.append(
                (
                    result["output_ids"],
                    logprobs,
                    meta["cached_tokens"],
                    {
                        tier: int(details.get(tier, 0))
                        for tier in ("device", "host", "storage")
                    },
                )
            )
        return outputs if batched else outputs[0]

    def oracle(self, token_ids, generated):
        logits = eager_beam_logits(self.config, self.state, token_ids + generated[:-1])
        return logits[len(token_ids) - 1 :].log_softmax(-1)

    def evict_to_host(self):
        type(self).filler_offset += EVICTION_FILLER
        _, _, cached, _ = self.generate(
            prompt(EVICTION_FILLER, offset=self.filler_offset), 1
        )
        self.assertEqual(cached, 0, "filler unexpectedly hit the cache")

    def assert_host_hit(self, tiers, tag):
        self.assertGreater(tiers["host"], 0, f"{tag}: no host-tier hit, got {tiers}")

    def test_hybrid_swa_memory_stays_enabled(self):
        info = self.engine.get_server_info()
        self.assertTrue(info["enable_hierarchical_cache"])
        self.assertFalse(info["disable_hybrid_swa_memory"])
        self.assertLess(info["swa_full_tokens_ratio"], 1.0)

    def test_round_trip_across_window(self):
        for length in ROUND_TRIP_LENGTHS:
            with self.subTest(prompt=length):
                self.engine.flush_cache()
                cold_ids, cold_logprobs, cached, _ = self.generate(prompt(length), 4)
                self.assertEqual(cached, 0)
                self.evict_to_host()
                ids, logprobs, cached, tiers = self.generate(prompt(length), 4)
                self.assert_host_hit(tiers, f"prompt={length}")
                self.assertEqual(cached, tiers["device"] + tiers["host"])
                self.assertEqual(ids, cold_ids)
                assert_logprobs_close(
                    self,
                    logprobs,
                    cold_logprobs,
                    tag=f"host hit vs cold prompt={length}",
                )
                assert_logprobs_close(
                    self,
                    logprobs,
                    self.oracle(prompt(length), ids),
                    tag=f"host hit vs oracle prompt={length}",
                )

    def test_partial_host_hit(self):
        self.generate(prompt(LONG_PROMPT), 4)
        self.evict_to_host()
        branch = prompt(PARTIAL_PREFIX) + prompt(64, offset=5000)
        ids, logprobs, cached, tiers = self.generate(branch, 4)
        self.assert_host_hit(tiers, "partial")
        # The shared prefix comes back in whole pages, ending inside the window.
        self.assertEqual(cached, PARTIAL_PREFIX // PAGE_SIZE * PAGE_SIZE)
        assert_logprobs_close(
            self, logprobs, self.oracle(branch, ids), tag="partial host hit"
        )

    def test_decode_tokens_served_from_host(self):
        first = prompt(FIRST_TURN)
        first_ids, _, _, _ = self.generate(first, FIRST_TURN_DECODE)
        self.evict_to_host()
        second = first + first_ids + prompt(64, offset=7000)
        ids, logprobs, cached, tiers = self.generate(second, 8)
        self.assert_host_hit(tiers, "second turn")
        # The first turn's decoded tokens were cached too, not just its prompt.
        self.assertGreater(cached, FIRST_TURN)
        assert_logprobs_close(
            self, logprobs, self.oracle(second, ids), tag="second turn over host"
        )

    def test_long_decode_after_load_back(self):
        # The cold run's distance to the oracle is test_beam_cache.py's gate;
        # here the host tier must not change the decode that follows it.
        cold_ids, cold_logprobs, _, _ = self.generate(prompt(LONG_PROMPT), LONG_DECODE)
        self.assertEqual(len(cold_ids), LONG_DECODE)
        self.evict_to_host()
        ids, logprobs, _, tiers = self.generate(prompt(LONG_PROMPT), LONG_DECODE)
        self.assert_host_hit(tiers, "long decode")
        self.assertEqual(ids, cold_ids)
        assert_logprobs_close(
            self,
            logprobs,
            cold_logprobs,
            tag=f"decode {LONG_DECODE} tokens after load-back vs cold",
        )

    def test_batched_load_back(self):
        prompts = [
            prompt(length, offset=300 * i) for i, length in enumerate(BATCH_LENGTHS)
        ]
        cold = self.generate(prompts, 8)
        self.evict_to_host()
        batch = self.generate(prompts, 8)
        for length, (ids, logprobs, _, _), (batch_ids, batch_logprobs, _, tiers) in zip(
            BATCH_LENGTHS, cold, batch
        ):
            self.assert_host_hit(tiers, f"batched prompt={length}")
            self.assertEqual(batch_ids, ids, f"batched ids differ at prompt={length}")
            assert_logprobs_close(
                self, batch_logprobs, logprobs, tag=f"batched load-back prompt={length}"
            )


class TestBeamHiCacheDirectIO(TestBeamHiCache):
    hicache_args = dict(
        hicache_io_backend="direct", hicache_mem_layout="page_first_direct"
    )


class TestBeamHiCachePP2(TestBeamHiCache):
    """Regression: host load-back on the second PP stage crashed the engine."""

    min_gpus = 2
    extra_args = dict(pp_size=2)


class TestBeamHiCacheFileStorage(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Beam HiCache test requires a CUDA GPU")
        if not is_flashinfer_available() or not (
            is_sm90_supported() or is_sm100_supported() or is_sm120_supported()
        ):
            raise unittest.SkipTest(
                "Beam TRT-LLM MHA test requires FlashInfer and SM90, SM100, or SM120"
            )
        cls.storage = tempfile.TemporaryDirectory(prefix="beam-hicache-l3-")
        cls.previous_dir = os.environ.get("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR")
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = cls.storage.name
        cls.checkpoint = tempfile.TemporaryDirectory(prefix="beam-fixture-")
        cls.config, state = make_beam_fixture(cls.checkpoint.name)
        cls.state = {name: value.cuda() for name, value in state.items()}
        cls.engine = sgl.Engine(
            model_path=cls.checkpoint.name,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            tp_size=1,
            attention_backend="trtllm_mha",
            page_size=PAGE_SIZE,
            chunked_prefill_size=256,
            max_total_tokens=DEVICE_TOKENS,
            mem_fraction_static=0.2,
            enable_hierarchical_cache=True,
            hicache_ratio=4,
            hicache_write_policy="write_through",
            hicache_io_backend="direct",
            hicache_mem_layout="page_first_direct",
            hicache_storage_backend="file",
            hicache_storage_prefetch_policy="wait_complete",
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "engine", None) is not None:
            cls.engine.shutdown()
        if hasattr(cls, "previous_dir"):
            if cls.previous_dir is None:
                os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", None)
            else:
                os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = cls.previous_dir
        if hasattr(cls, "state"):
            del cls.state
        for name in ("checkpoint", "storage"):
            if hasattr(cls, name):
                getattr(cls, name).cleanup()

    generate = TestBeamHiCache.generate
    oracle = TestBeamHiCache.oracle

    def test_storage_hit_after_flush(self):
        self.engine.flush_cache()
        cold_ids, cold_logprobs, _, _ = self.generate(prompt(LONG_PROMPT), 4)
        # Flushing drops the device and host tiers; the file tier keeps the pages.
        self.engine.flush_cache()
        ids, logprobs, cached, tiers = self.generate(prompt(LONG_PROMPT), 4)
        self.assertGreater(tiers["storage"], 0, f"no storage-tier hit, got {tiers}")
        self.assertGreaterEqual(
            cached, LONG_PROMPT // PAGE_SIZE * PAGE_SIZE - PAGE_SIZE
        )
        self.assertEqual(ids, cold_ids)
        assert_logprobs_close(self, logprobs, cold_logprobs, tag="storage hit vs cold")
        assert_logprobs_close(
            self,
            logprobs,
            self.oracle(prompt(LONG_PROMPT), ids),
            tag="storage hit vs oracle",
        )


if __name__ == "__main__":
    unittest.main()
