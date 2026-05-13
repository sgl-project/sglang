import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.managers.prefix_cache_key import encode_prefix_cache_key_parts
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams


def _make_req(*, extra_key=None, lora_id=None):
    return Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=1),
        extra_key=extra_key,
        lora_id=lora_id,
    )


class TestPrefixCacheKey(unittest.TestCase):
    def test_base_extra_key_is_namespaced_without_lora(self):
        req = _make_req(extra_key="adapter-a")

        self.assertEqual(
            req.extra_key, encode_prefix_cache_key_parts([("extra_key", "adapter-a")])
        )

    def test_no_extra_key_and_no_lora_stays_none(self):
        req = _make_req()

        self.assertIsNone(req.extra_key)

    def test_lora_id_does_not_collide_with_base_extra_key(self):
        base_req = _make_req(extra_key="adapter-a")
        lora_req = _make_req(lora_id="adapter-a")

        self.assertNotEqual(base_req.extra_key, lora_req.extra_key)

    def test_encoded_lora_namespace_does_not_collide_with_base_extra_key(self):
        lora_req = _make_req(lora_id="adapter-a")
        base_req = _make_req(extra_key=lora_req.extra_key)

        self.assertNotEqual(base_req.extra_key, lora_req.extra_key)

    def test_base_request_does_not_hit_lora_prefix_cache_entry(self):
        cache = RadixCache.create_simulated()
        token_ids = [1, 2, 3, 4]
        lora_req = _make_req(lora_id="adapter-a")
        base_req = _make_req(extra_key=lora_req.extra_key)

        cache.insert(InsertParams(key=RadixKey(token_ids, lora_req.extra_key)))
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids, base_req.extra_key))
        )

        self.assertEqual(int(match.device_indices.numel()), 0)

    def test_extra_key_and_lora_id_boundaries_are_preserved(self):
        first = _make_req(extra_key="ab", lora_id="c")
        second = _make_req(extra_key="a", lora_id="bc")

        self.assertNotEqual(first.extra_key, second.extra_key)

    def test_type_error_identifies_component_name(self):
        with self.assertRaisesRegex(TypeError, "Prefix-cache namespace part lora_id"):
            encode_prefix_cache_key_parts([("lora_id", ["not", "a", "str"])])

    def test_no_parts_stays_none(self):
        self.assertIsNone(encode_prefix_cache_key_parts([]))
        self.assertIsNone(
            encode_prefix_cache_key_parts([("extra_key", None), ("lora_id", None)])
        )


if __name__ == "__main__":
    unittest.main()
