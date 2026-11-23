"""TOP DeepSeek-V4.1 server checks: what validate_dsv41 refuses instead of serving another program."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.true_on_policy.dsv41 import validate_dsv41, validate_dsv41_radix

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# The release's layer schedule: two SWA-only layers, 18 ratio-2 and 20 ratio-1 layers.
_RATIOS = [0] * 2 + [2] * 18 + [1] * 20


def _radix(*, page_size=256, images=False, streaming_session=False, **fields):
    memory = dict(
        disable_radix_cache=False,
        enable_hierarchical_cache=False,
        enable_lmcache=False,
        enable_flexkv=False,
        radix_cache_backend=None,
        enable_unified_cache_external_linker=False,
        enable_session_radix_cache=False,
    )
    disagg = dict(
        disaggregation_mode="null", disaggregation_decode_retraction_backup=None
    )
    for name, value in fields.items():
        (memory if name in memory else disagg)[name] = value
    validate_dsv41_radix(
        compress_ratios=_RATIOS,
        page_size=page_size,
        images=images,
        streaming_session=streaming_session,
        memory=SimpleNamespace(**memory),
        disagg=SimpleNamespace(**disagg),
    )


class TestDsv41Validator(CustomTestCase):
    def test_another_model_type_is_refused(self):
        with self.assertRaisesRegex(NotImplementedError, "covers deepseek_v41 only"):
            validate_dsv41(SimpleNamespace(model_type="deepseek_v4"), None)

    def test_radix_reuse_from_the_device_tree_is_admitted(self):
        """The qualified path: the engine's own tree at 256-token pages, and anything with radix off."""
        _radix()
        _radix(disable_radix_cache=True, page_size=1, enable_hierarchical_cache=True)

    def test_a_hit_inside_a_compression_pair_is_refused(self):
        """Hits are page-aligned; a page that splits a ratio-2 pair lets a hit start on the pair's second token, whose
        partner sits in another request's ring."""
        for page_size in (1, 255):
            with self.subTest(page_size=page_size):
                with self.assertRaisesRegex(
                    NotImplementedError, "whole 2-token compression blocks"
                ):
                    _radix(page_size=page_size)

    def test_prefix_sources_outside_the_device_tree_are_refused(self):
        """Every other place a prefix can come from (host or external caches, sessions, a PD peer, image prompts)
        is unqualified and refused instead of served."""
        for source, field in (
            ("HiCache", dict(enable_hierarchical_cache=True)),
            ("LMCache", dict(enable_lmcache=True)),
            ("FlexKV", dict(enable_flexkv=True)),
            ("custom radix-cache backend", dict(radix_cache_backend="mine")),
            ("external cache linker", dict(enable_unified_cache_external_linker=True)),
            ("session radix caching", dict(enable_session_radix_cache=True)),
            ("streaming sessions", dict(streaming_session=True)),
            ("PD disaggregation", dict(disaggregation_mode="prefill")),
            (
                "host-pool retraction",
                dict(disaggregation_decode_retraction_backup="host_pool"),
            ),
            ("vision tower", dict(images=True)),
        ):
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    NotImplementedError, f"{source}.*NOT_QUALIFIED"
                ):
                    _radix(**field)


if __name__ == "__main__":
    unittest.main()
