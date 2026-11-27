import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.true_on_policy.gdn_contract import validate_gdn_contract
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config():
    return SimpleNamespace(
        model_type="qwen3_5_text",
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        head_dim=256,
        hidden_act="silu",
    )


def _args(**overrides):
    fields = dict(
        page_size=64,
        mamba_track_interval=256,
        enable_int8_mamba_checkpoint=False,
        enable_linear_replayssm=False,
        enable_page_major_kv_layout=False,
        chunked_prefill_size=8192,
        enable_streaming_session=False,
        disaggregation_mode="null",
        disaggregation_decode_retraction_backup=None,
        enable_hierarchical_cache=False,
        disable_radix_cache=False,
        mamba_radix_cache_strategy="extra_buffer",
        enable_mixed_chunk=False,
        linear_attn_backend="triton",
        mamba_ssm_dtype="float32",
        linear_attn_prefill_backend=None,
        linear_attn_decode_backend=None,
        disable_shared_experts_fusion=True,
        enable_lmcache=False,
        enable_flexkv=False,
        radix_cache_backend=None,
        enable_unified_cache_external_linker=False,
        speculative_algorithm=None,
        speculative_eagle_topk=None,
        speculative_num_draft_tokens=None,
        pp_size=1,
        enable_two_batch_overlap=False,
        enable_linear_replayssm_spec=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _validate(args, *, context_len=262144, mode="chunked"):
    model_config = SimpleNamespace(hf_config=_config(), context_len=context_len)
    with (
        patch(
            "sglang.srt.arg_groups.model_override_base.model_config_of",
            return_value=model_config,
        ),
        # The FLA kernels were imported before the override; the import-order guard is not under test here.
        patch("sglang.srt.true_on_policy.gdn_contract.require_chunked_gdn_kernels"),
        envs.SGLANG_TOP_GDN_MODE.override(mode),
    ):
        validate_gdn_contract(args)


class TestGDNContract(unittest.TestCase):
    def test_the_chunked_program_is_required(self):
        """The mode pins the chunk prefill's kernel config at import; a process without it could autotune another."""
        with self.assertRaisesRegex(ValueError, "SGLANG_TOP_GDN_MODE=chunked"):
            _validate(_args(), mode="")

    def test_recipe_args_pass(self):
        _validate(_args())

    def test_chunked_prefill_off_passes(self):
        _validate(_args(chunked_prefill_size=-1))
        _validate(_args(chunked_prefill_size=None))

    def test_unaligned_chunked_prefill_is_refused(self):
        with self.assertRaisesRegex(ValueError, "64-aligned chunked prefill"):
            _validate(_args(chunked_prefill_size=1000))

    def test_external_state_caches_are_refused(self):
        """LMCache and FlexKV restore mamba states this engine did not build."""
        for overrides in (
            dict(enable_lmcache=True),
            dict(enable_flexkv=True),
            dict(radix_cache_backend="flexkv"),
        ):
            with self.subTest(**overrides):
                with self.assertRaisesRegex(ValueError, "LMCache, FlexKV"):
                    _validate(_args(**overrides))

    def test_external_cache_linker_is_refused(self):
        """The unified cache's external linker copies in states this engine did not build."""
        with self.assertRaisesRegex(ValueError, "external cache linkers"):
            _validate(_args(enable_unified_cache_external_linker=True))

    def test_decode_split_limits_are_checked_at_launch(self):
        """A page size that does not divide the split, or a context needing more pieces than FA's
        combine merges, is refused at launch, not at CUDA graph capture or mid-serving."""
        with self.assertRaisesRegex(ValueError, "multiple of the page size 192"):
            _validate(_args(page_size=192))
        _validate(_args(), context_len=256 * 4096)
        with self.assertRaisesRegex(ValueError, "more than 256 pieces"):
            _validate(_args(), context_len=256 * 4096 + 1)
        with envs.SGLANG_TOP_DECODE_KV_SPLIT.override(512):
            with self.assertRaisesRegex(ValueError, "more than 256 pieces"):
                _validate(_args(), context_len=262144)


if __name__ == "__main__":
    unittest.main()
