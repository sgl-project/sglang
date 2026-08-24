import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.mem_cache.allocation_sizing import (
    get_req_to_token_extra_context_len,
)
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dvr.server_args import (
    DVR_DFLASH_SPECULATIVE_ALGORITHM,
    DVR_EAGLE_SPECULATIVE_ALGORITHM,
    DVR_SPECULATIVE_ALGORITHM,
    _handle_dvr_speculative_decoding,
    _is_dvr_gated_linear_state_model,
    handle_dvr_cuda_graph_config,
    handle_dvr_defaults,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Args:
    speculative_algorithm = DVR_SPECULATIVE_ALGORITHM
    device = "cuda"
    enable_dp_attention = False
    enable_pdmux = False
    disaggregation_mode = "null"
    decoupled_spec_role = "null"
    pp_size = 1
    speculative_adaptive = False
    speculative_adaptive_config = None
    speculative_draft_model_path = None
    speculative_draft_model_revision = "main"
    speculative_draft_attention_backend = "triton"
    speculative_dflash_block_size = None
    speculative_draft_window_size = None
    speculative_attention_mode = "prefill"
    speculative_token_map = None
    speculative_num_draft_tokens = 16
    max_speculative_num_draft_tokens = 16
    speculative_num_steps = None
    page_size = 64
    speculative_eagle_topk = None
    max_running_requests = 4
    disable_overlap_schedule = False
    disable_cuda_graph = False
    disable_draft_cuda_graph = False
    disable_cuda_graph_padding = False
    disable_radix_cache = False
    disable_custom_all_reduce = False
    enable_deterministic_inference = False
    enable_prefill_only_deterministic_inference = False
    flashinfer_allreduce_fusion_backend = None
    enforce_disable_flashinfer_allreduce_fusion = False
    mamba_radix_cache_strategy = "extra_buffer"
    mamba_track_interval = 1
    mamba_ssm_dtype = "float32"
    linear_attn_backend = "triton"
    linear_attn_prefill_backend = None
    attention_backend = "triton"
    prefill_attention_backend = None
    decode_attention_backend = None
    speculative_use_rejection_sampling = False
    speculative_accept_threshold_single = 1.0
    speculative_accept_threshold_acc = 1.0
    enable_multi_layer_eagle = False
    enable_streaming_session = False
    enable_int8_mamba_checkpoint = False
    enable_page_major_kv_layout = False
    enable_linear_replayssm = False
    enable_two_batch_overlap = False
    enable_mixed_chunk = False
    grammar_backend = None
    enable_custom_logit_processor = False
    enable_return_hidden_states = False
    trust_remote_code = False
    json_model_override_args = "{}"

    def __init__(self):
        self.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.FULL, bs=[1, 2, 4], max_bs=4),
            prefill=SimpleNamespace(backend=Backend.BREAKABLE),
        )
        self._cuda_graph_config_locked = set()

    def get_model_config(self):
        return SimpleNamespace(
            hf_config=SimpleNamespace(get_text_config=lambda: object())
        )


def model_args(text_config):
    return SimpleNamespace(
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(get_text_config=lambda: text_config)
        )
    )


class TestDVRServerArgs(CustomTestCase):
    def run_defaults(self, args, *, gdn=False):
        with patch(
            "sglang.srt.speculative.dvr.server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=gdn,
        ):
            handle_dvr_defaults(args)

    def test_linear_state_capability_is_explicit(self):
        gdn_config = Qwen3NextConfig(full_attention_interval=4)
        self.assertTrue(_is_dvr_gated_linear_state_model(model_args(gdn_config)))

        gdn_config.full_attention_interval = 1
        self.assertFalse(_is_dvr_gated_linear_state_model(model_args(gdn_config)))
        self.assertFalse(_is_dvr_gated_linear_state_model(model_args(object())))

        registry_path = (
            "sglang.srt.configs.linear_attn_model_registry.get_linear_attn_config"
        )
        for backend, message in (
            (
                "sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend",
                "does not yet install",
            ),
            (
                "sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend",
                "registered linear-state backend",
            ),
        ):
            with (
                self.subTest(backend=backend),
                patch(
                    registry_path,
                    return_value=(
                        SimpleNamespace(backend_class_name=backend),
                        object(),
                    ),
                ),
                self.assertRaisesRegex(ValueError, message),
            ):
                _is_dvr_gated_linear_state_model(model_args(object()))

    def test_unregistered_linear_state_config_is_rejected(self):
        class UnsupportedLinearConfig:
            @property
            def mamba2_cache_params(self):
                return object()

        with self.assertRaisesRegex(ValueError, "not linear-state config"):
            _is_dvr_gated_linear_state_model(model_args(UnsupportedLinearConfig()))

    def test_algorithm_defaults_define_one_chain_contract(self):
        cases = (
            (DVR_SPECULATIVE_ALGORITHM, None, 16, 15, None),
            (DVR_EAGLE_SPECULATIVE_ALGORITHM, "draft", 2, 1, 48),
            (DVR_DFLASH_SPECULATIVE_ALGORITHM, "draft", 16, 15, 48),
        )
        for algorithm, draft_path, draft_tokens, steps, max_requests in cases:
            with self.subTest(algorithm=algorithm):
                args = _Args()
                args.speculative_algorithm = algorithm
                args.speculative_draft_model_path = draft_path
                args.speculative_num_draft_tokens = None
                args.speculative_num_steps = None
                args.speculative_eagle_topk = None
                args.max_running_requests = None
                args.enable_mixed_chunk = algorithm == DVR_DFLASH_SPECULATIVE_ALGORITHM
                self.run_defaults(args)
                _handle_dvr_speculative_decoding(args)

                self.assertTrue(args.enable_deterministic_inference)
                self.assertEqual(args.grammar_backend, "none")
                self.assertEqual(args.speculative_num_draft_tokens, draft_tokens)
                self.assertEqual(args.speculative_num_steps, steps)
                self.assertEqual(args.speculative_eagle_topk, 1)
                self.assertTrue(args.speculative_use_rejection_sampling)
                self.assertEqual(args.max_running_requests, max_requests)
                if algorithm == DVR_DFLASH_SPECULATIVE_ALGORITHM:
                    self.assertFalse(args.enable_mixed_chunk)

    def test_defaults_preserve_scheduler_and_attention_selection(self):
        for disable_overlap in (False, True):
            with self.subTest(disable_overlap=disable_overlap):
                args = _Args()
                args.disable_overlap_schedule = disable_overlap
                self.run_defaults(args)
                _handle_dvr_speculative_decoding(args)
                self.assertIs(args.disable_overlap_schedule, disable_overlap)

        args = _Args()
        args.attention_backend = None
        self.run_defaults(args)
        self.assertIsNone(args.attention_backend)

    def test_generic_unsupported_modes_are_rejected(self):
        cases = (
            ("speculative_adaptive", True, handle_dvr_defaults, "fixed verify chain"),
            (
                "decoupled_spec_role",
                "verifier",
                _handle_dvr_speculative_decoding,
                "decoupled speculative",
            ),
            (
                "speculative_token_map",
                "tokens.json",
                _handle_dvr_speculative_decoding,
                "target-vocabulary",
            ),
            (
                "speculative_attention_mode",
                "decode",
                _handle_dvr_speculative_decoding,
                "target verify requires",
            ),
            (
                "enable_custom_logit_processor",
                True,
                _handle_dvr_speculative_decoding,
                "custom logit processors",
            ),
            (
                "enable_return_hidden_states",
                True,
                _handle_dvr_speculative_decoding,
                "hidden states",
            ),
            ("enable_pdmux", True, _handle_dvr_speculative_decoding, "PDMux"),
            (
                "enable_prefill_only_deterministic_inference",
                True,
                handle_dvr_defaults,
                "target prefill and verify",
            ),
        )
        for field, value, handler, message in cases:
            with self.subTest(field=field):
                args = _Args()
                setattr(args, field, value)
                with self.assertRaisesRegex((ValueError, NotImplementedError), message):
                    handler(args)

        args = _Args()
        args._resolved_overrides = [("test", {"enable_dp_attention": True})]
        with self.assertRaisesRegex(ValueError, "DP attention"):
            _handle_dvr_speculative_decoding(args)

    def test_invalid_chain_shapes_are_rejected(self):
        cases = (
            ("speculative_num_draft_tokens", 1, "draft_tokens >= 2"),
            ("speculative_eagle_topk", 2, "topk == 1"),
            ("speculative_num_steps", 3, "num_draft_tokens =="),
            ("speculative_accept_threshold_single", 0.9, "must remain 1.0"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                args = _Args()
                setattr(args, field, value)
                with self.assertRaisesRegex(ValueError, message):
                    _handle_dvr_speculative_decoding(args)

        args = _Args()
        args.speculative_draft_model_path = "draft"
        with self.assertRaisesRegex(ValueError, "does not use a draft model"):
            _handle_dvr_speculative_decoding(args)

        args = _Args()
        args.speculative_algorithm = DVR_DFLASH_SPECULATIVE_ALGORITHM
        with self.assertRaisesRegex(ValueError, "requires setting"):
            _handle_dvr_speculative_decoding(args)

        args.speculative_draft_model_path = "draft"
        args.speculative_dflash_block_size = 8
        with self.assertRaisesRegex(ValueError, "must match"):
            _handle_dvr_speculative_decoding(args)

    def test_eagle_rejects_multi_layer_tree(self):
        args = _Args()
        args.speculative_algorithm = DVR_EAGLE_SPECULATIVE_ALGORITHM
        args.speculative_draft_model_path = "draft"
        args.enable_multi_layer_eagle = True
        with self.assertRaisesRegex(NotImplementedError, "multi-layer EAGLE"):
            _handle_dvr_speculative_decoding(args)

    def test_cuda_graph_contract_and_capacity(self):
        for disable_draft, backend in (
            (True, Backend.FULL),
            (False, Backend.DISABLED),
        ):
            with self.subTest(disable_draft=disable_draft, backend=backend):
                args = _Args()
                args.disable_draft_cuda_graph = disable_draft
                args.cuda_graph_config.decode.backend = backend
                with self.assertRaisesRegex(ValueError, "requires draft CUDA graphs"):
                    handle_dvr_cuda_graph_config(args)

        args = _Args()
        args.max_running_requests = None
        handle_dvr_cuda_graph_config(args)
        self.assertEqual(args.max_running_requests, 4)

        args = _Args()
        args.max_running_requests = 5
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            handle_dvr_cuda_graph_config(args)

    def test_cuda_graph_keeps_user_decode_coverage(self):
        for padding_disabled, batch_sizes in (
            (False, [1, 2, 4]),
            (True, [1, 2, 3, 4]),
            (False, None),
        ):
            with self.subTest(padding_disabled=padding_disabled, bs=batch_sizes):
                args = _Args()
                args.disable_cuda_graph_padding = padding_disabled
                args.cuda_graph_config.decode.bs = batch_sizes
                handle_dvr_cuda_graph_config(args)
                self.assertEqual(args.cuda_graph_config.decode.bs, batch_sizes)
                self.assertEqual(args.cuda_graph_config.decode.max_bs, 4)

    def test_gdn_and_plain_attention_prefill_graph_policies(self):
        args = _Args()
        with patch(
            "sglang.srt.speculative.dvr.server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            handle_dvr_cuda_graph_config(args)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)

        args = _Args()
        args._cuda_graph_config_locked = {(Phase.PREFILL, "backend")}
        with (
            patch(
                "sglang.srt.speculative.dvr.server_args."
                "_is_dvr_gated_linear_state_model",
                return_value=True,
            ),
            self.assertRaisesRegex(ValueError, "prefill CUDA graphs"),
        ):
            handle_dvr_cuda_graph_config(args)

        args = _Args()
        with patch(
            "sglang.srt.speculative.dvr.server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=False,
        ):
            handle_dvr_cuda_graph_config(args)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.BREAKABLE)

    def test_gdn_defaults_match_chunk_state_contract(self):
        args = _Args()
        args.page_size = None
        args.mamba_radix_cache_strategy = "auto"
        args.mamba_track_interval = 256
        args.mamba_ssm_dtype = "bfloat16"
        self.run_defaults(args, gdn=True)

        self.assertEqual(args.page_size, 64)
        self.assertEqual(args.mamba_radix_cache_strategy, "auto")
        self.assertEqual(args.mamba_track_interval, 64)
        self.assertEqual(args.mamba_ssm_dtype, "float32")

        args = _Args()
        args.page_size = 32
        with self.assertRaisesRegex(ValueError, "page_size =="):
            self.run_defaults(args, gdn=True)

    def test_gdn_rejects_incompatible_state_lifecycles(self):
        cases = (
            ("enable_two_batch_overlap", "two-batch overlap"),
            ("enable_page_major_kv_layout", "page-major"),
            ("enable_linear_replayssm", "linear-replayssm"),
            ("enable_streaming_session", "streaming sessions"),
            ("enable_int8_mamba_checkpoint", "exact recurrent checkpoints"),
        )
        for field, message in cases:
            with self.subTest(field=field):
                args = _Args()
                setattr(args, field, True)
                with (
                    patch(
                        "sglang.srt.speculative.dvr.server_args."
                        "_is_dvr_gated_linear_state_model",
                        return_value=True,
                    ),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    _handle_dvr_speculative_decoding(args)

    def test_gdn_radix_requires_resolved_extra_buffer(self):
        for strategy in ("auto", "no_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                args = _Args()
                args.mamba_radix_cache_strategy = strategy
                with (
                    patch(
                        "sglang.srt.speculative.dvr.server_args."
                        "_is_dvr_gated_linear_state_model",
                        return_value=True,
                    ),
                    self.assertRaisesRegex(ValueError, "resolved.*extra_buffer"),
                ):
                    _handle_dvr_speculative_decoding(args)

        args = _Args()
        args.mamba_radix_cache_strategy = "auto"
        args._resolved_overrides = [
            ("mamba", {"mamba_radix_cache_strategy": "extra_buffer"})
        ]
        with patch(
            "sglang.srt.speculative.dvr.server_args."
            "_is_dvr_gated_linear_state_model",
            return_value=True,
        ):
            _handle_dvr_speculative_decoding(args)

    def test_gdn_requires_boundary_exporting_prefill_backend(self):
        args = _Args()
        args.linear_attn_prefill_backend = "fla"
        with (
            patch(
                "sglang.srt.speculative.dvr.server_args."
                "_is_dvr_gated_linear_state_model",
                return_value=True,
            ),
            self.assertRaisesRegex(ValueError, "linear-attn-prefill-backend"),
        ):
            _handle_dvr_speculative_decoding(args)

    def test_attention_backend_validation_uses_effective_phase_backends(self):
        for backend in ("flashinfer", "fa4"):
            with self.subTest(backend=backend):
                args = _Args()
                args.attention_backend = backend
                with self.assertRaisesRegex(ValueError, "only Triton and FA3"):
                    _handle_dvr_speculative_decoding(args)

        args = _Args()
        args.prefill_attention_backend = "fa3"
        args.decode_attention_backend = "triton"
        _handle_dvr_speculative_decoding(args)
        args.decode_attention_backend = "flashinfer"
        with self.assertRaisesRegex(ValueError, "effective decode backend"):
            _handle_dvr_speculative_decoding(args)

    def test_radix_disabled_and_plain_attention_keep_upstream_options(self):
        args = _Args()
        args.disable_radix_cache = True
        args.mamba_radix_cache_strategy = "auto"
        self.run_defaults(args, gdn=True)
        self.assertTrue(args.disable_radix_cache)
        self.assertEqual(args.mamba_radix_cache_strategy, "auto")
        self.assertFalse(ServerArgs.enable_mamba_extra_buffer(args))

        args = _Args()
        args.page_size = 32
        args.speculative_num_draft_tokens = 65
        args.speculative_num_steps = 64
        args.enable_streaming_session = True
        args.enable_int8_mamba_checkpoint = True
        args.enable_page_major_kv_layout = True
        args.enable_linear_replayssm = True
        self.run_defaults(args)
        _handle_dvr_speculative_decoding(args)
        self.assertEqual(args.page_size, 32)
        self.assertEqual(args.speculative_num_draft_tokens, 65)

    def test_request_row_headroom_does_not_change_ordinary_spec(self):
        args = _Args()
        self.assertEqual(get_req_to_token_extra_context_len(args), 95)

        args.speculative_algorithm = "EAGLE"
        args.speculative_eagle_topk = 1
        self.assertEqual(get_req_to_token_extra_context_len(args), 95)

    def test_algorithm_is_normalized_before_generic_spec_handling(self):
        args = _Args()
        args.speculative_algorithm = DVR_SPECULATIVE_ALGORITHM.lower()
        self.run_defaults(args)
        self.assertEqual(args.speculative_algorithm, DVR_SPECULATIVE_ALGORITHM)


if __name__ == "__main__":
    unittest.main()
