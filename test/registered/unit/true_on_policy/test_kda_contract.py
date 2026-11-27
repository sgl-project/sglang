"""TOP KDA admission refuses what its chunked program does not run, instead of serving another program."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.kernels.ops.attention.fla import kda_top
from sglang.srt.environ import envs
from sglang.srt.true_on_policy.kda_chunked import validate_kda_contract
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config(**lin):
    fields = dict(
        head_dim=128, short_conv_kernel_size=4, gate_lower_bound=-5.0, num_heads=64
    )
    fields.update(lin)
    return SimpleNamespace(model_type="glm5_next_text", linear_attn_config=fields)


def _args(**overrides):
    fields = dict(
        speculative_algorithm=None,
        speculative_eagle_topk=None,
        speculative_num_draft_tokens=None,
        enable_linear_replayssm_spec=False,
        disable_radix_cache=False,
        mamba_radix_cache_strategy="extra_buffer",
        page_size=256,
        mamba_track_interval=256,
        enable_unified_cache_external_linker=False,
        enable_mixed_chunk=False,
        enable_two_batch_overlap=False,
        pp_size=1,
        chunked_prefill_size=8192,
        linear_attn_backend="triton",
        mamba_ssm_dtype="float32",
        linear_attn_prefill_backend=None,
        linear_attn_decode_backend=None,
        linear_attn_verify_backend=None,
        enable_streaming_session=False,
        disaggregation_mode="null",
        enable_hierarchical_cache=False,
        enable_lmcache=False,
        enable_flexkv=False,
        radix_cache_backend=None,
        enable_int8_mamba_checkpoint=False,
        enable_linear_replayssm=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestKdaContract(unittest.TestCase):
    def setUp(self):
        # The FLA kernels were imported before the mode overrides; the import-order guard is not under test here.
        guard = patch.object(kda_top, "CHUNKED_AT_IMPORT", True)
        guard.start()
        self.addCleanup(guard.stop)

    def test_admits_the_gate_layout(self):
        with envs.SGLANG_TOP_KDA_MODE.override("chunked"):
            validate_kda_contract(_args(), _config())

    def test_without_radix_reuse_the_track_grid_is_not_checked(self):
        with envs.SGLANG_TOP_KDA_MODE.override("chunked"):
            validate_kda_contract(
                _args(
                    disable_radix_cache=True,
                    mamba_radix_cache_strategy="no_buffer",
                    mamba_track_interval=100,
                ),
                _config(),
            )

    def test_refuses_radix_states_off_the_64_token_grid(self):
        # A radix hit resumes E where the cache holds a state; the row cache holds no rows before it, so it must be a
        # chunk boundary: no_buffer caches a request's state where it stops, a page or interval off 64 tracks there.
        cases = {
            "extra_buffer": dict(mamba_radix_cache_strategy="no_buffer"),
            "64-aligned pages and Mamba tracking": dict(mamba_track_interval=96),
            "external cache linkers": dict(enable_unified_cache_external_linker=True),
        }
        with envs.SGLANG_TOP_KDA_MODE.override("chunked"):
            for message, overrides in cases.items():
                with self.subTest(message):
                    with self.assertRaisesRegex(ValueError, message):
                        validate_kda_contract(_args(**overrides), _config())

    def test_refuses_without_the_chunked_program(self):
        # Without the mode the prefill autotunes and picks the batch-size-dependent intra kernel.
        with envs.SGLANG_TOP_KDA_MODE.override(""):
            with self.assertRaisesRegex(ValueError, "SGLANG_TOP_KDA_MODE"):
                validate_kda_contract(_args(), _config())

    def test_refuses_without_the_safe_gate(self):
        # The fused intra kernel's per-block gate reference is only in FP32 range under the lower bound.
        with envs.SGLANG_TOP_KDA_MODE.override("chunked"):
            for bound in (None, -6.0, 0.0):
                with self.assertRaisesRegex(ValueError, "lower bound"):
                    validate_kda_contract(_args(), _config(gate_lower_bound=bound))

    def test_admits_an_eagle_chain(self):
        with envs.SGLANG_TOP_KDA_MODE.override("chunked"):
            validate_kda_contract(
                _args(
                    speculative_algorithm="EAGLE",
                    speculative_eagle_topk=1,
                    speculative_num_draft_tokens=6,
                ),
                _config(),
            )

    def test_refuses_paths_that_skip_the_row_cache(self):
        # Tree verify and mixed batches start or verify rows the row engine never cached.
        chain = dict(speculative_algorithm="EAGLE", speculative_eagle_topk=1)
        cases = {
            "EAGLE chains": dict(
                speculative_algorithm="EAGLE",
                speculative_eagle_topk=4,
                speculative_num_draft_tokens=8,
            ),
            "2 to 64 tokens": dict(**chain, speculative_num_draft_tokens=65),
            "ReplaySSM": dict(
                **chain,
                speculative_num_draft_tokens=2,
                enable_linear_replayssm_spec=True,
            ),
            "mixed": dict(enable_mixed_chunk=True),
            "64-aligned": dict(chunked_prefill_size=8100),
            "Triton KDA kernels": dict(linear_attn_decode_backend="flashinfer"),
            "FP32 recurrent state": dict(mamba_ssm_dtype="bfloat16"),
        }
        with envs.SGLANG_TOP_KDA_MODE.override("chunked"):
            for message, overrides in cases.items():
                with self.subTest(message):
                    with self.assertRaisesRegex(ValueError, message):
                        validate_kda_contract(_args(**overrides), _config())


class TestExtendPlanTracks(unittest.TestCase):
    """The radix track an extend hands to the cache names the prefill's state at a 64-token boundary: a snapshot at
    a chunk start inside the prefill call, or its final state (E at the last boundary, before the tails)."""

    def plan(self, *, prefix, lens, track_lens):
        import torch

        from sglang.srt.true_on_policy.kda_chunked import ChunkedKDAEngine

        engine = ChunkedKDAEngine(
            num_layers=1, num_slots=4, H=1, K=128, device="cpu", max_bs=4
        )
        n = len(lens)
        plan = engine.build_extend_plan(
            prefix=prefix,
            lens=lens,
            ring=list(range(n)),
            state=list(range(n)),
            track_mask=[t is not None for t in track_lens],
            track_lens=[t or -1 for t in track_lens],
        )
        self.assertEqual(engine.caches[0].e_chunk.dtype, torch.int32)
        return plan

    def test_tracks_resolve_to_chunk_starts_or_the_final_state(self):
        # Request 0: a radix hit at 256 extends 300 rows, tracked at 448 (inside: chunk 3 of the call); request 1:
        # tracked at its extend's last boundary 256 through the forced-h length 257 (the final state, not a chunk
        # index the state kernel never stores); request 2: aligned end; request 3: untracked.
        plan = self.plan(
            prefix=[256, 0, 0, 64],
            lens=[300, 300, 128, 100],
            track_lens=[449, 257, 128, None],
        )
        self.assertEqual(plan.fla_track_idx.tolist(), [3, -1, -1, -1])
        self.assertEqual(plan.final_track, [(1, 1), (2, 2)])

    def test_refuses_tracks_the_slot_does_not_hold(self):
        cases = {
            # An aligned track length names the slot's final state; only the extend's end is there.
            "not its final state": dict(prefix=[0], lens=[300], track_lens=[128]),
            "outside the extend": dict(prefix=[64], lens=[30], track_lens=[200]),
            "off the 64-token grid": dict(prefix=[100], lens=[300], track_lens=[None]),
        }
        for message, kwargs in cases.items():
            with self.subTest(message):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.plan(**kwargs)


if __name__ == "__main__":
    unittest.main()
