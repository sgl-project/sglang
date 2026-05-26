import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOptimisticPrefillServerArgs(unittest.TestCase):
    def test_default_value_is_zero(self):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs.__dataclass_fields__["optimistic_prefill_retries"]
        self.assertEqual(args.default, 0)

    def test_validation_requires_prefill_mode(self):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs.from_kwargs(
            model_path="meta-llama/Llama-3.1-8B",
            disaggregation_mode="decode",
            optimistic_prefill_retries=1,
        )
        with self.assertRaises(ValueError, msg="requires.*prefill"):
            args._handle_pd_disaggregation()

    def test_validation_rejects_pp(self):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs.from_kwargs(
            model_path="meta-llama/Llama-3.1-8B",
            disaggregation_mode="prefill",
            optimistic_prefill_retries=1,
            pp_size=2,
        )
        with self.assertRaises(ValueError, msg="pipeline parallelism"):
            args._handle_pd_disaggregation()

    def test_validation_passes_for_valid_config(self):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs.from_kwargs(
            model_path="meta-llama/Llama-3.1-8B",
            disaggregation_mode="prefill",
            optimistic_prefill_retries=3,
        )
        args._handle_pd_disaggregation()

    def test_zero_retries_passes_any_mode(self):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs.from_kwargs(
            model_path="meta-llama/Llama-3.1-8B",
            disaggregation_mode="decode",
            optimistic_prefill_retries=0,
        )
        args._handle_pd_disaggregation()


class TestReqOptimisticFields(unittest.TestCase):
    def test_default_fields(self):
        from sglang.srt.managers.schedule_batch import Req

        req = Req.__new__(Req)
        req.optimistic_prefill = False
        req.optimistic_prefill_remaining = 0
        req.optimistic_stop = False
        self.assertFalse(req.optimistic_prefill)
        self.assertEqual(req.optimistic_prefill_remaining, 0)
        self.assertFalse(req.optimistic_stop)


class TestResetForRetractCoversOptimisticFields(unittest.TestCase):
    def test_reset_for_retract_resets_kv_fields(self):
        from sglang.srt.managers.schedule_batch import Req

        req = Req.__new__(Req)
        req.retraction_count = 0
        req.kv_committed_len = 100
        req.kv_allocated_len = 200
        req.kv_committed_freed = True
        req.kv_overallocated_freed = True
        req.inflight_middle_chunks = 3
        req.prefix_indices = None
        req.last_node = "node"
        req.cache_protected_len = 50
        req.swa_uuid_for_lock = "uuid"
        req.swa_prefix_lock_released = True
        req.extend_input_len = 10
        req.is_retracted = False
        req.retracted_stain = False
        req.input_token_logprobs = [1, 2]
        req.temp_input_top_logprobs_val = [1]
        req.temp_input_top_logprobs_idx = [1]
        req.extend_logprob_start_len = 5
        req.mamba_pool_idx = "idx"
        req.mamba_ping_pong_track_buffer = "buf"
        req.mamba_next_track_idx = 1
        req.mamba_last_track_seqlen = 1
        req.mamba_branching_seqlen = 1
        req.mamba_cow_src_index = 1
        req.mamba_needs_clear = True
        req.already_computed = 10
        req.swa_evicted_seqlen = 5
        req.extend_batch_idx = 1
        req.decode_batch_idx = 1
        req.routed_experts = "experts"
        req.indexer_topk = "topk"
        req.input_embeds = None

        req.reset_for_retract()

        self.assertEqual(req.kv_committed_len, 0)
        self.assertEqual(req.kv_allocated_len, 0)
        self.assertFalse(req.kv_committed_freed)
        self.assertFalse(req.kv_overallocated_freed)
        self.assertEqual(req.inflight_middle_chunks, 0)
        self.assertIsNone(req.last_node)
        self.assertEqual(req.cache_protected_len, 0)
        self.assertEqual(req.extend_input_len, 0)
        self.assertTrue(req.is_retracted)
        self.assertEqual(req.retraction_count, 1)


if __name__ == "__main__":
    unittest.main()
