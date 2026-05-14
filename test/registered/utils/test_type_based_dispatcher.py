# tests/benchmarks/test_type_dispatcher_e2e.py
"""
E2E test for TypeBasedDispatcher optimization.
Tests real-world scenarios with actual request types.
"""

import timeit
import unittest

from sglang.srt.managers.io_struct import SamplingParams
from sglang.test.ci.ci_register import register_amd_ci
from sglang.utils import TypeBasedDispatcher

register_amd_ci(est_time=10, suite="stage-b-test-small-1-gpu-amd")


class TestTypeBasedDispatcher(unittest.TestCase):
    """Unit tests for TypeBasedDispatcher e2e performance."""

    def test_type_dispatcher_e2e_performance(self):
        """End-to-end performance test with real request types"""
        print("E2E Performance Test for TypeBasedDispatcher")
        print("=" * 50)

        from sglang.srt.managers.io_struct import (
            AbortReq,
            BatchTokenizedEmbeddingReqInput,
            BatchTokenizedGenerateReqInput,
            ClearHiCacheReqInput,
            CloseSessionReqInput,
            DestroyWeightsUpdateGroupReqInput,
            ExpertDistributionReq,
            FlushCacheReqInput,
            FreezeGCReq,
            GetInternalStateReq,
            GetLoadReqInput,
            GetWeightsByNameReqInput,
            InitWeightsSendGroupForRemoteInstanceReqInput,
            InitWeightsUpdateGroupReqInput,
            LoadLoRAAdapterReqInput,
            OpenSessionReqInput,
            ProfileReq,
            ReleaseMemoryOccupationReqInput,
            ResumeMemoryOccupationReqInput,
            RpcReqInput,
            SendWeightsToRemoteInstanceReqInput,
            SetInternalStateReq,
            SlowDownReqInput,
            TokenizedEmbeddingReqInput,
            TokenizedGenerateReqInput,
            UnloadLoRAAdapterReqInput,
            UpdateWeightFromDiskReqInput,
            UpdateWeightsFromIPCReqInput,
            UpdateWeightsFromTensorReqInput,
        )

        mapping = [
            (TokenizedGenerateReqInput, lambda req: "generate_handled"),
            (TokenizedEmbeddingReqInput, lambda req: "embedding_handled"),
            (BatchTokenizedGenerateReqInput, lambda req: "batch_generate_handled"),
            (
                BatchTokenizedEmbeddingReqInput,
                lambda req: "batch_generate_embedding_handled",
            ),
            (FlushCacheReqInput, lambda req: "flush_cache_handled"),
            (ClearHiCacheReqInput, lambda req: "clear_hicache_handled"),
            (AbortReq, lambda req: "abort_handled"),
            (OpenSessionReqInput, lambda req: "open_session_handled"),
            (CloseSessionReqInput, lambda req: "close_session_handled"),
            (
                UpdateWeightFromDiskReqInput,
                lambda req: "update_weights_from_disk_handled",
            ),
            (
                InitWeightsUpdateGroupReqInput,
                lambda req: "init_weights_update_group_handled",
            ),
            (
                DestroyWeightsUpdateGroupReqInput,
                lambda req: "destroy_weights_update_group_handled",
            ),
            (
                InitWeightsSendGroupForRemoteInstanceReqInput,
                lambda req: "init_weights_send_group_for_remote_instance_handled",
            ),
            (
                SendWeightsToRemoteInstanceReqInput,
                lambda req: "send_weights_to_remote_instance_handled",
            ),
            (
                UpdateWeightsFromTensorReqInput,
                lambda req: "update_weights_from_tensor_handled",
            ),
            (
                UpdateWeightsFromIPCReqInput,
                lambda req: "update_weights_from_ipc_handled",
            ),
            (GetWeightsByNameReqInput, lambda req: "get_weights_by_name_handled"),
            (
                ReleaseMemoryOccupationReqInput,
                lambda req: "release_memory_occupation_handled",
            ),
            (
                ResumeMemoryOccupationReqInput,
                lambda req: "resume_memory_occupation_handled",
            ),
            (SlowDownReqInput, lambda req: "slow_down_handled"),
            (ProfileReq, lambda req: "profile_handled"),
            (FreezeGCReq, lambda req: "freeze_gc_handled"),
            (GetInternalStateReq, lambda req: "get_internal_state_handled"),
            (SetInternalStateReq, lambda req: "set_internal_state_handled"),
            (RpcReqInput, lambda req: "rpc_request_handled"),
            (ExpertDistributionReq, lambda req: "expert_distribution_handled"),
            (LoadLoRAAdapterReqInput, lambda req: "load_lora_adapter_handled"),
            (UnloadLoRAAdapterReqInput, lambda req: "unload_lora_adapter_handled"),
            (GetLoadReqInput, lambda req: "get_load_handled"),
        ]

        # Create requests that conforms to the real distribution
        test_requests = []

        test_requests.append(
            TokenizedGenerateReqInput(
                input_text="",
                input_ids=[1, 2],
                mm_inputs=dict(),
                sampling_params=SamplingParams(),
                return_logprob=False,
                logprob_start_len=0,
                top_logprobs_num=0,
                token_ids_logprob=[1, 2],
                stream=False,
            )
        )

        test_requests.append(
            TokenizedEmbeddingReqInput(
                input_text="",
                input_ids=[1, 2],
                image_inputs=dict(),
                token_type_ids=[1, 2],
                sampling_params=SamplingParams(),
            )
        )

        test_requests.append(
            BatchTokenizedGenerateReqInput(
                batch=[
                    TokenizedGenerateReqInput(
                        input_text="",
                        input_ids=[1, 2],
                        mm_inputs=dict(),
                        sampling_params=SamplingParams(),
                        return_logprob=False,
                        logprob_start_len=0,
                        top_logprobs_num=0,
                        token_ids_logprob=[1, 2],
                        stream=False,
                    )
                ]
            )
        )
        test_requests.append(
            BatchTokenizedEmbeddingReqInput(
                batch=[
                    TokenizedEmbeddingReqInput(
                        input_text="",
                        input_ids=[1, 2],
                        image_inputs=dict(),
                        token_type_ids=[1, 2],
                        sampling_params=SamplingParams(),
                    )
                ]
            )
        )

        test_requests.append(FlushCacheReqInput())
        test_requests.append(ClearHiCacheReqInput())
        test_requests.append(AbortReq())
        test_requests.append(OpenSessionReqInput(capacity_of_str_len=0))
        test_requests.append(CloseSessionReqInput(session_id=""))
        test_requests.append(UpdateWeightFromDiskReqInput(model_path=""))
        test_requests.append(
            InitWeightsUpdateGroupReqInput(
                master_address="",
                master_port=0,
                rank_offset=0,
                world_size=0,
                group_name="",
            )
        )
        test_requests.append(DestroyWeightsUpdateGroupReqInput())
        test_requests.append(
            InitWeightsSendGroupForRemoteInstanceReqInput(
                master_address="", ports="", group_name="", world_size=0, group_rank=0
            )
        )
        test_requests.append(
            SendWeightsToRemoteInstanceReqInput(master_address="", ports="")
        )
        test_requests.append(
            UpdateWeightsFromTensorReqInput(serialized_named_tensors=[])
        )
        test_requests.append(GetWeightsByNameReqInput(name=""))
        test_requests.append(ReleaseMemoryOccupationReqInput())
        test_requests.append(RpcReqInput(method=""))
        test_requests.append(GetLoadReqInput())

        dispatcher = TypeBasedDispatcher(mapping)

        # test
        time_taken = timeit.timeit(
            lambda: [dispatcher(req) for req in test_requests],
            number=100,  # Average of 100 runs
        )

        print(f"Total requests: {len(test_requests)}")
        print(f"Time taken: {time_taken:.4f}s")
        print(f"Requests per second: {len(test_requests) * 100 / time_taken:.0f}")

        return time_taken


if __name__ == "__main__":
    unittest.main()
