"""CPU regression for the PR46980 transfer-counter ownership migration."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch, sentinel

import torch

from sglang.kernels.ops.speculative.q38_mk_mtp import metadata
from sglang.srt.mem_cache.kv.state_storage import StateStorage
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool


class _CudaMetadata:
    # Only exercise the host eligibility predicate; the CUDA kernel is mocked.
    is_cuda = True
    dtype = torch.int64
    device = torch.device("cpu")

    def numel(self):
        return 4

    def is_contiguous(self):
        return True


class TestMTPStateTransferOwner(unittest.TestCase):
    def _run(self, counter):
        pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
        pool.state = StateStorage.empty("mamba")
        pool.req_index_to_mamba_index_mapping = torch.tensor([0, 1], dtype=torch.int32)
        pool.ngram_pool = SimpleNamespace(context=torch.zeros((2, 2), dtype=torch.int64))
        self.assertFalse(hasattr(pool, "layer_transfer_counter"))
        mode = SimpleNamespace(is_target_verify=lambda: True)
        batch = SimpleNamespace(
            forward_mode=mode,
            spec_info=SimpleNamespace(draft_token_num=4),
            req_pool_indices=torch.tensor([1], dtype=torch.int32),
            out_cache_loc=torch.ones(4, dtype=torch.int64),
            extend_seq_lens=torch.tensor([4], dtype=torch.int32),
        )
        with (
            patch.object(metadata, "enabled", return_value=True),
            patch("sglang.srt.runtime_context.get_parallel",
                  return_value=SimpleNamespace(tp_size=4, pp_size=1)),
            patch("sglang.srt.layers.cp.utils.is_cp_active", return_value=False),
            patch("sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.is_in_breakable_cuda_graph",
                  return_value=False),
            patch("sglang.srt.model_executor.forward_context.get_token_to_kv_pool",
                  return_value=SimpleNamespace(layer_transfer_counter=counter)),
            patch.object(metadata, "prepare_verify_layout", return_value=sentinel.layout) as prepare,
        ):
            result = metadata.try_prepare_verify_layout(
                _CudaMetadata(), batch, mode, pool,
                ngram_size=3, ngram_eos_token_id=0, processed_tokens=4,
            )
        return result, prepare

    def test_no_transfer_keeps_the_fused_path(self):
        result, prepare = self._run(None)
        self.assertIs(result, sentinel.layout)
        prepare.assert_called_once()

    def test_active_transfer_keeps_the_native_fallback(self):
        result, prepare = self._run(object())
        self.assertIsNone(result)
        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
