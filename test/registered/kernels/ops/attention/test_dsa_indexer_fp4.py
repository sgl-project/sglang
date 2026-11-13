import unittest
from unittest.mock import patch

import test_dsa_indexer as base
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_fp4_indexer import Fp4Indexer
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="nightly", runner_config="4-gpu-b200")


class TestDSAIndexerFp4(CustomTestCase):
    setUpClass = classmethod(base.TestDSAIndexer.setUpClass.__func__)
    setUp = base.TestDSAIndexer.setUp
    _init_model_runner = base.TestDSAIndexer._init_model_runner
    _create_indexer = base.TestDSAIndexer._create_indexer
    _create_forward_batch = base.TestDSAIndexer._create_forward_batch
    _verify_topk_output = base.TestDSAIndexer._verify_topk_output

    def test_forward_extend_then_decode_fp4(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("DeepGEMM FP4 MQA requires SM100")

        self.enterContext(get_context().override_server_args(dsa_topk_backend="torch"))
        self.enterContext(patch.object(base, "Indexer", Fp4Indexer))
        self.enterContext(envs.SGLANG_DSA_FUSE_TOPK.override(False))
        self.batch_size = 1
        self._init_model_runner({"use_fp4_indexer": True})
        indexer = self._create_indexer()
        for parameter in indexer.parameters():
            torch.nn.init.normal_(parameter, std=0.02)

        for forward_mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
            forward_batch = self._create_forward_batch(forward_mode)
            forward_batch.out_cache_loc += self.config["page_size"]
            self.backend.init_forward_metadata(forward_batch)
            q_len = self.seq_len if forward_mode == ForwardMode.EXTEND else 1
            hidden_states = torch.randn(
                q_len,
                self.config["hidden_size"],
                dtype=self.dtype,
                device=self.device,
            )
            q_lora = torch.randn(
                q_len,
                self.config["q_lora_rank"],
                dtype=self.dtype,
                device=self.device,
            )
            seq_len = forward_batch.seq_lens_cpu[0].item()
            positions = torch.arange(seq_len - q_len, seq_len, device=self.device)

            topk_indices = indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.config["layer_id"],
            )

            self._verify_topk_output(
                topk_indices, self.batch_size, q_len, self.config["index_topk"]
            )
            self.assertEqual(topk_indices.dtype, torch.int32)
            torch.testing.assert_close(
                (topk_indices >= 0).sum(dim=-1),
                (positions + 1).clamp(max=self.config["index_topk"]),
            )


if __name__ == "__main__":
    unittest.main()
