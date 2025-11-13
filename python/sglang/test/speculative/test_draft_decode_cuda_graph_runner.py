import unittest

import os
import torch
from types import MethodType

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import find_available_port


TARGET_MODEL_PATH = "/shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct"
DRAFT_MODEL_PATH = "/shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714"


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required for this test")
@unittest.skipIf(
    not (os.path.isdir(TARGET_MODEL_PATH) and os.path.isdir(DRAFT_MODEL_PATH)),
    "Required local model paths not found",
)
class TestEAGLEDecodeCudaGraphRunnerCapture(unittest.TestCase):
    def test_capture_decode_graph(self):

        os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
        os.environ.pop("SGLANG_CI_SMALL_KV_SIZE", None)

        server_args = ServerArgs(
            model_path=TARGET_MODEL_PATH,
            speculative_draft_model_path=DRAFT_MODEL_PATH,
            speculative_algorithm="EAGLE3",
            device="cuda",
            dtype="float16",
            tp_size=1,
            ep_size=1,
            pp_size=1,
            trust_remote_code=True,
            disable_cuda_graph=False,
            cuda_graph_bs=[1],
            # Enable decode cuda-graph capture in eagle_worker.init_cuda_graphs
            speculative_batch_size_threshold=1,
            # Reduce memory footprint to avoid OOM in CI
            mem_fraction_static=0.75,
            max_total_tokens=131072,
            max_running_requests=1,
            enable_torch_compile=False,
            enable_p2p_check=False,
            enable_pdmux=True,
            log_level="error",
        )

        nccl_port = find_available_port(20000)

        # Initialize target worker (single rank)
        target_worker = TpModelWorker(
            server_args=server_args,
            gpu_id=0,
            tp_rank=0,
            moe_ep_rank=0,
            pp_rank=0,
            dp_rank=None,
            nccl_port=nccl_port,
        )

        def _noop_set_kv_buffer(self, *args, **kwargs):
            return None

        kv_pool = target_worker.model_runner.token_to_kv_pool
        kv_pool.set_kv_buffer = MethodType(_noop_set_kv_buffer, kv_pool)

        # Initialize EAGLE worker; this captures draft/extend and decode graphs
        eagle_worker = EAGLEWorker(
            server_args=server_args,
            gpu_id=0,
            tp_rank=0,
            dp_rank=None,
            moe_ep_rank=0,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )

        runner = eagle_worker.cuda_graph_runner_for_decode
        self.assertIsNotNone(runner, "Decode cuda graph runner should be created")

        # Should capture at least for bs=1
        self.assertIn(1, runner.capture_bs)
        self.assertIn(1, runner.graphs, "Graph for bs=1 not captured")
        self.assertIn(1, runner.output_buffers, "Output buffer for bs=1 missing")

        # Sanity on graph and buffer objects
        self.assertIsNotNone(runner.graphs[1])
        self.assertIsNotNone(runner.output_buffers[1])


if __name__ == "__main__":
    unittest.main()
