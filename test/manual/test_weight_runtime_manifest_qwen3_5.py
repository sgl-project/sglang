"""Manual GPU smoke test for the Qwen3.5 runtime weight manifest exporter.

Run with:

SGLANG_WEIGHT_MANIFEST_MODEL=/path/to/Qwen3.5-0.8B \
python -m unittest test/manual/test_weight_runtime_manifest_qwen3_5.py
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path


@unittest.skipUnless(
    os.getenv("SGLANG_WEIGHT_MANIFEST_MODEL"),
    "set SGLANG_WEIGHT_MANIFEST_MODEL to run the GPU smoke test",
)
class TestQwen35WeightRuntimeManifest(unittest.TestCase):
    def test_real_model_runner_exports_resident_cuda_parameters(self) -> None:
        import torch.distributed as dist

        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.distributed.parallel_state_wrapper import ParallelState
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.server_args import PortArgs, ServerArgs

        def destroy_process_group() -> None:
            if dist.is_initialized():
                dist.destroy_process_group()

        self.addCleanup(destroy_process_group)
        model_path = os.environ["SGLANG_WEIGHT_MANIFEST_MODEL"]
        server_args = ServerArgs(
            model_path=model_path,
            tokenizer_path=model_path,
            disable_cuda_graph=True,
            enable_weight_runtime_manifest=True,
            mem_fraction_static=0.1,
            tp_size=1,
        )
        port_args = PortArgs.init_new(server_args)
        model_config = ModelConfig.from_server_args(server_args)
        runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            ps=ParallelState.trivial(tp_size=1),
            nccl_port=port_args.nccl_port,
            server_args=server_args,
        )

        manifest = runner.get_weight_runtime_manifest(
            model_id=Path(model_path).name,
            revision="runtime-smoke",
            instance_id="qwen3.5-smoke-tp0",
            worker_id="qwen3.5-smoke-tp0",
            endpoint="127.0.0.1:19001",
        )
        try:
            self.assertGreater(len(manifest.tensors), 0)
            self.assertTrue(all(tensor.address > 0 for tensor in manifest.tensors))
            self.assertEqual({tensor.device for tensor in manifest.tensors}, {"cuda"})
            self.assertEqual(
                len(
                    {
                        (
                            tensor.tensor_id,
                            tensor.global_offset,
                            tensor.local_shape,
                        )
                        for tensor in manifest.tensors
                    }
                ),
                len(manifest.tensors),
            )
            tensor_ids = {tensor.tensor_id for tensor in manifest.tensors}
            required_suffixes = {
                "embed_tokens.weight",
                "lm_head.weight",
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "gate_proj.weight",
                "up_proj.weight",
                "down_proj.weight",
            }
            for suffix in required_suffixes:
                self.assertTrue(
                    any(tensor_id.endswith(suffix) for tensor_id in tensor_ids),
                    suffix,
                )
            print(
                "weight runtime manifest:",
                f"tensors={len(manifest.tensors)}",
                f"bytes={sum(tensor.nbytes for tensor in manifest.tensors)}",
                f"generation={manifest.generation}",
            )
        finally:
            runner.release_weight_runtime_manifest(manifest.lease_id)


if __name__ == "__main__":
    unittest.main()
