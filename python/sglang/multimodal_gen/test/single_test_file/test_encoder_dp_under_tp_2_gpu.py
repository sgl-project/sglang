"""Encoder data-parallel batched encode under DiT tensor parallelism.

encoder_parallel=dp used to be rejected whenever tp_size > 1, but the DiT's
TP layout never shards an encoder — only folding does, and on a pure-TP
replica (replica == tp) folding is never proposed, so every rank holds a
full encoder replica it can run alone. This test initializes the real
model-parallel state with tp=2, checks the dp gate engages against the live
world group, and asserts the sharded-and-gathered encode is bit-identical
to the replicated forward (including the odd-batch padding path).

The toy encoder is elementwise on purpose: the dp plumbing (shard / pad /
gather) is what this guards, not kernel tiling invariance across batch
shapes.

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_encoder_dp_under_tp_2_gpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase

_WORLD = 2


def _worker() -> int:
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch.distributed as dist

    from sglang.multimodal_gen.configs.models.encoders import (
        BaseEncoderOutput,
        TextEncoderConfig,
    )
    from sglang.multimodal_gen.runtime.distributed import (
        get_replica_group,
        get_world_group,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_tp_group,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
    from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
        TextEncodingStage,
        _data_parallel_text_encode,
    )

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    # pure-TP replica: tp == world — exactly the shape the old gate rejected
    maybe_init_distributed_environment_and_model_parallel(tp_size=_WORLD, sp_size=1)

    failures: list[str] = []
    world = get_world_group()
    if world.world_size != _WORLD:
        failures.append(f"world group size {world.world_size} != {_WORLD}")
    if get_tp_group().world_size != _WORLD:
        failures.append(f"tp group size {get_tp_group().world_size} != {_WORLD}")

    # --- the gate must engage for a replicated dp-capable encoder at tp>1 ---
    encoder_config = TextEncoderConfig()
    encoder_config.hidden_size = 2048
    encoder_config.num_attention_heads = 16
    encoder_config.intermediate_size = 8192
    encoder_config.parallel_folding_mode = None
    toy = Mock(spec=TextEncoder)
    toy.supports_dp_encode = True
    server_args = SimpleNamespace(encoder_parallel="dp", tp_size=_WORLD, dp_size=1)
    stage = SimpleNamespace(_log_dp_choice=lambda batch, world_size: None)
    group = TextEncodingStage._text_encode_dp_group(
        stage, server_args, encoder_config, batch_size=5, text_encoder=toy
    )
    if group is None:
        failures.append("dp gate refused a replicated encoder under tp>1")
    elif group is not get_replica_group():
        failures.append("dp gate returned a group other than the replica group")
    if get_replica_group().world_size != _WORLD:
        failures.append("replica group size mismatch (dp_size==1: replica == world)")

    # --- sharded + gathered encode must equal the replicated forward ---
    if group is not None:
        vocab, hidden, seq, bs = 512, 2048, 16, 5  # odd bs: exercises padding
        emb = torch.randn(
            vocab,
            hidden,
            generator=torch.Generator(device="cuda").manual_seed(7),
            device="cuda",
            dtype=torch.bfloat16,
        )

        def forward_fn(kwargs: dict) -> BaseEncoderOutput:
            hidden_states = emb[kwargs["input_ids"]]
            out = hidden_states * 2 + torch.sin(hidden_states)
            return BaseEncoderOutput(
                last_hidden_state=out,
                hidden_states=(hidden_states, out),
                attention_mask=kwargs["attention_mask"],
            )

        ids_gen = torch.Generator(device="cuda").manual_seed(23)
        forward_kwargs = {
            "input_ids": torch.randint(
                0, vocab, (bs, seq), generator=ids_gen, device="cuda"
            ),
            "attention_mask": torch.ones(bs, seq, device="cuda", dtype=torch.long),
        }
        expected = forward_fn(forward_kwargs)
        got = _data_parallel_text_encode(forward_fn, forward_kwargs, group)

        if not torch.equal(expected.last_hidden_state, got.last_hidden_state):
            failures.append("last_hidden_state diverged from replicated forward")
        if not torch.equal(expected.attention_mask, got.attention_mask):
            failures.append("attention_mask diverged from replicated forward")
        for i, (want, have) in enumerate(
            zip(expected.hidden_states, got.hidden_states)
        ):
            if not torch.equal(want, have):
                failures.append(f"hidden_states[{i}] diverged")
        if got.pooler_output is not None:
            failures.append("pooler_output should stay None through the gather")

    verdict = torch.tensor([len(failures)], device="cuda")
    dist.all_reduce(verdict)
    if failures:
        print(f"rank{rank} FAIL {failures}", flush=True)
    if rank == 0:
        print(
            f"ENCODER_DP_TP {'FAIL' if verdict.item() else 'PASS'}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 1 if verdict.item() else 0


class TestEncoderDpUnderTp(CustomTestCase):
    def test_dp_encode_matches_replicated_under_tp(self):
        if not current_platform.is_cuda():
            self.skipTest("exercised on CUDA only")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={_WORLD}",
                "--master-port=29527",
                __file__,
                "--worker",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = proc.stdout + proc.stderr
        self.assertIn("ENCODER_DP_TP PASS", output, output)
        self.assertEqual(proc.returncode, 0, output)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.exit(_worker())
    unittest.main()
