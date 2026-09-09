"""TP4 integration checks for live draft pointers and SGLang eagle_sample."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.speculative import eagle_utils
from sglang.srt.speculative.compact_verify import engine
from sglang.srt.speculative.compact_verify.core import Verify


def fixture(b, k, v, rank, tp, kind):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(1847)
    # Identical full fixture on all ranks; release target full logits before timing.
    target = torch.randn((b, k + 1, v), generator=gen, device=device).to(torch.bfloat16)
    q = torch.softmax(
        target[:, :k].float()
        + 0.5 * torch.randn((b, k, v), generator=gen, device=device),
        -1,
    )
    candidates = torch.randint(
        v, (b, k + 1), generator=gen, device=device, dtype=torch.int32
    )
    coins = torch.rand((b, k + 1), generator=gen, device=device)
    final = torch.rand(b, generator=gen, device=device)
    if kind == "identical":
        q = torch.softmax(target[:, :k].float(), -1)
    candidates[:, 1:] = (
        torch.multinomial(q.reshape(-1, v), 1, generator=gen).view(b, k).int()
    )
    if kind == "all_accept":
        coins.zero_()
    elif kind in ("first_reject", "last_reject"):
        step = 0 if kind == "first_reject" else k - 1
        coins.zero_()
        coins[:, step] = 0.999
        for i in range(b):
            target[i, step, candidates[i, step + 1]] = -20
    local = target[..., rank * (v // tp) : (rank + 1) * (v // tp)].contiguous()
    return Verify(local, q, candidates, coins, final)


def compare(a, b):
    for actual, expected in zip(a[:3], b[:3]):
        torch.testing.assert_close(actual.flatten(), expected.flatten(), atol=0, rtol=0)


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    root = Path(__file__).resolve().parents[2]
    assert Path(engine.__file__).resolve().is_relative_to(root / "python")
    result = {
        "tests": [],
        "status": "FAIL",
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }
    runner = None
    try:
        assert engine.runtime_supported()
        obj = fixture(4, 3, 154880, rank, 4, "random")
        runner = engine.ServingVerifier(
            obj.local, obj.q, obj.candidates, obj.coins, obj.final_coins, obj.idx
        )
        for step in range(4):
            # Fresh q allocation each time: the captured graph must follow the
            # updated pointer without copying a full B*K*V staging tensor.
            q = torch.softmax(obj.q.log() + 0.05 * step, -1).contiguous()
            local = (-obj.local if step % 2 else obj.local).contiguous()
            candidates = (obj.candidates + step * 13) % obj.v
            coins = 1 - obj.coins if step % 2 else obj.coins
            indices = obj.idx.flip(1).contiguous() if step % 2 else obj.idx
            runner.bind(local, q, candidates, coins, obj.final_coins, indices)
            actual = runner.run()
            compare(actual, runner.baseline())
            result["tests"].append(f"pointer_and_indices_{step}")
        shifted = (obj.local + 32).contiguous()
        runner.bind(shifted, obj.q, obj.candidates, obj.coins, obj.final_coins, obj.idx)
        compare(runner.run(), runner.baseline())
        assert not bool(runner.domain_invalid_rows.any())
        result["tests"].append("shift_invariant_domain")
        nan_q = obj.q.clone()
        nan_q[0, 0, obj.candidates[0, 1]] = float("nan")
        runner.bind(
            obj.local, nan_q, obj.candidates, obj.coins, obj.final_coins, obj.idx
        )
        compare(runner.run(), runner.baseline())
        result["tests"].append("reference_nan_q_rule")
        tiny_local = obj.local.clone()
        tiny_local[0, 0].fill_(-90)
        if rank == 0:
            tiny_local[0, 0, 1] = 0
        tiny_q = obj.q.clone()
        tiny_q[0, 0, 0] = 0
        tiny_candidates = obj.candidates.clone()
        tiny_candidates[0, 1] = 0
        runner.bind(
            tiny_local, tiny_q, tiny_candidates, obj.coins, obj.final_coins, obj.idx
        )
        compare(runner.run(), runner.baseline())
        result["tests"].append("subnormal_reference_probability")
        runner.bind(
            obj.local, obj.q, obj.candidates, obj.coins, obj.final_coins, obj.idx
        )
        for forced in (0, 1, 8, 9, 16):
            runner.force_flags.zero_()
            runner.force_flags[:forced] = True
            compare(runner.run(), runner.baseline())
        result["tests"].append("repair_chunk_boundaries")
        runner.close()
        runner = None
        # Enter through the actual eagle_sample function at a qualifying size.
        obj = fixture(256, 3, 154880, rank, 4, "random")
        sampling = SimpleNamespace(
            is_any_greedy=False,
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            need_min_p_sampling=False,
            has_custom_logit_processor=False,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            return_sampling_masks=None,
        )
        verify = SimpleNamespace(
            draft_probs=obj.q,
            tree_topk=1,
            max_tree_depth=4,
            draft_token_num=4,
            draft_token=obj.candidates.flatten(),
            retrieve_index=obj.idx,
        )
        batch = SimpleNamespace(
            device="cuda",
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            seq_lens=torch.ones(256, device="cuda", dtype=torch.int32),
            sampling_info=sampling,
            return_logprob=False,
            reqs=[SimpleNamespace(sampling_params=SimpleNamespace(temperature=1.0))]
            * 256,
        )
        original_coins = eagle_utils._verify_coins
        eagle_utils._verify_coins = lambda **kwargs: (obj.coins, obj.final_coins)
        logits = LogitsProcessorOutput(
            next_token_logits=obj.local.flatten(0, 1), compact_verify_sharded=True
        )
        predict, lengths, indices = eagle_utils.eagle_sample(verify, batch, logits)
        ref_predict, ref_indices, ref_counts, _ = obj.baseline()
        compare((predict, indices, lengths - 1), (ref_predict, ref_indices, ref_counts))
        assert engine._cache[(256, 4)].q is None
        cached = engine._cache[(256, 4)]
        cached.bind(
            obj.local, obj.q, obj.candidates, obj.coins, obj.final_coins, obj.idx
        )
        cached.force_flags.fill_(True)
        compare(cached.run(), cached.baseline())
        result["tests"].append("all_1024_rows_repaired")
        eagle_utils._verify_coins = original_coins
        result["tests"].append("actual_eagle_sample_1024")
        result["status"] = "PASS"
    finally:
        if runner is not None:
            runner.close()
        engine.close_all()
        dist.barrier()
        dist.destroy_process_group()
        result["normal_shutdown"] = True
        if rank == 0:
            Path(os.environ["COMPACT_TEST_OUTPUT"]).write_text(
                json.dumps(result, indent=2)
            )
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
