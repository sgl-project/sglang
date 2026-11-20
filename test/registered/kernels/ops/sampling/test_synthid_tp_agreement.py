"""Two TP ranks whose sampling RNG and logits differ still draw the same tokens
and keep the same SynthID state: under SGLANG_SYNC_TOKEN_IDS_ACROSS_TP the stock
sampler all-reduces the token ids, and SynthID commits the synced ids."""

import socket
import traceback
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.sampling.watermarking.synthid import adapter
from sglang.srt.sampling.watermarking.synthid.algorithm import REFERENCE_SYNTHID_KEYS
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_BATCH = 4
_VOCAB = 4096
_STEPS = 6
_STATE_FIELDS = ("context", "context_history", "history_cursor", "num_calls")
_FILTERS = {
    "top_ks": [8, TOP_K_ALL, 32, 4],
    "top_ps": [0.9, 0.8, 1.0, 0.95],
    "min_ps": [0.0, 0.05, 0.1, 0.0],
}
# name -> (watermarked rows, request filters)
_CASES = {
    "all_rows": ([True] * 4, False),
    "request_filters": ([True] * 4, True),
    "mixed_rows": ([True, False, True, False], True),
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _publish_runtime_context(group) -> None:
    from sglang.srt.runtime_context import get_flags, get_parallel
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", tp_size=2, sampling_backend="flashinfer")
    )
    get_parallel().override_permanently(
        tp_size=2, tp_group=SimpleNamespace(device_group=group)
    )
    get_flags().dp.enabled = False


def _sampling_info(device, on, *, filters):
    values = _FILTERS if filters else {}
    top_ks = torch.tensor(values.get("top_ks", [TOP_K_ALL] * _BATCH))
    top_ps = torch.tensor(values.get("top_ps", [1.0] * _BATCH))
    min_ps = torch.tensor(values.get("min_ps", [0.0] * _BATCH))
    return SamplingBatchInfo(
        temperatures=torch.linspace(0.7, 1.3, _BATCH, device=device).view(-1, 1),
        top_ps=top_ps.to(device),
        top_ks=top_ks.to(device, torch.int32),
        min_ps=min_ps.to(device),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=bool((top_ps < 1).any()),
        need_top_k_sampling=bool((top_ks != TOP_K_ALL).any()),
        need_min_p_sampling=bool((min_ps > 0).any()),
        vocab_size=_VOCAB,
        watermark=adapter._rows(on, device),
    )


def _run_case(rank, device, on, filters):
    """One rank's tokens and fused flag at each step, and its final SynthID
    state."""
    torch.manual_seed(rank)
    features = SimpleNamespace(
        watermark_keys=REFERENCE_SYNTHID_KEYS[:8],
        watermark_ngram_len=None,
        watermark_top_k=64,
        watermark_context_history_size=16,
        watermark_skip_first_ngram_calls=None,
    )
    engine = adapter.make_engine(
        features,
        max_num_reqs=_BATCH + 1,
        max_contexts_per_req=0,
        vocab_size=_VOCAB,
        device=device,
    )
    synthid = engine.sampler
    stock_sampler = Sampler()
    info = _sampling_info(device, on, filters=filters)
    req_pool_indices = torch.arange(1, _BATCH + 1, device=device)
    seed_rows = [row for row, row_on in enumerate(on) if row_on]
    logits_generator = torch.Generator().manual_seed(1)
    perturb_generator = torch.Generator().manual_seed(2)

    tokens, fused = [], []
    for step in range(_STEPS):
        forward_batch = SimpleNamespace(
            sampling_info=info,
            req_pool_indices=req_pool_indices,
            watermark_seed=(seed_rows, [[]] * len(seed_rows)) if step == 0 else None,
        )
        logits = 2 * torch.randn(_BATCH, _VOCAB, generator=logits_generator)
        if rank == 1:
            logits += 1e-3 * torch.randn(_BATCH, _VOCAB, generator=perturb_generator)
        logits = logits.to(device)
        sample = engine.bind_sample(forward_batch)
        sample_step = sample.apply(logits, keep_logits=False)
        token_ids = sample_step.token_ids
        fused.append(token_ids is not None)
        if token_ids is None:
            token_ids = stock_sampler(
                LogitsProcessorOutput(next_token_logits=logits),
                sample_step.sampling_info,
                False,
                [0] * _BATCH,
                [None] * _BATCH,
                torch.full((_BATCH,), step, device=device),
            )
        sample.commit(token_ids)
        tokens.append(token_ids.cpu().to(torch.int64))
    state = {name: getattr(synthid.state, name).cpu() for name in _STATE_FIELDS}
    return {"tokens": tokens, "fused": fused, "state": state}


def _worker(rank, port, results):
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=60),
        )
        device = f"cuda:{rank % torch.cuda.device_count()}"
        torch.cuda.set_device(device)
        _publish_runtime_context(dist.group.WORLD)
        for name, (on, filters) in _CASES.items():
            local = _run_case(rank, device, on, filters)
            gathered = [None, None]
            dist.all_gather_object(gathered, local)
            other = gathered[1 - rank]
            # all_rows fits the fused sample, which make_engine refuses at TP 2.
            assert not any(local["fused"]), (name, local["fused"])
            for step, (mine, theirs) in enumerate(
                zip(local["tokens"], other["tokens"], strict=True)
            ):
                assert torch.equal(mine, theirs), (name, step, mine, theirs)
            for field in _STATE_FIELDS:
                assert torch.equal(local["state"][field], other["state"][field]), (
                    name,
                    field,
                )
        results[rank] = "ok"
    except BaseException:
        results[rank] = traceback.format_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestSynthIDTPAgreement(CustomTestCase):
    def test_ranks_draw_the_same_tokens_and_keep_the_same_state(self):
        ctx = mp.get_context("spawn")
        # Spawned ranks inherit the environment the stock sampler reads at import.
        with (
            envs.SGLANG_SYNC_TOKEN_IDS_ACROSS_TP.override(True),
            ctx.Manager() as manager,
        ):
            results = manager.dict()
            port = _free_port()
            procs = [
                ctx.Process(target=_worker, args=(rank, port, results))
                for rank in range(2)
            ]
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(timeout=300)
            for proc in procs:
                if proc.is_alive():
                    proc.kill()
                    proc.join()
            self.assertEqual(
                dict(results), {0: "ok", 1: "ok"}, "\n".join(results.values())
            )
            self.assertEqual([proc.exitcode for proc in procs], [0, 0])


if __name__ == "__main__":
    unittest.main()
