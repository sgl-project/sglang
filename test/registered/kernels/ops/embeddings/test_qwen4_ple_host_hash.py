from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.qwen4_ple import can_fuse_qwen4_ngram_hash
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.qwen4_ple_utils import assert_host_hash_matches, make_embedding

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_host_hash_matches_cuda():
    embedding = make_embedding(fused=True)
    assert can_fuse_qwen4_ngram_hash(
        torch.ones(1, 3, dtype=torch.long, device="cuda"),
        embedding.layer_multipliers,
        embedding.ngram_heads_vocab_sizes,
        embedding.ngram_heads_offsets,
    )
    assert_host_hash_matches(embedding)


@pytest.mark.parametrize("decode", [False, True])
def test_prepared_batch_ids_match_device(decode):
    from sglang.srt.models import qwen4_exp as model
    from sglang.srt.models.qwen4_exp_ple_staging import EagerHashKey, PleHostStaging

    embedding = make_embedding()
    source = SimpleNamespace(
        row_bytes=2,
        close=lambda: None,
        fetch_rows=lambda ids, out: out.fill(0),
        prefetch_rows=lambda ids: None,
    )
    staging = PleHostStaging(source, 0, 60)
    embedding.ngram_embedding = SimpleNamespace(host_staging=staging)
    embedding.uses_host_staging = True
    contexts = torch.tensor([[7, 0, 11, 13], [3, 5, 17, 19]], device="cuda")
    batch = model._PLEBatch(
        mode=model.ForwardMode.DECODE if decode else model.ForwardMode.EXTEND,
        use_decode_fast_path=decode,
        physical_tokens=2,
        processed_tokens=2,
        lengths=torch.ones(2, device="cuda", dtype=torch.long),
        row_width=1,
        req_indices=torch.tensor([0, 1], device="cuda"),
        token_offsets=torch.tensor([0, 1], device="cuda"),
        valid_tokens=torch.ones(2, device="cuda", dtype=torch.bool),
        state_indices=torch.arange(2, device="cuda"),
        ngram_context=contexts[:, :3] if decode else contexts,
        ngram_eos_token_id=0,
    )
    owner = SimpleNamespace(
        _ple_staged_layers=None,
        _ple_ready_event=None,
        _ple_host_contexts=None,
        start_layer=0,
        end_layer=1,
        ple_ngram_size=3,
        layers=[SimpleNamespace(ple=SimpleNamespace(ple_embedding=embedding))],
    )
    with mock.patch.object(
        model,
        "get_req_to_token_pool",
        return_value=SimpleNamespace(ple_window_cache=None),
    ):
        model.Qwen4ExpModel.prepare_ple_rows(
            owner, SimpleNamespace(input_ids=contexts[:, -1]), ple_batch=batch
        )
        model.get_req_to_token_pool().ple_window_cache = None
        device_ids = embedding.compute_ngram_ids(batch)
        with pytest.raises(RuntimeError, match="first differing index 0"):
            staging.verify(device_ids + 1, EagerHashKey(batch.mode, decode, False))
        embedding.verify_ids_once(batch)
        with mock.patch.object(
            embedding, "compute_ngram_ids", side_effect=AssertionError
        ):
            embedding.verify_ids_once(batch)
    staging.close()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
