from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams


@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("topk", [False, True])
@pytest.mark.parametrize("dsa", [False, True])
@pytest.mark.parametrize("same", [False, True])
def test_single_relay(hidden, topk, dsa, same):
    f = object.__new__(FutureMap)
    f.spec_algo = SimpleNamespace(is_ngram=lambda: False)
    f.device = "cuda"
    f.need_topk = topk
    f.need_hidden_states = hidden
    f.output_tokens_buf = torch.arange(9, device="cuda", dtype=torch.int64)
    f.topk_p_buf = torch.randn(9, 3, device="cuda")
    f.topk_index_buf = torch.randint(0, 100, (9, 3), device="cuda")
    f.hidden_states_buf = torch.randn(9, 17, device="cuda", dtype=torch.bfloat16)
    f.draft_probs_buf = torch.randn(9, 23, device="cuda")
    f.dsa_topk_indices_buf = torch.randint(0, 100, (9, 4), device="cuda")
    ids = torch.tensor([3], device="cuda", dtype=torch.int64)
    d = SimpleNamespace(
        future_indices=ids if same else ids.clone(),
        draft_probs=torch.empty(0, device="cuda"),
        future_dsa_topk_indices_available=dsa,
    )
    request = Req(
        rid="relay",
        origin_input_text="",
        origin_input_ids=array("q", [1]),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=16),
    )
    request.kv.req_pool_idx = 3
    b = SimpleNamespace(spec_info=d, req_pool_indices=ids, reqs=[request])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        f._resolve_spec_extras(b)
        fields = [("bonus_tokens", "output_tokens_buf")]
        if topk:
            fields.extend(
                [
                    ("topk_p", "topk_p_buf"),
                    ("topk_index", "topk_index_buf"),
                    ("draft_probs", "draft_probs_buf"),
                ]
            )
        if hidden:
            fields.append(("hidden_states", "hidden_states_buf"))
        if dsa:
            fields.append(("dsa_topk_indices", "dsa_topk_indices_buf"))
        for k, v in fields:
            assert (getattr(d, k).data_ptr() == getattr(f, v)[3:4].data_ptr()) == same
        copies = [getattr(d, k).clone() for k, v in fields]
        references = [getattr(f, v)[3:4].clone() for k, v in fields]
        for k, v in fields:
            getattr(f, v).fill_(-1)
    stream.synchronize()
    for x, y in zip(copies, references):
        assert torch.equal(x, y)
