import torch

from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_dispatch_preserves_topk_weight_dtype():
    class FakeInitRouting:
        def _init_routing(self, hidden_states, topk_ids, num_experts, top_k):
            self.args = (hidden_states, topk_ids, num_experts, top_k)
            return (
                hidden_states,
                torch.empty((hidden_states.shape[0], top_k), dtype=torch.int32),
                torch.empty(num_experts, dtype=torch.int64),
                None,
            )

    dispatcher = object.__new__(AscendTPDispatcher)
    dispatcher.num_experts = 2
    dispatcher.init = FakeInitRouting()
    dispatcher.group_list_type = 1

    hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.12345678, 0.87654321], [0.33333334, 0.66666667]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=torch.empty((2, 2), dtype=torch.float32),
    )

    output = dispatcher.dispatch(hidden_states, topk_output)

    assert output.topk_weights.dtype == torch.float32
    torch.testing.assert_close(output.topk_weights, topk_weights)
    assert output.topk_ids.dtype == torch.int32
    assert dispatcher.init.args[1].dtype == torch.int32


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
