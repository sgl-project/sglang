"""Custom metadata must stay fresh without adding work to default DSA replay."""

import sys
from types import SimpleNamespace

import pytest

from sglang.srt.layers.attention import dsa_backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

DSA = dsa_backend.DeepseekSparseAttnBackend


def test_default_multistep_replay_does_not_allocate_hook_batch(monkeypatch):
    """Default DSA replay must not allocate a batch view for empty hooks."""
    from sglang.srt.model_executor import forward_batch_info

    backend = dsa_backend.DeepseekSparseAttnMultiStepBackend.__new__(
        dsa_backend.DeepseekSparseAttnMultiStepBackend
    )
    backend.speculative_num_steps = 3
    backend.attn_backends = [DSA.__new__(DSA), DSA.__new__(DSA)]
    refreshed = []
    backend._init_forward_metadata_replay = refreshed.append

    def unexpected_view(*args, **kwargs):
        pytest.fail("default replay allocated a hook batch view")

    monkeypatch.setattr(forward_batch_info, "build_inner_fb_view", unexpected_view)
    batch = SimpleNamespace(batch_size=2)
    backend.init_forward_metadata_out_graph(batch)
    assert refreshed == [batch]


@pytest.mark.parametrize("multistep", [False, True])
def test_custom_metadata_is_never_frozen_in_glue_graph(multistep):
    from sglang.srt.model_executor.runner.metadata_glue_graph import MetadataGlueGraph

    class CustomDSA(DSA):
        def init_kernel_metadata(self, batch):
            self.observed.append(list(batch.requests))

        def init_forward_metadata_out_graph(self, batch):
            self.init_kernel_metadata(batch)

    child = CustomDSA.__new__(CustomDSA)
    child.observed = []
    backend = (
        SimpleNamespace(
            requires_eager_metadata=False,
            attn_backend_list=[child],
            init_forward_metadata_out_graph=child.init_forward_metadata_out_graph,
        )
        if multistep
        else child
    )
    glue = MetadataGlueGraph("cpu")
    for requests in ([2, 0], [1, 3], [3, 2], [0, 1]):
        glue.run(backend, SimpleNamespace(requests=requests), (2,))
    assert child.observed == [[2, 0], [1, 3], [3, 2], [0, 1]]
    assert glue.disabled
    assert not glue.is_captured((2,))
    assert not DSA.__new__(DSA).requires_eager_metadata


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
