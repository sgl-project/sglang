from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models import minicpm
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_lightning_output_norm_matches_unsharded_reference(monkeypatch, tp_size):
    """Splitting heads across ranks must preserve full-width RMS normalization,
    including rows where one rank's shard has zero energy."""
    values = torch.tensor(
        [
            [0.0, 0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            [3.0, -1.0, 2.0, -4.0, 0.0, 0.0, 1.0, 5.0],
        ]
    )
    weight = torch.tensor([1.0, 0.5, -1.0, 2.0, 0.25, 1.5, 3.0, -0.5])
    expected = F.rms_norm(values, (8,), weight, eps=1e-6)
    shard_width = values.shape[-1] // tp_size
    shard_variances = [
        shard.square().mean(-1, keepdim=True) for shard in values.chunk(tp_size, dim=-1)
    ]
    outputs = []
    for rank, shard in enumerate(values.chunk(tp_size, dim=-1)):
        remote_variance = sum(shard_variances[:rank] + shard_variances[rank + 1 :])
        monkeypatch.setattr(
            minicpm,
            "tensor_model_parallel_all_reduce",
            lambda local_variance: local_variance + remote_variance,
        )
        monkeypatch.setattr(
            minicpm,
            "QKVParallelLinear",
            Mock(
                return_value=Mock(return_value=(torch.zeros(2, shard_width * 3), None))
            ),
        )
        monkeypatch.setattr(
            minicpm,
            "RowParallelLinear",
            Mock(return_value=lambda output: (output, None)),
        )
        monkeypatch.setattr(
            minicpm, "RadixAttention", Mock(return_value=Mock(return_value=shard))
        )
        # Keep CPU tensors on native RMSNorm dispatch on CUDA hosts.
        monkeypatch.setattr(
            minicpm,
            "RMSNorm",
            lambda hidden_size, eps: RMSNorm(hidden_size, eps=eps, force_native=True),
        )
        with get_parallel().override(
            tp_size=tp_size, tp_rank=rank, attn_tp_size=tp_size, attn_tp_rank=rank
        ):
            mixer = minicpm.MiniCPMLightningMixer(
                hidden_size=8,
                num_heads=8,
                num_kv_heads=8,
                head_dim=1,
                use_rope=False,
                qk_norm=False,
                use_output_norm=True,
            )
            mixer.o_norm.weight.weight_loader(mixer.o_norm.weight, weight)
            batch = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))
            outputs.append(mixer(torch.arange(2), torch.zeros_like(values), batch))
    torch.testing.assert_close(torch.cat(outputs, dim=-1), expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
