"""Small, deterministic target-KV draft fixtures without a target decoder."""

from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.speculative.dspark_components.dspark_target_kv_contract import (
    KVCompatibility,
    KVEncoderConfig,
    KVSequenceContract,
    KVTrainingContract,
    KVValidation,
    TargetKVDraftContract,
)
from sglang.srt.speculative.dspark_components.dspark_target_kv_inject import (
    TargetKVInjector,
)
from sglang.test.training_capture_utils import make_snapshot


def make_target_kv_contract():
    manifest, _ = make_snapshot()
    return TargetKVDraftContract.decode(
        TargetKVDraftContract(
            teacher=manifest.teacher,
            kv=manifest.kv,
            encoder=KVEncoderConfig(hidden_size=8, rms_norm_eps=1e-6),
            sequence=KVSequenceContract(
                prediction_count=3, input_length=3, mask_token_id=255
            ),
            training=KVTrainingContract(lambda_tv=0.5),
            compatibility=KVCompatibility(
                sglang_revision="test", specforge_revision="test"
            ),
            validation=KVValidation(
                golden_fixture_sha256="3" * 64, parity_rtol=1e-5, parity_atol=1e-5
            ),
        )
    )


def make_target_kv_config(contract=None):
    contract = contract or make_target_kv_contract()
    return {
        "architectures": ["DSparkTargetKVDraftModel"],
        "input_mode": "target_kv",
        "target_kv_contract": msgspec.to_builtins(contract),
        "hidden_size": contract.encoder.hidden_size,
        "enable_confidence_head": False,
    }


class RecordingTargetKVDraft:
    def __init__(self):
        self.writes = []

    def write_target_kv(self, *, target_kv, pool, positions, cache_loc):
        self.writes.append(
            (
                {name: value.clone() for name, value in target_kv.items()},
                positions.clone(),
                cache_loc.clone(),
            )
        )


def make_target_kv_injector():
    writer = RecordingTargetKVDraft()
    mapping = torch.tensor([[7, 3, 9, 2, 8, 4, 1, 6], [12, 10, 15, 11, 14, 13, 16, 17]])
    injector = TargetKVInjector(
        draft_model=writer,
        draft_model_runner=SimpleNamespace(token_to_kv_pool=object()),
        model_runner=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=mapping)
        ),
    )
    injector.weights_digest = "bound-test-weights"
    injector.sources = {"target_k.3": torch.arange(40).reshape(20, 1, 2)}
    reqs = [
        SimpleNamespace(req_pool_idx=i, dspark_projected_context=None) for i in range(2)
    ]
    batch = SimpleNamespace(reqs=reqs, seq_lens_cpu=torch.tensor([3, 2]))
    return injector, writer, batch


def make_minimal_kv_weight_loader(*, tp_size=1, tp_rank=0):
    """Exercise the real loader on a small parameter tree without distributed init."""
    from sglang.srt.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from sglang.srt.models.dspark_target_kv import DSparkTargetKVDraftModel

    model = DSparkTargetKVDraftModel.__new__(DSparkTargetKVDraftModel)
    torch.nn.Module.__init__(model)
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.qkv_proj = QKVParallelLinear(
        2,
        head_size=2,
        total_num_heads=2,
        total_num_kv_heads=1,
        bias=False,
        params_dtype=torch.float32,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    layer.mlp = torch.nn.Module()
    layer.mlp.gate_up_proj = MergedColumnParallelLinear(
        2,
        [4, 4],
        bias=False,
        params_dtype=torch.float32,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    layer.self_attn.o_proj = RowParallelLinear(
        4, 2, bias=False, params_dtype=torch.float32, tp_size=tp_size, tp_rank=tp_rank
    )
    model.layers = torch.nn.ModuleList([layer])
    model.kv_encoder = torch.nn.Linear(2, 2, bias=False)
    model.norm = torch.nn.Linear(2, 2, bias=False)
    model.markov_head = torch.nn.Linear(2, 2, bias=False)
    model.confidence_head = None
    model._fused_kv_write_cache = object()
    model._stacked_ctx_kv_cache = object()
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    return model


def graph_verify_hidden_mode(*, use_aux_hidden):
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    runner = SimpleNamespace(
        model_runner=SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            is_draft_worker=False,
            spec_aux_config=SimpleNamespace(dflash_use_aux_hidden_state=use_aux_hidden),
            attn_backend=None,
        ),
        captured_req_width=4,
        _capture_ragged_verify_layout=lambda num_tokens: None,
    )
    with patch(
        "sglang.srt.speculative.dflash_utils.resolve_dflash_verify_mask_policy",
        return_value=(None, False),
    ):
        return DecodeCudaGraphRunner.get_spec_info(runner, 4).capture_hidden_mode
