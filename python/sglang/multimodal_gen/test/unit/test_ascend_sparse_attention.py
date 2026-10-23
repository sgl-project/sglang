# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for configuration, dense reuse, and packed isolation.

MindIE calls are mocked; these tests do not establish native accuracy.
"""

import builtins
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

MODULE = "sglang.multimodal_gen.runtime.layers.attention.backends.ascend_sparse_attn"


def _reference(query, key, value, *, scale=0.37, **options):
    return F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=scale
    ).transpose(1, 2)


@pytest.fixture
def backend(monkeypatch):
    vendor = ModuleType("mindiesd")
    vendor.sparse_attention = Mock(side_effect=_reference)
    monkeypatch.setitem(sys.modules, "mindiesd", vendor)
    module = importlib.import_module(MODULE)
    monkeypatch.setattr(module, "sparse_attention", vendor.sparse_attention)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(get_soc_version=lambda: 260), raising=False
    )
    executors = []

    def make_dense(**config):
        executor = SimpleNamespace(config=config)
        executor.forward = Mock(
            side_effect=lambda q, k, v, metadata: _reference(q, k, v)
        )
        executors.append(executor)
        return executor

    factory = Mock(side_effect=make_dense)
    monkeypatch.setattr(module, "AscendFAImpl", factory)
    return SimpleNamespace(
        module=module,
        sparse=vendor.sparse_attention,
        dense=factory,
        executors=executors,
    )


def _impl(backend, **options):
    config = dict(num_heads=2, head_size=128, causal=False, softmax_scale=0.37)
    config.update(options)
    return backend.module.AscendSparseAttentionImpl(**config)


def _metadata(backend, **options):
    config = dict(
        current_timestep=10, video_spans=[{"start": 0, "latent_shape": [1, 8, 16]}]
    )
    config.update(options)
    return backend.module.AscendSparseAttentionMetadataBuilder().build(**config)


def _context(monkeypatch, metadata):
    from sglang.multimodal_gen.runtime.managers import forward_context

    monkeypatch.setattr(
        forward_context,
        "get_forward_context_or_none",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )


def test_defaults_work_without_a5_quantization(backend):
    metadata = _metadata(backend)
    assert metadata.precision == "bf16"
    assert metadata.dense_kwargs == {"precision": "bf16"}
    assert metadata.sparsity == 0.6
    assert metadata.skip_last_steps == 0
    assert not hasattr(metadata, "inner_precise")
    assert not hasattr(metadata, "sparse_kwargs")


@pytest.mark.parametrize("sparse_type", ["rf_v2", "rf_v3", "ada_bsa"])
@pytest.mark.parametrize("precision", ["bf16", "mix", "fp8", "mxfp4"])
def test_explicit_sparse_routes_and_builder_blocks(backend, sparse_type, precision):
    metadata = _metadata(backend, sparse_type=sparse_type, precision=precision)
    expected_block = (
        256 if sparse_type == "rf_v3" and precision in ("fp8", "mxfp4") else 128
    )
    assert metadata.block_size == expected_block
    query = torch.zeros(1, 128, 2, 128)
    output = _impl(backend).forward(query, query, query, metadata)
    assert output.shape == query.shape
    options = backend.sparse.call_args.kwargs
    assert options["sparse_type"] == sparse_type
    assert options["precision"] == precision
    assert options["block_size"] == expected_block
    assert options["input_layout"] == "BSND"
    assert options["head_num"] == 2 and options["scale"] == 0.37
    backend.dense.assert_not_called()


@pytest.mark.parametrize("soc, expected", [(220, 0), (250, 0), (260, 4), (270, 4)])
def test_soc_controls_inner_precision(backend, monkeypatch, soc, expected):
    monkeypatch.setattr(torch.npu, "get_soc_version", lambda: soc)
    query = torch.zeros(1, 128, 2, 128)
    _impl(backend).forward(query, query, query, _metadata(backend))
    assert backend.sparse.call_args.kwargs["inner_precise"] == expected


def test_documented_sparse_options_are_forwarded(backend):
    query = torch.zeros(1, 128, 2, 128)
    metadata = _metadata(
        backend,
        block_size=256,
        keep_sink=False,
        keep_recent=False,
        cdf_threshold=0.7,
        mxfp4_dst_type_max=7.25,
        mxfp4_scale_alg=1,
    )
    _impl(backend).forward(query, query, query, metadata)
    options = backend.sparse.call_args.kwargs
    for name in (
        "block_size",
        "keep_sink",
        "keep_recent",
        "cdf_threshold",
        "mxfp4_dst_type_max",
        "mxfp4_scale_alg",
    ):
        assert options[name] == getattr(metadata, name)


@pytest.mark.parametrize(
    "options, match",
    [
        ({"sparse_type": None}, "sparse_type"),
        ({"sparse_type": "rf_v4"}, "sparse_type"),
        ({"precision": "mxfp8"}, "precision"),
        ({"sparsity": float("nan")}, "sparsity"),
        ({"sparsity": float("inf")}, "sparsity"),
        ({"sparsity": -0.1}, "sparsity"),
        ({"sparsity": 1.1}, "sparsity"),
        ({"sparsity": True}, "sparsity"),
        ({"skip_first_steps": True}, "skip_first_steps"),
        ({"skip_last_steps": 2}, "num_steps"),
        ({"dense_last_steps": 2}, "num_steps"),
        ({"skip_last_steps": -1}, "skip_last_steps"),
        ({"current_timestep": -1}, "current_timestep"),
        ({"current_timestep": 1.5}, "current_timestep"),
        ({"num_steps": 10}, "current_timestep"),
        ({"num_steps": 0}, "num_steps"),
        ({"block_size": 0}, "block_size"),
        ({"block_size": 128.0}, "block_size"),
        ({"keep_sink": 1}, "keep_sink"),
        ({"keep_recent": "false"}, "keep_recent"),
        ({"cdf_threshold": 1.1}, "cdf_threshold"),
        ({"mxfp4_dst_type_max": -1}, "mxfp4_dst_type_max"),
        ({"mxfp4_scale_alg": False}, "mxfp4_scale_alg"),
        ({"dense_kwargs": []}, "dense_kwargs"),
        ({"dense_kwargs": {"precision": "float"}}, "precision"),
        ({"sprasity": 0.5}, "Unknown"),
        ({"sparse_kwargs": {}}, "Unknown"),
        ({"inner_precise": 4}, "Unknown"),
        ({"scale": 99}, "Unknown"),
        ({"input_layout": "BNSD"}, "Unknown"),
        (
            {"dense_kwargs": {"unknown_option": 1}},
            "Unknown|Unsupported|unsupported|unknown",
        ),
    ],
)
def test_bad_user_configuration_fails_before_vendor(backend, options, match):
    with pytest.raises(ValueError, match=match):
        _metadata(backend, **options)
    backend.sparse.assert_not_called()
    backend.dense.assert_not_called()


@pytest.mark.parametrize(
    "spans",
    [
        [{"start": -1, "latent_shape": [1, 2, 2]}],
        [{"start": 0, "latent_shape": [1, 2]}],
        [{"start": 0, "latent_shape": [1, 0, 2]}],
        [
            {"start": 0, "latent_shape": [1, 2, 2]},
            {"start": 3, "latent_shape": [1, 2, 2]},
        ],
    ],
)
def test_malformed_spans_are_rejected(backend, spans):
    with pytest.raises(ValueError):
        _metadata(backend, video_spans=spans)


def test_raw_grid_and_prefix_are_forwarded(backend):
    metadata = _metadata(
        backend,
        video_spans=None,
        raw_latent_shape=[1, 16, 1, 16, 32],
        patch_size=(1, 2, 2),
        txt_len=3,
    )
    query = torch.zeros(1, 131, 2, 128)
    _impl(backend).forward(query, query, query, metadata)
    assert backend.sparse.call_args.kwargs["video_spans"] == [
        {"start": 3, "latent_shape": [1, 8, 16]}
    ]


@pytest.mark.parametrize("patch", [(1, 2), (1, 0, 2), (1, 3, 2)])
def test_bad_patch_grid_is_rejected(backend, patch):
    with pytest.raises(ValueError):
        _metadata(
            backend, video_spans=None, raw_latent_shape=[1, 16, 32], patch_size=patch
        )


def test_migration_aliases_and_conflicts(backend):
    metadata = _metadata(
        backend, dense_last_steps=2, num_steps=50, dense_precision="fp8"
    )
    assert metadata.skip_last_steps == 2
    assert metadata.dense_kwargs["precision"] == "fp8"
    with pytest.raises(ValueError, match="disagree"):
        _metadata(backend, skip_last_steps=3, dense_last_steps=2, num_steps=50)
    with pytest.raises(ValueError, match="disagree"):
        _metadata(backend, dense_precision="fp8", dense_kwargs={"precision": "bf16"})


@pytest.mark.parametrize(
    "step, dense",
    [(0, True), (9, True), (10, False), (47, False), (48, True), (49, True)],
)
def test_schedule_boundaries(backend, step, dense):
    query = torch.zeros(1, 128, 2, 128)
    _impl(backend).forward(
        query,
        query,
        query,
        _metadata(backend, current_timestep=step, num_steps=50, skip_last_steps=2),
    )
    assert bool(backend.dense.call_count) == dense
    assert bool(backend.sparse.call_count) != dense


def test_dense_executor_uses_local_configuration_and_is_reused(backend):
    impl = _impl(backend)
    query = torch.zeros(1, 128, 2, 128)
    for _ in range(2):
        impl.forward(
            query,
            query,
            query,
            _metadata(backend, video_spans=[], dense_kwargs={"precision": "fp8"}),
        )
    backend.dense.assert_called_once()
    assert backend.executors[0].config["attention_config"]["precision"] == "fp8"
    assert backend.executors[0].forward.call_count == 2
    impl.forward(
        query,
        query,
        query,
        _metadata(backend, video_spans=[], dense_kwargs={"precision": "bf16"}),
    )
    assert backend.dense.call_count == 2
    backend.sparse.assert_not_called()


@pytest.mark.parametrize("dense", [False, True])
def test_batch_samples_remain_independent(backend, dense):
    query = torch.zeros(2, 128, 2, 128)
    value = torch.cat((torch.ones_like(query[:1]), torch.full_like(query[:1], 23)))
    output = _impl(backend).forward(
        query, query, value, _metadata(backend, current_timestep=0 if dense else 10)
    )
    torch.testing.assert_close(output, value)
    if not dense:
        assert backend.sparse.call_count == 2
        assert all(call.args[0].shape[0] == 1 for call in backend.sparse.call_args_list)


def test_packed_documents_rebase_spans_and_use_host_bounds(backend, monkeypatch):
    spans = [
        {"start": 3, "latent_shape": [1, 8, 16]},
        {"start": 136, "latent_shape": [1, 8, 16]},
    ]
    _context(monkeypatch, _metadata(backend, video_spans=spans))
    query = torch.zeros(264, 2, 128)
    value = torch.cat((torch.ones_like(query[:131]), torch.full_like(query[131:], 23)))
    bounds = torch.tensor([0, 131, 264], dtype=torch.int32)
    monkeypatch.setattr(
        bounds, "tolist", Mock(side_effect=AssertionError("host bounds required"))
    )
    output = _impl(backend).forward_varlen(
        query,
        query,
        value,
        cu_seqlens=bounds,
        cu_seqlens_host=(0, 131, 264),
        max_seqlen=133,
    )
    torch.testing.assert_close(output, value)
    assert [
        call.kwargs["video_spans"][0]["start"] for call in backend.sparse.call_args_list
    ] == [3, 5]
    assert spans[1]["start"] == 136


@pytest.mark.parametrize("flag", [False, True])
def test_padding_is_skipped_only_when_explicitly_flagged(backend, monkeypatch, flag):
    _context(monkeypatch, _metadata(backend, video_spans=[]))
    query = torch.zeros(64, 2, 128)
    value = torch.cat((torch.ones_like(query[:33]), torch.full_like(query[33:], 31)))
    output = _impl(backend, packed_trailing_padding=flag).forward_varlen(
        query, query, value, cu_seqlens=torch.tensor([0, 33, 64]), max_seqlen=33
    )
    torch.testing.assert_close(output[:33], value[:33])
    if flag:
        assert output[33:].count_nonzero() == 0
        assert backend.executors[0].forward.call_count == 1
    else:
        torch.testing.assert_close(output[33:], value[33:])
        assert backend.executors[0].forward.call_count == 2


@pytest.mark.parametrize(
    "context",
    [
        None,
        SimpleNamespace(attn_metadata=None),
        SimpleNamespace(attn_metadata=object()),
    ],
)
def test_missing_packed_metadata_raises_explicit_error(backend, monkeypatch, context):
    from sglang.multimodal_gen.runtime.managers import forward_context

    monkeypatch.setattr(forward_context, "get_forward_context_or_none", lambda: context)
    query = torch.zeros(5, 2, 128)
    with pytest.raises(RuntimeError, match="metadata"):
        _impl(backend).forward_varlen(
            query, query, query, cu_seqlens=torch.tensor([0, 5]), max_seqlen=5
        )


@pytest.mark.parametrize(
    "bounds, maximum",
    [([0, -1, 5], 5), ([0, 6, 5], 6), ([0, 5], 4), ([0, 5], True), ([1, 5], 5)],
)
def test_bad_packed_boundaries_are_rejected(backend, monkeypatch, bounds, maximum):
    _context(monkeypatch, _metadata(backend, video_spans=[]))
    query = torch.zeros(5, 2, 128)
    with pytest.raises(ValueError):
        _impl(backend).forward_varlen(
            query, query, query, cu_seqlens=torch.tensor(bounds), max_seqlen=maximum
        )


def test_cross_document_video_span_is_rejected(backend, monkeypatch):
    _context(
        monkeypatch,
        _metadata(backend, video_spans=[{"start": 3, "latent_shape": [1, 2, 2]}]),
    )
    query = torch.zeros(10, 2, 128)
    with pytest.raises(ValueError, match="cross"):
        _impl(backend).forward_varlen(
            query, query, query, cu_seqlens=torch.tensor([0, 5, 10]), max_seqlen=5
        )


def test_global_grid_cannot_be_reused_for_packed_documents(backend, monkeypatch):
    _context(
        monkeypatch,
        _metadata(
            backend, video_spans=None, raw_latent_shape=[1, 2, 2], patch_size=(1, 1, 1)
        ),
    )
    query = torch.zeros(8, 2, 128)
    with pytest.raises(ValueError, match="explicit video_spans"):
        _impl(backend).forward_varlen(
            query, query, query, cu_seqlens=torch.tensor([0, 4, 8]), max_seqlen=4
        )


def test_span_in_flagged_padding_is_rejected(backend, monkeypatch):
    _context(
        monkeypatch,
        _metadata(backend, video_spans=[{"start": 33, "latent_shape": [1, 2, 2]}]),
    )
    query = torch.zeros(64, 2, 128)
    with pytest.raises(ValueError, match="live packed"):
        _impl(backend, packed_trailing_padding=True).forward_varlen(
            query, query, query, cu_seqlens=torch.tensor([0, 33, 64]), max_seqlen=33
        )


def test_truthy_padding_flag_is_not_accepted(backend):
    with pytest.raises(ValueError, match="boolean"):
        _impl(backend, packed_trailing_padding="false")


@pytest.mark.parametrize("dense", [False, True])
def test_execution_errors_propagate_without_retry(backend, dense):
    query = torch.zeros(1, 128, 2, 128)
    impl = _impl(backend)
    metadata = _metadata(
        backend, video_spans=[] if dense else [{"start": 0, "latent_shape": [1, 8, 16]}]
    )
    if dense:
        impl.forward(query, query, query, metadata)
        operator = backend.executors[0].forward
    else:
        operator = backend.sparse
    operator.side_effect = ValueError("native execution failure")
    with pytest.raises(ValueError, match="native execution"):
        impl.forward(query, query, query, metadata)


def test_registry_and_packed_capability(backend):
    from sglang.multimodal_gen.runtime.platforms.npu import NPUPlatformBase

    cls = backend.module.AscendSparseAttentionBackend
    assert cls.get_enum() is AttentionBackendEnum.ASCEND_SPARSE_ATTN
    assert cls.unsupported_requirements(AttentionRequirements(packed_varlen=True)) == ()
    assert (
        NPUPlatformBase.get_attn_backend_cls_str(
            AttentionBackendEnum.ASCEND_SPARSE_ATTN, 128, torch.bfloat16
        )
        == f"{MODULE}.AscendSparseAttentionBackend"
    )


@pytest.mark.parametrize("name", ["attn_mask", "metadata", "block_table", "sinks"])
def test_dense_tensor_options_are_checked_before_packed_splitting(
    backend, monkeypatch, name
):
    _context(
        monkeypatch,
        _metadata(
            backend,
            video_spans=[],
            dense_kwargs={"precision": "mxfp4", name: torch.ones(1)},
        ),
    )
    query = torch.zeros(8, 2, 128)
    with pytest.raises(ValueError, match="single|one unpadded"):
        _impl(backend).forward_varlen(
            query, query, query, cu_seqlens=torch.tensor([0, 4, 8]), max_seqlen=4
        )
    backend.dense.assert_not_called()
    backend.sparse.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"skip_last_steps": True, "dense_last_steps": 1},
        {"skip_last_steps": 1, "dense_last_steps": True},
    ],
)
def test_migration_aliases_do_not_hide_invalid_boolean_steps(backend, options):
    with pytest.raises(ValueError, match="steps"):
        _metadata(backend, num_steps=50, **options)


def test_missing_sparse_sdk_raises_during_platform_resolution(backend, monkeypatch):
    from sglang.multimodal_gen.runtime.platforms.npu import NPUPlatformBase

    original_import = builtins.__import__

    def without_mindie(name, *args, **kwargs):
        if name == "mindiesd":
            raise ImportError("MindIE-SD is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, MODULE)
    monkeypatch.setattr(builtins, "__import__", without_mindie)
    with pytest.raises(ImportError, match="MindIE-SD"):
        NPUPlatformBase.get_attn_backend_cls_str(
            AttentionBackendEnum.ASCEND_SPARSE_ATTN, 128, torch.bfloat16
        )
