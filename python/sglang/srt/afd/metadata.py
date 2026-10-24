"""Attention-backend metadata ownership for AFD role-graph capture.

One guard class per attention backend family. The guard owns the padded,
address-stable ForwardBatch that a captured role graph is bound to, and
delegates private metadata allocation and refresh to the backend, retaining
stage activation and restoration here.
"""

from __future__ import annotations

import copy
from typing import Any

from .contracts import AFDError, MetadataContract, validate_non_speculative_batch


def _new_padded_tensor(value: Any, *, rows: int, fill: int = 0) -> Any:
    import torch

    target = torch.full(
        (rows,) + tuple(value.shape[1:]),
        fill,
        dtype=value.dtype,
        device=value.device,
    )
    return target


def _update_padded_tensor(
    target: Any,
    source: Any,
    *,
    rows: int,
    fill: int = 0,
) -> None:
    if source.shape[0] != rows or rows > target.shape[0]:
        raise AFDError(
            "AFD_PADDED_METADATA_ROWS_INVALID",
            f"source={source.shape[0]} rows={rows} capacity={target.shape[0]}",
        )
    if rows < target.shape[0]:
        target[rows:].fill_(fill)
    if rows:
        target[:rows].copy_(source)


_UNSET = object()


class AFDPaddedStaticBatchGuard:
    """Own one padded, address-stable decode batch for a (bucket, stage)."""

    metadata_contract: MetadataContract
    backends: tuple[str, ...] = ()
    backend_class: str = ""

    def __init__(
        self,
        *,
        backend: Any,
        stage: Any,
        bucket_rows: int,
    ) -> None:
        self._identity = self.validate_backend(backend)
        self._backend = backend
        self._stage = stage
        self._bucket_rows = bucket_rows
        self._original_metadata = None
        self._original_stage_batch = _UNSET
        self._original_stage_metadata = _UNSET
        self._capture_metadata = None
        self._capture_batch = None
        self._capture_batch_size: int | None = None
        self._active = False

    @classmethod
    def validate_backend(cls, backend: Any) -> tuple[Any, ...]:
        raise NotImplementedError

    @classmethod
    def initialize_shared_state(cls, *, backend: Any, max_rows: int) -> int:
        """Allocate whatever graph state is shared by every capture."""

        raise NotImplementedError

    @classmethod
    def prepare_stage_metadata(
        cls, *, backend: Any, metadata: Any, stage_index: int
    ) -> None:
        """Bind any backend-specific scratch needed outside graph capture."""

    @property
    def graph_forward_batch(self) -> Any:
        if self._capture_batch is None:
            raise AFDError("AFD_PADDED_BATCH_MISSING")
        return self._capture_batch

    def _prepare_padded_batch(self, forward_batch: Any) -> None:
        from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer

        validate_non_speculative_batch(forward_batch)
        real_requests = forward_batch.batch_size
        real_rows = real_requests
        if real_rows < 1 or real_rows > self._bucket_rows:
            raise AFDError(
                "AFD_PADDED_BATCH_ROWS_INVALID",
                f"real={real_rows} capacity={self._bucket_rows}",
            )
        if forward_batch.positions.ndim != 1:
            raise AFDError(
                "AFD_POSITION_LAYOUT_UNSUPPORTED",
                f"ndim={forward_batch.positions.ndim}",
            )
        requests = self._bucket_rows
        non_padded_tokens = TboForwardBatchPreparer._get_num_token_non_padded_cpu(
            forward_batch
        )
        if self._capture_batch is None:
            padded = copy.copy(forward_batch)
            # Capture owns these scalar addresses. A later live stage must not
            # replace them or retain a parent/S2 sibling's count storage.
            global_count = forward_batch.global_num_token_non_padded
            local_count = forward_batch.num_token_non_padded
            if global_count is not None:
                padded.global_num_token_non_padded = global_count.clone()
            if local_count is not None:
                padded.num_token_non_padded = (
                    padded.global_num_token_non_padded
                    if local_count is global_count
                    else local_count.clone()
                )
            for name in ("input_ids", "positions", "out_cache_loc"):
                setattr(
                    padded,
                    name,
                    _new_padded_tensor(
                        getattr(forward_batch, name), rows=self._bucket_rows
                    ),
                )
            padded.req_pool_indices = _new_padded_tensor(
                forward_batch.req_pool_indices, rows=requests
            )
            padded.seq_lens = _new_padded_tensor(
                forward_batch.seq_lens, rows=requests, fill=1
            )
            padded.seq_lens_cpu = (
                None
                if forward_batch.seq_lens_cpu is None
                else _new_padded_tensor(
                    forward_batch.seq_lens_cpu, rows=requests, fill=1
                )
            )
            padded.batch_size = requests
            self._capture_batch = padded
            self._capture_batch_size = requests
        padded = self._capture_batch
        for name in ("input_ids", "positions", "out_cache_loc"):
            _update_padded_tensor(
                getattr(padded, name), getattr(forward_batch, name), rows=real_rows
            )
        _update_padded_tensor(
            padded.req_pool_indices, forward_batch.req_pool_indices, rows=real_requests
        )
        _update_padded_tensor(
            padded.seq_lens, forward_batch.seq_lens, rows=real_requests, fill=1
        )
        if padded.seq_lens_cpu is not None and forward_batch.seq_lens_cpu is not None:
            _update_padded_tensor(
                padded.seq_lens_cpu,
                forward_batch.seq_lens_cpu,
                rows=real_requests,
                fill=1,
            )
        else:
            padded.seq_lens_cpu = None
        padded.seq_lens_sum = (
            None
            if forward_batch.seq_lens_sum is None
            else forward_batch.seq_lens_sum + requests - real_requests
        )
        if forward_batch.kv_loc_plan is not None:
            self._backend.kv_index_translator.bind_runner_slots(padded)
        padded.residual_stream = None
        padded.num_padding = requests - real_requests
        padded.global_num_token_non_padded_cpu = non_padded_tokens
        if padded.global_num_token_non_padded is not None:
            padded.global_num_token_non_padded.fill_(non_padded_tokens)
        if (
            padded.num_token_non_padded is not None
            and padded.num_token_non_padded is not padded.global_num_token_non_padded
        ):
            padded.num_token_non_padded.fill_(non_padded_tokens)
        padded.forward_metadata_ready = False
        padded.forward_metadata_planned_bs = None
        padded.forward_metadata_planned_num_tokens = None

    def activate_in_graph(self) -> None:
        if not self._active or self._capture_batch is None:
            raise AFDError("AFD_METADATA_CAPTURE_NOT_ACTIVE")
        self._backend.forward_metadata = self._capture_metadata
        if self._original_stage_batch is _UNSET:
            self._original_stage_batch = self._stage.forward_batch
            self._original_stage_metadata = self._stage.attention_metadata
        self._stage.forward_batch = self._capture_batch
        # The stage has to carry it too. A whole-role graph interleaves stages,
        # and the adapter re-points the backend from stage.attention_metadata on
        # every layer, so a stage left holding unpadded metadata would quietly
        # undo this activation from the second layer onwards.
        self._stage.attention_metadata = self._capture_metadata
        self._bind_in_graph()

    def _bind_in_graph(self) -> None:
        raise NotImplementedError

    def restore(self) -> None:
        if self._active:
            self._backend.forward_metadata = self._original_metadata
        if self._original_stage_batch is not _UNSET:
            self._stage.forward_batch = self._original_stage_batch
            self._original_stage_batch = _UNSET
        if self._original_stage_metadata is not _UNSET:
            self._stage.attention_metadata = self._original_stage_metadata
            self._original_stage_metadata = _UNSET
        self._active = False

    def close(self) -> None:
        if self._backend is None:
            return
        try:
            self.restore()
        finally:
            try:
                self._release_metadata()
            finally:
                self._capture_batch = self._capture_metadata = None
                self._original_metadata = None
                self._backend = self._stage = None

    def _release_metadata(self) -> None:
        pass


class FlashAttentionMetadataGuard(AFDPaddedStaticBatchGuard):
    """FA3/FA4 shared-metadata capture with cache identity drift detection."""

    metadata_contract = MetadataContract.STANDARD_FA
    backends = ("fa3", "fa4")
    backend_class = (
        "sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot = None

    @classmethod
    def validate_backend(cls, backend: Any) -> tuple[Any, ...]:
        identity = (
            type(backend).__module__,
            type(backend).__name__,
            backend.prefill_attention_backend_str,
            backend.decode_attention_backend_str,
        )
        if identity[:2] != (
            "sglang.srt.layers.attention.flashattention_backend",
            "FlashAttentionBackend",
        ) or identity[2:] not in (("fa3", "fa3"), ("fa4", "fa4")):
            raise AFDError(
                "AFD_FA_METADATA_BACKEND_UNSUPPORTED",
                f"identity={identity!r}",
            )
        return identity

    @classmethod
    def initialize_shared_state(cls, *, backend: Any, max_rows: int) -> int:
        try:
            return backend.init_interleaved_graph_state(max_rows)
        except RuntimeError as exc:
            cls._raise_backend_error(exc)

    def capture(self, forward_batch: Any) -> None:
        if self._active:
            raise AFDError("AFD_FA_METADATA_GUARD_REENTERED")
        self._original_metadata = self._backend.forward_metadata
        self._prepare_padded_batch(forward_batch)
        self._active = True
        try:
            if self._snapshot is None:
                self._snapshot = self._backend.create_graph_metadata_snapshot(
                    self._capture_batch
                )
                self._capture_metadata = self._snapshot.metadata
            else:
                self._snapshot.refresh(self._capture_batch)
        except Exception as exc:
            self.restore()
            self._raise_backend_error(exc)

    def _bind_in_graph(self) -> None:
        self._backend.init_forward_metadata_in_graph(self._capture_batch)

    def prepare_replay(self, forward_batch: Any) -> None:
        if self._snapshot is None:
            self.assert_stable()
        self._original_metadata = self._backend.forward_metadata
        self._active = True
        try:
            self._prepare_padded_batch(forward_batch)
            self._snapshot.refresh(self._capture_batch)
        except Exception as exc:
            self.restore()
            self._raise_backend_error(exc)

    def _release_metadata(self) -> None:
        if self._snapshot is not None:
            self._snapshot.close()
            self._snapshot = None

    def assert_stable(self) -> None:
        if self._snapshot is None or not self._snapshot.is_valid():
            raise AFDError(
                "AFD_FA_METADATA_CACHE_DRIFT",
                "FA3/FA4 graph cache identity changed",
            )

    @staticmethod
    def _raise_backend_error(exc: Exception) -> None:
        # Import only on failure; the other backend families stay CPU-importable.
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionMetadataError,
        )

        if isinstance(exc, FlashAttentionMetadataError):
            raise AFDError("AFD_" + exc.args[0], *exc.args[1:]) from exc
        raise exc


class TRTLLMMHAMetadataGuard(FlashAttentionMetadataGuard):
    """Own TRTLLM decode metadata and bind each stage's private scratch."""

    # Padded stage ownership is shared with FA; the concrete backend identity
    # and implementation remain peer-visible.
    backends = ("trtllm_mha",)
    backend_class = "sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend"

    @classmethod
    def validate_backend(cls, backend: Any) -> tuple[Any, ...]:
        identity = (
            type(backend).__module__,
            type(backend).__name__,
            backend.prefill_attention_backend_str,
            backend.decode_attention_backend_str,
        )
        if identity != (
            "sglang.srt.layers.attention.trtllm_mha_backend",
            "TRTLLMHAAttnBackend",
            "trtllm_mha",
            "trtllm_mha",
        ):
            raise AFDError("AFD_TRTLLM_METADATA_BACKEND_UNSUPPORTED", repr(identity))
        backend.validate_interleaved_graph_support()
        return identity

    @classmethod
    def prepare_stage_metadata(
        cls, *, backend: Any, metadata: Any, stage_index: int
    ) -> None:
        # Prefill/extend stages need private scratch just like captured decode.
        backend.bind_interleaved_stage_workspace(metadata, stage_index=stage_index)

    def capture(self, forward_batch: Any) -> None:
        if self._active:
            raise AFDError("AFD_TRTLLM_METADATA_GUARD_REENTERED")
        self._original_metadata = self._backend.forward_metadata
        self._prepare_padded_batch(forward_batch)
        self._active = True
        try:
            if self._snapshot is None:
                self._snapshot = self._backend.create_graph_metadata_snapshot(
                    self._capture_batch, stage_index=self._stage.index
                )
                self._capture_metadata = self._snapshot.metadata
            else:
                self._snapshot.refresh(self._capture_batch)
        except BaseException:
            self.restore()
            raise

    def _bind_in_graph(self) -> None:
        self._snapshot.bind(self._capture_batch)

    def assert_stable(self) -> None:
        if self._snapshot is None or not self._snapshot.is_valid():
            raise AFDError("AFD_TRTLLM_METADATA_CACHE_DRIFT")

    @staticmethod
    def _raise_backend_error(exc: Exception) -> None:
        raise exc


class DSAMetadataGuard(AFDPaddedStaticBatchGuard):
    """DSA capture against address-private graph state owned by this guard."""

    metadata_contract = MetadataContract.PRIVATE_DSA
    backends = ("nsa",)
    backend_class = "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend"

    @classmethod
    def prepare_stage_metadata(cls, *, backend, metadata, stage_index):
        backend.bind_interleaved_stage_workspace(metadata, stage_index=stage_index)

    @classmethod
    def validate_backend(cls, backend: Any) -> tuple[Any, ...]:
        identity = (
            type(backend).__module__,
            type(backend).__name__,
            backend.prefill_attention_backend_str,
            backend.decode_attention_backend_str,
        )
        if identity[:2] != (
            "sglang.srt.layers.attention.dsa_backend",
            "DeepseekSparseAttnBackend",
        ) or identity[2:] != ("nsa", "nsa"):
            raise AFDError(
                "AFD_DSA_METADATA_BACKEND_UNSUPPORTED",
                f"identity={identity!r}",
            )
        return identity

    @classmethod
    def initialize_shared_state(cls, *, backend: Any, max_rows: int) -> int:
        """DSA state is private per capture, so nothing is shared up front."""

        del backend, max_rows
        return 0

    def capture(self, forward_batch: Any) -> None:
        if self._active:
            raise AFDError("AFD_DSA_METADATA_GUARD_REENTERED")
        self._original_metadata = self._backend.forward_metadata
        self._prepare_padded_batch(forward_batch)
        self._active = True
        try:
            if self._capture_metadata is None:
                self._capture_metadata = (
                    self._backend.init_forward_metadata_for_afd_capture(
                        self._capture_batch
                    )
                )
                if self._capture_metadata is None:
                    raise AFDError("AFD_DSA_METADATA_CAPTURE_MISSING")
                self.prepare_stage_metadata(
                    backend=self._backend,
                    metadata=self._capture_metadata,
                    stage_index=self._stage.index,
                )
            else:
                self.assert_stable()
                self._refresh_private_metadata()
        except Exception:
            self.restore()
            raise

    def _bind_in_graph(self) -> None:
        self._backend.init_forward_metadata_in_graph(self._capture_batch)

    def prepare_replay(self, forward_batch: Any) -> None:
        self.assert_stable()
        self._original_metadata = self._backend.forward_metadata
        self._active = True
        try:
            self._prepare_padded_batch(forward_batch)
            self._refresh_private_metadata()
        except Exception:
            self.restore()
            raise

    def _refresh_private_metadata(self) -> None:
        self._backend.prepare_forward_metadata_for_afd_replay(
            self._capture_metadata,
            self._capture_batch,
            static_forward_batch=self._capture_batch,
        )

    def _release_metadata(self) -> None:
        self._backend.release_afd_capture_metadata(self._capture_metadata)

    def assert_stable(self) -> None:
        if self._capture_metadata is None:
            raise AFDError("AFD_DSA_METADATA_CAPTURE_NOT_ACTIVE")
        if not self._backend.owns_afd_capture_metadata(
            self._capture_metadata, batch_size=self._capture_batch_size
        ):
            raise AFDError(
                "AFD_DSA_METADATA_OWNERSHIP_LOST",
                f"bucket_rows={self._bucket_rows}",
            )


class DSV4MetadataGuard(AFDPaddedStaticBatchGuard):
    """Keep raw decode operands private; record native materialization in graph.

    V4's full metadata owns compressed page tables and lazy FlashMLA scheduling
    buffers. Copying/replacing that object between replays loses captured
    addresses. The native raw->full builder instead runs in each captured graph,
    reading this guard's stable request/length/write-location tensors.
    """

    metadata_contract = MetadataContract.PRIVATE_DSV4
    backends = ("dsv4",)
    backend_class = (
        "sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._raw_metadata = None
        self._raw_addresses = None
        self._saved_backend_state = None

    @classmethod
    def validate_backend(cls, backend: Any) -> tuple[Any, ...]:
        identity = (type(backend).__module__, type(backend).__name__)
        if (
            identity
            != (
                "sglang.srt.layers.attention.deepseek_v4_backend",
                "DeepseekV4AttnBackend",
            )
            or backend.trtllm_attn
            or backend.mega_attn
        ):
            raise AFDError("AFD_DSV4_METADATA_BACKEND_UNSUPPORTED", repr(identity))
        return (*identity, "dsv4", "dsv4")

    @classmethod
    def initialize_shared_state(cls, *, backend: Any, max_rows: int) -> int:
        backend.init_cuda_graph_state(max_rows, max_rows)
        return 0

    def _save_backend(self) -> None:
        self._original_metadata = self._backend.forward_metadata
        self._saved_backend_state = tuple(
            getattr(self._backend, name)
            for name in ("tail_forward_metadata", "encoder_replay", "encoder_row_floor")
        )

    def _refresh_raw(self, forward_batch: Any) -> None:
        self._prepare_padded_batch(forward_batch)
        if not self._capture_batch.forward_mode.is_decode_or_idle():
            raise AFDError("AFD_DSV4_GRAPH_DECODE_REQUIRED")
        raw = self._backend._build_forward_metadata(self._capture_batch)
        if type(raw).__name__ != "DSV4RawDecodeMetadata":
            raise AFDError("AFD_DSV4_RAW_DECODE_METADATA_REQUIRED")
        if self._raw_metadata is None:
            self._raw_metadata = raw
            self._raw_addresses = self._addresses()
        else:
            self._raw_metadata.copy_(raw)

    def _addresses(self) -> tuple[Any, ...]:
        return tuple(
            (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
            for tensor in (
                self._raw_metadata.req_pool_indices,
                self._raw_metadata.seq_lens,
                self._raw_metadata.out_cache_loc,
            )
        )

    def capture(self, forward_batch: Any) -> None:
        if self._active:
            raise AFDError("AFD_DSV4_METADATA_GUARD_REENTERED")
        self._save_backend()
        self._active = True
        try:
            self._refresh_raw(forward_batch)
        except BaseException:
            self.restore()
            raise

    def _bind_in_graph(self) -> None:
        # Called for every native warmup/capture invocation. Always regenerate
        # from raw so materialization belongs to the graph, not just warmup.
        for name in ("freqs_cis_c4", "freqs_cis_c128"):
            self._capture_batch.__dict__.pop(name, None)
        self._backend.forward_metadata = self._raw_metadata
        self._backend.init_forward_metadata_in_graph(self._capture_batch)
        self._capture_metadata = self._backend.forward_metadata
        self._stage.attention_metadata = self._capture_metadata

    def prepare_replay(self, forward_batch: Any) -> None:
        self.assert_stable()
        self._save_backend()
        self._active = True
        try:
            self._refresh_raw(forward_batch)
        except BaseException:
            self.restore()
            raise

    def assert_stable(self) -> None:
        if self._raw_metadata is None or self._addresses() != self._raw_addresses:
            raise AFDError("AFD_DSV4_METADATA_ADDRESS_DRIFT")

    def restore(self) -> None:
        active = self._active
        super().restore()
        if active and self._saved_backend_state is not None:
            (
                self._backend.tail_forward_metadata,
                self._backend.encoder_replay,
                self._backend.encoder_row_floor,
            ) = self._saved_backend_state
            if self._backend.forward_metadata is not None:
                self._backend._activate_request_window()
        self._saved_backend_state = None

    def _release_metadata(self) -> None:
        self._raw_metadata = self._raw_addresses = None


class DSV4TRTLLMMetadataGuard(DSV4MetadataGuard):
    """Native sparse TRTLLM metadata with stage-private execution scratch."""

    backend_class = (
        "sglang.srt.layers.attention.deepseek_v4_trtllm_backend."
        "DeepseekV4TrtllmAttnBackend"
    )

    @classmethod
    def validate_backend(cls, backend):
        identity = (type(backend).__module__, type(backend).__name__)
        if (
            ".".join(identity) != cls.backend_class
            or not backend.trtllm_attn
            or backend.mega_attn
        ):
            raise AFDError("AFD_DSV4_METADATA_BACKEND_UNSUPPORTED", repr(identity))
        return (*identity, "dsv4", "dsv4")

    @classmethod
    def prepare_stage_metadata(cls, *, backend, metadata, stage_index):
        backend.bind_interleaved_stage_workspace(metadata, stage_index=stage_index)

    def capture(self, forward_batch):
        # Allocate before entering CUDA capture, including the first bucket.
        self._backend.prepare_interleaved_stage(self._stage.index)
        super().capture(forward_batch)

    def _bind_in_graph(self):
        super()._bind_in_graph()
        self.prepare_stage_metadata(
            backend=self._backend,
            metadata=self._capture_metadata,
            stage_index=self._stage.index,
        )


def metadata_guard_for_backend(
    backend: str | None, *, default_guard: type[AFDPaddedStaticBatchGuard]
) -> type[AFDPaddedStaticBatchGuard]:
    """Select a backend implementation within the model's metadata contract."""
    if default_guard.metadata_contract is MetadataContract.PRIVATE_DSV4:
        from sglang.srt.runtime_context import get_exec

        if get_exec().kernel.dsv4_attn_backend == "trtllm":
            return DSV4TRTLLMMetadataGuard
    for guard in (
        FlashAttentionMetadataGuard,
        TRTLLMMHAMetadataGuard,
        DSAMetadataGuard,
        DSV4MetadataGuard,
    ):
        if (
            guard.metadata_contract is default_guard.metadata_contract
            and backend in guard.backends
        ):
            return guard
    # Retain the contract's validator and error for unsupported backends.
    return default_guard
