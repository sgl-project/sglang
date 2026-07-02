"""CP layer-split KV pool: routes each global layer_id to owned or transient inner pool."""

import logging

import torch

from sglang.srt.layers.utils.cp_utils import (
    cp_layersplit_layer_range,
    cp_layersplit_owner_rank,
    cp_layersplit_owns_layer,
    cp_layersplit_should_broadcast_prefix,
    is_cp_layersplit_active,
)

logger = logging.getLogger(__name__)


class _TransientLayer:
    """Layer view with ``layer_id == 0`` so object-keyed pool methods hit the transient slot."""

    __slots__ = ("_wrapped",)

    def __init__(self, layer):
        object.__setattr__(self, "_wrapped", layer)

    @property
    def layer_id(self):
        return 0

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_wrapped"), name)


class CpLayerSplitKVPool:
    """Routing wrapper: owned layers in ``owned_pool``, others via single-slot ``transient_pool``."""

    def __init__(
        self, num_layers, cp_size, cp_rank, inner_pool_cls, base_kwargs, layer_offset=0
    ):
        self._num_layers = num_layers
        self._cp_size = cp_size
        self._cp_rank = cp_rank
        self._layer_offset = layer_offset

        owned_start, owned_end = cp_layersplit_layer_range(
            num_layers, cp_size, cp_rank, layer_offset
        )
        self._owned_start = owned_start
        self._owned_end = owned_end

        self.owned_pool = inner_pool_cls(
            **{
                **base_kwargs,
                "layer_num": owned_end - owned_start,
                "start_layer": owned_start,
                "end_layer": owned_end - 1,
            }
        )
        self.transient_pool = inner_pool_cls(
            **{**base_kwargs, "layer_num": 1, "start_layer": 0, "end_layer": 0}
        )

        self.start_layer = layer_offset
        self.layer_num = num_layers
        self.layer_transfer_counter = None

        logger.info(
            f"CP layer-split rank {cp_rank}/{cp_size}: owns layers "
            f"[{owned_start}, {owned_end}) ({owned_end - owned_start} layers). "
            f"Allocates 1 transient slot to hold each non-owned layer's prefix KV "
            f"broadcast from its owner rank."
        )

    def _owns(self, layer_id: int) -> bool:
        return cp_layersplit_owns_layer(
            layer_id, self._num_layers, self._cp_size, self._cp_rank, self._layer_offset
        )

    def _route_layer(self, layer_id: int):
        if self._owns(layer_id):
            return self.owned_pool, layer_id
        return self.transient_pool, 0

    def _route_layer_obj(self, layer):
        if self._owns(layer.layer_id):
            return self.owned_pool, layer
        return self.transient_pool, _TransientLayer(layer)

    def get_key_buffer(self, layer_id: int):
        pool, lid = self._route_layer(layer_id)
        return pool.get_key_buffer(lid)

    def get_value_buffer(self, layer_id: int):
        pool, lid = self._route_layer(layer_id)
        return pool.get_value_buffer(lid)

    def get_kv_buffer(self, layer_id: int):
        pool, lid = self._route_layer(layer_id)
        return pool.get_kv_buffer(lid)

    def get_mla_kv_buffer(self, layer, loc, dst_dtype=None):
        pool, layer = self._route_layer_obj(layer)
        return pool.get_mla_kv_buffer(layer, loc, dst_dtype)

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, *args, **kwargs):
        pool, layer = self._route_layer_obj(layer)
        pool.set_kv_buffer(layer, loc, cache_k, cache_v, *args, **kwargs)

    def set_mla_kv_buffer(
        self, layer, loc, cache_k_nope, cache_k_rope, *args, **kwargs
    ):
        pool, layer = self._route_layer_obj(layer)
        pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope, *args, **kwargs)

    def get_index_k_with_scale_buffer(self, layer_id: int):
        pool, lid = self._route_layer(layer_id)
        return pool.get_index_k_with_scale_buffer(lid)

    def get_index_k_continuous(self, layer_id: int, seq_len, page_indices):
        pool, lid = self._route_layer(layer_id)
        return pool.get_index_k_continuous(lid, seq_len, page_indices)

    def get_index_k_scale_continuous(self, layer_id: int, seq_len, page_indices):
        pool, lid = self._route_layer(layer_id)
        return pool.get_index_k_scale_continuous(lid, seq_len, page_indices)

    def get_index_k_scale_buffer(
        self, layer_id: int, seq_len_tensor, page_indices, seq_len_sum, max_seq_len
    ):
        pool, lid = self._route_layer(layer_id)
        return pool.get_index_k_scale_buffer(
            lid, seq_len_tensor, page_indices, seq_len_sum, max_seq_len
        )

    def set_index_k_scale_buffer(self, layer_id: int, loc, index_k, index_k_scale):
        pool, lid = self._route_layer(layer_id)
        return pool.set_index_k_scale_buffer(lid, loc, index_k, index_k_scale)

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter
        self.owned_pool.register_layer_transfer_counter(None)
        self.transient_pool.register_layer_transfer_counter(None)

    def get_contiguous_buf_infos(self):
        return self.owned_pool.get_contiguous_buf_infos()

    def broadcast_owner_layer_prefix(
        self, layer_id: int, forward_batch, kind: str
    ) -> None:
        """Broadcast prefix KV for layer_id from owner rank; kind is ``latent`` or ``indexer``."""
        from sglang.srt.layers.dp_attention import (
            get_attention_cp_group,
            get_attention_cp_rank,
            get_attention_cp_size,
        )
        from sglang.srt.model_executor.forward_context import get_req_to_token_pool

        cp_group = get_attention_cp_group()
        my_rank = get_attention_cp_rank()
        cp_size = get_attention_cp_size()

        owner_rank = cp_layersplit_owner_rank(
            layer_id, self._num_layers, cp_size, self._layer_offset
        )

        req_to_token = get_req_to_token_pool().req_to_token
        req_pool_indices = forward_batch.req_pool_indices
        prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu

        prefix_slots_list = []
        for i, plen in enumerate(prefix_lens_cpu):
            if plen > 0:
                ridx = req_pool_indices[i].item()
                prefix_slots_list.append(req_to_token[ridx, :plen])
        if not prefix_slots_list:
            return
        prefix_slots = torch.cat(prefix_slots_list, dim=0).long()

        owned_layer_idx = layer_id - self._owned_start

        # Owner waits for L2 load-back before using owned KV as broadcast source.
        if my_rank == owner_rank and self.owned_pool.layer_transfer_counter is not None:
            self.owned_pool.layer_transfer_counter.wait_until(owned_layer_idx)

        if kind == "latent":
            ref = self.transient_pool.kv_buffer[0]
            buf = ref.new_empty((len(prefix_slots),) + ref.shape[1:])
            row_indices = prefix_slots
            if my_rank == owner_rank:
                buf.copy_(self.owned_pool.kv_buffer[owned_layer_idx][row_indices])
            recv = cp_group.broadcast(buf, src=owner_rank)
            if my_rank != owner_rank:
                self.transient_pool.kv_buffer[0][row_indices] = recv

        elif kind == "indexer":
            page_size = self.transient_pool.page_size
            assert all(p % page_size == 0 for p in prefix_lens_cpu if p > 0), (
                f"cp-layersplit indexer prefix-gather requires page-aligned prefix_len "
                f"(got {[p for p in prefix_lens_cpu if p > 0]})"
            )
            row_indices = torch.unique(prefix_slots // page_size)
            ref = self.transient_pool.index_k_with_scale_buffer[0]
            buf = ref.new_empty((len(row_indices),) + ref.shape[1:])
            if my_rank == owner_rank:
                buf.copy_(
                    self.owned_pool.index_k_with_scale_buffer[owned_layer_idx][
                        row_indices
                    ]
                )
            recv = cp_group.broadcast(buf, src=owner_rank)
            if my_rank != owner_rank:
                self.transient_pool.index_k_with_scale_buffer[0][row_indices] = recv

        else:
            raise ValueError(f"broadcast_owner_layer_prefix: unknown kind={kind!r}")

    def __getattr__(self, name):
        if name in ("owned_pool", "transient_pool"):
            raise AttributeError(name)
        return getattr(self.owned_pool, name)


def cp_layersplit_broadcast_prefix_if_needed(
    layer_id: int, forward_batch, kind: str
) -> None:
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        is_in_tc_piecewise_cuda_graph,
    )

    if is_in_tc_piecewise_cuda_graph():
        return
    if not cp_layersplit_should_broadcast_prefix(forward_batch):
        return
    pool = get_token_to_kv_pool()
    if not isinstance(pool, CpLayerSplitKVPool):
        return
    pool.broadcast_owner_layer_prefix(layer_id, forward_batch, kind)


def build_kv_pool_maybe_layersplit(
    server_args,
    num_layers: int,
    attn_cp_size: int,
    attn_cp_rank: int,
    inner_pool_cls,
    base_kwargs: dict,
    layer_offset: int = 0,
):
    if not is_cp_layersplit_active(server_args, attn_cp_rank):
        return inner_pool_cls(**base_kwargs)

    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    if not (
        inner_pool_cls is MLATokenToKVPool
        or issubclass(inner_pool_cls, MLATokenToKVPool)
    ):
        raise ValueError(
            f"--enable-dsa-prefill-cp-layersplit requires an MLA or DSA pool class "
            f"(MLATokenToKVPool or a subclass), got {inner_pool_cls.__name__!r}. "
            "Enable this flag only with MLA/DSA attention architectures."
        )

    return CpLayerSplitKVPool(
        num_layers=num_layers,
        cp_size=attn_cp_size,
        cp_rank=attn_cp_rank,
        inner_pool_cls=inner_pool_cls,
        base_kwargs=base_kwargs,
        layer_offset=layer_offset,
    )


def unwrap_cp_layersplit_kv_pool(token_to_kv_pool):
    if isinstance(token_to_kv_pool, CpLayerSplitKVPool):
        return token_to_kv_pool.owned_pool
    return token_to_kv_pool
