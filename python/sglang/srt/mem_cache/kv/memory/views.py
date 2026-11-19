"""Public layout view types.

A view is one sub-pool's per-layer storage in one layout. Code outside the KV
package may check a view's type (a backend asserting the layouts it serves),
never a pool class. A view's per-layer methods take the layer id the router's
`KVLayer.view_layer_id` names.
"""

import abc


class KVLayoutView(abc.ABC):
    """Per-layer token KV storage."""


class MHAKVView(KVLayoutView):
    """Separate K and V buffers per layer."""


class MLAKVView(KVLayoutView):
    """One latent buffer per layer. ``set_mla_kv_buffer`` and
    ``get_mla_kv_buffer`` take ``layer_id_override``, so a caller need not
    rewrite ``RadixAttention.layer_id`` to address the view's layer."""

    storage_rank_replicated = True
