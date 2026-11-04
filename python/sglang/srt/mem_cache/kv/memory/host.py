"""Host memory in the unified buffer's layouts.

HiCache keeps one host pool per sub-pool, laid out like the sub-pool's view:
token KV rows per layer (MHA or MLA), or one state entry per slot. The two end
sub-pools may instead share one host backing laid out in device page
envelopes, when both hold MHA envelopes: they grow from opposite ends of the
buffer, so their host pages can share one arena the same way. A floating
sub-pool has no host-side relocation and always keeps a host pool of its own.
"""

from __future__ import annotations

from typing import Optional, Tuple

from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
from sglang.srt.mem_cache.kv.plan import KVStoragePlan


def page_envelope_pair(plan: KVStoragePlan) -> Optional[Tuple[str, str]]:
    """The owner and the other end sub-pool, owner first, when both lay out
    MHA page envelopes; None otherwise."""
    if len(plan.sub_pools) < 2:
        return None
    up, down = plan.sub_pools[0], plan.sub_pools[-1]
    if not all(isinstance(s.spec, MHASubPoolSpec) for s in (up, down)):
        return None
    owner = plan.owner().name
    assert owner in (up.name, down.name), "the owner sits at an end of the chain"
    return (owner, down.name if owner == up.name else up.name)
