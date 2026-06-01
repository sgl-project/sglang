from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext

# Synthetic "starving adapter" id used as the is_draining_for marker. The real
# drainer stores the id of the adapter being drained-for; any non-None value
# activates the draining branch of LoRADrainer.can_schedule, so a sentinel is
# enough to exercise the real rejection path.
_FORCED_DRAIN_FOR = "scripted-forced-drain"


def _resolve_lora_id(ctx: "ScriptedContext", adapter: str) -> Optional[str]:
    # The scheduler's reqs already carry a resolved lora_id; the name->id map
    # lives in the model runner's lora_manager.lora_refs (lora_id -> LoRARef with
    # lora_path / lora_name). Match on either so the test can pass an adapter path.
    lora_manager = ctx._scheduler.tp_worker.model_runner.lora_manager
    for lora_id, lora_ref in lora_manager.lora_refs.items():
        if adapter in (lora_ref.lora_path, lora_ref.lora_name):
            return lora_id
    return None


def force_lora_drainer_reject(ctx: "ScriptedContext", *, adapter: str) -> None:
    # Drive the real LoRADrainer into a draining state for `adapter` so that the
    # scheduler's existing can_schedule rejection path runs for that adapter's
    # reqs, instead of faking a scheduler decision. There is no clean public
    # entrypoint to force draining (the drainer derives it from starvation each
    # iteration), so we minimally reach into the real drainer object and set the
    # same durable field its own _update_draining_loras would set. is_draining_for
    # persists while the adapter still has running reqs; the drainer only clears
    # it via _update_fully_drained_loras once the adapter has fully drained.
    drainer = ctx._scheduler.lora_drainer
    assert drainer is not None, "force_lora_drainer_reject requires LoRA enabled"

    lora_id = _resolve_lora_id(ctx, adapter)
    assert (
        lora_id is not None
    ), f"force_lora_drainer_reject: no loaded LoRA adapter matches {adapter!r}"

    drainer.adapter_to_stats[lora_id].is_draining_for = _FORCED_DRAIN_FOR
