"""PD disaggregation hooks: run SGLang's disaggregated path without a network.

`FAKE_BOOTSTRAP_HOST` selects SGLang's fake KV sender and receiver. It is only
applied after `Req.__init__`, which would otherwise keep the request out of the
radix cache as it does for warmup requests. The PD runner models transfer time;
decode holds each fake transfer until its modelled completion.
"""

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager.state import StateManager


class C_PrefillBootstrapQueueHook(BaseHook):
    """Select FakeKVSender on a prefill instance."""

    HOOK_CLASS_NAME = "PrefillBootstrapQueue"
    HOOK_MODULE_NAME = "sglang.srt.disaggregation.prefill"
    REQUIRED = True

    @classmethod
    def hook(cls, target) -> None:
        original_create_sender = target.create_sender

        def wrapped_create_sender(self, req, *args, **kwargs):
            from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

            # Scoped to sender selection: the request keeps being scheduled.
            original_host = req.bootstrap_host
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
            try:
                created = original_create_sender(self, req, *args, **kwargs)
            finally:
                req.bootstrap_host = original_host
            if created and req.disagg_kv_sender is not None:
                # Simulated queueing is not paced by the fake sender's wall clock.
                req.disagg_kv_sender.waiting_timeout = float("inf")
            return created

        target.create_sender = wrapped_create_sender


class C_DecodeReceiverHook(BaseHook):
    """Select FakeKVReceiver on a decode instance."""

    HOOK_CLASS_NAME = "DecodePreallocQueue"
    HOOK_MODULE_NAME = "sglang.srt.disaggregation.decode"
    REQUIRED = True

    @classmethod
    def hook(cls, target) -> None:
        original_add = target.add

        def wrapped_add(self, req, *args, **kwargs):
            from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

            # A router hands decode the prefill's real host; nothing is sent here.
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
            return original_add(self, req, *args, **kwargs)

        target.add = wrapped_add


class C_DecodeTransferQueueHook(BaseHook):
    """Gate fake KV completion on the simulator clock."""

    HOOK_CLASS_NAME = "DecodeTransferQueue"
    HOOK_MODULE_NAME = "sglang.srt.disaggregation.decode"
    REQUIRED = True

    @classmethod
    def hook(cls, target) -> None:
        original_extend = target.extend
        original_poll_with_metadata_gate = target._poll_with_metadata_gate
        original_poll_with_staging = target._poll_with_staging

        def wrapped_extend(self, decode_reqs):
            now = StateManager.get_global_clock()
            for decode_req in decode_reqs:
                custom_params = decode_req.req.sampling_params.custom_params or {}
                simulation = custom_params.get("simulation") or {}
                done = float(simulation.get("created_time", now)) + float(
                    simulation.get("kv_xfer_visible_s", 0.0)
                )
                StateManager.set_pending_transfer_completion(
                    decode_req.req.rid, max(now, done)
                )
            return original_extend(self, decode_reqs)

        def _gate_polls(polls, decode_reqs):
            from sglang.srt.disaggregation.base.conn import KVPoll

            now = StateManager.get_global_clock()
            for index, (poll, decode_req) in enumerate(zip(polls, decode_reqs)):
                rid = decode_req.req.rid
                done = StateManager.get_pending_transfer_completion(rid)
                if poll == KVPoll.Success and done is not None and now < done:
                    polls[index] = KVPoll.Transferring
                elif poll in (KVPoll.Success, KVPoll.Failed):
                    StateManager.clear_pending_transfer_completion(rid)
            return polls

        # The polls follow the queue's DecodeRequests; iterating the queue
        # itself yields their Reqs.
        def wrapped_poll_with_metadata_gate(self):
            return _gate_polls(
                original_poll_with_metadata_gate(self), list(self.queue.items())
            )

        def wrapped_poll_with_staging(self):
            return _gate_polls(
                original_poll_with_staging(self), list(self.queue.items())
            )

        target.extend = wrapped_extend
        target._poll_with_metadata_gate = wrapped_poll_with_metadata_gate
        target._poll_with_staging = wrapped_poll_with_staging
