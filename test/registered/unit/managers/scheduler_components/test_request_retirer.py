"""Exits that skipped a release part, now retired through RequestRetirer."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.disaggregation.req_state import DisaggReqState
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import prefill_owner_shell
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _retirer(**deps) -> RequestRetirer:
    args = dict(
        kv=SimpleNamespace(tree_cache=MagicMock()),
        beam_coordinator=MagicMock(),
        hisparse_coordinator=None,
        output_streamer=MagicMock(),
        ipc_channels=MagicMock(),
    )
    args.update(deps)
    return RequestRetirer(**args)


def _image_item() -> MultimodalDataItem:
    return MultimodalDataItem(modality=Modality.IMAGE, feature=torch.ones(2))


class TestBootstrapFailureReleasesMedia(CustomTestCase):
    """A PD prefill bootstrap failure releases the request's media like every
    other abort."""

    def _fail_bootstrap(self, req: Req) -> None:
        retirer = _retirer()
        owner = prefill_owner_shell(
            tree_cache=retirer.tree_cache,
            request_retirer=retirer,
            req_to_metadata_buffer_idx_allocator=Mock(),
            output_streamer=Mock(),
            metrics_sink=SimpleNamespace(enable_metrics=False),
        )
        owner.kv_senders[req.rid] = SimpleNamespace(failure_exception=Mock())
        released = []
        with (
            get_parallel().override(tp_rank=0),
            patch.object(
                MultimodalDataItem,
                "release_transport_proxies",
                autospec=True,
                side_effect=released.append,
            ),
        ):
            owner.handle_bootstrap_failure(req)
        self.assertNotIn(req.rid, owner.kv_senders)
        self.assertIsNotNone(req.finished_reason)
        return released

    def test_releases_this_turns_media(self):
        item = _image_item()
        req = Req("r", "", [1, 2], SamplingParams())
        req.disagg = DisaggReqState()
        req.multimodal_inputs = MultimodalInputs(mm_items=[item])

        released = self._fail_bootstrap(req)

        self.assertEqual(released, [item])
        self.assertIsNone(item.feature)
        self.assertIsNone(req.multimodal_inputs)

    def test_session_turn_releases_its_media_not_the_history(self):
        session = Session(100, streaming=True)
        session._inflight = True
        history = _image_item()
        item = _image_item()
        req = Req("s", "", [1, 2], SamplingParams(), session=session)
        req.disagg = DisaggReqState()
        req.multimodal_inputs = MultimodalInputs(mm_items=[history])
        req.extend_image_inputs(MultimodalInputs(mm_items=[item]))

        released = self._fail_bootstrap(req)

        self.assertEqual(released, [item])
        self.assertIsNone(item.feature)
        self.assertTrue(torch.equal(history.feature, torch.ones(2)))
        self.assertFalse(session._inflight)


if __name__ == "__main__":
    unittest.main()
