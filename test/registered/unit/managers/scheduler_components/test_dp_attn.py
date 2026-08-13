import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _GatherReached(Exception):
    pass


class TestMLPSyncTransportLog(CustomTestCase):
    def test_reports_backend_and_rank_once(self):
        group = object()
        sync_info = SimpleNamespace(num_tokens=64, local_forward_mode=1)
        previous_logged = dp_attn._MLP_SYNC_TRANSPORT_LOGGED
        dp_attn._MLP_SYNC_TRANSPORT_LOGGED = False

        try:
            with (
                patch.object(
                    dp_attn.torch.distributed, "get_backend", return_value="gloo"
                ),
                patch.object(
                    dp_attn.torch.distributed, "get_world_size", return_value=16
                ),
                patch.object(dp_attn.torch.distributed, "get_rank", side_effect=[3, 3]),
                patch.object(dp_attn.logger, "info") as log_info,
            ):
                for _ in range(2):
                    dp_attn._log_mlp_sync_transport_once(
                        group=group,
                        group_kind="cpu",
                        device="cpu",
                        overlap_schedule=True,
                        sync_info=sync_info,
                    )
        finally:
            dp_attn._MLP_SYNC_TRANSPORT_LOGGED = previous_logged

        log_info.assert_called_once()
        self.assertEqual(
            log_info.call_args.args[1:],
            ("gloo", "cpu", 16, 3, 3, "cpu", True, 64, 1),
        )


class TestPrepareMLPSyncBatchRaw(CustomTestCase):
    def _run_group_selection(self, *, force_device_group: bool):
        tp_group = SimpleNamespace(
            cpu_group=object(),
            device_group=object(),
            device="cuda:0",
        )

        with (
            dp_attn.envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            dp_attn.envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.override(
                force_device_group
            ),
            patch.object(dp_attn, "world_dp_gather_enabled", return_value=False),
            patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
            patch.object(
                dp_attn.TboDPAttentionPreparer,
                "prepare_all_gather",
                return_value=(True, dp_attn.ForwardMode.IDLE.value),
            ),
            patch.object(dp_attn, "_log_mlp_sync_transport_once") as log_transport,
            patch.object(
                dp_attn.MLPSyncBatchInfo,
                "all_gather",
                side_effect=_GatherReached,
            ) as all_gather,
            self.assertRaises(_GatherReached),
        ):
            dp_attn.prepare_mlp_sync_batch_raw(
                local_batch=None,
                model_runner=None,
                dp_size=2,
                attn_tp_size=8,
                attn_cp_size=1,
                tp_group=tp_group,
                get_idle_batch=lambda: None,
                disable_cuda_graph=True,
                require_mlp_tp_gather=True,
                disable_overlap_schedule=False,
                offload_tags=set(),
            )

        return tp_group, log_transport, all_gather

    def test_overlap_default_selects_cpu_group(self):
        tp_group, log_transport, all_gather = self._run_group_selection(
            force_device_group=False
        )

        all_gather.assert_called_once_with(
            device="cpu",
            group=tp_group.cpu_group,
            use_all_reduce=False,
        )
        self.assertEqual(log_transport.call_args.kwargs["group_kind"], "cpu")

    def test_overlap_env_selects_device_group(self):
        tp_group, log_transport, all_gather = self._run_group_selection(
            force_device_group=True
        )

        all_gather.assert_called_once_with(
            device="cuda:0",
            group=tp_group.device_group,
            use_all_reduce=False,
        )
        self.assertEqual(log_transport.call_args.kwargs["group_kind"], "device")


if __name__ == "__main__":
    unittest.main()
