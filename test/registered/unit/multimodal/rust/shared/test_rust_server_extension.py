"""Model packages use one extension for server arguments and worker startup."""

import json
import os
import unittest
from array import array
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch, sentinel

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.rust_server import server as server_module  # noqa: E402
from sglang.srt.rust_server.server import RustServer  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# The Rust frontend decodes these headers and renders their `meta_info` keys
# (rust/sglang-server/src/api_server/core/frame.rs); they must match Python.
RUST_FRAME_METADATA_FIXTURE = os.path.join(
    os.path.dirname(__file__),
    *[os.pardir] * 6,
    "rust/sglang-server/testdata/frame_metadata.json",
)


def build_rust_frame_metadata_fixture() -> dict:
    """The headers `push_generation` and `push_abort` send for the weight-version,
    media-count and decode-latency columns, and the `meta_info` keys Python's
    TokenizerManager derives from the same values, with "launch" as the
    server's weight version.
    Regenerate the fixture with ``json.dump(build_rust_frame_metadata_fixture(),
    f, indent=2)`` after changing either side."""
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.utils.weight_versions import (
        WeightVersionSpan,
        add_weight_versions_to_meta_info,
    )

    def meta_info(spans, completion_tokens, image=0, audio=0, video=0):
        # TokenizerManager._handle_batch_output / _handle_abort_req.
        meta = {"weight_version": "launch"}
        if spans is not None:
            add_weight_versions_to_meta_info(meta, spans, completion_tokens)
        for key, count in (
            ("image_tokens", image),
            ("audio_tokens", audio),
            ("video_tokens", video),
        ):
            if count:
                meta[key] = count
        return meta

    no_extras = dict.fromkeys(
        [
            f"{family}_{part}"
            for family in (
                "output_token_logprobs",
                "input_token_logprobs",
                "output_top_logprobs",
                "input_top_logprobs",
                "output_token_ids_logprobs",
                "input_token_ids_logprobs",
            )
            for part in ("val", "idx")
        ]
        + ["output_hidden_states"]
    )

    def push_generation(multimodal, decode_latencies, **columns):
        server = RustServer(Mock(), http_port=0)
        server._multimodal_enabled = multimodal
        payload = SimpleNamespace(**{**no_extras, **columns})
        server.push_generation(payload, decode_latencies)
        header, data = server.server.push_decode_result_batch.call_args.args
        return {
            "header": header.hex(),
            "data": b"".join(data).hex(),
            "decode_latencies": {
                columns["rids"][row]: latency
                for row, latency in decode_latencies.items()
            },
        }

    swapped = [
        WeightVersionSpan(version="old", start=0, end=2),
        WeightVersionSpan(version="new", start=2, end=3),
    ]
    batch = push_generation(
        True,
        {0: 0.25},
        rids=["done", "running"],
        finished_reasons=[{"type": "length", "length": 3}, None],
        prompt_tokens=[5, 6],
        output_ids=[array("q", [1, 2, 3]), array("q", [4])],
        cached_tokens=[0, 0],
        reasoning_tokens=[0, 0],
        weight_versions=[swapped, None],
        image_tokens=[8, 0],
        audio_tokens=[0, 2],
        video_tokens=[0, 0],
    )
    batch["meta_info"] = {
        "done": meta_info(swapped, 3, image=8),
        "running": meta_info(None, 1, audio=2),
    }
    # A text model's frame with output logprobs: the spans and the decode
    # latency follow the extras columns rather than their padding.
    single = [WeightVersionSpan(version="v1", start=0, end=2)]
    logprob_batch = push_generation(
        False,
        {0: 0.5},
        rids=["logprobs", "plain"],
        finished_reasons=[{"type": "length", "length": 2}, None],
        prompt_tokens=[3, 4],
        output_ids=[array("q", [7, 8]), array("q", [9])],
        cached_tokens=[0, 0],
        reasoning_tokens=[0, 0],
        weight_versions=[single, None],
        image_tokens=[0, 0],
        audio_tokens=[0, 0],
        video_tokens=[0, 0],
        output_token_logprobs_val=[[-0.5, -0.25], []],
        output_token_logprobs_idx=[[7, 8], []],
    )
    logprob_batch["meta_info"] = {
        "logprobs": meta_info(single, 2),
        "plain": meta_info(None, 1),
    }
    # The scheduler's spans run past the three tokens the client received.
    aborted = [
        WeightVersionSpan(version="old", start=0, end=2),
        WeightVersionSpan(version="new", start=2, end=9),
    ]
    server = RustServer(Mock(), http_port=0)
    req = SimpleNamespace(origin_input_ids=[1, 2], cached_tokens=0, reasoning_tokens=0)
    server.push_abort(AbortReq(rid="aborted", weight_versions=aborted), req)
    abort_header, _ = server.server.push_decode_result_batch.call_args.args
    return {
        "batch": batch,
        "logprob_batch": logprob_batch,
        "abort": {
            "header": abort_header.hex(),
            "completion_tokens": 3,
            "meta_info": meta_info(aborted, 3),
        },
    }


@contextmanager
def launch_context(server_cls):
    """Patch what `launch` reads so it runs as DP rank 1 of a two-rank node,
    unpinned; yields the `_build_server_args` mock."""
    with (
        patch.object(
            server_module,
            "get_exec",
            return_value=SimpleNamespace(moe=SimpleNamespace(is_ep_scale_joiner=False)),
        ),
        patch.object(
            server_module,
            "get_parallel",
            return_value=SimpleNamespace(
                nnodes=1,
                pp_size=1,
                num_dp_ranks=2,
                attn_dp_rank=1,
                tp_size=2,
                tp_rank=1,
                attn_tp_size=1,
                attn_cp_size=1,
            ),
        ),
        patch.object(server_cls, "_partition_cores", return_value=(None, None)),
        patch.object(
            server_module,
            "get_mm",
            return_value=SimpleNamespace(mm_processor_worker_num=8),
        ),
        patch.object(
            server_module,
            "get_serving",
            return_value=SimpleNamespace(host="::", port=30000),
        ),
        patch.object(
            server_module, "_build_server_args", return_value=sentinel.args
        ) as build_args,
    ):
        yield build_args


class TestRustServerExtension(CustomTestCase):
    def test_launch_uses_the_model_extension_and_instance_worker_state(self):
        extension = ModuleType("model_server")
        extension.Server = Mock()

        class ModelServer(RustServer):
            @classmethod
            def _load_extension(cls):
                return extension

            @classmethod
            def _server_kwargs(cls, scheduler):
                return {"model_setting": sentinel.setting}

            @classmethod
            def _resolve_multimodal(cls, scheduler):
                return None

            def _start_multimodal(self, scheduler):
                self.server.start_mm_workers(sentinel.spec, 8)

        scheduler = SimpleNamespace(
            model_config=SimpleNamespace(is_multimodal=True),
        )
        with launch_context(ModelServer) as build_args:
            instance = ModelServer.launch(scheduler)

        build_args.assert_called_once_with(
            scheduler,
            extension=extension,
            health_generate_headers={},
            default_max_new_tokens=128,
        )
        extension.Server.assert_called_once_with(
            sentinel.args,
            cores=None,
            port_offset=1,
            model_setting=sentinel.setting,
        )
        instance.server.start_mm_workers.assert_called_once_with(sentinel.spec, 8)
        self.assertIsInstance(instance, ModelServer)
        self.assertEqual(instance.http_port, 30001)
        self.assertTrue(instance._multimodal_enabled)

    def test_startup_time_takes_the_engine_shape(self):
        """The frontend's end-to-end phase runs from its process's spawn."""
        server = RustServer(Mock(), http_port=30000)
        self.assertIsNone(server.startup_time)
        process = SimpleNamespace(create_time=lambda: 100.0)
        with (
            patch.object(server_module, "time", SimpleNamespace(time=lambda: 110.0)),
            patch.object(
                server_module, "psutil", SimpleNamespace(Process=lambda: process)
            ),
        ):
            server.set_startup_time(
                {
                    "load_weight": 1.0,
                    "kv_cache_allocation": 0.5,
                    "scheduler_e2e": 3.0,
                    "cuda_graph": {"decode": 2.0},
                }
            )
        startup_time = server.startup_time
        self.assertEqual(startup_time["tokenizer_e2e"], 10.0)
        self.assertEqual(startup_time["scheduler_e2e"], 3.0)
        self.assertEqual(startup_time["load_weight"], 1.0)
        self.assertEqual(startup_time["cuda_graph"]["decode"], 2.0)

    def test_scheduler_aborts_reach_the_rust_egress(self):
        """An abort the scheduler sends outside the output stream becomes a
        one-request terminal batch in Rust mode, keeping the request's counts
        and marked as outside the stream; without the Rust server it still
        goes to the tokenizer socket."""
        import msgspec

        from sglang.srt.managers.io_struct import AbortReq
        from sglang.srt.managers.scheduler import Scheduler

        req = SimpleNamespace(
            origin_input_ids=[1, 2, 3], cached_tokens=2, reasoning_tokens=1
        )
        abort = AbortReq(rid="r#1", abort_message="gone")
        server = RustServer(Mock(), http_port=30000)
        scheduler = SimpleNamespace(rust_server=server, ipc_channels=Mock())
        Scheduler.send_abort_output(scheduler, abort, req)
        header, cols = server.server.push_decode_result_batch.call_args.args
        self.assertEqual(
            msgspec.msgpack.decode(header),
            [["r#1"], [{"type": "abort", "message": "gone"}], [3], [0], [2], [1]]
            + [[]] * 17
            + [[True]],
        )
        self.assertEqual(cols, [b""])
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

        scheduler.rust_server = None
        Scheduler.send_abort_output(scheduler, abort, req)
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once_with(
            abort, req
        )

    def test_frame_metadata_fixture_matches_python(self):
        with open(RUST_FRAME_METADATA_FIXTURE) as f:
            self.assertEqual(json.load(f), build_rust_frame_metadata_fixture())

    def test_launch_resolves_the_mm_pipeline_before_the_server_starts(self):
        """The listener answers once the server is constructed, so a model
        without a Rust MM pipeline must be refused before that; a supported
        one gets workers for the spec resolved up front."""
        extension = ModuleType("server")
        extension.Server = Mock()

        class BuiltinServer(RustServer):
            @classmethod
            def _load_extension(cls):
                return extension

        scheduler = SimpleNamespace(
            server_args=sentinel.server_args,
            processor=sentinel.processor,
            model_config=SimpleNamespace(
                is_multimodal=True,
                hf_config=SimpleNamespace(model_type="unsupported_vlm"),
            ),
        )
        with (
            launch_context(BuiltinServer),
            patch.object(server_module, "RustMmProcessor") as mm_processor,
            patch.object(
                BuiltinServer, "_build_mm_spec", return_value=sentinel.worker_spec
            ),
        ):
            mm_processor.return_value.resolve_spec.return_value = None
            with self.assertRaisesRegex(
                RuntimeError, "no Rust MM pipeline for model_type='unsupported_vlm'"
            ):
                BuiltinServer.launch(scheduler)
            extension.Server.assert_not_called()

            mm_processor.return_value.resolve_spec.return_value = sentinel.spec
            instance = BuiltinServer.launch(scheduler)

        extension.Server.assert_called_once()
        instance.server.start_mm_workers.assert_called_once_with(
            sentinel.worker_spec, 8
        )
        self.assertIs(instance.mm_spec, sentinel.spec)
        self.assertTrue(instance._multimodal_enabled)


if __name__ == "__main__":
    unittest.main()
