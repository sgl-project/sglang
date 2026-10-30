"""Object-store metadata preparation is independent of the weight loader."""

import unittest
from unittest import mock

from sglang.srt.arg_groups.model_path_hook import (
    handle_load_format,
    handle_model_source_paths,
)
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestObjectStorageLoadFormat(unittest.TestCase):
    def test_explicit_modelexpress_survives_object_storage_resolution(self):
        for scheme in ("s3", "gs", "az", "S3"):
            uri = f"{scheme}://bucket/model"
            with self.subTest(uri=uri):
                args = ServerArgs(
                    model_path=uri,
                    load_format="remote_instance",
                    remote_instance_weight_loader_backend="modelexpress",
                    modelexpress_config='{"transport":"nixl"}',
                    device="cuda",
                )

                handle_load_format(args)

                cfg = resolving_view(args)
                self.assertEqual(cfg.load_format, "remote_instance")
                self.assertEqual(
                    cfg.remote_instance_weight_loader_backend, "modelexpress"
                )
                self.assertEqual(cfg.modelexpress_config, '{"transport":"nixl"}')
                self.assertEqual(cfg.model_path, uri)

    def test_object_storage_uses_streamer_only_for_auto(self):
        for load_format, backend in (
            ("auto", "nccl"),
            ("auto", "modelexpress"),
            ("auto", "transfer_engine"),
        ):
            with self.subTest(load_format=load_format, backend=backend):
                args = ServerArgs(
                    model_path="s3://bucket/model",
                    load_format=load_format,
                    remote_instance_weight_loader_backend=backend,
                    device="cuda",
                )
                with mock.patch(
                    "sglang.srt.arg_groups.model_path_hook.is_mistral_native_format",
                    return_value=False,
                ):
                    handle_load_format(args)
                self.assertEqual(resolving_view(args).load_format, "runai_streamer")

    def test_object_storage_preserves_explicit_weight_formats(self):
        for load_format, backend in (
            ("runai_streamer", "modelexpress"),
            ("safetensors", "nccl"),
            ("dummy", "nccl"),
            ("remote", "nccl"),
        ):
            with self.subTest(load_format=load_format, backend=backend):
                args = ServerArgs(
                    model_path="s3://bucket/model",
                    load_format=load_format,
                    remote_instance_weight_loader_backend=backend,
                    device="cuda",
                )
                handle_load_format(args)

                self.assertEqual(resolving_view(args).load_format, load_format)

    def test_object_storage_preserves_configured_remote_backends(self):
        for backend in ("nccl", "transfer_engine", "modelexpress"):
            with self.subTest(backend=backend):
                args = ServerArgs(
                    model_path="s3://bucket/model",
                    load_format="remote_instance",
                    remote_instance_weight_loader_backend=backend,
                    remote_instance_weight_loader_seed_instance_ip="127.0.0.1",
                    remote_instance_weight_loader_seed_instance_service_port=30000,
                    remote_instance_weight_loader_send_weights_group_ports=[5000],
                    speculative_draft_model_path="s3://bucket/draft",
                    device="cuda",
                )
                # Only hardware/package availability is replaced; the real
                # backend validation and loader resolution both execute.
                with mock.patch(
                    "sglang.srt.arg_groups.model_path_hook.importlib.util.find_spec",
                    return_value=object(),
                ):
                    handle_load_format(args)
                cfg = resolving_view(args)
                self.assertEqual(cfg.load_format, "remote_instance")
                self.assertEqual(cfg.remote_instance_weight_loader_backend, backend)
                self.assertIsNone(cfg.speculative_draft_load_format)

    def test_invalid_remote_configuration_resolves_object_storage_default(self):
        for backend, missing in (
            ("nccl", "remote_instance_weight_loader_seed_instance_ip"),
            ("nccl", "remote_instance_weight_loader_seed_instance_service_port"),
            ("nccl", "remote_instance_weight_loader_send_weights_group_ports"),
            ("transfer_engine", "remote_instance_weight_loader_seed_instance_ip"),
            (
                "transfer_engine",
                "remote_instance_weight_loader_seed_instance_service_port",
            ),
        ):
            with self.subTest(backend=backend, missing=missing):
                settings = dict(
                    model_path="s3://bucket/model",
                    load_format="remote_instance",
                    remote_instance_weight_loader_backend=backend,
                    remote_instance_weight_loader_seed_instance_ip="127.0.0.1",
                    remote_instance_weight_loader_seed_instance_service_port=30000,
                    remote_instance_weight_loader_send_weights_group_ports=[5000],
                    speculative_draft_model_path="s3://bucket/draft",
                    device="cuda",
                )
                settings[missing] = None
                args = ServerArgs(**settings)
                handle_load_format(args)
                cfg = resolving_view(args)
                self.assertEqual(cfg.load_format, "runai_streamer")
                self.assertIsNone(cfg.speculative_draft_load_format)

    def test_unavailable_transfer_engine_resolves_object_storage_default(self):
        args = ServerArgs(
            model_path="s3://bucket/model",
            load_format="remote_instance",
            remote_instance_weight_loader_backend="transfer_engine",
            remote_instance_weight_loader_seed_instance_ip="127.0.0.1",
            remote_instance_weight_loader_seed_instance_service_port=30000,
            device="cuda",
        )
        with mock.patch(
            "sglang.srt.arg_groups.model_path_hook.importlib.util.find_spec",
            return_value=None,
        ):
            handle_load_format(args)
        self.assertEqual(resolving_view(args).load_format, "runai_streamer")

    def test_local_modelexpress_selection_is_unchanged(self):
        args = ServerArgs(
            model_path="/local/model",
            load_format="remote_instance",
            remote_instance_weight_loader_backend="modelexpress",
            device="cuda",
        )
        handle_load_format(args)
        self.assertEqual(resolving_view(args).load_format, "remote_instance")

    def test_other_remote_schemes_keep_remote_connector(self):
        args = ServerArgs(
            model_path="redis://host:6379/model",
            load_format="remote_instance",
            remote_instance_weight_loader_backend="modelexpress",
            device="cuda",
        )
        handle_load_format(args)
        self.assertEqual(resolving_view(args).load_format, "remote")

    def test_object_storage_draft_inherits_remote_connector_target(self):
        args = ServerArgs(
            model_path="redis://host:6379/model",
            load_format="remote_instance",
            remote_instance_weight_loader_backend="modelexpress",
            speculative_draft_model_path="s3://bucket/draft",
            device="cuda",
        )
        handle_load_format(args)
        cfg = resolving_view(args)
        self.assertEqual(cfg.load_format, "remote")
        self.assertIsNone(cfg.speculative_draft_load_format)

    def test_object_storage_draft_inherits_modelexpress_unless_explicit(self):
        for draft_format, expected in (
            (None, None),
            ("auto", "runai_streamer"),
            ("remote_instance", "remote_instance"),
            ("runai_streamer", "runai_streamer"),
            ("dummy", "dummy"),
        ):
            with self.subTest(draft_format=draft_format):
                args = ServerArgs(
                    model_path="s3://bucket/model",
                    load_format="remote_instance",
                    remote_instance_weight_loader_backend="modelexpress",
                    speculative_draft_model_path="s3://bucket/draft",
                    speculative_draft_load_format=draft_format,
                    device="cuda",
                )
                handle_load_format(args)
                cfg = resolving_view(args)
                self.assertEqual(cfg.load_format, "remote_instance")
                self.assertEqual(cfg.speculative_draft_load_format, expected)

    def test_object_storage_draft_inherits_local_modelexpress_target(self):
        args = ServerArgs(
            model_path="/local/model",
            load_format="remote_instance",
            remote_instance_weight_loader_backend="modelexpress",
            speculative_draft_model_path="s3://bucket/draft",
            device="cuda",
        )
        handle_load_format(args)
        self.assertIsNone(resolving_view(args).speculative_draft_load_format)

    def test_object_storage_draft_inherits_explicit_weight_format(self):
        args = ServerArgs(
            model_path="/local/model",
            load_format="safetensors",
            speculative_draft_model_path="s3://bucket/draft",
            device="cuda",
        )
        handle_load_format(args)
        self.assertIsNone(resolving_view(args).speculative_draft_load_format)

    def test_object_storage_draft_auto_uses_streamer_for_local_auto_target(self):
        args = ServerArgs(
            model_path="/local/model",
            load_format="auto",
            speculative_draft_model_path="s3://bucket/draft",
            device="cuda",
        )
        with mock.patch(
            "sglang.srt.arg_groups.model_path_hook.is_mistral_native_format",
            return_value=False,
        ):
            handle_load_format(args)
        cfg = resolving_view(args)
        self.assertEqual(cfg.load_format, "auto")
        self.assertEqual(cfg.speculative_draft_load_format, "runai_streamer")

    def test_modelexpress_still_prepares_metadata_once_per_uri(self):
        uri = "s3://bucket/model"
        draft_uri = "s3://bucket/draft"
        args = ServerArgs(
            model_path=uri,
            tokenizer_path=uri,
            speculative_draft_model_path=draft_uri,
            load_format="remote_instance",
            remote_instance_weight_loader_backend="modelexpress",
            device="cuda",
        )
        # Replace only the external metadata download; run both real hooks.
        with mock.patch(
            "sglang.srt.arg_groups.model_path_hook.ObjectStorageModel.download_and_get_path",
            side_effect=["/cache/model", "/cache/draft"],
        ) as download:
            handle_model_source_paths(args)
            handle_load_format(args)

        self.assertEqual(
            download.call_args_list, [mock.call(uri), mock.call(draft_uri)]
        )
        cfg = resolving_view(args)
        self.assertEqual(cfg.load_format, "remote_instance")
        self.assertEqual(cfg.model_path, uri)
        self.assertEqual(cfg.tokenizer_path, uri)
        self.assertEqual(cfg.speculative_draft_model_path, draft_uri)


if __name__ == "__main__":
    unittest.main()
