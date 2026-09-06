# SPDX-License-Identifier: Apache-2.0

"""Fake out-of-tree platform package used by ``test_worker_bootstrap``.

The test copies this file into a temporary directory alongside a generated
``sgl_fake_plugin-0.1.dist-info`` and puts that directory on ``sys.path``, so a
spawned child discovers it through real entry-point metadata.

It records which diffusion modules were already imported at each bootstrap
boundary, which is why it must stay free of diffusion runtime imports beyond
``Platform``: importing an observed module here would corrupt the measurement.
"""

import sys

from sglang.multimodal_gen.runtime.platforms import Platform, PlatformEnum

WORKER_MODULE = "sglang.multimodal_gen.runtime.managers.gpu_worker"
GENERATOR_MODULE = "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator"
SERVER_ARGS_MODULE = "sglang.multimodal_gen.runtime.server_args.server_args"

worker_imported_when_plugin_ran = None
generator_imported_when_plugin_ran = None
server_args_imported_when_plugin_ran = None
backend_initialized_when_plugin_ran = None
worker_imported_when_backend_initialized = None
server_args_imported_when_backend_initialized = None
backend_initialized = False


class FakePlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "fake"
    device_type = "fake"
    dispatch_key = "PrivateUse1"

    def init_backend(self):
        global backend_initialized, server_args_imported_when_backend_initialized
        global worker_imported_when_backend_initialized
        worker_imported_when_backend_initialized = WORKER_MODULE in sys.modules
        server_args_imported_when_backend_initialized = (
            SERVER_ARGS_MODULE in sys.modules
        )
        backend_initialized = True


def activate():
    return "sgl_fake_plugin.FakePlatform"


def replacement(pipe_writer, *args, **kwargs):
    pipe_writer.send(
        {
            "override_ran": True,
            "worker_imported_when_plugin_ran": worker_imported_when_plugin_ran,
            "generator_imported_when_plugin_ran": generator_imported_when_plugin_ran,
            "server_args_imported_when_plugin_ran": (
                server_args_imported_when_plugin_ran
            ),
            "backend_initialized_when_plugin_ran": backend_initialized_when_plugin_ran,
            "worker_imported_when_backend_initialized": (
                worker_imported_when_backend_initialized
            ),
            "server_args_imported_when_backend_initialized": (
                server_args_imported_when_backend_initialized
            ),
            "backend_initialized": backend_initialized,
        }
    )
    pipe_writer.close()


def register():
    global backend_initialized_when_plugin_ran
    global generator_imported_when_plugin_ran, server_args_imported_when_plugin_ran
    global worker_imported_when_plugin_ran
    backend_initialized_when_plugin_ran = backend_initialized
    worker_imported_when_plugin_ran = WORKER_MODULE in sys.modules
    generator_imported_when_plugin_ran = GENERATOR_MODULE in sys.modules
    server_args_imported_when_plugin_ran = SERVER_ARGS_MODULE in sys.modules

    from sglang.multimodal_gen.plugins import HookRegistry, HookType

    HookRegistry.register(
        WORKER_MODULE + ".run_scheduler_process", replacement, HookType.REPLACE
    )
