# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Shared test doubles for Ray-backed observability code paths.

These fakes let tests exercise :mod:`sglang.srt.observability.ray_wrappers`,
and any DI surface that wires through it, without requiring Ray to be
installed. The module is consumed by:

* ``test/registered/unit/observability/test_ray_wrappers.py`` — wrapper unit
  tests.
* ``test/registered/unit/observability/test_stat_loggers_di.py`` — DI
  integration tests that verify emissions flow through to the metric instance.
"""

from __future__ import annotations

import importlib
import sys
import types

RAY_FAKE_MODULE_NAMES = (
    "ray",
    "ray.util",
    "ray.util.metrics",
    "ray.serve",
    "ray.serve.exceptions",
    "sglang.srt.observability.ray_wrappers",
)


class FakeRayMetric:
    """Stand-in for ``ray.util.metrics.{Counter,Gauge,Histogram}``.

    Records every ``inc`` / ``set`` / ``observe`` call so tests can assert on
    the forwarded value and tags dict.
    """

    def __init__(
        self,
        name: str = "",
        description: str = "",
        tag_keys: tuple = (),
        boundaries=None,
    ):
        self.name = name
        self.description = description
        self._tag_keys = tuple(tag_keys)
        self.boundaries = list(boundaries) if boundaries is not None else None
        self.calls = []  # list of (op, value, tags)

    def inc(self, value, tags=None):
        self.calls.append(("inc", value, dict(tags or {})))

    def set(self, value, tags=None):
        self.calls.append(("set", value, dict(tags or {})))

    def observe(self, value, tags=None):
        self.calls.append(("observe", value, dict(tags or {})))


class FakeRayServeException(Exception):
    pass


def make_fake_ray_modules(replica_id: str = "test-replica") -> dict:
    """Build a dict of fake ray modules suitable for ``sys.modules.update``.

    Each call returns fresh module objects so different tests can use
    different ``replica_id`` values without cross-contamination.
    """
    ray_pkg = types.ModuleType("ray")
    ray_util = types.ModuleType("ray.util")
    ray_util_metrics = types.ModuleType("ray.util.metrics")
    ray_util_metrics.Counter = FakeRayMetric
    ray_util_metrics.Gauge = FakeRayMetric
    ray_util_metrics.Histogram = FakeRayMetric
    ray_util_metrics.Metric = FakeRayMetric

    ray_serve = types.ModuleType("ray.serve")
    ray_serve_exc = types.ModuleType("ray.serve.exceptions")
    ray_serve_exc.RayServeException = FakeRayServeException
    ray_serve.exceptions = ray_serve_exc

    class _ReplicaCtx:
        class _Id:
            unique_id = replica_id

        replica_id = _Id()

    ray_serve.get_replica_context = lambda: _ReplicaCtx()

    return {
        "ray": ray_pkg,
        "ray.util": ray_util,
        "ray.util.metrics": ray_util_metrics,
        "ray.serve": ray_serve,
        "ray.serve.exceptions": ray_serve_exc,
    }


def load_ray_wrappers_with_fake_ray(replica_id: str = "test-replica"):
    """Inject fake ray modules into ``sys.modules`` and (re)import ``ray_wrappers``."""
    fake = make_fake_ray_modules(replica_id=replica_id)
    sys.modules.update(fake)
    sys.modules.pop("sglang.srt.observability.ray_wrappers", None)
    return importlib.import_module("sglang.srt.observability.ray_wrappers")


def load_ray_wrappers_without_ray():
    """Make ``import ray`` fail and (re)import ``ray_wrappers`` cleanly."""
    for name in (
        "ray",
        "ray.util",
        "ray.util.metrics",
        "ray.serve",
        "ray.serve.exceptions",
    ):
        sys.modules.pop(name, None)
        sys.modules[name] = None  # type: ignore[assignment]
    sys.modules.pop("sglang.srt.observability.ray_wrappers", None)
    return importlib.import_module("sglang.srt.observability.ray_wrappers")


def clear_fake_ray_modules() -> None:
    """Remove the fake ray modules and the cached ray_wrappers import.

    Tests call this from ``tearDown`` so module state does not leak between
    tests.
    """
    for name in RAY_FAKE_MODULE_NAMES:
        sys.modules.pop(name, None)
