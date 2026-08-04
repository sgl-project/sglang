# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from sglang.multimodal_gen.runtime.entrypoints.realtime_gpu_scaler import (
    KubernetesScaleClient,
)


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b'{"status":"ok"}'


def test_scaler_patches_only_the_named_deployment_scale_subresource(tmp_path):
    token = tmp_path / "token"
    token.write_text("service-account-token")
    calls = []

    def open_request(request, **kwargs):
        calls.append((request, kwargs))
        return _Response()

    client = KubernetesScaleClient(
        host="kubernetes.default.svc",
        port=443,
        namespace="minwm-realtime",
        token_path=token,
        ca_path=Path("/missing-test-ca"),
        opener=open_request,
        ssl_context=object(),
    )
    client.scale("minwm-async-vae", 3)

    request, kwargs = calls[0]
    assert request.full_url.endswith(
        "/apis/apps/v1/namespaces/minwm-realtime/"
        "deployments/minwm-async-vae/scale"
    )
    assert request.method == "PATCH"
    assert request.headers["Authorization"] == "Bearer service-account-token"
    assert json.loads(request.data) == {"spec": {"replicas": 3}}
    assert kwargs["timeout"] == 10


@pytest.mark.parametrize("replicas", [-1, 9])
def test_scaler_enforces_the_production_gpu_replica_bound(tmp_path, replicas):
    token = tmp_path / "token"
    token.write_text("token")
    client = KubernetesScaleClient(
        host="kubernetes.default.svc",
        port=443,
        namespace="minwm-realtime",
        token_path=token,
        ca_path=Path("/missing-test-ca"),
        opener=lambda *_args, **_kwargs: _Response(),
        ssl_context=object(),
    )

    with pytest.raises(ValueError, match="between 0 and 8"):
        client.scale("minwm-async-denoiser", replicas)
