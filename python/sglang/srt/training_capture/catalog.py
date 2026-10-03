"""Producer-side client for the versioned external CaptureCatalog API."""

from __future__ import annotations

import math
import time
import urllib.error
import urllib.request
from typing import Protocol

import msgspec
from sglang.srt.training_capture.protocol import (
    CaptureError,
    Identifier,
    Positive,
    StrictStruct,
    canonical_bytes,
)

MAX_CATALOG_REQUEST_BYTES = 8 << 20
MAX_CATALOG_RESPONSE_BYTES = 1 << 20


class CatalogError(CaptureError):
    pass


class CatalogConflict(CatalogError):
    pass


class CatalogUnavailable(CatalogError):
    pass


class CatalogContract(StrictStruct):
    contract_id: Identifier
    schema_version: Positive
    payload_format: str
    kv_codecs: list[str]


class CatalogCapabilities(StrictStruct):
    protocol_version: Positive
    contracts: list[CatalogContract]
    store_protocols: list[str]
    hard_pin: bool
    retention_policy: str
    max_request_bytes: Positive


class CaptureLease(StrictStruct):
    capture_id: Identifier
    fencing_token: Positive
    dataset_id: Identifier
    sample_id: Identifier
    generation_id: Identifier
    expires_in_seconds: float
    renew_after_seconds: float

    def __post_init__(self):
        if not (
            math.isfinite(self.expires_in_seconds)
            and 0 < self.renew_after_seconds < self.expires_in_seconds
        ):
            raise ValueError("invalid capture lease renewal interval")

    def credentials(self) -> dict:
        return {"capture_id": self.capture_id, "fencing_token": self.fencing_token}


class Catalog(Protocol):
    def check_compatibility(
        self, *, contract_id: str, kv_codec: str, store_protocol: str
    ) -> CatalogCapabilities: ...

    def begin(self, identity: dict) -> CaptureLease: ...

    def heartbeat(self, lease: CaptureLease) -> CaptureLease: ...

    def objects(self, lease: CaptureLease, payload: dict) -> dict: ...

    def seal(self, lease: CaptureLease, payload: dict) -> dict: ...

    def publish(self, payload: dict) -> dict: ...

    def fail(self, lease: CaptureLease, reason: str) -> dict: ...


def manifest_descriptor(manifest, *, nbytes, sha256):
    return {
        "object_id": "manifest",
        "kind": "manifest",
        "key": manifest.key_prefix + "manifest",
        "nbytes": nbytes,
        "sha256": sha256,
        "owner_id": manifest.topology.aux_owner,
    }


def seal_payload(manifest, descriptor, lease, *, manifest_base64):
    return {
        "owner_id": manifest.topology.aux_owner,
        "sequence": msgspec.to_builtins(manifest.sequence),
        "manifest": descriptor,
        "manifest_base64": manifest_base64,
        "total_tensor_bytes": manifest.total_tensor_bytes,
        "idempotency_key": f"seal-{lease.capture_id}-{descriptor['sha256']}",
    }


def seal_request_nbytes(manifest, descriptor, lease):
    payload = seal_payload(manifest, descriptor, lease, manifest_base64="")
    envelope = canonical_bytes({**payload, **lease.credentials()})
    # Standard Base64 needs no JSON escaping and has four bytes per input triple.
    return len(envelope) + 4 * ((descriptor["nbytes"] + 2) // 3)


def check_seal_capacity(manifest, descriptor, lease):
    if seal_request_nbytes(manifest, descriptor, lease) > MAX_CATALOG_REQUEST_BYTES:
        raise CatalogError("manifest exceeds Catalog seal request budget")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "Catalog redirects are disabled", headers, fp
        )


class HTTPCaptureCatalog:
    """Bounded HTTP calls, with identical bodies and idempotency keys on retry.

    Run on a background thread. The caller owns renewals and durable recovery;
    exhausting a publish retry is not evidence that publication did not happen.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        bearer_token: str | None = None,
        timeout: float = 5,
        attempts: int = 3,
    ):
        if (
            not endpoint.startswith(("http://", "https://"))
            or timeout <= 0
            or attempts < 1
        ):
            raise ValueError("invalid Catalog endpoint or retry limits")
        self.endpoint = endpoint.rstrip("/")
        self.bearer_token = bearer_token
        self.timeout = timeout
        self.attempts = attempts
        # Internal control traffic must not inherit an external-network proxy.
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )

    def _request(self, method: str, route: str, payload: dict | None = None) -> dict:
        body = canonical_bytes(payload) if payload is not None else None
        if body is not None and len(body) > MAX_CATALOG_REQUEST_BYTES:
            raise CatalogError("Catalog request exceeds metadata budget")
        headers = {"X-Training-Capture-Protocol": "1"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.bearer_token:
            headers["Authorization"] = "Bearer " + self.bearer_token
        for attempt in range(self.attempts):
            request = urllib.request.Request(
                self.endpoint + route, data=body, headers=headers, method=method
            )
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    content = response.read(MAX_CATALOG_RESPONSE_BYTES + 1)
                    if len(content) > MAX_CATALOG_RESPONSE_BYTES:
                        raise CatalogError("Catalog response exceeds metadata budget")
                    result = msgspec.json.decode(content)
                    if not isinstance(result, dict):
                        raise CatalogError("Catalog response must be a JSON object")
                    return result
            except urllib.error.HTTPError as error:
                error.close()
                if error.code in (409, 410, 422):
                    raise CatalogConflict(
                        f"Catalog rejected {route}: HTTP {error.code}"
                    ) from error
                if error.code not in (429, 500, 502, 503, 504):
                    raise CatalogError(
                        f"Catalog rejected {route}: HTTP {error.code}"
                    ) from error
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            except msgspec.DecodeError as error:
                raise CatalogError("invalid Catalog response JSON") from error
            if attempt + 1 < self.attempts:
                time.sleep(min(0.1 * 2**attempt, 1))
        raise CatalogUnavailable(
            f"Catalog operation has no confirmed result after bounded retries: {route}"
        )

    def _post(self, route: str, payload: dict) -> dict:
        return self._request("POST", route, payload)

    def capabilities(self) -> CatalogCapabilities:
        result = self._request("GET", "/capabilities")
        try:
            return msgspec.convert(result, type=CatalogCapabilities)
        except msgspec.ValidationError as error:
            raise CatalogError("invalid Catalog capabilities") from error

    def check_compatibility(
        self, *, contract_id: str, kv_codec: str, store_protocol: str
    ) -> CatalogCapabilities:
        capabilities = self.capabilities()
        if capabilities.protocol_version != 1:
            raise CatalogConflict("unsupported Catalog producer protocol")
        if not any(
            contract.contract_id == contract_id
            and contract.schema_version == 1
            and contract.payload_format == "maas_target_kv_v1"
            and kv_codec in contract.kv_codecs
            for contract in capabilities.contracts
        ):
            raise CatalogConflict("Catalog does not support the capture contract/codec")
        if store_protocol not in capabilities.store_protocols:
            raise CatalogConflict("Catalog does not support the Store transport")
        if not capabilities.hard_pin:
            raise CatalogConflict("Catalog must require hard-pinned payloads")
        if capabilities.retention_policy != "retain_until_checkpoint":
            raise CatalogConflict("Catalog must retain payloads through checkpoints")
        if capabilities.max_request_bytes < MAX_CATALOG_REQUEST_BYTES:
            raise CatalogConflict("Catalog request budget is below the producer limit")
        return capabilities

    def begin(self, identity: dict) -> CaptureLease:
        result = self._post("/captures:begin", identity)
        return msgspec.convert(result, type=CaptureLease)

    def heartbeat(self, lease: CaptureLease) -> CaptureLease:
        result = self._post(
            f"/captures/{lease.capture_id}/heartbeat", lease.credentials()
        )
        renewed = msgspec.convert(result, type=CaptureLease)
        if (
            renewed.capture_id,
            renewed.fencing_token,
            renewed.dataset_id,
            renewed.sample_id,
            renewed.generation_id,
        ) != (
            lease.capture_id,
            lease.fencing_token,
            lease.dataset_id,
            lease.sample_id,
            lease.generation_id,
        ):
            raise CatalogConflict("heartbeat changed the capture identity or fence")
        return renewed

    def objects(self, lease: CaptureLease, payload: dict) -> dict:
        result = self._post(
            f"/captures/{lease.capture_id}/objects", {**payload, **lease.credentials()}
        )
        expected = sorted(obj["object_id"] for obj in payload["objects"])
        if (
            result.get("phase") != payload["phase"]
            or result.get("accepted_object_ids") != expected
        ):
            raise CatalogConflict("Catalog did not acknowledge the exact object set")
        return result

    def seal(self, lease: CaptureLease, payload: dict) -> dict:
        return self._post(
            f"/captures/{lease.capture_id}/seal", {**payload, **lease.credentials()}
        )

    def publish(self, payload: dict) -> dict:
        return self._post("/samples:publish", payload)

    def fail(self, lease: CaptureLease, reason: str) -> dict:
        return self._post(
            f"/captures/{lease.capture_id}/fail",
            {
                **lease.credentials(),
                "reason": reason,
                "idempotency_key": f"fail-{lease.capture_id}-{lease.fencing_token}",
            },
        )
