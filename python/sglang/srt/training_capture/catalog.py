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


class CatalogError(CaptureError):
    pass


class CatalogConflict(CatalogError):
    pass


class CatalogUnavailable(CatalogError):
    pass


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
    def begin(self, identity: dict) -> CaptureLease: ...

    def heartbeat(self, lease: CaptureLease) -> CaptureLease: ...

    def objects(self, lease: CaptureLease, payload: dict) -> dict: ...

    def seal(self, lease: CaptureLease, payload: dict) -> dict: ...

    def publish(self, payload: dict) -> dict: ...

    def fail(self, lease: CaptureLease, reason: str) -> dict: ...


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

    def _post(self, route: str, payload: dict) -> dict:
        body = canonical_bytes(payload)
        if len(body) > 8 << 20:
            raise CatalogError("Catalog request exceeds metadata budget")
        headers = {
            "Content-Type": "application/json",
            "X-Training-Capture-Protocol": "1",
        }
        if self.bearer_token:
            headers["Authorization"] = "Bearer " + self.bearer_token
        for attempt in range(self.attempts):
            request = urllib.request.Request(
                self.endpoint + route, data=body, headers=headers, method="POST"
            )
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    content = response.read((1 << 20) + 1)
                    if len(content) > 1 << 20:
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
