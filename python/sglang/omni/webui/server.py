# SPDX-License-Identifier: Apache-2.0
"""Tiny local UI/proxy for SenseNova U1 interleaved omni sessions."""

from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent


class U1OmniUIHandler(BaseHTTPRequestHandler):
    api_base: str
    request_model: str
    served_model: str
    opener: urllib.request.OpenerDirector

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._serve_file(ROOT / "index.html", "text/html; charset=utf-8")
        elif self.path == "/api/config":
            self._json(
                200,
                {
                    "api_base": self.api_base,
                    "request_model": self.request_model,
                    "served_model": self.served_model,
                },
            )
        elif self.path == "/api/health":
            self._proxy("GET", "/model_info", None)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("content-length", "0") or 0))
        if self.path == "/api/omni/generate":
            if _is_stream_request(body):
                self._proxy_stream("POST", "/v1/omni/generate", body)
            else:
                self._proxy("POST", "/v1/omni/generate", body)
        elif self.path == "/api/omni/close":
            payload = json.loads(body.decode("utf-8") or "{}")
            session_id = str(payload.get("session_id", "")).strip()
            if not session_id:
                self._json(400, {"error": "session_id is required"})
                return
            self._proxy("POST", f"/v1/omni/sessions/{quote(session_id)}/close", b"{}")
        else:
            self.send_error(404)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[u1-ui] {self.address_string()} {fmt % args}")

    def _serve_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header(
            "content-type",
            content_type
            or mimetypes.guess_type(path.name)[0]
            or "application/octet-stream",
        )
        self.send_header("content-length", str(len(data)))
        if content_type and "text/html" in content_type:
            self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _proxy(self, method: str, path: str, body: bytes | None) -> None:
        url = f"{self.api_base.rstrip('/')}{path}"
        headers = {"content-type": "application/json"} if body is not None else {}
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        timeout = 5 if method == "GET" and path == "/health" else 900
        try:
            with self.opener.open(request, timeout=timeout) as response:
                data = response.read()
                self.send_response(response.status)
                self.send_header(
                    "content-type",
                    response.headers.get("content-type", "application/json"),
                )
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as exc:
            data = exc.read()
            self.send_response(exc.code)
            self.send_header(
                "content-type",
                exc.headers.get("content-type", "application/json"),
            )
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            self._json(502, {"error": f"{exc.__class__.__name__}: {exc}"})

    def _proxy_stream(self, method: str, path: str, body: bytes | None) -> None:
        url = f"{self.api_base.rstrip('/')}{path}"
        headers = {"content-type": "application/json"} if body is not None else {}
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=900) as response:
                self.send_response(response.status)
                self.send_header(
                    "content-type",
                    response.headers.get("content-type", "text/event-stream"),
                )
                self.send_header("cache-control", "no-cache")
                self.end_headers()
                # upstream SSE frames are small; large reads buffer until completion
                for chunk in response:
                    if not chunk:
                        continue
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as exc:
            data = exc.read()
            self.send_response(exc.code)
            self.send_header(
                "content-type",
                exc.headers.get("content-type", "application/json"),
            )
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            self._json(502, {"error": f"{exc.__class__.__name__}: {exc}"})

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _is_stream_request(body: bytes) -> bool:
    if not body:
        return False
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(payload.get("stream", False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--api-base", default="http://127.0.0.1:30000")
    parser.add_argument("--request-model", default="sensenova-u1")
    parser.add_argument("--served-model", default="sensenova/SenseNova-U1-8B-MoT")
    args = parser.parse_args()

    U1OmniUIHandler.api_base = args.api_base
    U1OmniUIHandler.request_model = args.request_model
    U1OmniUIHandler.served_model = args.served_model
    U1OmniUIHandler.opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({})
    )
    server = ThreadingHTTPServer((args.host, args.port), U1OmniUIHandler)
    print(f"Omni Lab: http://{args.host}:{args.port}")
    print(f"Proxy target: {args.api_base}")
    print(f"Served model: {args.served_model}")
    server.serve_forever()


if __name__ == "__main__":
    main()
