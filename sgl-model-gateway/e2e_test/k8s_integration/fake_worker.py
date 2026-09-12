"""Minimal fake worker that mimics an SGLang worker for integration testing.

Responds to:
  GET  /health                        -> 200 OK
  GET  /v1/models                     -> {"data": [{"id": "fake-model", "owned_by": "sglang"}]}
  GET  /server_info, /get_server_info -> {"model_path": ..., "version": ..., "tp_size": ..., "dp_size": ...}
  GET  /model_info, /get_model_info   -> {"model_path": ..., "is_generation": true}
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8000


class FakeWorkerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")

        elif self.path == "/v1/models":
            body = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "fake-model",
                            "object": "model",
                            "created": 0,
                            "owned_by": "sglang",
                        }
                    ],
                }
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

        elif self.path in ("/server_info", "/get_server_info"):
            body = json.dumps(
                {
                    "model_path": "fake-model",
                    "version": "0.0.0-test",
                    "tp_size": 1,
                    "dp_size": 1,
                }
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

        elif self.path in ("/model_info", "/get_model_info"):
            body = json.dumps(
                {
                    "model_path": "fake-model",
                    "is_generation": True,
                }
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress per-request logs to keep test output clean
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), FakeWorkerHandler)
    print(f"Fake worker listening on port {PORT}", flush=True)
    server.serve_forever()
