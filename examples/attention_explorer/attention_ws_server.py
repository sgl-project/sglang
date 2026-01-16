#!/usr/bin/env python3
"""
WebSocket Server for Real-Time Attention Streaming

Bridges SGLang API to WebSocket clients for live token-by-token
attention visualization.

Usage:
    python attention_ws_server.py --sglang-url http://localhost:8000 --port 8765
"""

import asyncio
import json
import argparse
from typing import Set
import aiohttp
from aiohttp import web
import weakref

# Connected WebSocket clients
CLIENTS: Set[web.WebSocketResponse] = weakref.WeakSet()


async def stream_completion(request: web.Request, ws: web.WebSocketResponse, data: dict):
    """Stream a completion request to SGLang and forward tokens to WebSocket."""

    sglang_url = request.app["sglang_url"]

    # Prepare the request for SGLang
    payload = {
        "model": data.get("model", "default"),
        "messages": data.get("messages", []),
        "max_tokens": data.get("max_tokens", 100),
        "temperature": data.get("temperature", 0.6),
        "stream": True,  # Enable streaming
        "return_attention_tokens": True,
        "attention_tokens_top_k": data.get("attention_top_k", 10),
    }

    # Send start message
    await ws.send_json({
        "type": "stream_start",
        "prompt": data.get("messages", [])[-1].get("content", "") if data.get("messages") else ""
    })

    token_index = 0
    full_content = ""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{sglang_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    await ws.send_json({
                        "type": "error",
                        "message": f"SGLang API error: {response.status} - {error_text}"
                    })
                    return

                # Process SSE stream
                async for line in response.content:
                    line = line.decode('utf-8').strip()

                    if not line or not line.startswith('data: '):
                        continue

                    data_str = line[6:]  # Remove 'data: ' prefix

                    if data_str == '[DONE]':
                        break

                    try:
                        chunk = json.loads(data_str)

                        # Extract delta content
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        choice = choices[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")

                        if content:
                            full_content += content

                            # Send token update
                            token_msg = {
                                "type": "token",
                                "index": token_index,
                                "content": content,
                                "full_content": full_content,
                            }

                            # Include attention data if available
                            attention = choice.get("attention_tokens")
                            if attention:
                                token_msg["attention"] = attention

                            # Include finish reason
                            if choice.get("finish_reason"):
                                token_msg["finish_reason"] = choice["finish_reason"]

                            await ws.send_json(token_msg)
                            token_index += 1

                        # Check for attention-only chunks (some implementations send separately)
                        attention_tokens = choice.get("attention_tokens")
                        if attention_tokens and not content:
                            await ws.send_json({
                                "type": "attention_update",
                                "index": token_index - 1,
                                "attention": attention_tokens
                            })

                    except json.JSONDecodeError:
                        continue

        # Send completion message
        await ws.send_json({
            "type": "stream_end",
            "total_tokens": token_index,
            "full_content": full_content
        })

    except Exception as e:
        await ws.send_json({
            "type": "error",
            "message": str(e)
        })


async def websocket_handler(request: web.Request):
    """Handle WebSocket connections."""

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    CLIENTS.add(ws)
    print(f"Client connected. Total clients: {len(CLIENTS)}")

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    action = data.get("action")

                    if action == "ping":
                        await ws.send_json({"type": "pong"})

                    elif action == "analyze":
                        # Start streaming completion
                        await stream_completion(request, ws, data)

                    elif action == "get_models":
                        # Forward to SGLang
                        async with aiohttp.ClientSession() as session:
                            async with session.get(f"{request.app['sglang_url']}/v1/models") as resp:
                                models = await resp.json()
                                await ws.send_json({
                                    "type": "models",
                                    "data": models
                                })

                    else:
                        await ws.send_json({
                            "type": "error",
                            "message": f"Unknown action: {action}"
                        })

                except json.JSONDecodeError:
                    await ws.send_json({
                        "type": "error",
                        "message": "Invalid JSON"
                    })

            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"WebSocket error: {ws.exception()}")

    finally:
        CLIENTS.discard(ws)
        print(f"Client disconnected. Total clients: {len(CLIENTS)}")

    return ws


async def health_handler(request: web.Request):
    """Health check endpoint."""
    return web.json_response({
        "status": "ok",
        "clients": len(CLIENTS),
        "sglang_url": request.app["sglang_url"]
    })


async def index_handler(request: web.Request):
    """Serve info page."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Attention WebSocket Server</title>
        <style>
            body { font-family: monospace; background: #1a1a2e; color: #eee; padding: 2rem; }
            h1 { color: #00d4ff; }
            code { background: #16213e; padding: 0.2rem 0.5rem; border-radius: 4px; }
            .endpoint { margin: 1rem 0; padding: 1rem; background: #16213e; border-radius: 8px; }
        </style>
    </head>
    <body>
        <h1>Attention WebSocket Server</h1>
        <p>Real-time attention streaming for SGLang</p>

        <div class="endpoint">
            <h3>WebSocket Endpoint</h3>
            <code>ws://localhost:PORT/ws</code>
        </div>

        <div class="endpoint">
            <h3>Send Message Format</h3>
            <pre>{
  "action": "analyze",
  "messages": [{"role": "user", "content": "What is AI?"}],
  "max_tokens": 100,
  "temperature": 0.6,
  "attention_top_k": 10
}</pre>
        </div>

        <div class="endpoint">
            <h3>Receive Message Types</h3>
            <ul>
                <li><code>stream_start</code> - Generation started</li>
                <li><code>token</code> - New token with attention data</li>
                <li><code>attention_update</code> - Attention data for token</li>
                <li><code>stream_end</code> - Generation complete</li>
                <li><code>error</code> - Error message</li>
            </ul>
        </div>

        <div class="endpoint">
            <h3>Status</h3>
            <p>Connected clients: <span id="clients">?</span></p>
            <script>
                fetch('/health').then(r => r.json()).then(d => {
                    document.getElementById('clients').textContent = d.clients;
                });
            </script>
        </div>
    </body>
    </html>
    """
    return web.Response(text=html, content_type='text/html')


def create_app(sglang_url: str) -> web.Application:
    """Create the aiohttp application."""
    app = web.Application()
    app["sglang_url"] = sglang_url

    # Add CORS headers
    async def cors_middleware(app, handler):
        async def middleware_handler(request):
            if request.method == "OPTIONS":
                response = web.Response()
            else:
                response = await handler(request)

            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response
        return middleware_handler

    app.middlewares.append(cors_middleware)

    app.router.add_get("/", index_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ws", websocket_handler)

    return app


def main():
    parser = argparse.ArgumentParser(description="WebSocket server for attention streaming")
    parser.add_argument("--sglang-url", default="http://localhost:8000", help="SGLang server URL")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    print(f"Starting Attention WebSocket Server")
    print(f"  SGLang URL: {args.sglang_url}")
    print(f"  WebSocket:  ws://{args.host}:{args.port}/ws")
    print(f"  Info page:  http://{args.host}:{args.port}/")

    app = create_app(args.sglang_url)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
