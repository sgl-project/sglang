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
    """Stream a completion request to SGLang and forward tokens to WebSocket WITH attention data.

    Uses non-streaming API to get full attention data, then sends tokens progressively
    to simulate streaming while providing complete attention information.
    """

    sglang_url = request.app["sglang_url"]
    stream_delay = data.get("stream_delay", 0.05)  # Delay between tokens for visual effect

    # Prepare the request for SGLang - NON-streaming to get attention data
    payload = {
        "model": data.get("model", "default"),
        "messages": data.get("messages", []),
        "max_tokens": data.get("max_tokens", 100),
        "temperature": data.get("temperature", 0.6),
        "stream": False,  # Non-streaming to get attention data
        "return_attention_tokens": True,
        "attention_tokens_top_k": data.get("attention_top_k", 10),
    }

    prompt_text = data.get("messages", [])[-1].get("content", "") if data.get("messages") else ""

    # Send start message
    await ws.send_json({
        "type": "stream_start",
        "prompt": prompt_text
    })

    try:
        async with aiohttp.ClientSession() as session:
            # First, tokenize the prompt to get token count
            tokenize_payload = {"text": prompt_text}
            prompt_token_count = 0

            try:
                async with session.post(
                    f"{sglang_url}/v1/tokenize",
                    json=tokenize_payload,
                    headers={"Content-Type": "application/json"}
                ) as tok_response:
                    if tok_response.status == 200:
                        tok_data = await tok_response.json()
                        prompt_token_count = len(tok_data.get("tokens", []))
            except:
                pass  # Continue without exact prompt token count

            # Make the completion request
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

                result = await response.json()

        # Parse the response
        choices = result.get("choices", [])
        if not choices:
            await ws.send_json({
                "type": "error",
                "message": "No choices in response"
            })
            return

        choice = choices[0]
        content = choice.get("message", {}).get("content", "")
        attention_tokens = choice.get("attention_tokens", [])
        usage = result.get("usage", {})

        # Send prompt token count
        await ws.send_json({
            "type": "prompt_info",
            "token_count": usage.get("prompt_tokens", prompt_token_count),
            "text": prompt_text
        })

        # Tokenize the response to get individual tokens
        # We'll use the attention_tokens array which has one entry per generated token
        if attention_tokens:
            # Each attention_tokens entry corresponds to one generated token
            # We need to detokenize to get the actual token text

            # First, let's get all token texts via detokenize
            token_texts = []

            # Try to detokenize token by token using the content
            # Split content into approximate tokens (this is a heuristic)
            # Better approach: use the actual token IDs if available

            # For now, send tokens based on attention_tokens count
            num_tokens = len(attention_tokens)

            # Simple heuristic: split content into roughly equal parts
            # This isn't perfect but gives a reasonable approximation
            words = content.split()
            tokens_per_word = max(1, num_tokens / len(words)) if words else 1

            current_word_idx = 0
            accumulated_text = ""

            for i, attn in enumerate(attention_tokens):
                # Estimate which part of content this token represents
                # Add approximately one token's worth of content
                if current_word_idx < len(words):
                    token_text = words[current_word_idx]
                    if i % int(tokens_per_word) == 0 and current_word_idx < len(words) - 1:
                        token_text += " "
                        current_word_idx += 1
                    elif (i + 1) % max(1, int(tokens_per_word)) == 0:
                        current_word_idx += 1
                else:
                    token_text = ""

                accumulated_text = content[:int((i + 1) / num_tokens * len(content))] if num_tokens > 0 else content

                # Build token message with attention
                token_msg = {
                    "type": "token",
                    "index": i,
                    "content": token_text if token_text else f"[tok_{i}]",
                    "full_content": accumulated_text,
                    "attention": attn,
                    "zone": attn.get("manifold_zone", "unknown"),
                    "fingerprint": attn.get("fingerprint"),
                }

                # Extract attention edges
                if attn.get("token_positions") and attn.get("attention_scores"):
                    token_msg["attends_to"] = [
                        {"position": pos, "score": score}
                        for pos, score in zip(attn["token_positions"], attn["attention_scores"])
                    ]

                await ws.send_json(token_msg)

                # Small delay for visual streaming effect
                if stream_delay > 0:
                    await asyncio.sleep(stream_delay)

        else:
            # No attention data - just send the content as a single message
            await ws.send_json({
                "type": "token",
                "index": 0,
                "content": content,
                "full_content": content,
            })

        # Send completion message
        await ws.send_json({
            "type": "stream_end",
            "total_tokens": len(attention_tokens) if attention_tokens else 1,
            "full_content": content,
            "usage": usage
        })

    except Exception as e:
        import traceback
        await ws.send_json({
            "type": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
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
