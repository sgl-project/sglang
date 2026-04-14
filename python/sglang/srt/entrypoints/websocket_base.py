import json
import uuid
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect


@dataclass(kw_only=True)
class WebSocketSessionBase:
    """Minimal base for persistent WebSocket sessions.

    Provides JSON send / accept / safe close. Subclasses are responsible
    for the receive loop, message dispatch, error event format, and any
    protocol-specific state.
    """

    websocket: WebSocket
    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")

    async def accept(self) -> None:
        await self.websocket.accept()

    async def send_json(self, data: dict) -> None:
        await self.websocket.send_text(json.dumps(data))

    async def safe_close(self) -> None:
        try:
            await self.websocket.close()
        except (WebSocketDisconnect, RuntimeError):
            pass
