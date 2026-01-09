"""
WebSocket Server for Live Manifold Discovery Updates

Streams real-time updates to the UI during long-running discovery jobs:
- Progress updates with ETA
- New embeddings as they're computed
- Cluster updates as they evolve
- Zone distribution changes
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import weakref

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketServerProtocol = Any

logger = logging.getLogger(__name__)


# =============================================================================
# MESSAGE TYPES
# =============================================================================

class MessageType(str, Enum):
    """WebSocket message types for manifold discovery updates."""

    # Connection lifecycle
    CONNECTED = "connected"
    HEARTBEAT = "heartbeat"

    # Progress updates
    PROGRESS = "progress"
    STAGE_START = "stage_start"
    STAGE_COMPLETE = "stage_complete"

    # Data updates
    BATCH_COMPLETE = "batch_complete"
    CLUSTER_UPDATE = "cluster_update"
    ZONE_STATS = "zone_stats"
    EMBEDDING_SAMPLE = "embedding_sample"

    # Run lifecycle
    RUN_START = "run_start"
    RUN_COMPLETE = "run_complete"
    RUN_ERROR = "run_error"
    RUN_PAUSED = "run_paused"
    RUN_RESUMED = "run_resumed"


@dataclass
class ProgressMessage:
    """Progress update message."""
    run_id: str
    stage: int
    stage_name: str
    percent_complete: float
    items_processed: int
    total_items: int
    eta_seconds: Optional[float] = None
    memory_used_gb: Optional[float] = None


@dataclass
class BatchCompleteMessage:
    """Batch of embeddings completed."""
    run_id: str
    batch_idx: int
    new_points: List[Dict[str, Any]]  # [{x, y, cluster_id, zone, fingerprint_id}]
    total_points: int


@dataclass
class ClusterUpdateMessage:
    """Cluster state update."""
    run_id: str
    clusters: List[Dict[str, Any]]  # [{id, centroid_x, centroid_y, size, zone}]
    total_clusters: int


@dataclass
class ZoneStatsMessage:
    """Zone distribution statistics."""
    run_id: str
    distribution: Dict[str, int]  # {zone_name: count}
    percentages: Dict[str, float]  # {zone_name: percentage}


@dataclass
class StageMessage:
    """Stage lifecycle message."""
    run_id: str
    stage: int
    stage_name: str
    timestamp: float


@dataclass
class RunMessage:
    """Run lifecycle message."""
    run_id: str
    status: str
    timestamp: float
    message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


# =============================================================================
# CONNECTION MANAGER
# =============================================================================

class ConnectionManager:
    """
    Manages WebSocket connections with automatic cleanup.

    Supports:
    - Multiple concurrent connections
    - Topic-based subscriptions (by run_id)
    - Automatic dead connection cleanup
    - Broadcast and targeted messaging
    """

    def __init__(self):
        self._connections: Set[WebSocketServerProtocol] = set()
        self._subscriptions: Dict[str, Set[WebSocketServerProtocol]] = {}
        self._lock = asyncio.Lock()
        self._connection_info: Dict[WebSocketServerProtocol, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocketServerProtocol) -> None:
        """Register a new connection."""
        async with self._lock:
            self._connections.add(websocket)
            self._connection_info[websocket] = {
                'connected_at': time.time(),
                'subscriptions': set(),
            }

        logger.info(f"WebSocket connected: {len(self._connections)} total connections")

    async def disconnect(self, websocket: WebSocketServerProtocol) -> None:
        """Unregister a connection."""
        async with self._lock:
            self._connections.discard(websocket)

            # Remove from all subscriptions
            info = self._connection_info.pop(websocket, {})
            for run_id in info.get('subscriptions', set()):
                if run_id in self._subscriptions:
                    self._subscriptions[run_id].discard(websocket)
                    if not self._subscriptions[run_id]:
                        del self._subscriptions[run_id]

        logger.info(f"WebSocket disconnected: {len(self._connections)} remaining")

    async def subscribe(self, websocket: WebSocketServerProtocol, run_id: str) -> None:
        """Subscribe connection to a run's updates."""
        async with self._lock:
            if run_id not in self._subscriptions:
                self._subscriptions[run_id] = set()
            self._subscriptions[run_id].add(websocket)

            if websocket in self._connection_info:
                self._connection_info[websocket]['subscriptions'].add(run_id)

        logger.debug(f"Connection subscribed to run: {run_id}")

    async def unsubscribe(self, websocket: WebSocketServerProtocol, run_id: str) -> None:
        """Unsubscribe connection from a run's updates."""
        async with self._lock:
            if run_id in self._subscriptions:
                self._subscriptions[run_id].discard(websocket)

            if websocket in self._connection_info:
                self._connection_info[websocket]['subscriptions'].discard(run_id)

    async def broadcast(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all connected clients."""
        if not self._connections:
            return 0

        data = json.dumps(message)
        sent = 0
        dead_connections = []

        for websocket in list(self._connections):
            try:
                await websocket.send(data)
                sent += 1
            except Exception:
                dead_connections.append(websocket)

        # Cleanup dead connections
        for ws in dead_connections:
            await self.disconnect(ws)

        return sent

    async def broadcast_to_run(self, run_id: str, message: Dict[str, Any]) -> int:
        """Broadcast message to subscribers of a specific run."""
        subscribers = self._subscriptions.get(run_id, set())
        if not subscribers:
            return 0

        data = json.dumps(message)
        sent = 0
        dead_connections = []

        for websocket in list(subscribers):
            try:
                await websocket.send(data)
                sent += 1
            except Exception:
                dead_connections.append(websocket)

        # Cleanup dead connections
        for ws in dead_connections:
            await self.disconnect(ws)

        return sent

    @property
    def connection_count(self) -> int:
        """Number of active connections."""
        return len(self._connections)

    def get_subscription_counts(self) -> Dict[str, int]:
        """Get subscriber counts per run."""
        return {run_id: len(subs) for run_id, subs in self._subscriptions.items()}


# =============================================================================
# WEBSOCKET SERVER
# =============================================================================

class ManifoldWebSocketServer:
    """
    WebSocket server for streaming manifold discovery updates.

    Features:
    - Real-time progress updates
    - Incremental embedding streaming
    - Cluster evolution tracking
    - Zone distribution updates
    - Automatic reconnection handling

    Usage:
        server = ManifoldWebSocketServer(port=9010)
        await server.start()

        # From discovery job:
        await server.broadcast_progress(run_id, stage=3, percent=45.2)
        await server.broadcast_batch_complete(run_id, batch_idx=5, points=[...])
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9010,
        heartbeat_interval: float = 30.0,
        max_message_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        """
        Initialize WebSocket server.

        Args:
            host: Host to bind to
            port: Port to listen on
            heartbeat_interval: Seconds between heartbeat messages
            max_message_size: Maximum message size in bytes
        """
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets library required. Install with: pip install websockets"
            )

        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.max_message_size = max_message_size

        self._manager = ConnectionManager()
        self._server = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

        # Callbacks for custom handling
        self._on_subscribe: Optional[Callable[[str, str], None]] = None
        self._on_command: Optional[Callable[[str, Dict], None]] = None

    async def start(self) -> None:
        """Start the WebSocket server."""
        if self._running:
            logger.warning("WebSocket server already running")
            return

        self._running = True

        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            max_size=self.max_message_size,
            ping_interval=20,
            ping_timeout=60,
        )

        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(f"Manifold WebSocket server started on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        logger.info("Manifold WebSocket server stopped")

    async def _handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a WebSocket connection."""
        await self._manager.connect(websocket)

        # Send connected message
        await websocket.send(json.dumps({
            'type': MessageType.CONNECTED,
            'timestamp': time.time(),
            'server_info': {
                'heartbeat_interval': self.heartbeat_interval,
            }
        }))

        try:
            async for message in websocket:
                await self._handle_message(websocket, message)
        except websockets.ConnectionClosed:
            pass
        finally:
            await self._manager.disconnect(websocket)

    async def _handle_message(
        self,
        websocket: WebSocketServerProtocol,
        message: str,
    ) -> None:
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')

            if msg_type == 'subscribe':
                run_id = data.get('run_id')
                if run_id:
                    await self._manager.subscribe(websocket, run_id)
                    await websocket.send(json.dumps({
                        'type': 'subscribed',
                        'run_id': run_id,
                        'timestamp': time.time(),
                    }))
                    if self._on_subscribe:
                        self._on_subscribe(run_id, 'subscribe')

            elif msg_type == 'unsubscribe':
                run_id = data.get('run_id')
                if run_id:
                    await self._manager.unsubscribe(websocket, run_id)
                    await websocket.send(json.dumps({
                        'type': 'unsubscribed',
                        'run_id': run_id,
                        'timestamp': time.time(),
                    }))
                    if self._on_subscribe:
                        self._on_subscribe(run_id, 'unsubscribe')

            elif msg_type == 'command':
                if self._on_command:
                    self._on_command(data.get('command', ''), data.get('params', {}))

            elif msg_type == 'ping':
                await websocket.send(json.dumps({
                    'type': 'pong',
                    'timestamp': time.time(),
                }))

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received: {message[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to all connections."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                message = {
                    'type': MessageType.HEARTBEAT,
                    'timestamp': time.time(),
                    'connections': self._manager.connection_count,
                }

                await self._manager.broadcast(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    # =========================================================================
    # BROADCAST METHODS
    # =========================================================================

    async def broadcast_progress(
        self,
        run_id: str,
        stage: int,
        stage_name: str,
        percent_complete: float,
        items_processed: int,
        total_items: int,
        eta_seconds: Optional[float] = None,
        memory_used_gb: Optional[float] = None,
    ) -> int:
        """
        Broadcast progress update.

        Args:
            run_id: Discovery run ID
            stage: Current stage number (0-9)
            stage_name: Human-readable stage name
            percent_complete: Progress percentage (0-100)
            items_processed: Items processed so far
            total_items: Total items to process
            eta_seconds: Estimated time remaining
            memory_used_gb: Current memory usage

        Returns:
            Number of clients notified
        """
        message = {
            'type': MessageType.PROGRESS,
            'timestamp': time.time(),
            'data': asdict(ProgressMessage(
                run_id=run_id,
                stage=stage,
                stage_name=stage_name,
                percent_complete=percent_complete,
                items_processed=items_processed,
                total_items=total_items,
                eta_seconds=eta_seconds,
                memory_used_gb=memory_used_gb,
            ))
        }

        return await self._manager.broadcast_to_run(run_id, message)

    async def broadcast_stage_start(
        self,
        run_id: str,
        stage: int,
        stage_name: str,
    ) -> int:
        """Broadcast stage start notification."""
        message = {
            'type': MessageType.STAGE_START,
            'timestamp': time.time(),
            'data': asdict(StageMessage(
                run_id=run_id,
                stage=stage,
                stage_name=stage_name,
                timestamp=time.time(),
            ))
        }

        return await self._manager.broadcast_to_run(run_id, message)

    async def broadcast_stage_complete(
        self,
        run_id: str,
        stage: int,
        stage_name: str,
    ) -> int:
        """Broadcast stage completion notification."""
        message = {
            'type': MessageType.STAGE_COMPLETE,
            'timestamp': time.time(),
            'data': asdict(StageMessage(
                run_id=run_id,
                stage=stage,
                stage_name=stage_name,
                timestamp=time.time(),
            ))
        }

        return await self._manager.broadcast_to_run(run_id, message)

    async def broadcast_batch_complete(
        self,
        run_id: str,
        batch_idx: int,
        new_points: List[Dict[str, Any]],
        total_points: int,
    ) -> int:
        """
        Broadcast batch of new embeddings.

        Args:
            run_id: Discovery run ID
            batch_idx: Batch index
            new_points: List of point dicts with {x, y, cluster_id, zone, fingerprint_id}
            total_points: Total points computed so far

        Returns:
            Number of clients notified
        """
        message = {
            'type': MessageType.BATCH_COMPLETE,
            'timestamp': time.time(),
            'data': asdict(BatchCompleteMessage(
                run_id=run_id,
                batch_idx=batch_idx,
                new_points=new_points,
                total_points=total_points,
            ))
        }

        return await self._manager.broadcast_to_run(run_id, message)

    async def broadcast_cluster_update(
        self,
        run_id: str,
        clusters: List[Dict[str, Any]],
    ) -> int:
        """
        Broadcast cluster state update.

        Args:
            run_id: Discovery run ID
            clusters: List of cluster dicts with {id, centroid_x, centroid_y, size, zone}

        Returns:
            Number of clients notified
        """
        message = {
            'type': MessageType.CLUSTER_UPDATE,
            'timestamp': time.time(),
            'data': asdict(ClusterUpdateMessage(
                run_id=run_id,
                clusters=clusters,
                total_clusters=len(clusters),
            ))
        }

        return await self._manager.broadcast_to_run(run_id, message)

    async def broadcast_zone_stats(
        self,
        run_id: str,
        distribution: Dict[str, int],
    ) -> int:
        """
        Broadcast zone distribution statistics.

        Args:
            run_id: Discovery run ID
            distribution: Dict mapping zone names to counts

        Returns:
            Number of clients notified
        """
        total = sum(distribution.values()) or 1
        percentages = {
            zone: (count / total) * 100
            for zone, count in distribution.items()
        }

        message = {
            'type': MessageType.ZONE_STATS,
            'timestamp': time.time(),
            'data': asdict(ZoneStatsMessage(
                run_id=run_id,
                distribution=distribution,
                percentages=percentages,
            ))
        }

        return await self._manager.broadcast_to_run(run_id, message)

    async def broadcast_embedding_sample(
        self,
        run_id: str,
        points: List[Dict[str, Any]],
        sample_type: str = "random",
    ) -> int:
        """
        Broadcast a sample of embeddings for visualization.

        Args:
            run_id: Discovery run ID
            points: Sample points for visualization
            sample_type: Type of sample (random, stratified, etc.)

        Returns:
            Number of clients notified
        """
        message = {
            'type': MessageType.EMBEDDING_SAMPLE,
            'timestamp': time.time(),
            'data': {
                'run_id': run_id,
                'points': points,
                'sample_type': sample_type,
                'count': len(points),
            }
        }

        return await self._manager.broadcast_to_run(run_id, message)

    async def broadcast_run_start(
        self,
        run_id: str,
        config: Dict[str, Any],
    ) -> int:
        """Broadcast run start notification."""
        message = {
            'type': MessageType.RUN_START,
            'timestamp': time.time(),
            'data': asdict(RunMessage(
                run_id=run_id,
                status='started',
                timestamp=time.time(),
                metrics=config,
            ))
        }

        # Broadcast to all connections (run hasn't started subscribing yet)
        return await self._manager.broadcast(message)

    async def broadcast_run_complete(
        self,
        run_id: str,
        metrics: Dict[str, Any],
    ) -> int:
        """Broadcast run completion notification."""
        message = {
            'type': MessageType.RUN_COMPLETE,
            'timestamp': time.time(),
            'data': asdict(RunMessage(
                run_id=run_id,
                status='complete',
                timestamp=time.time(),
                metrics=metrics,
            ))
        }

        return await self._manager.broadcast_to_run(run_id, message)

    async def broadcast_run_error(
        self,
        run_id: str,
        error_message: str,
    ) -> int:
        """Broadcast run error notification."""
        message = {
            'type': MessageType.RUN_ERROR,
            'timestamp': time.time(),
            'data': asdict(RunMessage(
                run_id=run_id,
                status='error',
                timestamp=time.time(),
                message=error_message,
            ))
        }

        return await self._manager.broadcast_to_run(run_id, message)

    # =========================================================================
    # PROPERTIES AND CALLBACKS
    # =========================================================================

    @property
    def connection_count(self) -> int:
        """Number of active WebSocket connections."""
        return self._manager.connection_count

    @property
    def is_running(self) -> bool:
        """Whether server is running."""
        return self._running

    def set_subscribe_callback(
        self,
        callback: Callable[[str, str], None],
    ) -> None:
        """Set callback for subscribe/unsubscribe events."""
        self._on_subscribe = callback

    def set_command_callback(
        self,
        callback: Callable[[str, Dict], None],
    ) -> None:
        """Set callback for client commands."""
        self._on_command = callback


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_websocket_server(
    port: int = 9010,
    enabled: bool = True,
) -> Optional[ManifoldWebSocketServer]:
    """
    Create a WebSocket server if enabled and dependencies available.

    Args:
        port: Port to listen on
        enabled: Whether to create server

    Returns:
        ManifoldWebSocketServer or None if disabled/unavailable
    """
    if not enabled:
        logger.info("WebSocket server disabled")
        return None

    if not HAS_WEBSOCKETS:
        logger.warning(
            "websockets library not available. "
            "Install with: pip install websockets"
        )
        return None

    if port == 0:
        logger.info("WebSocket server disabled (port=0)")
        return None

    return ManifoldWebSocketServer(port=port)


# =============================================================================
# TESTING
# =============================================================================

async def _test_server():
    """Test the WebSocket server."""
    import random

    server = ManifoldWebSocketServer(port=9010)
    await server.start()

    print(f"WebSocket server running on ws://localhost:9010")
    print("Connect with: websocat ws://localhost:9010")
    print("Send: {\"type\": \"subscribe\", \"run_id\": \"test-001\"}")

    # Simulate discovery updates
    run_id = "test-001"

    try:
        for stage in range(5):
            stage_name = ["extract", "pca", "umap", "cluster", "zones"][stage]

            await server.broadcast_stage_start(run_id, stage, stage_name)

            for i in range(10):
                await asyncio.sleep(0.5)

                await server.broadcast_progress(
                    run_id=run_id,
                    stage=stage,
                    stage_name=stage_name,
                    percent_complete=(i + 1) * 10,
                    items_processed=(i + 1) * 100,
                    total_items=1000,
                    eta_seconds=(9 - i) * 5,
                )

                # Simulate batch complete
                if i % 3 == 0:
                    points = [
                        {
                            'x': random.uniform(-10, 10),
                            'y': random.uniform(-10, 10),
                            'cluster_id': random.randint(0, 5),
                            'zone': random.choice(['syntax_floor', 'semantic_bridge', 'long_range']),
                            'fingerprint_id': i * 10 + j,
                        }
                        for j in range(10)
                    ]
                    await server.broadcast_batch_complete(run_id, i, points, (i + 1) * 100)

            await server.broadcast_stage_complete(run_id, stage, stage_name)

        await server.broadcast_run_complete(run_id, {'total_points': 1000})

        # Keep server running
        while True:
            await asyncio.sleep(60)

    except KeyboardInterrupt:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(_test_server())
