"""
Unit tests for ManifoldWebSocketServer and related classes

Tests WebSocket connection management, message broadcasting, and message types.
"""

import asyncio
import json
import pytest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discovery.websocket_server import (
    MessageType,
    ProgressMessage,
    BatchCompleteMessage,
    ClusterUpdateMessage,
    ZoneStatsMessage,
    StageMessage,
    RunMessage,
    ConnectionManager,
    ManifoldWebSocketServer,
    create_websocket_server,
    HAS_WEBSOCKETS,
)


class TestMessageType:
    """Tests for MessageType enum."""

    def test_message_types_exist(self):
        """Test all expected message types exist."""
        assert MessageType.CONNECTED == "connected"
        assert MessageType.HEARTBEAT == "heartbeat"
        assert MessageType.PROGRESS == "progress"
        assert MessageType.STAGE_START == "stage_start"
        assert MessageType.STAGE_COMPLETE == "stage_complete"
        assert MessageType.BATCH_COMPLETE == "batch_complete"
        assert MessageType.CLUSTER_UPDATE == "cluster_update"
        assert MessageType.ZONE_STATS == "zone_stats"
        assert MessageType.RUN_START == "run_start"
        assert MessageType.RUN_COMPLETE == "run_complete"
        assert MessageType.RUN_ERROR == "run_error"

    def test_message_type_is_string(self):
        """Test message types are strings."""
        assert isinstance(MessageType.PROGRESS.value, str)
        # MessageType inherits from str, so value equals the enum itself
        assert MessageType.PROGRESS.value == "progress"


class TestProgressMessage:
    """Tests for ProgressMessage dataclass."""

    def test_create_progress_message(self):
        """Test creating a progress message."""
        msg = ProgressMessage(
            run_id="run-001",
            stage=3,
            stage_name="umap",
            percent_complete=45.5,
            items_processed=4550,
            total_items=10000,
            eta_seconds=120.5,
            memory_used_gb=4.2,
        )

        assert msg.run_id == "run-001"
        assert msg.stage == 3
        assert msg.stage_name == "umap"
        assert msg.percent_complete == 45.5
        assert msg.items_processed == 4550
        assert msg.total_items == 10000
        assert msg.eta_seconds == 120.5
        assert msg.memory_used_gb == 4.2

    def test_progress_message_optional_fields(self):
        """Test progress message with optional fields as None."""
        msg = ProgressMessage(
            run_id="run-002",
            stage=1,
            stage_name="extract",
            percent_complete=10.0,
            items_processed=100,
            total_items=1000,
        )

        assert msg.eta_seconds is None
        assert msg.memory_used_gb is None

    def test_progress_message_to_dict(self):
        """Test progress message serialization."""
        msg = ProgressMessage(
            run_id="run-003",
            stage=2,
            stage_name="pca",
            percent_complete=100.0,
            items_processed=5000,
            total_items=5000,
        )

        data = asdict(msg)
        assert data["run_id"] == "run-003"
        assert data["stage"] == 2
        assert json.dumps(data)  # Should be JSON serializable


class TestBatchCompleteMessage:
    """Tests for BatchCompleteMessage dataclass."""

    def test_create_batch_message(self):
        """Test creating a batch complete message."""
        points = [
            {"x": 0.5, "y": 0.3, "cluster_id": 1, "zone": "syntax_floor"},
            {"x": -0.2, "y": 0.8, "cluster_id": 2, "zone": "semantic_bridge"},
        ]

        msg = BatchCompleteMessage(
            run_id="run-001",
            batch_idx=5,
            new_points=points,
            total_points=500,
        )

        assert msg.run_id == "run-001"
        assert msg.batch_idx == 5
        assert len(msg.new_points) == 2
        assert msg.total_points == 500

    def test_batch_message_serializable(self):
        """Test batch message is JSON serializable."""
        msg = BatchCompleteMessage(
            run_id="run-001",
            batch_idx=1,
            new_points=[{"x": 0, "y": 0}],
            total_points=100,
        )

        data = asdict(msg)
        json_str = json.dumps(data)
        assert "new_points" in json_str


class TestClusterUpdateMessage:
    """Tests for ClusterUpdateMessage dataclass."""

    def test_create_cluster_message(self):
        """Test creating a cluster update message."""
        clusters = [
            {"id": 0, "centroid_x": 0.5, "centroid_y": 0.3, "size": 100, "zone": "syntax_floor"},
            {"id": 1, "centroid_x": -0.2, "centroid_y": 0.8, "size": 50, "zone": "long_range"},
        ]

        msg = ClusterUpdateMessage(
            run_id="run-001",
            clusters=clusters,
            total_clusters=2,
        )

        assert msg.run_id == "run-001"
        assert len(msg.clusters) == 2
        assert msg.total_clusters == 2


class TestZoneStatsMessage:
    """Tests for ZoneStatsMessage dataclass."""

    def test_create_zone_stats_message(self):
        """Test creating a zone stats message."""
        msg = ZoneStatsMessage(
            run_id="run-001",
            distribution={"syntax_floor": 500, "semantic_bridge": 300, "long_range": 200},
            percentages={"syntax_floor": 50.0, "semantic_bridge": 30.0, "long_range": 20.0},
        )

        assert msg.run_id == "run-001"
        assert msg.distribution["syntax_floor"] == 500
        assert msg.percentages["syntax_floor"] == 50.0


class TestStageMessage:
    """Tests for StageMessage dataclass."""

    def test_create_stage_message(self):
        """Test creating a stage message."""
        msg = StageMessage(
            run_id="run-001",
            stage=3,
            stage_name="umap",
            timestamp=1234567890.5,
        )

        assert msg.run_id == "run-001"
        assert msg.stage == 3
        assert msg.stage_name == "umap"
        assert msg.timestamp == 1234567890.5


class TestRunMessage:
    """Tests for RunMessage dataclass."""

    def test_create_run_message(self):
        """Test creating a run message."""
        msg = RunMessage(
            run_id="run-001",
            status="started",
            timestamp=1234567890.0,
        )

        assert msg.run_id == "run-001"
        assert msg.status == "started"
        assert msg.message is None
        assert msg.metrics is None

    def test_run_message_with_optionals(self):
        """Test run message with optional fields."""
        msg = RunMessage(
            run_id="run-002",
            status="error",
            timestamp=1234567890.0,
            message="Database connection failed",
            metrics={"processed": 1000},
        )

        assert msg.message == "Database connection failed"
        assert msg.metrics["processed"] == 1000


class TestConnectionManager:
    """Tests for ConnectionManager class."""

    def test_connect(self):
        """Test connecting a websocket."""
        async def run_test():
            manager = ConnectionManager()
            ws = AsyncMock()
            ws.send = AsyncMock()
            await manager.connect(ws)

            assert manager.connection_count == 1
            assert ws in manager._connections

        asyncio.run(run_test())

    def test_disconnect(self):
        """Test disconnecting a websocket."""
        async def run_test():
            manager = ConnectionManager()
            ws = AsyncMock()
            ws.send = AsyncMock()
            await manager.connect(ws)
            await manager.disconnect(ws)

            assert manager.connection_count == 0
            assert ws not in manager._connections

        asyncio.run(run_test())

    def test_subscribe(self):
        """Test subscribing to a run."""
        async def run_test():
            manager = ConnectionManager()
            ws = AsyncMock()
            ws.send = AsyncMock()
            await manager.connect(ws)
            await manager.subscribe(ws, "run-001")

            assert "run-001" in manager._subscriptions
            assert ws in manager._subscriptions["run-001"]

        asyncio.run(run_test())

    def test_unsubscribe(self):
        """Test unsubscribing from a run."""
        async def run_test():
            manager = ConnectionManager()
            ws = AsyncMock()
            ws.send = AsyncMock()
            await manager.connect(ws)
            await manager.subscribe(ws, "run-001")
            await manager.unsubscribe(ws, "run-001")

            # Subscription set might still exist but be empty or not contain ws
            if "run-001" in manager._subscriptions:
                assert ws not in manager._subscriptions["run-001"]

        asyncio.run(run_test())

    def test_broadcast_to_all(self):
        """Test broadcasting to all connections."""
        async def run_test():
            manager = ConnectionManager()
            ws1 = AsyncMock()
            ws2 = AsyncMock()
            ws1.send = AsyncMock()
            ws2.send = AsyncMock()

            await manager.connect(ws1)
            await manager.connect(ws2)

            message = {"type": "test", "data": 123}
            sent = await manager.broadcast(message)

            assert sent == 2
            ws1.send.assert_called_once()
            ws2.send.assert_called_once()

        asyncio.run(run_test())

    def test_broadcast_to_run(self):
        """Test broadcasting to subscribers of a specific run."""
        async def run_test():
            manager = ConnectionManager()
            ws1 = AsyncMock()
            ws2 = AsyncMock()
            ws3 = AsyncMock()
            ws1.send = AsyncMock()
            ws2.send = AsyncMock()
            ws3.send = AsyncMock()

            await manager.connect(ws1)
            await manager.connect(ws2)
            await manager.connect(ws3)

            # Subscribe ws1 and ws2 to run-001
            await manager.subscribe(ws1, "run-001")
            await manager.subscribe(ws2, "run-001")

            message = {"type": "test"}
            sent = await manager.broadcast_to_run("run-001", message)

            assert sent == 2
            ws1.send.assert_called_once()
            ws2.send.assert_called_once()
            ws3.send.assert_not_called()

        asyncio.run(run_test())

    def test_broadcast_to_empty_run(self):
        """Test broadcasting to a run with no subscribers."""
        async def run_test():
            manager = ConnectionManager()
            sent = await manager.broadcast_to_run("nonexistent-run", {"type": "test"})
            assert sent == 0

        asyncio.run(run_test())

    def test_broadcast_handles_dead_connection(self):
        """Test broadcast handles and cleans up dead connections."""
        async def run_test():
            manager = ConnectionManager()
            ws1 = AsyncMock()
            ws2 = AsyncMock()
            ws1.send = AsyncMock()
            ws2.send = AsyncMock(side_effect=Exception("Connection closed"))

            await manager.connect(ws1)
            await manager.connect(ws2)

            sent = await manager.broadcast({"type": "test"})

            # ws1 succeeded, ws2 failed and should be removed
            assert sent == 1
            assert manager.connection_count == 1

        asyncio.run(run_test())

    def test_connection_count(self):
        """Test connection_count property."""
        manager = ConnectionManager()
        assert manager.connection_count == 0

    def test_get_subscription_counts(self):
        """Test getting subscription counts."""
        async def run_test():
            manager = ConnectionManager()
            ws1 = AsyncMock()
            ws2 = AsyncMock()
            ws1.send = AsyncMock()
            ws2.send = AsyncMock()

            await manager.connect(ws1)
            await manager.connect(ws2)

            await manager.subscribe(ws1, "run-001")
            await manager.subscribe(ws2, "run-001")
            await manager.subscribe(ws1, "run-002")

            counts = manager.get_subscription_counts()

            assert counts["run-001"] == 2
            assert counts["run-002"] == 1

        asyncio.run(run_test())


class TestManifoldWebSocketServer:
    """Tests for ManifoldWebSocketServer class."""

    @pytest.mark.skipif(not HAS_WEBSOCKETS, reason="websockets not installed")
    def test_initialization(self):
        """Test server initialization."""
        server = ManifoldWebSocketServer(
            host="127.0.0.1",
            port=9010,
            heartbeat_interval=30.0,
            max_message_size=5 * 1024 * 1024,
        )

        assert server.host == "127.0.0.1"
        assert server.port == 9010
        assert server.heartbeat_interval == 30.0
        assert server.is_running is False

    @pytest.mark.skipif(not HAS_WEBSOCKETS, reason="websockets not installed")
    def test_connection_count(self):
        """Test connection_count property."""
        server = ManifoldWebSocketServer(port=9010)
        assert server.connection_count == 0

    @pytest.mark.skipif(not HAS_WEBSOCKETS, reason="websockets not installed")
    def test_is_running(self):
        """Test is_running property."""
        server = ManifoldWebSocketServer(port=9010)
        assert server.is_running is False

    @pytest.mark.skipif(not HAS_WEBSOCKETS, reason="websockets not installed")
    def test_set_subscribe_callback(self):
        """Test setting subscribe callback."""
        server = ManifoldWebSocketServer(port=9010)

        callback = Mock()
        server.set_subscribe_callback(callback)

        assert server._on_subscribe is callback

    @pytest.mark.skipif(not HAS_WEBSOCKETS, reason="websockets not installed")
    def test_set_command_callback(self):
        """Test setting command callback."""
        server = ManifoldWebSocketServer(port=9010)

        callback = Mock()
        server.set_command_callback(callback)

        assert server._on_command is callback


class TestCreateWebsocketServer:
    """Tests for create_websocket_server factory function."""

    def test_create_disabled(self):
        """Test factory returns None when disabled."""
        result = create_websocket_server(port=9010, enabled=False)
        assert result is None

    def test_create_port_zero(self):
        """Test factory returns None when port is 0."""
        result = create_websocket_server(port=0, enabled=True)
        assert result is None

    @pytest.mark.skipif(not HAS_WEBSOCKETS, reason="websockets not installed")
    def test_create_enabled(self):
        """Test factory creates server when enabled."""
        result = create_websocket_server(port=9010, enabled=True)
        assert result is not None
        assert isinstance(result, ManifoldWebSocketServer)
        assert result.port == 9010


class TestHasWebsockets:
    """Tests for HAS_WEBSOCKETS flag."""

    def test_has_websockets_is_bool(self):
        """Test HAS_WEBSOCKETS is a boolean."""
        assert isinstance(HAS_WEBSOCKETS, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
