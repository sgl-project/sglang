#!/usr/bin/env python3
"""
Attention Fingerprint Visualization Dashboard

Real-time visualization of attention fingerprints from the SGLang semantic routing loop.

Usage:
    pip install streamlit plotly
    streamlit run scripts/attention_dashboard.py

The dashboard connects to:
- Port 9001: Receives fingerprints (ZMQ SUB)
- Port 9002: Sends steering commands (ZMQ PUB)
"""

import json
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Please install: pip install streamlit plotly")
    exit(1)

try:
    import zmq
except ImportError:
    print("Please install: pip install pyzmq")
    exit(1)


# ============================================================================
# Data Store (shared across Streamlit reruns)
# ============================================================================

@dataclass
class FingerprintData:
    request_id: str
    step: int
    vector: List[float]
    manifold: str
    timestamp: float
    think_phase: str = "unknown"


class DataStore:
    """Thread-safe data store for fingerprints."""

    def __init__(self, max_history: int = 500):
        self.fingerprints: deque = deque(maxlen=max_history)
        self.by_request: Dict[str, List[FingerprintData]] = {}
        self.lock = threading.Lock()
        self.stats = {
            "total_received": 0,
            "requests_seen": set(),
            "manifold_counts": {},
        }

    def add(self, data: dict):
        with self.lock:
            fp = FingerprintData(
                request_id=data.get("request_id", "unknown"),
                step=data.get("step", 0),
                vector=data.get("vector", []),
                manifold=data.get("manifold", "unknown"),
                timestamp=time.time(),
                think_phase=data.get("think_phase", "unknown"),
            )
            self.fingerprints.append(fp)

            # Track by request
            if fp.request_id not in self.by_request:
                self.by_request[fp.request_id] = []
            self.by_request[fp.request_id].append(fp)

            # Update stats
            self.stats["total_received"] += 1
            self.stats["requests_seen"].add(fp.request_id)
            self.stats["manifold_counts"][fp.manifold] = \
                self.stats["manifold_counts"].get(fp.manifold, 0) + 1

    def get_recent(self, n: int = 50) -> List[FingerprintData]:
        with self.lock:
            return list(self.fingerprints)[-n:]

    def get_request_history(self, request_id: str) -> List[FingerprintData]:
        with self.lock:
            return self.by_request.get(request_id, [])

    def get_stats(self) -> dict:
        with self.lock:
            return {
                "total_received": self.stats["total_received"],
                "unique_requests": len(self.stats["requests_seen"]),
                "manifold_counts": dict(self.stats["manifold_counts"]),
            }

    def get_request_ids(self) -> List[str]:
        with self.lock:
            return list(self.by_request.keys())[-20:]  # Last 20 requests


# Global data store
if "data_store" not in st.session_state:
    st.session_state.data_store = DataStore()

if "zmq_connected" not in st.session_state:
    st.session_state.zmq_connected = False

if "receiver_thread" not in st.session_state:
    st.session_state.receiver_thread = None


# ============================================================================
# ZMQ Receiver Thread
# ============================================================================

def zmq_receiver_thread(data_store: DataStore, stop_event: threading.Event):
    """Background thread to receive fingerprints via ZMQ."""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    try:
        socket.connect("tcp://localhost:9001")
        print("[Dashboard] Connected to fingerprint stream on port 9001")

        while not stop_event.is_set():
            try:
                message = socket.recv()
                data = json.loads(message.decode())
                data_store.add(data)
            except zmq.Again:
                continue  # Timeout, check stop_event
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"[Dashboard] Error: {e}")
                continue
    finally:
        socket.close()
        context.term()
        print("[Dashboard] ZMQ receiver stopped")


def start_receiver():
    """Start the ZMQ receiver thread."""
    if st.session_state.receiver_thread is None or not st.session_state.receiver_thread.is_alive():
        stop_event = threading.Event()
        st.session_state.stop_event = stop_event
        thread = threading.Thread(
            target=zmq_receiver_thread,
            args=(st.session_state.data_store, stop_event),
            daemon=True
        )
        thread.start()
        st.session_state.receiver_thread = thread
        st.session_state.zmq_connected = True


def stop_receiver():
    """Stop the ZMQ receiver thread."""
    if hasattr(st.session_state, 'stop_event'):
        st.session_state.stop_event.set()
        st.session_state.zmq_connected = False


# ============================================================================
# Visualization Components
# ============================================================================

def create_fingerprint_heatmap(fingerprints: List[FingerprintData]) -> go.Figure:
    """Create a heatmap of recent fingerprints."""
    if not fingerprints:
        return go.Figure()

    # Build matrix: rows = fingerprints, cols = vector dimensions
    vectors = [fp.vector for fp in fingerprints if fp.vector]
    if not vectors:
        return go.Figure()

    matrix = np.array(vectors)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale="Viridis",
        showscale=True,
    ))

    fig.update_layout(
        title="Fingerprint Heatmap (recent → oldest)",
        xaxis_title="Dimension",
        yaxis_title="Step",
        height=300,
    )

    return fig


def create_manifold_trajectory(fingerprints: List[FingerprintData]) -> go.Figure:
    """Create a 2D trajectory using first 2 PCA components."""
    if len(fingerprints) < 3:
        return go.Figure()

    vectors = np.array([fp.vector for fp in fingerprints if len(fp.vector) >= 2])
    if len(vectors) < 3:
        return go.Figure()

    # Simple projection: use first 2 dims (or could do PCA)
    x = vectors[:, 0] if vectors.shape[1] > 0 else np.zeros(len(vectors))
    y = vectors[:, 1] if vectors.shape[1] > 1 else np.zeros(len(vectors))

    # Color by manifold zone
    manifolds = [fp.manifold for fp in fingerprints if len(fp.vector) >= 2]

    fig = go.Figure()

    # Add trajectory line
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        marker=dict(size=8, color=list(range(len(x))), colorscale='Blues'),
        line=dict(width=1, color='lightgray'),
        text=[f"Step {fp.step}: {fp.manifold}" for fp in fingerprints if len(fp.vector) >= 2],
        hoverinfo='text',
    ))

    # Mark start and end
    if len(x) > 0:
        fig.add_trace(go.Scatter(
            x=[x[0]], y=[y[0]],
            mode='markers',
            marker=dict(size=15, color='green', symbol='star'),
            name='Start',
        ))
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y[-1]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='End',
        ))

    fig.update_layout(
        title="Semantic Manifold Trajectory",
        xaxis_title="Dim 0",
        yaxis_title="Dim 1",
        height=350,
        showlegend=True,
    )

    return fig


def create_dimension_timeseries(fingerprints: List[FingerprintData], dims: List[int] = [0, 1, 2, 3]) -> go.Figure:
    """Create time series of specific dimensions."""
    if not fingerprints:
        return go.Figure()

    fig = go.Figure()

    for dim in dims:
        values = [fp.vector[dim] if len(fp.vector) > dim else 0 for fp in fingerprints]
        steps = list(range(len(values)))

        fig.add_trace(go.Scatter(
            x=steps,
            y=values,
            mode='lines',
            name=f'Dim {dim}',
        ))

    fig.update_layout(
        title="Fingerprint Dimensions Over Time",
        xaxis_title="Step",
        yaxis_title="Value",
        height=250,
    )

    return fig


def create_manifold_pie(stats: dict) -> go.Figure:
    """Create pie chart of manifold zone distribution."""
    counts = stats.get("manifold_counts", {})
    if not counts:
        return go.Figure()

    fig = go.Figure(data=[go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        hole=0.4,
    )])

    fig.update_layout(
        title="Manifold Zone Distribution",
        height=250,
    )

    return fig


# ============================================================================
# Steering Controls
# ============================================================================

def send_steering_command(request_id: str, command: dict):
    """Send a steering command to the scheduler via ZMQ."""
    try:
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect("tcp://localhost:9002")

        message = {
            "schema_version": 1,
            "request_id": request_id,
            "seq": int(time.time() * 1000),
            "ts_ms": int(time.time() * 1000),
            **command
        }

        socket.send(json.dumps(message).encode(), zmq.NOBLOCK)
        socket.close()
        context.term()
        return True
    except Exception as e:
        st.error(f"Failed to send command: {e}")
        return False


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    st.set_page_config(
        page_title="Attention Fingerprint Dashboard",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 Attention Fingerprint Dashboard")
    st.markdown("Real-time visualization of semantic manifold fingerprints")

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")

        # Connection status
        if st.session_state.zmq_connected:
            st.success("🟢 Connected to fingerprint stream")
            if st.button("Disconnect"):
                stop_receiver()
                st.rerun()
        else:
            st.warning("🔴 Not connected")
            if st.button("Connect to Stream"):
                start_receiver()
                st.rerun()

        st.divider()

        # Stats
        stats = st.session_state.data_store.get_stats()
        st.metric("Total Fingerprints", stats["total_received"])
        st.metric("Unique Requests", stats["unique_requests"])

        st.divider()

        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_rate = st.slider("Refresh rate (sec)", 0.5, 5.0, 1.0)

        st.divider()

        # Steering controls
        st.header("Steering")
        request_ids = st.session_state.data_store.get_request_ids()
        if request_ids:
            selected_request = st.selectbox("Request ID", request_ids)

            st.subheader("Probe Layers")
            probe_layers = st.multiselect(
                "Layers to capture",
                options=list(range(48)),
                default=[3, 19, 35, 47],
            )

            if st.button("Send Probe Command"):
                cmd = {
                    "control": {
                        "next_capture_layer_ids": probe_layers,
                    }
                }
                if send_steering_command(selected_request, cmd):
                    st.success("Sent!")

            st.subheader("Attention Bias")
            bias_layer = st.number_input("Layer", 0, 47, 15)
            bias_token = st.number_input("Token Position", 0, 1000, 0)
            bias_value = st.slider("Bias Value", -1.0, 1.0, 0.5)

            if st.button("Apply Bias"):
                cmd = {
                    "control": {
                        "attention_biases": {
                            str(bias_layer): {str(bias_token): bias_value}
                        }
                    }
                }
                if send_steering_command(selected_request, cmd):
                    st.success("Bias applied!")

    # Main content
    data_store = st.session_state.data_store
    fingerprints = data_store.get_recent(100)

    if not fingerprints:
        st.info("👆 Click 'Connect to Stream' and send a request with `return_attention_tokens: true`")

        st.code('''
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "...", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50, "return_attention_tokens": true}'
        ''', language="bash")
    else:
        # Row 1: Heatmap and Trajectory
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                create_fingerprint_heatmap(fingerprints),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                create_manifold_trajectory(fingerprints),
                use_container_width=True,
            )

        # Row 2: Time series and Pie chart
        col3, col4 = st.columns([2, 1])

        with col3:
            st.plotly_chart(
                create_dimension_timeseries(fingerprints),
                use_container_width=True,
            )

        with col4:
            st.plotly_chart(
                create_manifold_pie(stats),
                use_container_width=True,
            )

        # Row 3: Recent fingerprints table
        st.subheader("Recent Fingerprints")
        recent = fingerprints[-10:][::-1]
        table_data = [
            {
                "Request": fp.request_id[:8] + "...",
                "Step": fp.step,
                "Manifold": fp.manifold,
                "Phase": fp.think_phase,
                "Vector (first 5)": str([round(v, 3) for v in fp.vector[:5]]),
            }
            for fp in recent
        ]
        st.table(table_data)

    # Auto-refresh
    if auto_refresh and st.session_state.zmq_connected:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
