"""Native GPU/RDMA regression probe; see the adjacent README."""

import argparse
import base64
import ctypes
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
import torch
from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

parser = argparse.ArgumentParser(description="Two-host native NIXL recovery probe")
parser.add_argument("--role", choices=["prefill", "decode"], required=True)
parser.add_argument("--decode-host", required=True)
parser.add_argument("--port", type=int, default=19098)
parser.add_argument("--fault-library", required=True)
parser.add_argument("--size-mib", type=int, default=2048)
args = parser.parse_args()
RANK = int(args.role == "decode")
IPS = [None, args.decode_host]
SIZE = args.size_mib * 1024 * 1024
PORT = args.port
CONFIG = nixl_agent_config(
    backends=[], num_threads=8, sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT
)
SESSION = requests.Session()
SESSION.trust_env = False


def make_agent(name):
    agent = nixl_agent(name, CONFIG)
    agent.create_backend("UCX", {"num_threads": "8" if name == "source" else "0"})
    return agent


def rpc(action, **kwargs):
    r = SESSION.post(f"http://{IPS[1]}:{PORT}/{action}", json=kwargs, timeout=60)
    r.raise_for_status()
    return r.json()


def remote():
    buffers = {
        name: torch.zeros(SIZE, dtype=torch.uint8, device="cuda:0")
        for name in ["bad", "healthy"]
    }
    agents = {name: make_agent(name) for name in buffers}
    regs = {
        name: agents[name].register_memory([(buf.data_ptr(), SIZE, 0, "")], "VRAM")
        for name, buf in buffers.items()
    }
    torch.cuda.synchronize()
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            args = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            action = self.path.strip("/")
            if action == "metadata":
                result = {
                    name: {
                        "ptr": buf.data_ptr(),
                        "metadata": base64.b64encode(
                            agents[name].get_agent_metadata()
                        ).decode(),
                    }
                    for name, buf in buffers.items()
                }
            elif action == "fill":
                buffers[args["peer"]].fill_(args["value"])
                torch.cuda.synchronize()
                result = {"ok": True}
            elif action == "check":
                buf = buffers[args["peer"]][: args.get("size", SIZE)]
                result = {
                    "matches": bool(torch.all(buf == args["value"]).item()),
                    "pid": os.getpid(),
                }
            elif action == "stop":
                result = {"ok": True}
                done.set()
            else:
                result = {"error": action}
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(json.dumps({"ready": True, "pid": os.getpid(), "rank": RANK}), flush=True)
    if not done.wait(800):
        raise TimeoutError("controller did not finish")
    server.shutdown()
    for name in agents:
        agents[name].deregister_memory(regs[name])
    agents.clear()
    print("receiver finished", flush=True)


def sender():
    deadline = time.monotonic() + 120
    while True:
        try:
            metadata = rpc("metadata")
            break
        except requests.RequestException:
            if time.monotonic() > deadline:
                raise
            time.sleep(1)
    agent = make_agent("source")
    buf = torch.full((SIZE,), 85, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize()
    reg = agent.register_memory([(buf.data_ptr(), SIZE, 0, "")], "VRAM")
    hook = ctypes.CDLL(args.fault_library)
    states = []
    recovery = None

    def post(peer, size=SIZE):
        step = 1024 * 1024
        src = agent.get_xfer_descs(
            [
                (buf.data_ptr() + i, min(step, size - i), 0)
                for i in range(0, size, step)
            ],
            "VRAM",
        )
        dst = agent.get_xfer_descs(
            [
                (metadata[peer]["ptr"] + i, min(step, size - i), 0)
                for i in range(0, size, step)
            ],
            "VRAM",
        )
        h = agent.initialize_xfer("WRITE", src, dst, peer)
        try:
            state = (
                recovery.batch.post(h, peer)
                if recovery is not None and recovery.batch is not None
                else agent.transfer(h)
            )
        except Exception as exc:
            state = type(exc).__name__ + ": " + str(exc)
        states.append(state)
        return h

    def wait(h):
        deadline = time.monotonic() + 30
        while True:
            state = agent.check_xfer_state(h)
            if state != "PROC":
                return state
            if time.monotonic() > deadline:
                raise TimeoutError("native transfer did not settle")
            time.sleep(0.0001)

    hook.fault_capture(1)
    agent.add_remote_agent(base64.b64decode(metadata["bad"]["metadata"]))
    h = post("bad")
    assert wait(h) == "DONE"
    agent.release_xfer_handle(h)
    hook.fault_capture(0)
    captured = hook.fault_count()
    assert captured > 0, "interposer did not capture any QPs"
    assert rpc("check", peer="bad", value=85)["matches"]
    agent.add_remote_agent(base64.b64decode(metadata["healthy"]["metadata"]))
    h = post("healthy")
    assert wait(h) == "DONE"
    agent.release_xfer_handle(h)
    healthy_pid = rpc("check", peer="healthy", value=85)["pid"]
    print(
        json.dumps(
            {
                "baseline": "ok",
                "captured_qps": captured,
                "sender_pid": os.getpid(),
                "receiver_pid": healthy_pid,
            }
        ),
        flush=True,
    )

    from nixl._bindings import nixlNotFoundError, nixlRemoteDisconnectError

    from sglang.srt.disaggregation.nixl.peer_recovery import PeerRecovery

    def fatal(reason):
        raise AssertionError(reason)

    recovery = PeerRecovery(
        agent,
        lambda peer: base64.b64decode(metadata[peer]["metadata"]),
        lambda peer: None,
        fatal,
    )
    rpc("fill", peer="bad", value=0)
    batch = recovery.begin(["bad"])
    handles = [post("bad") for _ in range(4)]
    changed = hook.fault_disconnect_one()
    assert changed > 0
    print(
        json.dumps(
            {"armed": True, "submit_states": states[-4:], "injected_qps": changed}
        ),
        flush=True,
    )
    result = batch.wait(
        failure_seen=False,
        timeout=5,
        poll_interval=0.001,
        disconnect_errors=(nixlRemoteDisconnectError,),
        missing_errors=(nixlNotFoundError,),
    )
    recovery.end()
    print(json.dumps({"batch_result": result}), flush=True)
    assert result == (True, True), "fault did not fail the batch"
    rpc("fill", peer="bad", value=204)
    for pause in [0.01, 0.1, 1.0]:
        time.sleep(pause)
        assert rpc("check", peer="bad", value=204)["matches"], (
            "write arrived after handle retirement"
        )
    old_buf = buf
    buf = torch.full((SIZE,), 119, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize()
    assert old_buf.data_ptr() != buf.data_ptr()
    new_reg = agent.register_memory([(buf.data_ptr(), SIZE, 0, "")], "VRAM")
    h = post("bad")
    assert wait(h) == "DONE"
    agent.release_xfer_handle(h)
    assert rpc("check", peer="bad", value=119)["matches"]
    rpc("fill", peer="healthy", value=0)
    h = post("healthy")
    assert wait(h) == "DONE"
    agent.release_xfer_handle(h)
    assert rpc("check", peer="healthy", value=119)["matches"]
    assert rpc("check", peer="bad", value=119)["pid"] == healthy_pid
    time.sleep(3)
    assert rpc("check", peer="bad", value=119)["matches"], (
        "old payload overwrote new request"
    )
    print(
        json.dumps(
            {
                "recovered": True,
                "healthy_peer": True,
                "same_receiver_pid": True,
                "no_late_writes": True,
            }
        ),
        flush=True,
    )
    agent.remove_remote_agent("bad")
    agent.remove_remote_agent("healthy")
    agent.deregister_memory(reg)
    agent.deregister_memory(new_reg)
    del h, handles
    rpc("stop")


if __name__ == "__main__":
    remote() if RANK else sender()
