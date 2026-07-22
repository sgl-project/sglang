import io
import logging
import os
import shlex
import time
import warnings
from typing import ClassVar, Optional
from urllib.parse import urlparse

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_pd_server,
    popen_with_error_check,
    start_subprocess_fail_fast_watcher,
)
from sglang.utils import wait_for_http_ready

logger = logging.getLogger(__name__)


def configure_nixl_pd_backend(test_cls):
    test_cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
    # NIXL backend/network selection is driven by NIXL environment variables
    # such as SGLANG_DISAGGREGATION_NIXL_BACKEND and backend params, not by the
    # Mooncake-specific --disaggregation-ib-device argument.
    test_cls.rdma_devices = []


def assert_process_healthy(test_case, name, process, url, health_path="/health"):
    test_case.assertIsNotNone(process, f"{name} process was not started")
    test_case.assertIsNone(
        process.poll(),
        f"{name} exited unexpectedly with code {process.returncode}",
    )
    try:
        response = requests.get(f"{url}{health_path}", timeout=10)
    except requests.RequestException as e:
        test_case.fail(f"Failed to connect to {name} health endpoint: {e}")
    test_case.assertEqual(response.status_code, 200, response.text)


class PDDisaggregationServerBase(CustomTestCase):
    capture_per_side_logs: ClassVar[bool] = False
    extra_prefill_env: ClassVar[dict[str, str]] = {}
    extra_decode_env: ClassVar[dict[str, str]] = {}
    _prefill_stdout_buf: ClassVar[Optional[io.StringIO]] = None
    _prefill_stderr_buf: ClassVar[Optional[io.StringIO]] = None
    _decode_stdout_buf: ClassVar[Optional[io.StringIO]] = None
    _decode_stderr_buf: ClassVar[Optional[io.StringIO]] = None

    @classmethod
    def setUpClass(cls):
        os.environ["MC_TCP_ENABLE_CONNECTION_POOL"] = "true"
        parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed_url.hostname
        base_port = str(parsed_url.port)
        cls.lb_port = base_port
        cls.prefill_port = f"{int(base_port) + 100}"
        cls.decode_port = f"{int(base_port) + 200}"
        cls.bootstrap_port = f"{int(base_port) + 500}"
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        cls.base_url = cls.lb_url
        print(
            f"{cls.base_host=} {cls.lb_port=} {cls.prefill_port=} {cls.decode_port=} {cls.bootstrap_port=}"
        )
        cls.process_lb, cls.process_decode, cls.process_prefill = None, None, None
        if cls.capture_per_side_logs:
            cls._prefill_stdout_buf = io.StringIO()
            cls._prefill_stderr_buf = io.StringIO()
            cls._decode_stdout_buf = io.StringIO()
            cls._decode_stderr_buf = io.StringIO()
        cls._fail_fast_stop = None

        # config transfer backend and rdma devices
        cls._mc_gid_index_set = False
        if is_in_ci():
            cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
            ib_devices = get_rdma_devices_args()
            cls.rdma_devices = ["--disaggregation-ib-device", ib_devices]
            cls._mc_gid_index_set = _maybe_set_roce_gid_index(ib_devices)
        else:
            cls.transfer_backend = [
                "--disaggregation-transfer-backend",
                envs.SGLANG_TEST_PD_DISAGG_BACKEND.get(),
            ]
            cls.rdma_devices = [
                "--disaggregation-ib-device",
                envs.SGLANG_TEST_PD_DISAGG_DEVICES.get(),
            ]
            if cls.rdma_devices[1] is None:
                cls.rdma_devices = []
                msg = "No RDMA devices specified for disaggregation test, using default settings."
                warnings.warn(msg)

    # Subclasses can set these to customize server args
    extra_prefill_args = []
    extra_decode_args = []

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
        ] + list(cls.extra_prefill_args)
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=dict(cls.extra_prefill_env),
            return_stdout_stderr=(
                (cls._prefill_stdout_buf, cls._prefill_stderr_buf)
                if cls.capture_per_side_logs
                else None
            ),
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
        ] + list(cls.extra_decode_args)
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=dict(cls.extra_decode_env),
            return_stdout_stderr=(
                (cls._decode_stdout_buf, cls._decode_stderr_buf)
                if cls.capture_per_side_logs
                else None
            ),
        )

    @classmethod
    def launch_all(cls):
        """Start prefill, decode, wait for health, and launch LB."""
        cls.start_prefill()
        cls.start_decode()
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)
        cls.launch_lb()
        cls._fail_fast_stop = start_subprocess_fail_fast_watcher(
            [
                ("prefill", cls.process_prefill),
                ("decode", cls.process_decode),
                ("lb", cls.process_lb),
            ]
        )

    @classmethod
    def launch_lb(cls):
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",  # FIXME: remove this
            "--prefill",
            cls.prefill_url,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        print("Starting load balancer:", shlex.join(lb_command))
        cls.process_lb = popen_with_error_check(lb_command)
        cls.wait_server_ready(cls.lb_url + "/health", process=cls.process_lb)

    @classmethod
    def wait_server_ready(
        cls, url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, process=None
    ):
        wait_for_http_ready(url=url, timeout=timeout, process=process)
        print(f"Server {url} is ready")

    @classmethod
    def tearDownClass(cls):
        # Stop the watcher BEFORE killing processes: kill_process_tree
        # below makes them exit with a negative signal rc, which would
        # otherwise trip the watcher and os._exit out of pytest mid-teardown.
        if cls._fail_fast_stop is not None:
            cls._fail_fast_stop.set()
        os.environ.pop("MC_TCP_ENABLE_CONNECTION_POOL")
        if getattr(cls, "_mc_gid_index_set", False):
            os.environ.pop("MC_GID_INDEX", None)
        for process in [cls.process_lb, cls.process_decode, cls.process_prefill]:
            if process:
                try:
                    kill_process_tree(process.pid, wait_timeout=60)
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")

        if cls.capture_per_side_logs:
            for buf in (
                cls._prefill_stdout_buf,
                cls._prefill_stderr_buf,
                cls._decode_stdout_buf,
                cls._decode_stderr_buf,
            ):
                if buf is not None:
                    buf.close()
            cls._prefill_stdout_buf = None
            cls._prefill_stderr_buf = None
            cls._decode_stdout_buf = None
            cls._decode_stderr_buf = None

        # wait for 5 seconds
        time.sleep(5)


def _get_available_ib_devices():
    """Auto-detect available high-speed RDMA devices from sysfs.

    Filters for devices that are:
    1. Not Ethernet NICs (excludes devices with 'eth' in the name like mlx5_eth0)
    2. Active (port state)
    3. High-speed (rate >= 100 Gbps to exclude regular Ethernet NICs)
    """
    ib_sysfs_path = "/sys/class/infiniband"
    if not os.path.isdir(ib_sysfs_path):
        logger.warning("IB sysfs path %s does not exist", ib_sysfs_path)
        return None

    all_devices = sorted(os.listdir(ib_sysfs_path))
    logger.warning("All IB devices in sysfs: %s", all_devices)

    devices = []
    for dev in all_devices:
        # Check port 1 state and rate (most devices have single port)
        port_path = os.path.join(ib_sysfs_path, dev, "ports", "1")
        if not os.path.isdir(port_path):
            logger.warning("Device %s: SKIPPED (no port 1)", dev)
            continue

        # Read state and rate for logging
        state = "unknown"
        rate = -1
        state_file = os.path.join(port_path, "state")
        rate_file = os.path.join(port_path, "rate")

        try:
            with open(state_file) as f:
                state = f.read().strip()
        except (OSError, IOError):
            pass

        try:
            with open(rate_file) as f:
                rate_str = f.read().strip()
                rate = int(rate_str.split()[0])
        except (OSError, IOError, ValueError, IndexError):
            pass

        # Log device properties for debugging
        logger.warning(
            "Device %s: state=%s, rate=%d Gbps, has_eth_in_name=%s",
            dev,
            state,
            rate,
            "eth" in dev.lower(),
        )

        # Skip devices with "eth" in the name - these are typically Ethernet NICs
        # that don't work properly with RDMA (e.g., mlx5_eth0)
        if "eth" in dev.lower():
            logger.warning("Device %s: SKIPPED (contains 'eth' in name)", dev)
            continue

        # Check if port is active
        # State format is like "4: ACTIVE" or just "ACTIVE"
        if "ACTIVE" not in state.upper():
            logger.warning("Device %s: SKIPPED (state=%s)", dev, state)
            continue

        # Check rate (filter out low-speed NICs like 10/25 Gbps Ethernet)
        if rate >= 0 and rate < 100:  # Skip devices slower than 100 Gbps
            logger.warning("Device %s: SKIPPED (rate=%d Gbps)", dev, rate)
            continue

        devices.append(dev)
        logger.warning("Device %s: INCLUDED", dev)

    logger.warning("Filtered IB devices: %s (count=%d)", devices, len(devices))
    return devices if devices else None


def get_rdma_devices_args():
    def _parse_list_env(var_name: str):
        val = os.getenv(var_name)
        if not val:
            return None
        items = [x.strip() for x in val.split(",") if x.strip()]
        return items or None

    def _pick_default_pair(rdma_all_devices):
        return [rdma_all_devices[0], rdma_all_devices[len(rdma_all_devices) // 2]]

    # Priority: env var > auto-detect > hardcoded fallback
    rdma_all_devices = (
        _parse_list_env("SGLANG_CI_RDMA_ALL_DEVICES")
        or _get_available_ib_devices()
        or [f"mlx5_roce{i}" for i in range(8)]
    )
    logger.warning("Resolved rdma_all_devices=%s", rdma_all_devices)

    n_rdma = len(rdma_all_devices)

    # 1. Get visible GPU indices
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if not cuda_visible_devices:
        warnings.warn("CUDA_VISIBLE_DEVICES is not set. Using default RDMA devices.")
        return ",".join(_pick_default_pair(rdma_all_devices))

    try:
        # Convert to list of integers (handling possible spaces and empty strings)
        gpu_indices = [
            int(idx.strip()) for idx in cuda_visible_devices.split(",") if idx.strip()
        ]
        if not gpu_indices or len(gpu_indices) > 4:
            return ",".join(_pick_default_pair(rdma_all_devices))
    except ValueError:
        warnings.warn(f"Invalid CUDA_VISIBLE_DEVICES format: {cuda_visible_devices}")
        return ",".join(_pick_default_pair(rdma_all_devices))

    # 2. Calculate base RDMA index group (each group of 4 GPUs uses consecutive devices)
    base_rdma_group = (min(gpu_indices) // 4) * 4
    for gpu_idx in gpu_indices:
        if not (base_rdma_group <= gpu_idx < base_rdma_group + 4):
            warnings.warn(
                f"GPU index {gpu_idx} is outside expected group "
                f"{base_rdma_group}-{base_rdma_group+3}"
            )

    # 3. Generate RDMA device names
    # Detect total GPUs on the node (not just visible ones)
    try:
        import torch

        total_gpus = torch.cuda.device_count()
    except Exception:
        total_gpus = 8  # Fallback to common 8-GPU setup

    # Handle edge cases
    if total_gpus == 0:
        total_gpus = 8
    if n_rdma > total_gpus:
        logger.warning(
            "More RDMA devices (%d) than GPUs (%d), using first and middle device",
            n_rdma,
            total_gpus,
        )
        return ",".join(_pick_default_pair(rdma_all_devices))

    # Calculate how many GPUs share each RDMA device
    gpus_per_rdma = max(1, total_gpus // n_rdma)
    logger.warning(
        "GPU-to-RDMA mapping: total_gpus=%d, n_rdma=%d, gpus_per_rdma=%d",
        total_gpus,
        n_rdma,
        gpus_per_rdma,
    )

    rdma_devices = []
    base_gpu = min(gpu_indices)
    for gpu_idx in gpu_indices:
        nic_index = min((gpu_idx - base_gpu) // gpus_per_rdma, n_rdma - 1)
        rdma_devices.append(rdma_all_devices[nic_index])

    if not rdma_devices:
        return ",".join(_pick_default_pair(rdma_all_devices))

    # Deduplicate while preserving order
    return ",".join(dict.fromkeys(rdma_devices))


_IB_SYSFS = "/sys/class/infiniband"


def _roce_v2_gid_index(device: str):
    """Return a RoCEv2 GID index for a device, preferring a global (routable)
    GID over a link-local (fe80::) one, or None if the device has no RoCEv2 GID.
    """
    port = os.path.join(_IB_SYSFS, device, "ports", "1")
    types_dir = os.path.join(port, "gid_attrs", "types")
    try:
        indices = sorted(int(x) for x in os.listdir(types_dir) if x.isdigit())
    except OSError:
        return None
    fallback = None
    for i in indices:
        try:
            with open(os.path.join(types_dir, str(i))) as f:
                if f.read().strip() != "RoCE v2":
                    continue
        except OSError:
            continue
        if fallback is None:
            fallback = i
        try:
            with open(os.path.join(port, "gids", str(i))) as f:
                gid = f.read().strip()
        except OSError:
            gid = ""
        # Prefer a global GID; link-local (fe80::) entries don't route between
        # NICs on some fabrics.
        if gid and not gid.lower().startswith("fe80"):
            return i
    return fallback


def _detect_roce_gid_index(devices):
    """Return a single RoCEv2 GID index shared by all `devices`, or None.

    None when any device is InfiniBand (mooncake selects the GID automatically
    there), when a device has no RoCEv2 GID, or when devices disagree on the
    index — MC_GID_INDEX is a single global value, so a divergent set can't be
    satisfied and is left to mooncake's own selection.
    """
    picked = None
    for device in [d.strip() for d in devices if d.strip()]:
        try:
            with open(os.path.join(_IB_SYSFS, device, "ports", "1", "link_layer")) as f:
                if f.read().strip() != "Ethernet":
                    return None
        except OSError:
            return None
        idx = _roce_v2_gid_index(device)
        if idx is None:
            return None
        if picked is None:
            picked = idx
        elif picked != idx:
            return None
    return picked


def _maybe_set_roce_gid_index(ib_devices) -> bool:
    """Export MC_GID_INDEX for a RoCE fabric; return True if this call set it.

    On RoCE-only hosts mooncake's automatic GID selection can come up empty
    ("GID is NULL, please check your GID index by specifying MC_GID_INDEX"),
    leaving the KV-transfer RDMA endpoint with no GID so every prefill->decode
    transfer fails and PD accuracy collapses to 0. InfiniBand hosts don't need
    this (auto GID works), and a user-provided MC_GID_INDEX is left untouched.
    """
    if not ib_devices or os.environ.get("MC_GID_INDEX"):
        return False
    gid_index = _detect_roce_gid_index(ib_devices.split(","))
    if gid_index is None:
        return False
    os.environ["MC_GID_INDEX"] = str(gid_index)
    logger.warning("RoCE fabric detected; set MC_GID_INDEX=%d for mooncake", gid_index)
    return True
