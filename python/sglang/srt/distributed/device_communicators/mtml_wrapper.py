import ctypes
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MTML_SUCCESS = 0
MTML_DEVICE_UUID_BUFFER_SIZE = 48
MTML_MTLINK_STATE_UP = 1

MtmlReturn = ctypes.c_int
MtmlDeviceP2PCaps = ctypes.c_int
MtmlDeviceP2PStatus = ctypes.c_int


class MtmlLibrary(ctypes.Structure):
    _fields_ = []


class MtmlDevice(ctypes.Structure):
    _fields_ = []


class MtmlMtLinkSpec(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("bandWidth", ctypes.c_uint),
        ("linkNum", ctypes.c_uint),
        ("rsvd", ctypes.c_uint * 4),
    ]


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class MTMLLibrary:
    class NVMLError(Exception):
        def __init__(self, code, extra_msg=None):
            self.code = code
            self.extra_msg = extra_msg
            msg = f"MTML error with code {code}"
            if extra_msg:
                msg += f": {extra_msg}"
            super().__init__(msg)

        def __str__(self):
            return self.args[0]

        def __repr__(self):
            return f"<MTMLError code={self.code} extra_msg={self.extra_msg}>"

    NVML_P2P_CAPS_INDEX_NVLINK = 0
    NVML_P2P_STATUS_OK = 0
    _NVML_P2P_STATUS_NOT_OK = 1

    exported_functions = [
        Function(
            "mtmlLibraryInit", MtmlReturn, [ctypes.POINTER(ctypes.POINTER(MtmlLibrary))]
        ),
        Function("mtmlLibraryShutDown", MtmlReturn, [ctypes.POINTER(MtmlLibrary)]),
        Function(
            "mtmlLibraryInitDeviceByIndex",
            MtmlReturn,
            [
                ctypes.POINTER(MtmlLibrary),
                ctypes.c_uint,
                ctypes.POINTER(ctypes.POINTER(MtmlDevice)),
            ],
        ),
        Function(
            "mtmlDeviceGetMtLinkSpec",
            MtmlReturn,
            [ctypes.POINTER(MtmlDevice), ctypes.POINTER(MtmlMtLinkSpec)],
        ),
        Function(
            "mtmlDeviceGetMtLinkState",
            MtmlReturn,
            [ctypes.POINTER(MtmlDevice), ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
        ),
        Function(
            "mtmlDeviceGetMtLinkRemoteDevice",
            MtmlReturn,
            [
                ctypes.POINTER(MtmlDevice),
                ctypes.c_uint,
                ctypes.POINTER(ctypes.POINTER(MtmlDevice)),
            ],
        ),
        Function(
            "mtmlDeviceGetUUID",
            MtmlReturn,
            [ctypes.POINTER(MtmlDevice), ctypes.c_char_p, ctypes.c_uint],
        ),
    ]

    path_to_library_cache: Dict[str, Any] = {}
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            so_file = "libmtml.so"
        if so_file not in MTMLLibrary.path_to_library_cache:
            try:
                lib = ctypes.CDLL(so_file)
            except OSError as e:
                raise RuntimeError(f"Failed to load MTML library from {so_file}: {e}")
            MTMLLibrary.path_to_library_cache[so_file] = lib
        self.lib = MTMLLibrary.path_to_library_cache[so_file]

        if so_file not in MTMLLibrary.path_to_dict_mapping:
            _funcs = {}
            for func in MTMLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            MTMLLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = MTMLLibrary.path_to_dict_mapping[so_file]

        self._lib_handle = None

    def MTML_CHECK(self, result: MtmlReturn, msg: str = "") -> None:
        if result != MTML_SUCCESS:
            raise self.NVMLError(result, msg)

    def _mtmlDeviceGetMtLinkSpec(self, device):
        spec = MtmlMtLinkSpec()
        ret = self.funcs["mtmlDeviceGetMtLinkSpec"](device, ctypes.byref(spec))
        self.MTML_CHECK(
            ret, "Failed to get MTLink spec for device (mtmlDeviceGetMtLinkSpec)"
        )
        return spec

    def _mtmlDeviceGetMtLinkState(self, device, linkId: int) -> int:
        link_state = ctypes.c_uint()
        ret = self.funcs["mtmlDeviceGetMtLinkState"](
            device, linkId, ctypes.byref(link_state)
        )
        self.MTML_CHECK(
            ret,
            f"Failed to get MTLink state for device (linkId={linkId}) (mtmlDeviceGetMtLinkState)",
        )
        return link_state.value

    def _mtmlDeviceGetMtLinkRemoteDevice(self, device, linkId: int):
        remote_dev_ptr = ctypes.POINTER(MtmlDevice)()
        ret = self.funcs["mtmlDeviceGetMtLinkRemoteDevice"](
            device, linkId, ctypes.byref(remote_dev_ptr)
        )
        self.MTML_CHECK(
            ret,
            f"Failed to get remote device for linkId={linkId} (mtmlDeviceGetMtLinkRemoteDevice)",
        )
        return remote_dev_ptr

    def _mtmlDeviceGetUUID(self, device) -> bytes:
        uuid_buf = (ctypes.c_char * MTML_DEVICE_UUID_BUFFER_SIZE)()
        ret = self.funcs["mtmlDeviceGetUUID"](
            device, uuid_buf, MTML_DEVICE_UUID_BUFFER_SIZE
        )
        self.MTML_CHECK(ret, "Failed to get device UUID (mtmlDeviceGetUUID)")
        return bytes(uuid_buf)

    def nvmlInit(self) -> None:
        lib_ptr = ctypes.POINTER(MtmlLibrary)()
        ret = self.funcs["mtmlLibraryInit"](ctypes.byref(lib_ptr))
        self.MTML_CHECK(ret, "Failed to initialize MTML library (mtmlLibraryInit)")
        self._lib_handle = lib_ptr

    def nvmlShutdown(self) -> None:
        if self._lib_handle is None:
            logger.warning(
                "MTML library shutdown called but library is not initialized."
            )
            return
        ret = self.funcs["mtmlLibraryShutDown"](self._lib_handle)
        self.MTML_CHECK(ret, "Failed to shut down MTML library (mtmlLibraryShutDown)")
        self._lib_handle = None

    def nvmlDeviceGetHandleByIndex(self, index: int):
        if self._lib_handle is None:
            raise self.NVMLError(
                self._NVML_P2P_STATUS_NOT_OK,
                f"MTML library not initialized. Call nvmlInit() before accessing device index {index}.",
            )
        dev_ptr = ctypes.POINTER(MtmlDevice)()
        ret = self.funcs["mtmlLibraryInitDeviceByIndex"](
            self._lib_handle, index, ctypes.byref(dev_ptr)
        )
        self.MTML_CHECK(
            ret,
            f"Failed to get device handle for index {index} (mtmlLibraryInitDeviceByIndex)",
        )
        return dev_ptr

    def nvmlDeviceGetP2PStatus(self, device1, device2, caps: MtmlDeviceP2PCaps) -> int:
        status = MtmlDeviceP2PStatus(self._NVML_P2P_STATUS_NOT_OK)
        try:
            device2_uuid = self._mtmlDeviceGetUUID(device2)
            spec = self._mtmlDeviceGetMtLinkSpec(device1)
            for link_id in range(spec.linkNum):
                link_state = self._mtmlDeviceGetMtLinkState(device1, link_id)
                if link_state != MTML_MTLINK_STATE_UP:
                    continue
                remote_dev_ptr = self._mtmlDeviceGetMtLinkRemoteDevice(device1, link_id)
                if not remote_dev_ptr:
                    continue
                remote_dev_uuid = self._mtmlDeviceGetUUID(remote_dev_ptr)
                if device2_uuid == remote_dev_uuid:
                    status.value = self.NVML_P2P_STATUS_OK
                    break
        except Exception as e:
            logger.exception(f"Unexpected error in nvmlDeviceGetP2PStatus: {e}")
        return status.value


# Singleton instance
pymtml = MTMLLibrary()
