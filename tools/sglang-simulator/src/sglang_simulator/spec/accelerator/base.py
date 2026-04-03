from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from sglang_simulator.spec.data_type import DataType
from sglang_simulator.utils import get_logger

_all_accs_: Dict[str, "AcceleratorInfo"] = {}
_acc_alias: Dict[str, str] = {}

logger = get_logger("sgl_simulator")


@dataclass
class AcceleratorInfo:
    name: str
    vendor: str
    hbm_capacity_gb: int
    hbm_bandwidth_gb: int
    intra_node_bandwidth_gb: Optional[int] = None  # scale up
    inter_node_bandwidth_gb: int = 64  # scale out
    device_alias: list = field(default_factory=list)
    tflops: dict = field(default_factory=dict)
    ref: str = ""

    @classmethod
    def from_dict(cls, config: Dict, save_to_registry: bool = False):
        acc = cls(**config)
        if save_to_registry:
            if acc.name in _acc_alias:
                logger.error(f"{acc.name} is already in registry")
            _all_accs_[acc.name.upper()] = acc
            for alias in acc.device_alias:
                if alias in _acc_alias:
                    logger.warning(f"Device alias [{alias}] is already in registry.")
                else:
                    _acc_alias[alias] = acc.name.upper()
        return acc

    def flops(self, datatype: Union[str, DataType] = DataType.FP16):
        if isinstance(datatype, DataType):
            datatype = datatype.value
        return self.tflops.get(datatype, 1) * 1e12

    def tensor_flops(self, datatype: Union[str, DataType] = DataType.FP16_TENSOR):
        if isinstance(datatype, DataType):
            datatype = datatype.value
        if not datatype.endswith(DataType.tensor_suffix()):
            datatype += DataType.tensor_suffix()
        tflops = self.tflops.get(datatype, None)
        return None if tflops is None else tflops * 1e12

    @property
    def hbm_io_bw(self):
        return self.hbm_bandwidth_gb * 1e9

    @property
    def hbm_bytes(self):
        return self.hbm_capacity_gb * 1e9

    @property
    def intra_node_bw(self) -> Optional[float]:
        if self.intra_node_bandwidth_gb is None:
            return None
        return self.intra_node_bandwidth_gb * 1e9

    @property
    def inter_node_bw(self):
        return self.inter_node_bandwidth_gb * 1e9

    @staticmethod
    def find_by_hw_name(hw_name: str) -> Union[None, "AcceleratorInfo"]:
        if hw_name in _acc_alias:
            hw = _all_accs_.get(_acc_alias[hw_name], None)
            if hw is not None:
                hw = deepcopy(hw)
                hw.name = hw_name
            return hw
        else:
            return _all_accs_.get(hw_name.upper(), None)

    @staticmethod
    def list_all_hws() -> Dict[str, "AcceleratorInfo"]:
        return _all_accs_

    @classmethod
    def from_config(cls, config: Dict):
        hw_info = cls.find_by_hw_name(config["name"])
        return cls(**config) if hw_info is None else hw_info

    def __eq__(self, value):
        if isinstance(value, str):
            value = self.find_by_hw_name(value)

        if isinstance(value, AcceleratorInfo):
            return _acc_alias.get(
                value.device_name, value.device_name
            ) == _acc_alias.get(self.device_name, self.device_name)

        return False
