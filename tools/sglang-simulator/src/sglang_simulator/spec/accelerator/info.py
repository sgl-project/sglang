from sglang_simulator.spec.accelerator.base import AcceleratorInfo


class NVIDIA:
    NVIDIA_H20 = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA H20",
            "device_alias": ["H20", "h20_sxm"],
            "tflops": {
                "FP8_TENSOR": 296,
                "INT8_TENSOR": 296,
                "FP16_TENSOR": 148,
                "BF16_TENSOR": 148,
                "FP32": 74,
            },
            "hbm_capacity_gb": 96,
            "hbm_bandwidth_gb": 4022,
            "inter_node_bandwidth_gb": 64,
            "intra_node_bandwidth_gb": 450,
            "vendor": "NVIDIA",
            "ref": "https://viperatech.com/product/nvidia-hgx-h20",
        },
        save_to_registry=True,
    )
