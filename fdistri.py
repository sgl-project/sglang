import torch
import requests
from transformers import AutoModelForCausalLM, AutoConfig

# --------------------------
# 配置（必须和 SGLang 服务完全一致）
# --------------------------
SGLANG_PORT = 6990
NEW_MODEL_PATH = "/home/wzy/qwen3-30b-a3b"
SGLANG_TP_SIZE = 2
SGLANG_BASE_NPU_ID = 8

# 通信组配置
MASTER_ADDRESS = "127.0.0.1"  # 训练进程的 IP 地址
MASTER_PORT = 29500  # 任意未被占用的端口
GROUP_NAME = "qwen3_weight_update_group"
BACKEND = "hccl"  # NPU 必须用 hccl，GPU 用 nccl

# --------------------------
# 1. 初始化 SGLang 权重更新通信组
# --------------------------
print("Initializing weight update group...")
init_url = f"http://localhost:{SGLANG_PORT}/init_weights_update_group"
init_data = {
    "master_address": MASTER_ADDRESS,
    "master_port": MASTER_PORT,
    "rank_offset": 0,  # SGLang 进程的 rank 从 0 开始
    "world_size": SGLANG_TP_SIZE,  # 总进程数 = SGLang TP 大小
    "group_name": GROUP_NAME,
    "backend": BACKEND
}

response = requests.post(init_url, json=init_data, timeout=60)
assert response.json()["success"] is True, "通信组初始化失败"
print("✅ 权重更新通信组初始化成功")

# --------------------------
# 2. 加载新权重并提取参数信息
# --------------------------
print("Loading new model weights...")
config = AutoConfig.from_pretrained(NEW_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    NEW_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=f"npu:{SGLANG_BASE_NPU_ID}",
    trust_remote_code=True
)

# 自动提取所有参数的名称、dtype 和形状
names = []
dtypes = []
shapes = []
named_tensors = {}

for name, param in model.named_parameters():
    names.append(name)
    dtypes.append(str(param.dtype).split(".")[-1])  # 转换为 "bfloat16" 字符串
    shapes.append(list(param.shape))
    named_tensors[name] = param.data

print(f"✅ 提取到 {len(names)} 个参数")

# --------------------------
# 3. 调用 update_weights_from_distributed
# --------------------------
print("Calling update_weights_from_distributed...")
update_url = f"http://localhost:{SGLANG_PORT}/update_weights_from_distributed"
update_data = {
    "names": names,
    "dtypes": dtypes,
    "shapes": shapes,
    "group_name": GROUP_NAME,
    "flush_cache": True,
    "abort_all_requests": False,
    "weight_version": "qwen3-30b-a3b-v1",
    "load_format": "flattened_bucket"  # 最快的批量加载格式
}

# 注意：这个调用会阻塞直到权重广播和更新完成
response = requests.post(update_url, json=update_data, timeout=1200)
result = response.json()
print(f"更新结果: {result}")

assert result["success"] is True, f"权重更新失败: {result['message']}"
print("✅ 分布式权重更新成功！")

# --------------------------
# 4. 销毁通信组
# --------------------------
print("Destroying weight update group...")
destroy_url = f"http://localhost:{SGLANG_PORT}/destroy_weights_update_group"
destroy_data = {"group_name": GROUP_NAME}
response = requests.post(destroy_url, json=destroy_data, timeout=60)
assert response.json()["success"] is True, "通信组销毁失败"
print("✅ 通信组销毁成功")

# 清理内存
del model, named_tensors
torch.npu.empty_cache()