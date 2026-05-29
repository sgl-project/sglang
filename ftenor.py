import requests
import torch
import base64
import gc
from transformers import AutoModelForCausalLM, AutoConfig
from sglang.srt.utils import MultiprocessingSerializer

# --------------------------
# 配置
# --------------------------
SGLANG_PORT = 6990
NEW_MODEL_PATH = "/home/wzy/qwen3-30b-a3b"
SGLANG_TP_SIZE = 2
SGLANG_BASE_NPU_ID = 12

# 自动读取模型配置
config = AutoConfig.from_pretrained(NEW_MODEL_PATH, trust_remote_code=True)
HIDDEN_SIZE = config.hidden_size  # 2048
NUM_ATTENTION_HEADS = config.num_attention_heads  # 32
NUM_KEY_VALUE_HEADS = config.num_key_value_heads  # 4
HEAD_DIM = config.head_dim  # 128
NUM_LAYERS = config.num_hidden_layers  # 32

# 计算QKV总维度
Q_DIM = NUM_ATTENTION_HEADS * HEAD_DIM  # 4096
KV_DIM = NUM_KEY_VALUE_HEADS * HEAD_DIM  # 512
QKV_TOTAL_DIM = Q_DIM + KV_DIM + KV_DIM  # 5120

print(f"✅ Qwen3-30B-A3B 配置：")
print(f"   HIDDEN_SIZE: {HIDDEN_SIZE}")
print(f"   HEAD_DIM: {HEAD_DIM}")
print(f"   TP={SGLANG_TP_SIZE} 时 qkv_proj 形状：[{QKV_TOTAL_DIM//SGLANG_TP_SIZE}, {HIDDEN_SIZE}] = [2560, 2048]")
print(f"   TP={SGLANG_TP_SIZE} 时 o_proj 形状：[{HIDDEN_SIZE}, {Q_DIM//SGLANG_TP_SIZE}] = [2048, 2048]")
print(f"\n⚠️  注意：本次更新将跳过所有MLP层，只更新注意力层、嵌入层和归一化层\n")

# --------------------------
# 1. 加载完整模型到CPU
# --------------------------
print("Loading Qwen3-30B-A3B to CPU...")
full_model = AutoModelForCausalLM.from_pretrained(
    NEW_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# --------------------------
# 2. 只更新非MLP层
# --------------------------
print(f"Splitting weights for {SGLANG_TP_SIZE} TP ranks...")
serialized_tensors = []
serializer = MultiprocessingSerializer()

for tp_rank in range(SGLANG_TP_SIZE):
    device = f"npu:{SGLANG_BASE_NPU_ID + tp_rank}"
    print(f"\nProcessing rank {tp_rank} on {device}...")
    
    raw_shards = {}
    for name, param in full_model.named_parameters():
        # ✅ 跳过所有MLP相关层
        if "mlp" in name:
            continue
            
        if any(layer in name for layer in ["q_proj", "k_proj", "v_proj"]):
            # 注意力输入投影层：沿输出维度（dim=0）分片
            shard = param.data.chunk(SGLANG_TP_SIZE, dim=0)[tp_rank]
        elif any(layer in name for layer in ["o_proj"]):
            # 注意力输出投影层：沿输入维度（dim=1）分片
            shard = param.data.chunk(SGLANG_TP_SIZE, dim=1)[tp_rank]
        elif "lm_head" in name or "embed_tokens" in name:
            # 嵌入层：沿词汇表维度（dim=0）分片
            shard = param.data.chunk(SGLANG_TP_SIZE, dim=0)[tp_rank]
        else:
            # 其他层（norm、bias等）全量复制
            shard = param.data.clone()
        
        raw_shards[name] = shard
    
    # 融合权重
    rank_named_tensors = []
    processed_names = set()
    
    # 1. 融合qkv_proj
    for layer_id in range(NUM_LAYERS):
        base_prefix = f"model.layers.{layer_id}.self_attn."
        q_name = f"{base_prefix}q_proj.weight"
        k_name = f"{base_prefix}k_proj.weight"
        v_name = f"{base_prefix}v_proj.weight"
        
        if q_name in raw_shards and k_name in raw_shards and v_name in raw_shards:
            q_shard = raw_shards[q_name]
            k_shard = raw_shards[k_name]
            v_shard = raw_shards[v_name]
            
            qkv_shard = torch.cat([q_shard, k_shard, v_shard], dim=0)
            
            expected_qkv_size = (QKV_TOTAL_DIM // SGLANG_TP_SIZE, HIDDEN_SIZE)
            assert qkv_shard.size() == expected_qkv_size, \
                f"Layer {layer_id} qkv_proj 形状错误：期望 {expected_qkv_size}，实际 {qkv_shard.size()}"
            
            fused_name = f"{base_prefix}qkv_proj.weight"
            rank_named_tensors.append((fused_name, qkv_shard.to(device)))
            
            processed_names.add(q_name)
            processed_names.add(k_name)
            processed_names.add(v_name)
    
    # 2. 添加所有未处理的非MLP权重
    for name, shard in raw_shards.items():
        if name not in processed_names:
            rank_named_tensors.append((name, shard.to(device)))
    
    # 验证：确保没有MLP层
    has_mlp = any("mlp" in name for name, _ in rank_named_tensors)
    assert not has_mlp, "不应该包含任何MLP层"
    
    # 打印更新的层数量
    print(f"✅ rank {tp_rank} 准备更新 {len(rank_named_tensors)} 个非MLP参数")
    
    # 序列化
    serialized_bytes = serializer.serialize(rank_named_tensors)
    serialized_str = base64.b64encode(serialized_bytes).decode("utf-8")
    serialized_tensors.append(serialized_str)
    
    del raw_shards, rank_named_tensors
    gc.collect()
    torch.npu.empty_cache()

# 验证TP数量
assert len(serialized_tensors) == SGLANG_TP_SIZE
print(f"\n✅ Successfully generated {SGLANG_TP_SIZE} weight shards (MLP skipped)")

# --------------------------
# 3. 调用接口
# --------------------------
url = f"http://localhost:{SGLANG_PORT}/update_weights_from_tensor"
data = {
    "serialized_named_tensors": serialized_tensors,
    "load_format": "direct",
    "flush_cache": True,
    "abort_all_requests": False,
    "weight_version": "qwen3-30b-a3b-non-mlp-v1"
}

print("\nCalling update_weights_from_tensor...")
response = requests.post(url, json=data, timeout=600)

# --------------------------
# 4. 验证结果
# --------------------------
print(f"Status code: {response.status_code}")
result = response.json()
print(f"Result: {result}")

assert result["success"] is True, f"Update failed: {result['message']}"
print("\n✅ Qwen3-30B-A3B TP=2 NPU 非MLP层热更新成功！")
print("\n⚠️  重要提示：MLP层权重未更新，推理结果可能不准确。")
print("   这是临时解决方案，后续需要修复MLP层的融合逻辑。")

import time
time.sleep(5)