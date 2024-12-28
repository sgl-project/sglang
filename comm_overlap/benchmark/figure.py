import matplotlib.pyplot as plt
import numpy as np

# 数据准备
batch_sizes = [1, 16, 64, 128]
configs = [
    (256, 32),
    (256, 256),
    (512, 32),
    (512, 256)
]

# torch的延迟数据
torch_latencies = [
    # batch_size=1, 16, 64, 128
    [0.14153, 0.17966, 0.62760, 0.47668],  # 256/32
    [0.14421, 0.21406, 0.45839, 0.78746],  # 512/32
    [0.99640, 1.13127, 1.49928, 1.81941],  # 256/256
    [0.99510, 1.18566, 1.67485, 2.18945]   # 512/256
]

# te的延迟数据
te_latencies = [
    # batch_size=1, 16, 64, 128
    [0.14115, 0.26962, 0.35437, 0.47692],  # 256/32
    [0.14150, 0.21293, 0.47643, 0.77235],  # 512/32
    [0.99611, 1.13072, 1.49399, 1.81369],  # 256/256
    [0.99560, 1.18268, 1.66728, 2.18201]   # 512/256
]

# 创建图形
plt.figure(figsize=(15, 10))

# 设置柱状图的位置
x = np.arange(len(batch_sizes))
width = 0.1  # 减小柱子宽度使图形更清晰

# 为每个配置绘制柱状图
for i, (input_len, output_len) in enumerate(configs):
    plt.bar(x + i*width*2, torch_latencies[i], 
            width, label=f'Torch-{input_len}/{output_len}', alpha=0.8)
    plt.bar(x + i*width*2 + width, te_latencies[i], 
            width, label=f'TE-{input_len}/{output_len}', alpha=0.8)

# 设置图形属性
plt.xlabel('Batch Size')
plt.ylabel('Total Latency (s)')
plt.title('Torch vs TE Total Latency Comparison')
plt.xticks(x + width*3.5, batch_sizes)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 调整布局以确保图例完全显示
plt.tight_layout()

# 保存图形
plt.savefig('latency_comparison.png', bbox_inches='tight')