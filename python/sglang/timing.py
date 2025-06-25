"""Timing for moe model"""
import time
import torch
import json
import os
from collections import defaultdict
from typing import Dict, Optional, List

class MoETimer:
    def __init__(self):
        super().__init__()
        self.timers = defaultdict(list)  # 存储所有计时结果
        self.starts = {}  # 当前正在计时的开始时间点
        self.use_cuda = torch.cuda.is_available()
        
    def start_timer(self, metric_name: str):
        """开始计时一个指标"""
        if self.use_cuda:
            torch.cuda.synchronize()  # 确保之前的所有CUDA操作完成
        self.starts[metric_name] = time.time()
    
    def end_timer(self, metric_name: str):
        """结束计时并保存结果"""
        if metric_name not in self.starts:
            raise ValueError(f"Timer not started for metric: {metric_name}")
            
        if self.use_cuda:
            torch.cuda.synchronize()
            
        elapsed_ms = (time.time() - self.starts[metric_name]) * 1000
        self.timers[metric_name].append(elapsed_ms)
        del self.starts[metric_name]
    
    def get_stats(self) -> Dict[str, Dict]:
        """获取所有指标的统计信息"""
        stats = {}
        for metric, times in self.timers.items():
            if not times: 
                continue
                
            sorted_times = sorted(times)
            count = len(times)
            stats[metric] = {
                "count": count,
                "mean(ms)": round(sum(times) / count, 3),
                "min(ms)": round(min(times), 3),
                "max(ms)": round(max(times), 3),
                "p95(ms)": round(sorted_times[int(0.95 * count)], 3),
            }
        return stats
    
    def save_results(self, file_name: str):
        """保存统计结果到JSON文件"""
        print(f"Saving results to {file_name}")
        with open(file_name, 'w') as f:
            json.dump(self.get_stats(), f, indent=4)
    
    def reset(self):
        """重置所有计时器"""
        self.timers.clear()
        self.starts.clear()
        
class DeriveClass(MoETimer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def run(self):
        for i in range(10):
            self.start_timer("test")
            time.sleep(1 + i * 0.1)
            self.end_timer("test")
        self.save_results("test.json")

if __name__ == "__main__":
    # test
    timer = MoETimer()
    for i in range(10):
        timer.start_timer("test")
        time.sleep(1 + i * 0.1)
        timer.end_timer("test")
    print(timer.get_stats())