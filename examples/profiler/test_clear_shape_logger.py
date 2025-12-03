"""
Quick test of the clear shape logger with a simple PyTorch model.
This demonstrates the output format without requiring a full LLM.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

# Import our clear shape logger
from profile_with_clear_shapes import ClearShapeLogger


class SimpleModel(nn.Module):
    """A simple model to demonstrate shape logging."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    print("="*80)
    print("Clear Shape Logger Test")
    print("="*80)
    print("This demonstrates the output format for kernel shapes.")
    print("="*80)
    
    # Create model and input
    model = SimpleModel()
    batch_size = 4
    x = torch.randn(batch_size, 1024)
    
    output_file = "test_clear_shapes.jsonl"
    
    # Run with shape logging
    print("\nRunning model with shape logging...\n")
    with ClearShapeLogger(output_file=output_file, verbose=True, tp_size=8) as logger:
        output = model(x)
    
    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
    print(f"Output shape: {list(output.shape)}")
    print(f"Shapes logged to: {output_file}")
    print("\nTo analyze:")
    print(f"  python analyze_clear_shapes.py {output_file}")
    print(f"  python analyze_clear_shapes.py {output_file} --detailed")
    print("="*80)


if __name__ == "__main__":
    main()
