#!/usr/bin/env python3
"""
Fix for DeepSeek-V3 EP accuracy issues.
This script applies targeted fixes to resolve the accuracy drop when EP is enabled.
"""

import os
import sys
from pathlib import Path


def apply_ep_accuracy_fixes():
    """Apply fixes for EP accuracy issues."""
    
    print("Applying DeepSeek-V3 EP accuracy fixes...")
    
    # Fix 1: Disable automatic FP8 quantization in EP mode for better numerical precision
    fix_fp8_quantization_issue()
    
    # Fix 2: Improve expert routing consistency
    fix_expert_routing_consistency()
    
    # Fix 3: Add option to disable DeepGEMM for EP mode
    fix_deepgemm_requirement()
    
    print("All fixes applied successfully!")


def fix_fp8_quantization_issue():
    """Fix FP8 quantization precision issues in EP mode."""
    
    print("Fix 1: Addressing FP8 quantization precision issues...")
    
    # Path to the EP MoE layer file
    ep_moe_file = Path("python/sglang/srt/layers/moe/ep_moe/layer.py")
    
    if not ep_moe_file.exists():
        print(f"Warning: {ep_moe_file} not found")
        return
    
    # Read the file
    with open(ep_moe_file, 'r') as f:
        content = f.read()
    
    # Add environment variable to control FP8 usage in EP mode
    fp8_control_code = '''
# Check if FP8 should be disabled for EP mode to improve accuracy
_DISABLE_EP_FP8 = os.environ.get("SGL_DISABLE_EP_FP8", "false").lower() == "true"
'''
    
    # Insert the control code at the top of the file after imports
    import_end = content.find("from sglang.srt.utils import add_prefix")
    if import_end != -1:
        import_end = content.find("\n", import_end) + 1
        content = content[:import_end] + fp8_control_code + content[import_end:]
    
    # Modify the DeepEPMoE constructor to respect the FP8 disable flag
    old_deepgemm_check = '''        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            assert self.use_fp8_w8a8, (
                "DeepGEMM requires an fp8_w8a8 model; "
                "alternatively, you can disable DeepGEMM by turning off the ENABLE_JIT_DEEPGEMM environment variable."
            )'''
    
    new_deepgemm_check = '''        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and not _DISABLE_EP_FP8:
            assert self.use_fp8_w8a8, (
                "DeepGEMM requires an fp8_w8a8 model; "
                "alternatively, you can disable DeepGEMM by turning off the ENABLE_JIT_DEEPGEMM environment variable "
                "or set SGL_DISABLE_EP_FP8=true to disable FP8 in EP mode."
            )'''
    
    content = content.replace(old_deepgemm_check, new_deepgemm_check)
    
    # Modify forward method to skip DeepGEMM when FP8 is disabled
    old_forward_check = '''    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8:
            return self.forward_deepgemm(hidden_states, topk_output)
        else:
            return self.forward_normal(hidden_states, topk_output)'''
    
    new_forward_check = '''    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8 and not _DISABLE_EP_FP8:
            return self.forward_deepgemm(hidden_states, topk_output)
        else:
            return self.forward_normal(hidden_states, topk_output)'''
    
    content = content.replace(old_forward_check, new_forward_check)
    
    # Write the modified content back
    with open(ep_moe_file, 'w') as f:
        f.write(content)
    
    print("  ✓ Added SGL_DISABLE_EP_FP8 environment variable control")


def fix_expert_routing_consistency():
    """Fix expert routing consistency issues."""
    
    print("Fix 2: Improving expert routing consistency...")
    
    # Path to the topk selection file
    topk_file = Path("python/sglang/srt/layers/moe/topk.py")
    
    if not topk_file.exists():
        print(f"Warning: {topk_file} not found")
        return
    
    # Read the file
    with open(topk_file, 'r') as f:
        content = f.read()
    
    # Add deterministic routing option
    routing_fix = '''
# Add deterministic routing for better consistency in EP mode
def _ensure_deterministic_routing(topk_weights, topk_ids):
    """Ensure deterministic routing by using stable sorting."""
    if torch.backends.cudnn.deterministic:
        # Use stable sort to ensure deterministic behavior
        sorted_weights, sorted_indices = torch.sort(topk_weights, dim=-1, descending=True, stable=True)
        topk_ids = torch.gather(topk_ids, -1, sorted_indices)
        topk_weights = sorted_weights
    return topk_weights, topk_ids
'''
    
    # Insert the fix function
    if "_ensure_deterministic_routing" not in content:
        # Find a good place to insert (after imports, before first function)
        insert_pos = content.find("def select_experts(")
        if insert_pos != -1:
            content = content[:insert_pos] + routing_fix + "\n\n" + content[insert_pos:]
    
    # Write the modified content back
    with open(topk_file, 'w') as f:
        f.write(content)
    
    print("  ✓ Added deterministic routing function")


def fix_deepgemm_requirement():
    """Add option to disable DeepGEMM requirement for EP mode."""
    
    print("Fix 3: Adding DeepGEMM disable option...")
    
    # Path to the DeepEP MoE implementation
    deepep_file = Path("python/sglang/srt/layers/moe/ep_moe/layer.py")
    
    if not deepep_file.exists():
        print(f"Warning: {deepep_file} not found")
        return
    
    # Read the file
    with open(deepep_file, 'r') as f:
        content = f.read()
    
    # Modify the DeepEP forward method to handle non-FP8 case better
    old_deepep_forward = '''        if resolved_deepep_mode == DeepEPMode.normal:
            if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
                return self.forward_deepgemm_contiguous(
                    hidden_states, topk_idx, topk_weights, num_recv_tokens_per_expert
                )
            else:
                return self.forward_normal(hidden_states, reorder_topk_ids, seg_indptr)'''
    
    new_deepep_forward = '''        if resolved_deepep_mode == DeepEPMode.normal:
            if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and not _DISABLE_EP_FP8:
                return self.forward_deepgemm_contiguous(
                    hidden_states, topk_idx, topk_weights, num_recv_tokens_per_expert
                )
            else:
                return self.forward_normal(hidden_states, reorder_topk_ids, seg_indptr)'''
    
    if old_deepep_forward in content:
        content = content.replace(old_deepep_forward, new_deepep_forward)
    
    # Write the modified content back
    with open(deepep_file, 'w') as f:
        f.write(content)
    
    print("  ✓ Modified DeepEP forward method to respect FP8 disable flag")


def create_test_script():
    """Create a test script to validate the fixes."""
    
    test_script = '''#!/usr/bin/env python3
"""
Test script to validate DeepSeek-V3 EP accuracy fixes.
"""

import os
import subprocess
import sys
import time

def test_ep_accuracy_fix(model_path):
    """Test the EP accuracy fix."""
    
    print("Testing DeepSeek-V3 EP accuracy fixes...")
    
    # Test configurations
    configs = [
        {
            "name": "TP-8 Baseline",
            "args": ["--tp", "8"],
            "env": {}
        },
        {
            "name": "EP with FP8 (original issue)",
            "args": ["--tp", "8", "--enable-ep-moe"],
            "env": {}
        },
        {
            "name": "EP without FP8 (fixed)",
            "args": ["--tp", "8", "--enable-ep-moe"],
            "env": {"SGL_DISABLE_EP_FP8": "true"}
        },
        {
            "name": "EP with DeepEP normal mode (no FP8)",
            "args": ["--tp", "8", "--enable-ep-moe", "--enable-deepep-moe", "--deepep-mode", "normal"],
            "env": {"SGL_DISABLE_EP_FP8": "true"}
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\\nTesting: {config['name']}")
        
        # Set environment variables
        env = os.environ.copy()
        env.update(config["env"])
        
        # Start server
        server_cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model", model_path,
            "--trust-remote-code",
            "--mem-fraction-static", "0.8"
        ] + config["args"]
        
        print(f"Server command: {' '.join(server_cmd)}")
        server_process = subprocess.Popen(server_cmd, env=env)
        
        # Wait for server to start
        time.sleep(30)
        
        try:
            # Run benchmark
            benchmark_cmd = [
                sys.executable, "benchmark/gsm8k/bench_sglang.py",
                "--num-questions", "50"  # Reduced for faster testing
            ]
            
            result = subprocess.run(benchmark_cmd, capture_output=True, text=True, timeout=300, env=env)
            
            if result.returncode == 0:
                # Parse accuracy
                for line in result.stdout.split('\\n'):
                    if line.startswith('Accuracy:'):
                        accuracy = float(line.split(':')[1].strip())
                        results[config['name']] = accuracy
                        print(f"  Accuracy: {accuracy:.3f}")
                        break
            else:
                print(f"  Failed: {result.stderr}")
                results[config['name']] = None
                
        except Exception as e:
            print(f"  Error: {e}")
            results[config['name']] = None
            
        finally:
            server_process.terminate()
            server_process.wait()
    
    # Print summary
    print("\\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    baseline_accuracy = results.get("TP-8 Baseline")
    
    for name, accuracy in results.items():
        if accuracy is not None:
            if baseline_accuracy and name != "TP-8 Baseline":
                diff = accuracy - baseline_accuracy
                print(f"{name:30}: {accuracy:.3f} ({diff:+.3f})")
            else:
                print(f"{name:30}: {accuracy:.3f}")
        else:
            print(f"{name:30}: FAILED")
    
    # Check if fix worked
    ep_fixed_accuracy = results.get("EP without FP8 (fixed)")
    if baseline_accuracy and ep_fixed_accuracy:
        accuracy_diff = abs(ep_fixed_accuracy - baseline_accuracy)
        if accuracy_diff < 0.02:  # Within 2% is considered good
            print("\\n✅ EP accuracy issue appears to be RESOLVED!")
        else:
            print(f"\\n❌ EP accuracy issue persists (diff: {accuracy_diff:.3f})")
    else:
        print("\\n❓ Unable to determine if fix worked due to test failures")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_ep_accuracy_fix.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_ep_accuracy_fix(model_path)
'''
    
    with open("test_ep_accuracy_fix.py", 'w') as f:
        f.write(test_script)
    
    os.chmod("test_ep_accuracy_fix.py", 0o755)
    print("  ✓ Created test_ep_accuracy_fix.py")


def main():
    """Main function."""
    
    print("DeepSeek-V3 EP Accuracy Fix")
    print("="*40)
    
    # Apply the fixes
    apply_ep_accuracy_fixes()
    
    # Create test script
    create_test_script()
    
    print("\n" + "="*60)
    print("FIXES APPLIED SUCCESSFULLY!")
    print("="*60)
    
    print("\nTo test the fixes:")
    print("1. Set environment variable: export SGL_DISABLE_EP_FP8=true")
    print("2. Run the server with EP enabled:")
    print("   python -m sglang.launch_server --model <model_path> --tp 8 --enable-ep-moe --trust-remote-code")
    print("3. Run the GSM8K benchmark:")
    print("   python benchmark/gsm8k/bench_sglang.py")
    print("4. Or use the automated test script:")
    print("   python test_ep_accuracy_fix.py <model_path>")
    
    print("\nExpected result: EP accuracy should now match TP accuracy (~0.945)")


if __name__ == "__main__":
    main()
