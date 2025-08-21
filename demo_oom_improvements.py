#!/usr/bin/env python3
"""
Demonstration of enhanced OOM handling in SGLang.

This script shows how the new memory management features work:
1. Enhanced error messages with specific suggestions
2. Memory pressure monitoring
3. Automatic parameter optimization suggestions
4. CUDA OOM exception handling
"""

import logging
from typing import Dict, Any

# Configure logging to show the improvements
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_memory_suggestions():
    """Demonstrate the memory optimization suggestion system."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Memory Optimization Suggestions")
    print("="*60)
    
    # Mock the memory utils (normally these would be imported)
    def suggest_memory_optimizations(config: Dict[str, Any], pressure: str, context: str) -> Dict[str, Any]:
        suggestions = {}
        mem_fraction = config.get("mem_fraction_static", 0.9)
        max_requests = config.get("max_running_requests", None)
        chunked_prefill = config.get("chunked_prefill_size", None)
        
        if pressure in ["high", "critical"] or "memory" in context.lower():
            if mem_fraction > 0.7:
                suggestions["mem_fraction_static"] = max(0.7, mem_fraction - 0.1)
            if max_requests is None or max_requests > 64:
                suggestions["max_running_requests"] = min(64, max_requests or 128)
            if "prefill" in context.lower() and (chunked_prefill is None or chunked_prefill > 4096):
                suggestions["chunked_prefill_size"] = 4096
        
        return suggestions
    
    # Example scenarios
    scenarios = [
        {
            "name": "High Memory Pressure During Prefill",
            "config": {
                "mem_fraction_static": 0.9,
                "max_running_requests": 128,
                "chunked_prefill_size": None,
                "max_prefill_tokens": 16384,
            },
            "pressure": "high",
            "context": "prefill"
        },
        {
            "name": "Critical Memory During Decode",
            "config": {
                "mem_fraction_static": 0.95,
                "max_running_requests": 256,
                "chunked_prefill_size": 8192,
                "max_prefill_tokens": 16384,
            },
            "pressure": "critical", 
            "context": "decode"
        },
        {
            "name": "Normal Operation",
            "config": {
                "mem_fraction_static": 0.8,
                "max_running_requests": 32,
                "chunked_prefill_size": 4096,
                "max_prefill_tokens": 8192,
            },
            "pressure": "medium",
            "context": "normal"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Current config: {scenario['config']}")
        print(f"Memory pressure: {scenario['pressure']}")
        print(f"Context: {scenario['context']}")
        
        suggestions = suggest_memory_optimizations(
            scenario['config'], 
            scenario['pressure'], 
            scenario['context']
        )
        
        if suggestions:
            print("💡 Suggested optimizations:")
            for param, value in suggestions.items():
                print(f"   --{param.replace('_', '-')} {value}")
        else:
            print("✅ No optimizations needed - configuration looks good!")


def demonstrate_enhanced_error_messages():
    """Demonstrate the enhanced OOM error messages."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Enhanced OOM Error Messages")
    print("="*60)
    
    # Mock the error message creation (normally imported)
    def create_enhanced_error_message(context: str, tokens_req: int, tokens_avail: int, config: Dict[str, Any]) -> str:
        memory_usage = "8.5GB / 10.0GB (85%)"  # Mock values
        pressure = "high"
        
        suggestions = []
        if config.get("mem_fraction_static", 0.9) > 0.7:
            suggestions.append("--mem-fraction-static 0.7")
        if config.get("chunked_prefill_size") is None and "prefill" in context.lower():
            suggestions.append("--chunked-prefill-size 4096") 
        if config.get("max_running_requests", 128) > 32:
            suggestions.append("--max-running-requests 32")
        
        error_msg = f"""
{context} out of memory.

Memory Status:
- GPU Memory Usage: {memory_usage}
- Memory Pressure: {pressure}
- Tokens Requested: {tokens_req}
- Tokens Available: {tokens_avail}

Current Configuration:
- mem_fraction_static: {config.get('mem_fraction_static', 'auto')}
- max_running_requests: {config.get('max_running_requests', 'auto')}
- chunked_prefill_size: {config.get('chunked_prefill_size', 'disabled')}

Suggested Optimizations:"""
        
        if suggestions:
            for suggestion in suggestions:
                error_msg += f"\n- Set {suggestion}"
        else:
            error_msg += "\n- Current configuration is optimal, consider reducing batch size"
            
        error_msg += """

Quick fixes:
- Restart server with --mem-fraction-static 0.7
- Use --chunked-prefill-size 4096 for long sequences
- Set --max-running-requests 32 to limit concurrency"""
        
        return error_msg
    
    # Example error scenarios
    examples = [
        {
            "context": "Prefill",
            "tokens_requested": 8192,
            "tokens_available": 4096,
            "config": {
                "mem_fraction_static": 0.9,
                "max_running_requests": 128,
                "chunked_prefill_size": None,
            }
        },
        {
            "context": "Decode", 
            "tokens_requested": 64,
            "tokens_available": 32,
            "config": {
                "mem_fraction_static": 0.85,
                "max_running_requests": 256,
                "chunked_prefill_size": 8192,
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i}: {example['context']} OOM ---")
        error_msg = create_enhanced_error_message(
            example['context'],
            example['tokens_requested'], 
            example['tokens_available'],
            example['config']
        )
        print(error_msg)


def demonstrate_memory_monitoring():
    """Demonstrate the memory monitoring features."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Memory Monitoring and Warnings")
    print("="*60)
    
    # Mock memory monitoring scenarios
    monitoring_scenarios = [
        {
            "name": "Server Startup - Safe Configuration",
            "memory_usage": "4.2GB / 10.0GB (42%)",
            "pressure": "low",
            "config": {
                "mem_fraction_static": 0.8,
                "max_running_requests": 64,
                "context_length": 8192,
            },
            "warnings": []
        },
        {
            "name": "Server Startup - Risky Configuration", 
            "memory_usage": "8.7GB / 10.0GB (87%)",
            "pressure": "high",
            "config": {
                "mem_fraction_static": 0.95,
                "max_running_requests": 512,
                "context_length": 65536,
            },
            "warnings": [
                "mem_fraction_static (0.95) is very high - consider using 0.8-0.9",
                "max_running_requests (512) is very high - may cause OOM",
                "Large context length (65536) detected - consider chunked prefill"
            ]
        },
        {
            "name": "Runtime - Memory Pressure Detected",
            "memory_usage": "9.2GB / 10.0GB (92%)",
            "pressure": "critical", 
            "config": {
                "mem_fraction_static": 0.9,
                "max_running_requests": 128,
                "context_length": 32768,
            },
            "warnings": [
                "GPU memory pressure is critical (9.2GB/10.0GB)",
                "Consider: --mem-fraction-static 0.7",
                "Consider: --max-running-requests 64"
            ]
        }
    ]
    
    for scenario in monitoring_scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Memory Usage: {scenario['memory_usage']}")
        print(f"Pressure Level: {scenario['pressure']}")
        print(f"Configuration: {scenario['config']}")
        
        if scenario['warnings']:
            print("⚠️  Warnings:")
            for warning in scenario['warnings']:
                print(f"   {warning}")
        else:
            print("✅ Configuration looks good!")


def main():
    """Main demonstration."""
    print("SGLang Enhanced OOM Handling Demonstration")
    print("This shows the new memory management features:")
    
    demonstrate_memory_suggestions()
    demonstrate_enhanced_error_messages()
    demonstrate_memory_monitoring()
    
    print("\n" + "="*60)
    print("SUMMARY OF IMPROVEMENTS")
    print("="*60)
    print("""
✅ Enhanced OOM Error Messages:
   - Show current memory usage and pressure level
   - Provide specific parameter suggestions
   - Include quick recovery commands

✅ Automatic Memory Optimization:
   - Detect memory pressure levels
   - Suggest optimal parameters based on context
   - Warn about risky configurations

✅ CUDA OOM Exception Handling:
   - Catch torch.cuda.OutOfMemoryError gracefully
   - Provide detailed recovery guidance
   - Log memory status for debugging

✅ Server Startup Validation:
   - Check memory-related parameters
   - Warn about potentially problematic settings
   - Suggest optimizations during initialization

✅ Memory Monitoring:
   - Real-time memory usage logging
   - Pressure level detection
   - Proactive warnings before OOM occurs

These improvements help users:
1. Quickly diagnose and fix OOM issues
2. Optimize memory usage automatically
3. Prevent OOM errors before they occur
4. Understand memory usage patterns better
""")


if __name__ == "__main__":
    main()