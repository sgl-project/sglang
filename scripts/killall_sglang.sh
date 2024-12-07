# Kill all SGLang processes and free the GPU memory.

nvidia-smi
kill -9 $(ps aux | grep 'multiprocessing.spawn' | grep -v 'grep' | awk '{print $2}')
kill -9 $(ps aux | grep 'sglang.launch_server' | grep -v 'grep' | awk '{print $2}')
kill -9 $(ps aux | grep 'sglang.bench' | grep -v 'grep' | awk '{print $2}')

# Kill all processes occupying GPU memory, and enable it when needed.
# kill -9 $(nvidia-smi | sed -n '/Processes:/,$p' | grep "   [0-9]" | awk '{print $5}')
