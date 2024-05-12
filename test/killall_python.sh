kill -9 $(ps aux | grep 'sglang' | grep -v 'grep' | awk '{print $2}')
