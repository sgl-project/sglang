ps -ef | grep sgl | grep -v grep | awk '{print $2}' | xargs kill -9
