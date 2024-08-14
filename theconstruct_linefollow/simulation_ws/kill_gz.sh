ps aux | grep gz | grep -v grep | awk '{print $2}' | xargs -r kill -9
