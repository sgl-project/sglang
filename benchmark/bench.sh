python3 send_req.py \
    --url "http://127.0.0.1:30300/v1/chat/completions" \
    --concurrency 100 \
    --total-requests 100 \
    --payload-file q.json \
    --output-file results.json