PORT=${PORT:-30000}

curl -X POST http://localhost:$PORT/start_profile \
    -H 'Content-Type: application/json' \
    -d '{
        "num_steps": 10,
        "output_dir": "./"
    }'
