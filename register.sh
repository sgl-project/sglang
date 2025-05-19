curl -X POST http://localhost:8000/register -H "Content-Type: application/json" -d '{
"mode": "prefill",
"url": "http://localhost:30000",
"bootstrap_port": 8998
}'
