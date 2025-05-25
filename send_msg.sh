curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{
  "text": "Where are you from?",
  "sampling_params": {
    "temperature": 0
  },
  "stream": false
}'

# curl http://127.0.0.1:8000/get_loads