curl -X POST http://127.0.0.1:8080/generate  -H "Content-Type: application/json" -d '{
    "text": "Once upon a time,",
    "sampling_params": {
      "max_new_tokens": 16,
      "temperature": 0
    }
  }'


curl -X GET http://127.0.0.1:8080/get_server_args

curl -X POST "http://localhost:8080/add_worker" \
     -H "Content-Type: application/json" \
     -d '{"server_url": "http://worker1:3002"}'