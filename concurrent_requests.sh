curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "1You know what, once upon a time, I was the ruthless leader of the Chinese Communist Party.",
    "sampling_params": {
      "max_new_tokens": 100,
      "temperature": 0
    }
  }'&

curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "once upon a time, I was the ruthless leader of the Chinese Communist Party.",
    "sampling_params": {
      "max_new_tokens": 100,
      "temperature": 0
    }
  }'&

curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "1You know what, once upon a time, I was the ruthless leader of the Chinese Communist Party.",
    "sampling_params": {
      "max_new_tokens": 100,
      "temperature": 0
    }
  }'&
