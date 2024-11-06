# sglang_triton

Build the docker image:
```
docker build -t sglang-triton .
```

Then do:
```
docker run -ti --gpus=all --network=host --name sglang-triton -v ./models:/mnt/models sglang-triton
```

inside the docker container:
```
cd sglang
python3 -m sglang.launch_server --model-path mistralai/Mistral-7B-Instruct-v0.2 --port 30000 --mem-fraction-static 0.9
```

with another shell, inside the docker container:
```
docker exec -ti sglang-triton /bin/bash
cd /mnt
tritonserver --model-repository=/mnt/models
```


Send request to the server:
```
curl -X POST http://localhost:8000/v2/models/character_generation/generate \
-H "Content-Type: application/json" \
-d '{
  "INPUT_TEXT": ["harry"]
}'

```
