# SGLang Triton Python Backend

We provide the [Triton Python Backend](https://github.com/triton-inference-server/python_backend/tree/main) for SGLang. Here is the guide to deploy server, test client and profile server performance.

## Server Deployment

### Step 1: Prepare the Model Repository

The Triton Inference Server requires a specific folder structure for serving models. We provided the necessary `model.py` and `config.pbtxt` files. Follow these steps to set up your model repository:

1. Create a directory named `model_repository/` in a suitable location.

2. Inside the `model_repository/` directory, create a subdirectory for your model. For this example, we will name it `sglang_model/`, which should be consistent with the name in `config.pbtxt`.

3. Within the `sglang_model/` directory, create another subdirectory named `1/`, which represents version 1 of your model. Triton supports model versioning, and each version should have its own directory.

4. Place the `model.py` file inside the `1/` directory. This file contains the logic for your model's inference.

5. The model weights should be placed in a folder named `weights/` inside the `1/` directory.

6. Place the `config.pbtxt` file directly inside the `sglang_model/` directory. This configuration file is essential for Triton to understand how to serve your model.

7. For advanced usage, you can modify the `config.pbtxt` to use your customized configurations. There are two important parameters you may change:

   - `model_name`: The name of the to-be-deployed model, such as `llama`, `llama2`, `internlm2`, `baichuan2` and etc. 
   - `tp`: GPU number used in tensor parallelism. Should be 2^n.

The final folder structure should look like this:

```
model_repository
└── sglang_model
    ├── 1
    │   ├── model.py
    │   └── weights
    └── config.pbtxt
```

You can find more details of the model repository in the [Triton Inference Server User Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html).

### Step 2: Run the Triton Server

The [memory leak](https://github.com/triton-inference-server/python_backend/pull/309) issue in TIS Python Backend prior to version r23.10 occurs when the backend is in decoupled mode. It's strongly recommended to upgrade to TIS Python Backend version [r23.10](https://github.com/triton-inference-server/python_backend/tree/r23.10) or higher to mitigate this issue.

Use the following command to start the Triton Inference Server with the specified model repository:

```
export PYTHONIOENCODING=UTF-8

tritonserver \
    --model-repository=/path/to/your/model_repository \
    --allow-grpc=1 \
    --grpc-port=33337
```

- Replace `/path/to/your/model_repository` with the actual path to the model repository you prepared in step 1.
- The `--allow-grpc=1` option enables gRPC support, which is used by our client to communicate with the server.
- The `--grpc-port` option sets the port for the Triton gRPC server. You can change this to any port that suits your environment. Here we use `33337` as an example for following client test and performance profiling.

## Client Test

After successfully deploying the server, you can test it using the provided `client.py` script:

```
python3 client.py
```

- Before running the client script, make sure that the server's address and port in the `client.py` file match the ones you configured for the Triton server.

## Performance Profiling

The `profile_triton_python_backend.py` script in the `benchmark/` folder is provided to help you profile the performance of this Triton Python backend.

```
python3 benchmark/profile_triton_python_backend.py 0.0.0.0:33337 /path/to/model /path/to/ShareGPT_V3_unfiltered_cleaned_split.json --num_prompts 1000 --concurrency=128
```

- Replace `0.0.0.0:33337` with the address and port of your Triton server.
- Replace `/path/to/model` with the path to your model weights.
- Replace `/path/to/ShareGPT_V3_unfiltered_cleaned_split.json` with the path to your dataset file.
- The `--num_prompts=1000` option specifies the number of prompts you want to use for profiling.
- The `--concurrency=128` option specifies the number of concurrent requests to send to the Triton server for profiling.