"""
All supported endpoints

*
[POST] /generate
[PUT] /generate
Usage: Handle a text generation request

*
[GET] /get_server_args
Usage: Get the server arguments

[GET] /health
Usage: Check the health of the HTTP server

[GET] /health_generate
Usage: Check the health of the inference server by generating one token

[GET] /get_model_info
Usage: Get information about the model being served

[GET] /flush_cache
Usage: Flush the radix cache

[GET] /start_profile
[POST] /start_profile  
Usage: Start profiling

[GET] /stop_profile
[POST] /stop_profile
Usage: Stop profiling

[POST] /update_weights
Usage: Update model weights without restarting the server

[POST] /encode
[PUT] /encode  
Usage: Handle an embedding request

[POST] /judge
[PUT] /judge
Usage: Handle a reward model request

OpenAI related:

[POST] /v1/completions
Usage: OpenAI-compatible completions endpoint

[POST] /v1/chat/completions  
Usage: OpenAI-compatible chat completions endpoint

[POST] /v1/embeddings
Usage: OpenAI-compatible embeddings endpoint

[GET] /v1/models
Usage: List available models

[POST] /v1/files
Usage: Upload a file for batch processing

[DELETE] /v1/files/{file_id}
Usage: Delete an uploaded file

[POST] /v1/batches
Usage: Create a new batch processing job

[POST] /v1/batches/{batch_id}/cancel
Usage: Cancel a batch processing job

[GET] /v1/batches/{batch_id}
Usage: Retrieve information about a batch job

[GET] /v1/files/{file_id}
Usage: Retrieve information about an uploaded file

[GET] /v1/files/{file_id}/content
Usage: Retrieve the content of an uploaded file
"""