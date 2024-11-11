from sglang_router import PolicyType, Router

router = Router(
    worker_urls=[
        "http://localhost:30000",
        "http://localhost:30001",
    ],
    policy=PolicyType.ApproxTree,
    tokenizer_path="/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/tokenizer.json",
)

router.start()
