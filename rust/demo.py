from sglang_router import PolicyType, Router

router = Router(
    worker_urls=[
        "http://localhost:30000",
        "http://localhost:30001",
    ]
)

router.start()
