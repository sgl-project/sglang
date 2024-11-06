import router

# Create a Router instance with:
# - host: the address to bind to (e.g., "127.0.0.1")
# - port: the port number (e.g., 3001)
# - worker_urls: list of worker URLs to distribute requests to
router = router.Router(
    host="127.0.0.1",
    port=3001,
    worker_urls=[
        "http://localhost:30000",
        "http://localhost:30002",
    ],
)

# Start the router - this will block and run the server
router.start()
