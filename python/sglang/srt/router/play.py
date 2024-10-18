import asyncio

async def faulty_task():
    await asyncio.sleep(1)
    print("About to raise an exception")
    raise ValueError("Oops!")

async def main():
    task = asyncio.create_task(faulty_task())
    await asyncio.sleep(2)
    print("Main coroutine still running")

asyncio.run(main())