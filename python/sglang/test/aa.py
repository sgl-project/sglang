import asyncio

async def concurrent_generator(*generators):
    queue = asyncio.Queue()
    tasks = []

    async def producer(gen):
        try:
            async for item in gen:
                await queue.put(item)
        finally:
            await queue.put(None)  # Sentinel value to indicate completion

    # Start a producer task for each generator
    for gen in generators:
        tasks.append(asyncio.create_task(producer(gen)))

    completed = 0
    total = len(tasks)

    while completed < total:
        print(completed, total)
        item = await queue.get()
        if item is None:
            completed += 1
        else:
            yield item

    # Optionally, ensure all tasks are done
    await asyncio.gather(*tasks)

# Example usage
async def gen1():
    for i in range(3):
        await asyncio.sleep(1)
        yield f"Gen1: {i}"

async def gen2():
    for i in range(3):
        await asyncio.sleep(1.5)
        yield f"Gen2: {i}"

async def gen3():
    for i in range(3):
        await asyncio.sleep(0.7)
        yield f"Gen3: {i}"

async def main():
    async for item in concurrent_generator(gen1(), gen2(), gen3()):
        print(item)

if __name__ == "__main__":
    asyncio.run(main())
