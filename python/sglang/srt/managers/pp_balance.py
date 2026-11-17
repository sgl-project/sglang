def plan_decode_transfers(sizes: list[int], source: int, capacity: int) -> list[int]:
    """How many rows the retired source can donate to each pipeline slot."""
    loads = sizes.copy()
    transfers = [0] * len(loads)
    while True:
        smallest = min(loads)
        if smallest >= capacity or loads[source] <= smallest + 1:
            return transfers
        target = loads.index(smallest)
        loads[source] -= 1
        loads[target] += 1
        transfers[target] += 1
