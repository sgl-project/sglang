import torch


class ReqToTokenPool:
    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,  # not used for now
    ):

        self.size = size
        self.max_context_len = max_context_len
        self.free_slots = list(range(size))  # should use rid as the key
        # add a dummy entry for capturing
        self.pool = {
            0: torch.empty((size, max_context_len), dtype=torch.int32, device=device)
        }
        self.device = device

    def write(self, indices, values: torch.Tensor):
        match indices:
            case int(rid):
                self.pool[rid][: len(values)] = values
            case (int(rid), slice() as s):
                self.pool[rid][s] = values
            case (list() as rids, torch.Tensor() as locs):
                for i, (rid, loc) in enumerate(zip(rids, locs.tolist())):
                    self.pool[rid][loc] = values[i]
            case _:
                raise ValueError(f"Invalid indices: {indices}")

    def read(self, indices) -> torch.Tensor:
        match indices:
            case int(rid):
                return self.pool[rid]
            case (int(rid), slice() as s):
                return self.pool[rid][s]
            case (list() as rids, list() as locs):
                result = torch.empty(len(rids), dtype=torch.int32, device=self.device)
                for i, (rid, loc) in enumerate(zip(rids, locs)):
                    result[i] = self.pool[rid][int(loc)]
                return result
            case _:
                raise ValueError(f"Invalid indices: {indices}")

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> list[int] | None:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        for index in select_index:
            # TODO: use empty tensor to reduce memory usage
            self.pool[index] = torch.empty(
                (self.max_context_len,), dtype=torch.int32, device=self.device
            )

        return select_index

    def free(self, free_index: int | list[int]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
            del self.pool[free_index]
        else:
            self.free_slots.extend(free_index)
            for index in free_index:
                del self.pool[index]

    def clear(self):
        self.free_slots = list(range(self.size))
        self.pool.clear()
