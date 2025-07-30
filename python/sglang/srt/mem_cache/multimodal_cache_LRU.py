from typing import Dict

import torch


# Define Node of Doubly Linked List for LRU Cache
class Node:
    def __init__(self, key: int, value: torch.Tensor):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class MultiModalCache:
    """MultiModalCache is used to store vlm encoder results"""

    def __init__(
        self,
        max_size: int,
    ):
        self.max_size = max_size
        self.mm_cache: Dict[int, Node] = {}
        self.current_size = 0
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head
    

    # define the delete function to remove a node from the DLL
    def delete(self, node: Node):
        # get prev and next ndoes of the node
        prv = node.prev
        nxt = node.next
        # link them together
        prv.next = nxt
        nxt.prev = prv

    # define the add function to insert the node at the first position of the DLL
    def add(self, node: Node):
        # get the next pointer of head
        nxt = self.head.next
        # link the node with head and nxt
        self.head.next = node
        nxt.prev = node
        node.next = nxt
        node.prev = self.head

    def get_free_memory_torch(self):
        torch.cuda.empty_cache()  # Clears unused memory in PyTorch's cache
        free, total = torch.cuda.mem_get_info()
        return free / 1024**3  # Return MiB

    def put(self, mm_hash: int, embedding: torch.Tensor) -> bool:
        
        print(f"Put Embedding with hash {mm_hash} into cache.")
        if mm_hash in self.mm_cache:
            node = self.mm_cache[mm_hash]
            self.delete(node)
            del self.mm_cache[mm_hash]
            self.current_size -= self._get_tensor_size(node.value)
        
        # check if eviction is needed
        while self.tail.prev != self.head and self.current_size + self._get_tensor_size(embedding) > self.max_size:
            least_used_embedding_size = self._get_tensor_size(self.tail.prev.value)
            print(f"Evict embedding with hash {self.tail.prev.key} from cache.")
            del self.mm_cache[self.tail.prev.key]
            self.delete(self.tail.prev)
            self.current_size -= least_used_embedding_size
            print(f"Current cache size: {self.current_size} bytes, Max size: {self.max_size} bytes")
        
        
        if self.current_size + self._get_tensor_size(embedding) > self.max_size:
            print(f"Embedding with hash {mm_hash} is too large to fit in cache. Skipping insertion.")
            return False
        
        new_node = Node(mm_hash, embedding)
        self.add(new_node)
        self.mm_cache[mm_hash] = self.head.next
        self.current_size += self._get_tensor_size(embedding)
        return True


            
    def get(self, mm_hash: int) -> torch.Tensor:
        
        if mm_hash in self.mm_cache:
            print(f"Embedding with hash {mm_hash} found in cache.")
            embedding = self.mm_cache[mm_hash].value
            node = self.mm_cache[mm_hash]
            del self.mm_cache[mm_hash]
            self.delete(node)
            self.add(node)
            self.mm_cache[mm_hash] = self.head.next
            return embedding
        else:
            print(f"Embedding with hash {mm_hash} not found in cache.")
            return None


    def free(self, mm_hash: int) -> bool:
        print("Multimodal Cache Free!!")
        if mm_hash not in self.mm_cache:
            return False
        node = self.mm_cache.pop(mm_hash)
        old_embedding = node.value
        self.delete(node)
        self.current_size -= self._get_tensor_size(old_embedding)
        return True

    def clear(self):
        print("Multimodal Cache Clear!!")
        self.mm_cache.clear()
        self.current_size = 0
        self.head.next = self.tail
        self.tail.prev = self.head

    def _get_tensor_size(self, embedding: torch.Tensor):
        return embedding.element_size() * embedding.numel()

    def __len__(self):
        return len(self.mm_cache)
