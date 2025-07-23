from sglang.srt.mem_cache.hicache_storage import MooncakeStore

store = MooncakeStore()

exist = store.exists(["key1", "key2"])
print(exist)