from memcache_hybrid import DistributedObjectStore, LocalConfig

config = LocalConfig()
config.meta_service_url = "tcp://127.0.0.1:5001"
config.config_store_url = "tcp://127.0.0.1:6000"
config.protocol = "host_shm"
config.dram_size = "10GB"
config.max_dram_size = "64GB"
config.hbm_size = "0"
config.log_level = "info"

store = DistributedObjectStore()
ret = store.setup(config)
print("setup ret:", ret)
ret = store.init(device_id=0, init_bm=True)


for size in [4096, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]:
    key = f"test_{size}"
    value = bytearray(size)

    put_ret = store.put(key, value)
    exist_ret = store.is_exist(key)
    get_val = store.get(key)

    print(
        key,
        "put_ret =", put_ret,
        "exist_ret =", exist_ret,
        "get_ok =", get_val == value if get_val is not None else False,
    )