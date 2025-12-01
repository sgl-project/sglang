## R3 usage

1. add `--enable-return-routed-experts` when launching SGLang server
2. add `"return_routed_experts":True` from client for request payload

### From Network
get "return_experts" from response. It is base64 encoded string, you need to reverse to tensor like following:
``` python
raw_bytes = pybase64.b64decode(ret["meta_info"]["routed_experts"])
numpy_array = np.frombuffer(raw_bytes, dtype=np.int32)
restored_tensor = torch.from_numpy(numpy_array).reshape(-1, layer_num, topk)
```

### From Disk
1. add extra params `--r3-use-storage-backup` and `--r3-storage-backup-path <r3_dir_path>` for launching SGLang, it will save expert_ids results to a file for pre-request asynchronously.
2. get `id` from response. For each `<id>`, find `<id>_routed_experts.pt` in `<r3_dir_path>`, and load it using `torch.load` (since it is asynchronously saved, you may need to wait until the file is created).
``` python
torch.load(f"<r3_dir_path>/{ret['meta_info']['id']}_routed_experts.pt")
```
3. after loading expert_ids, remember remove releated file in `<r3_dir_path>`, **the file needs to be deleted by RL system**.