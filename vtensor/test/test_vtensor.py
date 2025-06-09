import torch
import vTensor

if __name__ == "__main__":
    torch.tensor([0], device='cuda') # just to init cuda ctx

    #test_vmm_weight()
    vTensor.init_shared_phy_blocks(1, 4*1024*1024)
    vTensor.init_unique_phy_blocks(2, 4*1024*1024)
    v = vTensor.tensor((4*1024*1024, 1, 1), torch.bfloat16, 0, 2, 1)
    a = v.to_torch_tensor()
    print(a.shape)

    b = v.split_tensor((4*1024*1024 // 2, 1, 1), torch.bfloat16, 0)
    b[:,:,:] = 1
    print(b.shape)

    t = vTensor.tensor((4*1024*1024, 1, 1), torch.bfloat16, 0, 2, 0)
    c = t.to_torch_tensor()

    d = t.split_tensor((4*1024*1024 // 2, 1, 1), torch.bfloat16, 0)
    d[:,:,:] = 2
    import pdb
    pdb.set_trace()
