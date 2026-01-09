# import torch
# import triton
# import triton.language as tl

# @triton.jit
# def _get_ke_ks_triton_kernel(
#     seq_lens_ptr,
#     ks,
#     ke,
#     seq_lens_expanded,
#     prefix_sum_ptr,
#     prefix_sum_len:tl.constexpr,
#     extend_seq_lens_sum: tl.constexpr,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     '''
#     Get ke and ks fuse kernel.

#     :param seq_lens_ptr: every batch seq len, int32
#     :param ks: shape=[sum_extend_seq_len] int32
#     :param ke: shape=[sum_extend_seq_len] int32
#     :param seq_lens_expanded: shape=[sum_extend_seq_len] int32
#     :param prefix_sum_ptr: extend_seq_len offset(include 0)
#     :param prefix_sum_len: 
#     :param extend_seq_lens_sum: extend seq_len sum
#     :param BLOCK_SIZE: all batch s out tensor ptr
    
#     '''
#     # 1. 获取当前线程块的全局索引（按输出C的位置分配）
#     pid = tl.program_id(axis=0)
#     if pid >= extend_seq_lens_sum:
#         return
    
#     # 2. 计算当前线程块处理的C的位置范围
#     start_pos = pid * BLOCK_SIZE
#     # 3. 生成当前线程块处理的C的目标位置索引
#     c_pos = tl.arange(0, BLOCK_SIZE) + start_pos
#     # 4. 边界掩码（避免处理超出C_len的无效位置）
#     c_mask = c_pos < extend_seq_lens_sum
    
#     # 5. 对每个有效c_pos，通过二分查找前缀和数组反向映射输入索引i
#     # 初始化二分查找的上下界
#     low = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
#     high = tl.full((BLOCK_SIZE,), prefix_sum_len - 1, dtype=tl.int32)
    
#     # 二分查找核心逻辑（固定迭代次数，避免分支，更适合Triton）
#     for _ in range(8):
#         mid = (low + high) // 2
#         # 加载prefix_sum[mid]的值
#         prefix_mid = tl.load(prefix_sum_ptr + mid, mask=c_mask)
#         # 更新上下界：c_pos >= prefix_mid时，说明i在mid右侧，否则在左侧
#         cond = c_pos >= prefix_mid
#         low = tl.where(cond, mid, low)
#         high = tl.where(~cond, mid, high)
    
#     i = low
#     i_mask = (i >= 0) & (i < (prefix_sum_len - 1)) & c_mask
    
#     # 6. 加载对应输入B[i]和D[c_pos]（带有效掩码）
#     B_val = tl.load(seq_lens_ptr + i, mask=i_mask)
#     D_val = tl.load(seq_lens_expanded + c_pos, mask=i_mask)
    
#     # 7. 计算并存储结果（B[i] + seq_lens_expanded[c_pos]）
#     tl.store(ks + c_pos, B_val, mask=i_mask)
#     store_val = B_val + D_val
#     tl.store(ke + c_pos, store_val, mask=i_mask)

# def triton_repeat_add_optimized(extend_seq_lens: torch.Tensor, seq_lens: torch.Tensor, seq_lens_expanded: torch.Tensor) -> torch.Tensor:
#     # 输入校验
#     assert extend_seq_lens.shape == seq_lens.shape, "tensorA和tensorB的形状必须一致"
#     assert extend_seq_lens.dtype in (torch.int32, torch.int64), "tensorA必须为整数类型"
#     assert extend_seq_lens.is_cuda and seq_lens.is_cuda and seq_lens_expanded.is_cuda, "tensorA、tensorB和tensorD必须为CUDA张量"
#     assert extend_seq_lens.dtype == seq_lens.dtype == seq_lens_expanded.dtype, "tensorA、tensorB和tensorD的数据类型必须一致"
    
#     prefix_sum = torch.cat([torch.tensor([0], device=extend_seq_lens.device), torch.cumsum(extend_seq_lens, dim=0)])
#     seq_lens_sum = torch.cat([torch.tensor([0], device=seq_lens.device), torch.cumsum(seq_lens, dim=0)])
#     extend_seq_lens_sum = prefix_sum[-1].item()
    
#     assert seq_lens_expanded.shape == (extend_seq_lens_sum,), f"tensorD的形状必须为({extend_seq_lens_sum},)，当前为{seq_lens_expanded.shape}"
    
#     # 初始化输出张量ke ks
#     ks = torch.empty((extend_seq_lens_sum,), dtype=extend_seq_lens.dtype, device=extend_seq_lens.device)
#     ke = torch.empty((extend_seq_lens_sum,), dtype=extend_seq_lens.dtype, device=extend_seq_lens.device)
    
#     # 配置Triton Kernel启动参数（按输出C的长度分配网格）
#     BLOCK_SIZE = 128
#     grid = lambda meta: (triton.cdiv(extend_seq_lens_sum, meta['BLOCK_SIZE']),)

#     for _ in range(10):
#         _get_ke_ks_triton_kernel[grid](
#         seq_lens_ptr=seq_lens_sum,
#         ks=ks,
#         ke=ke,
#         seq_lens_expanded=seq_lens_expanded,
#         prefix_sum_ptr=prefix_sum,
#         prefix_sum_len=prefix_sum.shape[0],
#         extend_seq_lens_sum=extend_seq_lens_sum,
#         BLOCK_SIZE=BLOCK_SIZE)
    
#     torch.cuda.synchronize()
#     import time
#     start = time.perf_counter()

#     _get_ke_ks_triton_kernel[grid](
#     seq_lens_ptr=seq_lens_sum,
#     ks=ks,
#     ke=ke,
#     seq_lens_expanded=seq_lens_expanded,
#     prefix_sum_ptr=prefix_sum,
#     prefix_sum_len=prefix_sum.shape[0],
#     extend_seq_lens_sum=extend_seq_lens_sum,
#     BLOCK_SIZE=BLOCK_SIZE)

#     torch.cuda.synchronize()

#     end = time.perf_counter()
#     print(f"kernel infer time is {(end-start)*1000} ms")


#     return ks, ke


# def golden_torch_gen(extend_seq_lens:list, seq_lens:list, seq_lens_expanded:torch.tensor):
#     ks_list = []
#     ke_list = []
#     k_offset = 0
#     q_offset = 0
#     N = len(extend_seq_lens)
#     for i in range(N):
#         seq_len = seq_lens[i]
#         extend_seq_len = extend_seq_lens[i]
#         ks = torch.full(
#             (extend_seq_len,), k_offset, dtype=torch.int32, device="cuda"
#         )
#         ke = ks + seq_lens_expanded[q_offset : q_offset + extend_seq_len]
#         ks_list.append(ks)
#         ke_list.append(ke)

#         q_offset += extend_seq_len
#         k_offset += seq_len

#     ks = torch.cat(ks_list, dim=0)
#     ke = torch.cat(ke_list, dim=0)

#     return ks, ke


# extend_seq_lens = torch.tensor([1026,526,3528,4458], dtype=torch.int32, device="cuda")
# seq_lens = torch.tensor([2025,1258,4096,5256], dtype=torch.int32, device="cuda")
# extend_seq_lens_sum = torch.sum(extend_seq_lens).item()
# # seq_lens_expanded = torch.arange(0, extend_seq_lens_sum * 10, 10, dtype=torch.int32, device="cuda")
# seq_lens_expanded = torch.zeros(extend_seq_lens_sum, dtype=torch.int32, device="cuda")

# # 调用优化后的函数
# triton_ks, triton_ke = triton_repeat_add_optimized(extend_seq_lens, seq_lens, seq_lens_expanded)

# golden
# torch_ks, torch_ke = golden_torch_gen(extend_seq_lens, seq_lens, seq_lens_expanded)

# torch.testing.assert_close(triton_ks, torch_ks, rtol=0, atol=0, msg="ks outputs differ!")
# torch.testing.assert_close(triton_ke, torch_ke, rtol=0, atol=0, msg="ke outputs differ!")
# print("✓ ke ks tests passed.")
# # 打印结果
# print("tensorA:", extend_seq_lens.cpu())
# print("tensorB:", seq_lens.cpu())
# print("tensorD (前20个元素):", seq_lens_expanded.cpu())  # 大A[i]场景下，D过长，只打印前20个
# print("tensorKS (前20个元素，C = B重复 + seq_lens_expanded):", triton_ks.cpu())
# print("tensorKE(前20个元素，C = B重复 + seq_lens_expanded):", triton_ke.cpu())


# import torch
# import triton
# import triton.language as tl

# @triton.jit
# def _get_ke_ks_triton_kernel(
#     seq_lens_ptr,
#     ks,
#     ke,
#     seq_lens_expanded,
#     prefix_sum_ptr,
#     prefix_sum_len: tl.constexpr,
#     extend_seq_lens_sum: tl.constexpr,
#     BLOCK_SIZE: tl.constexpr,
#     seq_lens_len: tl.constexpr,  # seq_lens的长度
# ):
#     '''
#     Get ke and ks fuse kernel.

#     :param seq_lens_ptr: every batch seq len, int32
#     :param ks: shape=[sum_extend_seq_len] int32
#     :param ke: shape=[sum_extend_seq_len] int32
#     :param seq_lens_expanded: shape=[sum_extend_seq_len] int32
#     :param prefix_sum_ptr: extend_seq_len offset(include 0)
#     :param prefix_sum_len: 
#     :param extend_seq_lens_sum: extend seq_len sum
#     :param BLOCK_SIZE: all batch s out tensor ptr
#     :param seq_lens_len: seq_lens数组的长度
#     '''
#     pid = tl.program_id(axis=0)
#     if pid >= extend_seq_lens_sum:
#         return
    
#     start_pos = pid * BLOCK_SIZE
#     c_pos = tl.arange(0, BLOCK_SIZE) + start_pos
#     c_mask = c_pos < extend_seq_lens_sum
    
#     low = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
#     high = tl.full((BLOCK_SIZE,), prefix_sum_len - 1, dtype=tl.int32)
    
#     for _ in range(32):
#         mid = (low + high) // 2
#         prefix_mid = tl.load(prefix_sum_ptr + mid, mask=c_mask)
#         cond = c_pos >= prefix_mid
#         low = tl.where(cond, mid, low)
#         high = tl.where(~cond, mid, high)
    
#     i = low
#     i_mask = (i >= 0) & (i < (prefix_sum_len - 1)) & c_mask
    
#     seq_lens_sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    
#     for j in range(seq_lens_len):
#         j_lt_i = (j < i) & i_mask
#         seq_len_j = tl.load(seq_lens_ptr + j)
        
#         seq_lens_sum_val = tl.where(j_lt_i, seq_lens_sum_val + seq_len_j, seq_lens_sum_val)
    
#     D_val = tl.load(seq_lens_expanded + c_pos, mask=i_mask)
    
#     tl.store(ks + c_pos, seq_lens_sum_val, mask=i_mask)
#     store_val = seq_lens_sum_val + D_val
#     tl.store(ke + c_pos, store_val, mask=i_mask)

# def triton_repeat_add_optimized(extend_seq_lens: torch.Tensor, seq_lens: torch.Tensor, seq_lens_expanded: torch.Tensor) -> torch.Tensor:
#     # 输入校验
#     assert extend_seq_lens.shape == seq_lens.shape, "extend_seq_lens和seq_lens的形状必须一致"
#     assert extend_seq_lens.dtype in (torch.int32, torch.int64), "extend_seq_lens必须为整数类型"
#     assert extend_seq_lens.is_cuda and seq_lens.is_cuda and seq_lens_expanded.is_cuda, "所有张量必须为CUDA张量"
#     assert extend_seq_lens.dtype == seq_lens.dtype == seq_lens_expanded.dtype, "所有张量的数据类型必须一致"
    
#     # 计算extend_seq_lens的前缀和（用于二分查找）
#     prefix_sum = torch.cat([torch.tensor([0], device=extend_seq_lens.device), torch.cumsum(extend_seq_lens, dim=0)])

#     extend_seq_lens_sum = torch.sum(extend_seq_lens).item()
#     prefix_seq_num = extend_seq_lens.shape[0]
#     assert seq_lens_expanded.shape == (extend_seq_lens_sum,), f"seq_lens_expanded的形状必须为({extend_seq_lens_sum},)，当前为{seq_lens_expanded.shape}"
    
#     # 初始化输出张量
#     ks = torch.empty((extend_seq_lens_sum,), dtype=extend_seq_lens.dtype, device=extend_seq_lens.device)
#     ke = torch.empty((extend_seq_lens_sum,), dtype=extend_seq_lens.dtype, device=extend_seq_lens.device)
    
#     # 配置Triton启动参数
#     BLOCK_SIZE = 128
#     grid = lambda meta: (triton.cdiv(extend_seq_lens_sum, meta['BLOCK_SIZE']),)

#     _get_ke_ks_triton_kernel[grid](
#         seq_lens_ptr=seq_lens,
#         ks=ks,
#         ke=ke,
#         seq_lens_expanded=seq_lens_expanded,
#         prefix_sum_ptr=prefix_sum,
#         prefix_sum_len=prefix_seq_num,
#         extend_seq_lens_sum=extend_seq_lens_sum,
#         BLOCK_SIZE=BLOCK_SIZE,
#         seq_lens_len=seq_lens.shape[0],
#     )

#     return ks, ke


# def golden_torch_gen(extend_seq_lens:list, seq_lens:list, seq_lens_expanded:torch.tensor):
#     ks_list = []
#     ke_list = []
#     k_offset = 0
#     q_offset = 0
#     N = len(extend_seq_lens)
#     for i in range(N):
#         seq_len = seq_lens[i]
#         extend_seq_len = extend_seq_lens[i]
#         ks = torch.full(
#             (extend_seq_len,), k_offset, dtype=torch.int32, device="cuda"
#         )
#         ke = ks + seq_lens_expanded[q_offset : q_offset + extend_seq_len]
#         ks_list.append(ks)
#         ke_list.append(ke)

#         q_offset += extend_seq_len
#         k_offset += seq_len

#     ks = torch.cat(ks_list, dim=0)
#     ke = torch.cat(ke_list, dim=0)

#     return ks, ke


# # 测试代码
# extend_seq_lens = torch.tensor([3,2], dtype=torch.int32, device="cuda")
# seq_lens = torch.tensor([10,4], dtype=torch.int32, device="cuda")
# extend_seq_lens_sum = torch.sum(extend_seq_lens).item()
# seq_lens_expanded = torch.zeros(extend_seq_lens_sum, dtype=torch.int32, device="cuda")

# # 调用优化后的函数
# triton_ks, triton_ke = triton_repeat_add_optimized(extend_seq_lens, seq_lens, seq_lens_expanded)

# # 生成标准答案
# torch_ks, torch_ke = golden_torch_gen(extend_seq_lens.cpu().tolist(), seq_lens.cpu().tolist(), seq_lens_expanded)

# # 打印结果
# print("extend_seq_lens:", extend_seq_lens.cpu())
# print("seq_lens:", seq_lens.cpu())
# print("seq_lens_expanded:", seq_lens_expanded.cpu())
# print("triton_ks:", triton_ks.cpu())  # 正确输出：[0, 0, 0, 10, 10]
# print("triton_ke:", triton_ke.cpu())  # 正确输出：[0, 0, 0, 10, 10]

# # 验证结果
# torch.testing.assert_close(triton_ks, torch_ks, rtol=0, atol=0, msg="ks outputs differ!")
# torch.testing.assert_close(triton_ke, torch_ke, rtol=0, atol=0, msg="ke outputs differ!")
# print("✓ ke ks tests passed.")

import torch
import triton
import triton.language as tl
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

@triton.jit
def _get_ke_ks_triton_kernel(
    seq_lens_ptr,          # int64类型
    extend_seq_lens_ptr,   # int32类型
    seq_lens_expanded,     # int32类型
    ks_out_ptr,                    # int32类型
    ke_out_ptr,                    # int32类型
    seq_num: tl.constexpr,
    extend_seq_lens_sum: tl.constexpr,
    iter_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    '''
    Get ke and ks fuse kernel.
    仅seq_lens为int64，其他输入为int32的适配版本
    
    :param seq_lens_ptr: every batch seq len, int64
    :param extend_seq_lens_ptr: every batch extend seq len, int32
    :param ks: shape=[sum_extend_seq_len] int64
    :param ke: shape=[sum_extend_seq_len] int64
    :param seq_lens_expanded: shape=[sum_extend_seq_len] int32
    :param extend_seq_lens_sum: extend seq_len sum (sum_extend_seq_len)
    :param BLOCK_SIZE: block size
    :param seq_lens_len: seq_lens/extend_seq_lens数组的长度
    '''
    pid = tl.program_id(axis=0)
    if pid >= extend_seq_lens_sum:
        return
    
    start_pos = pid * BLOCK_SIZE
    out_pos = tl.arange(0, BLOCK_SIZE) + start_pos
    pos_mask = out_pos < extend_seq_lens_sum
    
    low = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    high = tl.full((BLOCK_SIZE,), seq_num, dtype=tl.int32)
    for _ in range(iter_num):
        mid = (low + high) // 2
        
        prefix_mid = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        for j in range(seq_num):
            j_lt_mid = (j < mid) & pos_mask
            extend_seq_len_j = tl.load(extend_seq_lens_ptr + j)
            prefix_mid = tl.where(j_lt_mid, prefix_mid + extend_seq_len_j, prefix_mid)
        
        cond = out_pos >= prefix_mid
        low = tl.where(cond, mid, low)
        high = tl.where(~cond, mid, high)
    
    i = low
    out_mask = (i >= 0) & (i < seq_num) & pos_mask
    
    seq_lens_sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for j in range(seq_num):
        j_lt_i = (j < i) & out_mask

        seq_len_j = tl.load(seq_lens_ptr + j)
        seq_len_j = tl.cast(seq_len_j, tl.int32)
        seq_lens_sum_val = tl.where(j_lt_i, seq_lens_sum_val + seq_len_j, seq_lens_sum_val)
    
    D_val = tl.load(seq_lens_expanded + out_pos, mask=out_mask)
    
    tl.store(ks_out_ptr + out_pos, seq_lens_sum_val, mask=out_mask)
    store_val = seq_lens_sum_val + D_val
    tl.store(ke_out_ptr + out_pos, store_val, mask=out_mask)

def triton_repeat_add_optimized() -> torch.Tensor:
    extend_seq_lens = torch.tensor([3528,3528,3528,4458,3528,3528,3528,4458], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([2025,1258,4096,5256,2025,1258,4096,5256], dtype=torch.int64, device="cuda") 
    extend_seq_lens_sum = torch.sum(extend_seq_lens).item()
    seq_lens_expanded = torch.zeros(extend_seq_lens_sum, dtype=torch.int32, device="cuda")

    extend_seq_lens_sum = torch.sum(extend_seq_lens).item()
    extend_seq_lens_max = torch.max(extend_seq_lens).item()
    seq_lens_num = seq_lens.shape[0]
    
    triton_ks = torch.empty((extend_seq_lens_sum,), dtype=torch.int32, device=extend_seq_lens.device)
    triton_ke = torch.empty((extend_seq_lens_sum,), dtype=torch.int32, device=extend_seq_lens.device)
    
    # acc test =====================
    BLOCK_SIZE = 256
    grid = grid = lambda meta: (triton.cdiv(extend_seq_lens_sum, meta['BLOCK_SIZE']),)
    max_iter = math.ceil(math.log2(extend_seq_lens_max)) + 1 if extend_seq_lens_max > 0 else 1

    torch_ks, torch_ke = golden_torch_gen(
        extend_seq_lens.cpu().tolist(), 
        seq_lens.cpu().tolist(), 
        seq_lens_expanded
    )

    _get_ke_ks_triton_kernel[grid](
    seq_lens_ptr=seq_lens,
    extend_seq_lens_ptr=extend_seq_lens,
    seq_lens_expanded=seq_lens_expanded,
    ks_out_ptr=triton_ks,
    ke_out_ptr=triton_ke,
    seq_num=seq_lens_num,
    extend_seq_lens_sum=extend_seq_lens_sum,
    iter_num=max_iter,
    BLOCK_SIZE=BLOCK_SIZE,)
    
    torch.testing.assert_close(triton_ks, torch_ks, rtol=0, atol=0, msg="ks outputs differ!")
    torch.testing.assert_close(triton_ke, torch_ke, rtol=0, atol=0, msg="ke outputs differ!")
    print(f"_get_ke_ks_triton_kernel test pass")

    # perf test =====================
    import time
    torch.cuda.synchronize()
    for _ in range(10):
        _get_ke_ks_triton_kernel[grid](
            seq_lens_ptr=seq_lens,
            extend_seq_lens_ptr=extend_seq_lens,
            seq_lens_expanded=seq_lens_expanded,
            ks_out_ptr=triton_ks,
            ke_out_ptr=triton_ke,
            seq_num=seq_lens_num,
            extend_seq_lens_sum=extend_seq_lens_sum,
            iter_num=max_iter,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    torch.cuda.synchronize()
    import time
    start = time.perf_counter()

    _get_ke_ks_triton_kernel[grid](
    seq_lens_ptr=seq_lens,
    extend_seq_lens_ptr=extend_seq_lens,
    seq_lens_expanded=seq_lens_expanded,
    ks_out_ptr=triton_ks,
    ke_out_ptr=triton_ke,
    seq_num=seq_lens_num,
    extend_seq_lens_sum=extend_seq_lens_sum,
    iter_num=max_iter,
    BLOCK_SIZE=BLOCK_SIZE,)

    torch.cuda.synchronize()

    end = time.perf_counter()
    print(f"_get_ke_ks_triton_kernel triton infer time is {((end-start)*1000):.4f} ms\n")


def golden_torch_gen(extend_seq_lens:list, seq_lens:list, seq_lens_expanded:torch.tensor):
    ks_list = []
    ke_list = []
    k_offset = 0
    q_offset = 0
    N = len(extend_seq_lens)
    for i in range(N):
        seq_len = seq_lens[i]
        extend_seq_len = extend_seq_lens[i]
        ks = torch.full(
            (extend_seq_len,), k_offset, dtype=torch.int32, device="cuda"
        )
        ke = ks + seq_lens_expanded[q_offset : q_offset + extend_seq_len]
        ks_list.append(ks)
        ke_list.append(ke)

        q_offset += extend_seq_len
        k_offset += seq_len

    ks = torch.cat(ks_list, dim=0)
    ke = torch.cat(ke_list, dim=0)

    return ks, ke

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping tests.")
        exit(0)

    print("Start test cases...\n")

    triton_repeat_add_optimized()

    print("End test cases...\n")