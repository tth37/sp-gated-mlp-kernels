import triton
import triton.language as tl
import torch

@triton.jit
def get_m_n(pid, M, N, GROUP_SIZE_M):
    num_groups_m = (M + GROUP_SIZE_M - 1) // GROUP_SIZE_M  # Ceiling division
    max_group_index = num_groups_m - 1  # Maximum valid group index
    group_pids_capacity = N * GROUP_SIZE_M
    g = min(pid // group_pids_capacity, max_group_index)
    cumulative_pids_before_group_g = group_pids_capacity * g
    group_start_row = GROUP_SIZE_M * g
    group_size_m = min(GROUP_SIZE_M, M - group_start_row)
    pid_in_group = pid - cumulative_pids_before_group_g
    n = pid_in_group // group_size_m
    delta_m = pid_in_group % group_size_m
    m = group_start_row + delta_m
    return (m, n) # equiv to blockIdx.m, blockIdx.n




@triton.jit
def sparsemm_gemm_bcsr_kernel(
    a_row_ptr, a_col_ind, a_val_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    m_blocks = tl.cdiv(M, BLOCK_SIZE_M)
    n_blocks = tl.cdiv(N, BLOCK_SIZE_N)

    n, m = get_m_n(pid, n_blocks, m_blocks, GROUP_SIZE_N)

    m_offs = tl.arange(0, BLOCK_SIZE_M) + m * BLOCK_SIZE_M
    n_offs = tl.arange(0, BLOCK_SIZE_N) + n * BLOCK_SIZE_N

    col_start = tl.load(a_row_ptr + m).to(tl.int32)
    col_end = tl.load(a_row_ptr + m + 1).to(tl.int32)

    # if m == 1 and n == 0:
    #     tl.device_print("col_start", col_start)
        # tl.device_print("col_end", col_end)

    a_k_offs = tl.arange(0, BLOCK_SIZE_K)
    a_m_offs = tl.arange(0, BLOCK_SIZE_M)
    ind_offs = col_start + tl.arange(0, BLOCK_SIZE_K)
    ind_ptrs = a_col_ind + ind_offs

    a_offs = a_m_offs[:, None] + a_k_offs[None, :] * BLOCK_SIZE_M
    a_ptrs = a_val_ptr + col_start * BLOCK_SIZE_M + a_offs

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(col_end - col_start, BLOCK_SIZE_K)):
        a_mask = (a_offs < (col_end - col_start) * BLOCK_SIZE_M)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # if m == 1 and n == 0:
        #     tl.device_print("a", a)
        
        ind_mask = ind_offs < col_end
        b_k_offs = tl.load(ind_ptrs, mask=ind_mask, other=0).to(tl.int32)
        # if m == 0 and n == 0:
        #     if k == 1:
        #         tl.device_print("b_k_offs", b_k_offs)
        b_mask = (b_k_offs[:, None] < K) & (n_offs[None, :] < N)
        # if m == 0 and n == 0:
        #     tl.device_print("b_mask", b_mask)
        b_ptrs = b_ptr + b_k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn
        # if m == 0 and n == 0:
        #     tl.device_print("b_offs", b_k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # if m == 0 and n == 0:
        #     if k == 1:
        #         tl.device_print("b", b)
        acc = tl.dot(a, b, acc)
        # if m == 0 and n == 0:
        #     if k == 0:
        #         tl.device_print("acc", acc)
        a_offs += BLOCK_SIZE_K * BLOCK_SIZE_M
        a_ptrs += BLOCK_SIZE_K * BLOCK_SIZE_M
        ind_offs += BLOCK_SIZE_K
        ind_ptrs += BLOCK_SIZE_K
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    c_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def dense_to_bcsr(a, BLOCK_SIZE_M):
    bsr = a.to_sparse_bsr((BLOCK_SIZE_M, 1))
    a_row_ptr = bsr.crow_indices()
    a_col_ind = bsr.col_indices()
    a_val = bsr.values().flatten()
    return a_row_ptr, a_col_ind, a_val

def generate_bcsr(a, BLOCK_SIZE_M, sparsity):
    nnz = int(a.numel() * (1.0 - sparsity))
    M, K = a.shape
    num_blocks_M = M // BLOCK_SIZE_M

    # Initialize a_row_ptr and mask
    a_row_ptr = torch.zeros(num_blocks_M + 1, dtype=torch.int32, device=a.device)
    mask = torch.zeros(num_blocks_M, K, dtype=torch.bool, device=a.device)

    # Randomly select nnz elements to be True
    flat_mask = mask.flatten()
    random_indices = torch.randperm(flat_mask.numel(), device=a.device)[:nnz]
    flat_mask[random_indices] = True
    mask = flat_mask.view(num_blocks_M, K)

    # Calculate a_row_ptr based on the number of True values in each row
    a_row_ptr[1:] = mask.sum(dim=1).cumsum(dim=0)

    # Calculate a_col_ind as the column indices for each True element in the mask
    a_col_ind = mask.nonzero(as_tuple=False)[:, 1]  # Only need the column index

    # Initialize a_val as a tensor of ones of size (nnz * BLOCK_SIZE_M,)
    a_val = torch.ones(nnz * BLOCK_SIZE_M, dtype=a.dtype, device=a.device)

    return a_row_ptr, a_col_ind, a_val

# Example usage
# a = torch.rand(8, 8, device='cuda')  # Example dense matrix
# BLOCK_SIZE_M = 2
# sparsity = 0.25
# a_row_ptr, a_col_ind, a_val = dense_to_bcsr(a, BLOCK_SIZE_M, sparsity)

# print("a_row_ptr:", a_row_ptr)
# print("a_col_ind:", a_col_ind)
# print("a_val:", a_val)
    


def sparsemm_gemm_bcsr(a, b, BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16, GROUP_SIZE_N=4, num_warps=4, num_stages=2):
    M, K = a.shape
    K, N = b.shape

    a_row_ptr, a_col_ind, a_val = dense_to_bcsr(a, BLOCK_SIZE_M)

    # BLOCK_SIZE_M = 16
    # BLOCK_SIZE_N = 16
    # BLOCK_SIZE_K = 16
    # GROUP_SIZE_N = 4

    c = torch.empty((M, N), device=a_row_ptr.device, dtype=torch.float16)

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )

    sparsemm_gemm_bcsr_kernel[grid](
        a_row_ptr, a_col_ind, a_val, b, c,
        M, N, K,
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        GROUP_SIZE_N,
        num_warps=num_warps, num_stages=num_stages
    )

    return c

def bench_sparsemm_gemm_bcsr(
    batch_size, embed_dim, hidden_dim,
    sparsity=0.9, BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16, GROUP_SIZE_N=4,
    num_warps=4, num_stages=2
):
    A = torch.ones((batch_size, embed_dim), device="cuda", dtype=torch.float16)
    # Randomly select 90% of the elements to be zero
    mask = torch.rand((batch_size // BLOCK_SIZE_M, embed_dim), device=A.device) < sparsity
    expanded_mask = torch.repeat_interleave(mask, BLOCK_SIZE_M, dim=0)
    A[expanded_mask] = 0
    B = torch.ones((hidden_dim, embed_dim), device="cuda", dtype=torch.float16)
    
    # a_row_ptr, a_col_ind, a_val = dense_to_bcsr(A, BLOCK_SIZE_M)
    # a_row_ptr, a_col_ind, a_val = generate_bcsr(A, BLOCK_SIZE_M, sparsity)

    try:
        ms = triton.testing.do_bench(lambda: sparsemm_gemm_bcsr(A, B, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_N, num_warps, num_stages))
        return ms
    except Exception as e:
        print(e)
        return float('inf')
    

# # a = torch.tensor([
# #     [1, 2, 0, 4],
# #     [5, 6, 0, 8],
# #     [9, 0, 11, 12],
# #     [13, 0, 15, 16],
# # ], dtype=torch.float16)
# # b = torch.tensor([
# #     [1, 2, 0, 4],
# #     [5, 6, 0, 8],
# #     [9, 0, 11, 12],
# #     [13, 0, 15, 16],
# # ], dtype=torch.float16)

# torch.manual_seed(0)
# # a = torch.randn(16, 16, dtype=torch.float16, device='cuda')
# # b = torch.randn(16, 16, dtype=torch.float16, device='cuda')
# a = torch.randn(32, 32, dtype=torch.float16, device='cuda')
# b = torch.randn(32, 32, dtype=torch.float16, device='cuda')

# # a[:, 3] = 0
# # a[0, 0] = 2
# # a[0, 1] = 3
# # a[0, 2] = 4
# # a[1, 1] = 5
# # a[1, 2] = 6
# # a[1, 3] = 7

# # b[0][0] = 2
# b[16][0] = 0
# # b[16][1] = 2

# c = sparsemm_gemm_bcsr(a, b)
# c_ref = a @ b

# print(c)
# print(c_ref)
# print(torch.allclose(c, c_ref, atol=1e-3), torch.max(torch.abs(c - c_ref)))

# M = 64
# K = 512
# N = 512

# BLOCK_SIZE_M = 16
# BLOCK_SIZE_N = 64
# BLOCK_SIZE_K = 128
# GROUP_SIZE_N = 4

# a = torch.ones((M, K), dtype=torch.float16, device='cuda')
# b = torch.ones((K, N), dtype=torch.float16, device='cuda')
# a_row_ptr, a_col_ind, a_val = dense_to_bcsr(a, BLOCK_SIZE_M)
# # print(a_row_ptr)

# c = sparsemm_gemm_bcsr(a_row_ptr, a_col_ind, a_val, b, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_N)
# c_ref = a @ b

# # print(torch.allclose(c, c_ref, atol=1e-3), torch.max(torch.abs(c - c_ref)))

# bcsr_ms = triton.testing.do_bench(lambda: sparsemm_gemm_bcsr(a_row_ptr, a_col_ind, a_val, b, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_N))
# dense_ms = triton.testing.do_bench(lambda: a @ b)

# print(f'BCSR: {bcsr_ms} ms')
# print(f'Dense: {dense_ms} ms')
