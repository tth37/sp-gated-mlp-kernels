import triton
import triton.language as tl
import torch
from loguru import logger



@triton.jit
def sparsemm_down_dejavu_kernel(
    a_ptr, b_ptr, c_ptr, idx_ptr,
    M, N, K, P, Q,
    stride_am, stride_aq,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_idxp, stride_idxq,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    m_blocks = tl.cdiv(M, BLOCK_SIZE_M)
    n_blocks = tl.cdiv(N, BLOCK_SIZE_N)

    p = 0
    n = pid % n_blocks
    m = pid // n_blocks

    m_offs = tl.arange(0, BLOCK_SIZE_M) + m * BLOCK_SIZE_M
    n_offs = tl.arange(0, BLOCK_SIZE_N) + n * BLOCK_SIZE_N
    q_offs = tl.arange(0, BLOCK_SIZE_Q) # implicit in the loop

    # Compute A @ B
    a_ptrs = a_ptr + m_offs[:, None] * stride_am + q_offs[None, :] * stride_aq
    idx_ptrs = idx_ptr + p * stride_idxp + q_offs * stride_idxq
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(Q, BLOCK_SIZE_Q)):
        idx_mask = q_offs < Q
        k_offs = tl.load(idx_ptrs, mask=idx_mask).to(tl.int32)
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn
        a_mask = (m_offs[:, None] < M) & (q_offs[None, :] < Q)
        b_mask = (k_offs[:, None] < K) & (n_offs[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
        q_offs += BLOCK_SIZE_Q
        a_ptrs += BLOCK_SIZE_Q * stride_aq
        idx_ptrs += BLOCK_SIZE_Q * stride_idxq
    c = acc.to(tl.float16)

    # Store the result
    c_ptrs = c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    c_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def sparsemm_down_dejavu(
    A, B, IDX, tune=False,
    BLOCK_SIZE_M=None, BLOCK_SIZE_Q=None, BLOCK_SIZE_N=None,
    num_stages=None, num_warps=None
):
    assert A.dtype == B.dtype == torch.float16, "A and B must be of dtype torch.float16"
    assert IDX.dtype == torch.int32, "IDX must be of dtype torch.int32"
    assert A.device == B.device == IDX.device, "A, B, and IDX must be on the same device"
    assert A.is_cuda, "A must be a CUDA tensor"

    M, Q = A.shape
    K, N = B.shape
    P, Q = IDX.shape

    assert A.shape[1] == IDX.shape[1], "A and IDX must have the same embedding dimension"
    assert P == 1, "Deja Vu only supports one row-partition"

    C = torch.zeros((M, N), device=A.device, dtype=A.dtype)

    if tune:
        assert BLOCK_SIZE_M is not None, "BLOCK_SIZE_M must be provided for tuning"
        assert BLOCK_SIZE_Q is not None, "BLOCK_SIZE_Q must be provided for tuning"
        assert BLOCK_SIZE_N is not None, "BLOCK_SIZE_N must be provided for tuning"
        assert num_stages is not None, "num_stages must be provided for tuning"
        assert num_warps is not None, "num_warps must be provided for tuning"
    else:
        assert False, "Tuned operators are not yet supported"

    # grid = (((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N),)
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    sparsemm_down_dejavu_kernel[grid](
        A, B, C, IDX,
        M, N, K, P, Q,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        IDX.stride(0), IDX.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_Q, BLOCK_SIZE_N,
        num_stages=num_stages, num_warps=num_warps
    )
    return C

def bench_sparsemm_down_dejavu(
    batch_size, embed_dim, hidden_dim,
    P, Q,
    BLOCK_SIZE_M=None, BLOCK_SIZE_Q=None, BLOCK_SIZE_N=None,
    num_stages=None, num_warps=None
):
    A = torch.randn((batch_size, Q), device="cuda", dtype=torch.float16)
    B = torch.randn((hidden_dim, embed_dim), device="cuda", dtype=torch.float16)
    IDX = torch.randint(0, hidden_dim, (P, Q), device="cuda", dtype=torch.int32)
    IDX = torch.sort(IDX, dim=1)[0]
    
    try:
        ms = triton.testing.do_bench(lambda: sparsemm_down_dejavu(
            A, B, IDX, True,
            BLOCK_SIZE_M, BLOCK_SIZE_Q, BLOCK_SIZE_N,
            num_stages, num_warps
        ))
        return ms
    except Exception as e:
        logger.warning(f"Error in bench_sparsemm_down_dejavu: {e}")
        return float("inf")