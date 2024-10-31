import triton
import triton.language as tl
import torch
from loguru import logger

from .utils import act_fn, get_m_n

@triton.jit
def sparsemm_up_neo_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, idx_ptr,
    M, N, K, P, Q,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cn, stride_ck,
    stride_dm, stride_dq,
    stride_idxp, stride_idxq,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_Q: tl.constexpr,
    ACT_TYPE: tl.constexpr = "fatrelu"
):
    pid = tl.program_id(axis=0)

    m_blocks = tl.cdiv(M, BLOCK_SIZE_M)
    q_blocks = tl.cdiv(Q, BLOCK_SIZE_Q)

    q, m = get_m_n(pid, q_blocks, m_blocks, GROUP_SIZE_Q)

    p = m * P // m_blocks

    m_offs = tl.arange(0, BLOCK_SIZE_M) + m * BLOCK_SIZE_M
    q_offs = tl.arange(0, BLOCK_SIZE_Q) + q * BLOCK_SIZE_Q
    k_offs = tl.arange(0, BLOCK_SIZE_K) # implicit in the loop

    idx_ptrs = idx_ptr + p * stride_idxp + q_offs * stride_idxq
    idx_mask = q_offs < Q
    n_offs = tl.load(idx_ptrs, mask=idx_mask).to(tl.int32)

    a_ptrs = a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
    b_ptrs = b_ptr + n_offs[:, None] * stride_bn + k_offs[None, :] * stride_bk
    c_ptrs = c_ptr + n_offs[:, None] * stride_cn + k_offs[None, :] * stride_ck
    acc_ab = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_Q), dtype=tl.float32)
    acc_ac = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_Q), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        bc_mask = (n_offs[:, None] < N) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=bc_mask, other=0.0)
        c = tl.load(c_ptrs, mask=bc_mask, other=0.0)
        acc_ab = tl.dot(a, b.T, acc_ab)
        acc_ac = tl.dot(a, c.T, acc_ac)
        k_offs += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        c_ptrs += BLOCK_SIZE_K * stride_ck
    
    acc_ab = acc_ab.to(tl.float16)
    acc_ab = act_fn(acc_ab, ACT_TYPE)
    acc_ac = acc_ac.to(tl.float16)
    
    # Compute ActFn(A @ B) * A @ C
    acc = acc_ab * acc_ac
    d = acc.to(tl.float16)

    # Store the result
    d_ptrs = d_ptr + m_offs[:, None] * stride_dm + q_offs[None, :] * stride_dq
    d_mask = (m_offs[:, None] < M) & (q_offs[None, :] < Q)
    tl.store(d_ptrs, d, mask=d_mask)

def sparsemm_up_neo(
    A, B, C, IDX, ACT_TYPE="fatrelu", tune=False,
    BLOCK_SIZE_M=None, BLOCK_SIZE_Q=None, BLOCK_SIZE_K=None, GROUP_SIZE_Q=None,
    num_stages=None, num_warps=None
):
    assert A.dtype == B.dtype == C.dtype == torch.float16, "A, B, and C must be of dtype torch.float16"
    assert IDX.dtype == torch.int32, "IDX must be of dtype torch.int32"
    assert A.device == B.device == C.device == IDX.device, "A, B, C, and IDX must be on the same device"
    assert A.is_cuda, "A must be a CUDA tensor"

    M, K = A.shape
    N, K = B.shape
    N, K = C.shape
    P, Q = IDX.shape

    assert A.shape[1] == B.shape[1] == C.shape[1], "The second dimension of A, B, and C must be the same"
    assert B.shape[0] == C.shape[0], "The first dimension of B and C must be the same"
    assert M % P == 0, "The first dimension of A must be divisible by P"
    assert (M // P) % BLOCK_SIZE_M == 0, "The first dimension of A must be divisible by BLOCK_SIZE_M"

    D = torch.zeros((M, Q), device=A.device, dtype=A.dtype)

    if tune:
        assert BLOCK_SIZE_M is not None, "BLOCK_SIZE_M must be provided for tuning"
        assert BLOCK_SIZE_Q is not None, "BLOCK_SIZE_Q must be provided for tuning"
        assert BLOCK_SIZE_K is not None, "BLOCK_SIZE_K must be provided for tuning"
        assert GROUP_SIZE_Q is not None, "GROUP_SIZE_Q must be provided for tuning"
        assert num_stages is not None, "num_stages must be provided for tuning"
        assert num_warps is not None, "num_warps must be provided for tuning"
    else:
        assert False, "Tuned operators are not yet supported"

    # grid = (((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * ((Q + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q),)
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(Q, BLOCK_SIZE_Q), )
    sparsemm_up_neo_kernel[grid](
        A, B, C, D, IDX,
        M, N, K, P, Q,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        D.stride(0), D.stride(1),
        IDX.stride(0), IDX.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_Q, BLOCK_SIZE_K, GROUP_SIZE_Q, ACT_TYPE,
        num_stages=num_stages, num_warps=num_warps
    )
    return D

def bench_sparsemm_up_neo(
    batch_size, embed_dim, hidden_dim,
    P, Q, ACT_TYPE="fatrelu",
    BLOCK_SIZE_M=None, BLOCK_SIZE_Q=None, BLOCK_SIZE_K=None, GROUP_SIZE_Q=None,
    num_stages=None, num_warps=None
):
    A = torch.randn((batch_size, embed_dim), device="cuda", dtype=torch.float16)
    B = torch.randn((hidden_dim, embed_dim), device="cuda", dtype=torch.float16)
    C = torch.randn((hidden_dim, embed_dim), device="cuda", dtype=torch.float16)
    IDX = torch.randint(0, hidden_dim, (P, Q), device="cuda", dtype=torch.int32)
    IDX = torch.sort(IDX, dim=1)[0]

    try:
        ms = triton.testing.do_bench(lambda: sparsemm_up_neo(
            A, B, C, IDX, ACT_TYPE, True,
            BLOCK_SIZE_M, BLOCK_SIZE_Q, BLOCK_SIZE_K, GROUP_SIZE_Q,
            num_stages, num_warps
        ))
        return ms
    except Exception as e:
        logger.warning(f"Error in bench_sparsemm_up_neo: {e}")
        return float("inf")