import torch
import triton

def sparsemm_up_torchsparse(
    A, B, C, IDX, ACT_TYPE="fatrelu"
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
    assert M % P == 0, "The first dimension of A must be divisible by the number of row-partitions"

    D = []
    for p in range(P):
        AA = A[p * (M // P):(p + 1) * (M // P), :]
        BB = B[IDX[p], :]
        CC = C[IDX[p], :]
        DD = torch.maximum(torch.tensor(0.01), AA @ BB.T) * (AA @ CC.T)
        D.append(DD)
    D = torch.cat(D, dim=0)
    return D

def bench_sparsemm_up_torchsparse(
    batch_size, embed_dim, hidden_dim,
    P, Q, ACT_TYPE="fatrelu",
):
    A = torch.randn((batch_size, embed_dim), device="cuda", dtype=torch.float16)
    B = torch.randn((hidden_dim, embed_dim), device="cuda", dtype=torch.float16)
    C = torch.randn((hidden_dim, embed_dim), device="cuda", dtype=torch.float16)
    IDX = torch.randint(0, hidden_dim, (P, Q), device="cuda", dtype=torch.int32)

    ms = triton.testing.do_bench(lambda: sparsemm_up_torchsparse(A, B, C, IDX))
    return ms