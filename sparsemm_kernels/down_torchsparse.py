import torch
import triton

def sparsemm_down_torchsparse(
    A, B, IDX
):
    assert A.dtype == B.dtype == torch.float16, "A and B must be of dtype torch.float16"
    assert IDX.dtype == torch.int32, "IDX must be of dtype torch.int32"
    assert A.device == B.device == IDX.device, "A, B, and IDX must be on the same device"
    assert A.is_cuda, "A must be a CUDA tensor"

    M, Q = A.shape
    K, N = B.shape
    P, Q = IDX.shape

    assert A.shape[1] == IDX.shape[1], "A and IDX must have the same embedding dimension"
    assert M % P == 0, "M must be divisible by P"

    C = []
    for p in range(P):
        AA = A[p * (M // P):(p + 1) * (M // P), :]
        BB = B[IDX[p], :]
        CC = AA @ BB
        C.append(CC)
    C = torch.cat(C, dim=0)
    return C

def bench_sparsemm_down_torchsparse(
    batch_size, embed_dim, hidden_dim,
    P, Q,
):
    A = torch.randn((batch_size, Q), device="cuda", dtype=torch.float16)
    B = torch.randn((hidden_dim, embed_dim), device="cuda", dtype=torch.float16)
    IDX = torch.randint(0, hidden_dim, (P, Q), device="cuda", dtype=torch.int32)
    IDX = torch.sort(IDX, dim=1)[0]

    ms = triton.testing.do_bench(lambda: sparsemm_down_torchsparse(A, B, IDX))
    return ms