import torch
import triton

def sparsemm_up_dense(
    A, B, C, ACT_TYPE="fatrelu"
):
    return torch.maximum(torch.tensor(0.01), A @ B.T) * (A @ C.T)

def bench_sparsemm_up_dense(
    batch_size, embed_dim, hidden_dim,
    P, Q, ACT_TYPE="fatrelu",
):
    A = torch.randn((batch_size, embed_dim), device="cuda", dtype=torch.float16)
    B = torch.randn((hidden_dim, embed_dim), device="cuda", dtype=torch.float16)
    C = torch.randn((hidden_dim, embed_dim), device="cuda", dtype=torch.float16)

    def dense():
        return torch.maximum(torch.tensor(0.01), A @ B.T) * (A @ C.T)

    ms = triton.testing.do_bench(lambda: dense())
    return ms