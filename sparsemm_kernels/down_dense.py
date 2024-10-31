import torch
import triton

def bench_sparsemm_down_dense(
    batch_size, embed_dim, hidden_dim,
    P, Q
):
    A = torch.randn((batch_size, hidden_dim), device="cuda", dtype=torch.float16)
    B = torch.randn((hidden_dim, embed_dim), device="cuda", dtype=torch.float16)

    def dense():
        return A @ B
    
    ms = triton.testing.do_bench(lambda: dense())
    return ms