import argparse
from loguru import logger

from sparsemm_kernels.down_splitk import bench_sparsemm_down_splitk
from sparsemm_kernels.up_cats import bench_sparsemm_up_cats
from sparsemm_kernels.up_dense import bench_sparsemm_up_dense
from sparsemm_kernels.up_dejavu import bench_sparsemm_up_dejavu
from sparsemm_kernels.up_neo import bench_sparsemm_up_neo
from sparsemm_kernels.down_dense import bench_sparsemm_down_dense
from sparsemm_kernels.down_dejavu import bench_sparsemm_down_dejavu
from sparsemm_kernels.down_neo import bench_sparsemm_down_neo
from sparsemm_kernels.gemm_bcsr import bench_sparsemm_gemm_bcsr

from sparsemm_kernels.autotune import autotune

import torch
import triton
import json


parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str, required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--embed-dim", type=int, required=True)
parser.add_argument("--hidden-dim", type=int, required=True)
parser.add_argument("--sparsity", type=float, required=True)
parser.add_argument("--block-size-m", type=int, required=True)
parser.add_argument("--op", type=str, required=True)
parser.add_argument("--num-trials", type=int, default=300)
args = parser.parse_args()

# print("BLOCK_SIZE_M:", args.block_size_m)
BATCH_SIZE = args.batch_size
EMBED_DIM = args.embed_dim
HIDDEN_DIM = args.hidden_dim
SPARSITY = args.sparsity
N_TRIALS = args.num_trials

logger.info(f"Tuning kernels for {args.arch} with batch size {BATCH_SIZE}, embed dim {EMBED_DIM}, hidden dim {HIDDEN_DIM}, SPARSITY {SPARSITY}")

def write_result(op, result):
    with open(f"sbatch-results/kernel-results-b{BATCH_SIZE}-e{EMBED_DIM}-h{HIDDEN_DIM}-b{args.block_size_m}.txt", "a") as f:
        f.write(
            f"SPARSITY={SPARSITY}\t"
            f"{op}={result:.6f}\n"
        )

if args.op == "gemm_dense":
    A = torch.randn((BATCH_SIZE, EMBED_DIM), device="cuda", dtype=torch.float16)
    B = torch.randn((EMBED_DIM, HIDDEN_DIM), device="cuda", dtype=torch.float16)
    ms = triton.testing.do_bench(lambda: A @ B, warmup=250, rep=1000)
    write_result("GEMM_DENSE", ms)

if args.op == "gemm_bcsr":
    cfgs, gemm_bcsr = autotune(
        bench_sparsemm_gemm_bcsr,
        (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, SPARSITY),
        {
            "BLOCK_SIZE_M": [args.block_size_m],
            "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_K": [16, 32, 64, 128, 256],
            "GROUP_SIZE_N": [1, 2, 4, 8],
            "num_stages": [2, 3, 4, 5],
            "num_warps": [4, 8, 16, 32],
        },
        n_trials=N_TRIALS
    )
    write_result("GEMM_BCSR", gemm_bcsr)