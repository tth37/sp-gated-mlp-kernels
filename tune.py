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
args = parser.parse_args()

# print("BLOCK_SIZE_M:", args.block_size_m)
logger.info(f"BLOCK_SIZE_M: {args.block_size_m}")
BATCH_SIZE = args.batch_size
EMBED_DIM = args.embed_dim
HIDDEN_DIM = args.hidden_dim
# P = 1
# Q = args.Q
SPARSITY = args.sparsity

N_TRIALS = 300

logger.info(f"Tuning kernels for {args.arch} with batch size {BATCH_SIZE}, embed dim {EMBED_DIM}, hidden dim {HIDDEN_DIM}, SPARSITY {SPARSITY}")

def write_result(op, result):
    with open(f"sbatch-results/kernel-results.txt", "a") as f:
        f.write(
            f"ARCH={args.arch}\tBATCH_SIZE={BATCH_SIZE}\tEMBED_DIM={EMBED_DIM}\tHIDDEN_DIM={HIDDEN_DIM}\tSPARSITY={SPARSITY}\t"
            f"{op}={result:.6f}\n"
        )

def write_configs(cfgs):
    del cfgs["BLOCK_SIZE_M"]

    try:
        with open("kernel_cache.json", "r") as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}
    
    arch = str(args.arch)
    BLOCK_SIZE_M = str(args.block_size_m)
    sparsity = str(args.sparsity)

    if arch not in cache:
        cache[arch] = {}
    if BLOCK_SIZE_M not in cache[arch]:
        cache[arch][BLOCK_SIZE_M] = {}
    if sparsity not in cache[arch][BLOCK_SIZE_M]:
        cache[arch][BLOCK_SIZE_M][sparsity] = {}

    for cfg in cfgs:
        cache[arch][BLOCK_SIZE_M][sparsity][cfg] = cfgs[cfg]

    with open("kernel_cache.json", "w") as f:
        json.dump(cache, f, indent=4)


if args.op == "gemm_dense":
    A = torch.randn((BATCH_SIZE, EMBED_DIM), device="cuda", dtype=torch.float16)
    B = torch.randn((EMBED_DIM, HIDDEN_DIM), device="cuda", dtype=torch.float16)
    ms = triton.testing.do_bench(lambda: A @ B)
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
    write_configs(cfgs)

# if args.op == "up_dense":
#     up_dense = bench_sparsemm_up_dense(BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q)
#     write_result("UP_DENSE", up_dense)

# if args.op == "up_dejavu":
#     _, up_dejavu = autotune(
#         bench_sparsemm_up_dejavu,
#         (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
#         {
#             "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
#             "BLOCK_SIZE_K": [16, 32, 64, 128, 256],
#             "BLOCK_SIZE_Q": [16, 32, 64, 128, 256],
#             "num_stages": [2, 3, 4, 5],
#             "num_warps": [4, 8, 16, 32],
#         },
#         n_trials=N_TRIALS
#     )
#     write_result("UP_DEJAVU", up_dejavu)

# if args.op == "up_neo":
#     _, up_neo = autotune(
#         bench_sparsemm_up_neo,
#         (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
#         {
#             "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
#             "BLOCK_SIZE_K": [16, 32, 64, 128, 256],
#             "BLOCK_SIZE_Q": [16, 32, 64, 128, 256],
#             "GROUP_SIZE_Q": [1, 2, 4, 8, 16],
#             "num_stages": [2, 3, 4, 5],
#             "num_warps": [4, 8, 16, 32],
#         },
#         n_trials=N_TRIALS
#     )
#     write_result("UP_NEO", up_neo)

# if args.op == "up_cats":
#     _, up_cats = autotune(
#         bench_sparsemm_up_cats,
#         (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
#         {
#             "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
#             "BLOCK_SIZE_K": [16, 32, 64, 128, 256],
#             "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
#             "GROUP_SIZE_N": [1, 2, 4, 8, 16],
#             "num_stages": [2, 3, 4, 5],
#             "num_warps": [4, 8, 16, 32],
#         },
#         n_trials=N_TRIALS
#     )
#     write_result("UP_CATS", up_cats)

# if args.op == "down_dense": 
#     down_dense = bench_sparsemm_down_dense(BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q)
#     write_result("DOWN_DENSE", down_dense)

# if args.op == "down_dejavu":
#     _, down_dejavu = autotune(
#         bench_sparsemm_down_dejavu,
#         (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
#         {
#             "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
#             "BLOCK_SIZE_Q": [16, 32, 64, 128, 256],
#             "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
#             "num_stages": [2, 3, 4, 5],
#             "num_warps": [4, 8, 16, 32],
#         },
#         n_trials=N_TRIALS
#     )
#     write_result("DOWN_DEJAVU", down_dejavu)

# if args.op == "down_neo":
#     _, down_neo = autotune(
#         bench_sparsemm_down_neo,
#         (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
#         {
#             "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
#             "BLOCK_SIZE_Q": [16, 32, 64, 128, 256],
#             "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
#             "GROUP_SIZE_N": [1, 2, 4, 8, 16],
#             "num_stages": [2, 3, 4, 5],
#             "num_warps": [4, 8, 16, 32],
#         },
#         n_trials=N_TRIALS
#     )
#     write_result("DOWN_NEO", down_neo)

# if args.op == "down_splitk":
#     _, down_splitk = autotune(
#         bench_sparsemm_down_splitk,
#         (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
#         {
#             "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
#             "BLOCK_SIZE_Q": [16, 32, 64, 128, 256],
#             "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
#             "GROUP_SIZE_Q": [1, 2, 4, 8, 16],
#             "num_stages": [2, 3, 4, 5],
#             "num_warps": [4, 8, 16, 32],
#         },
#         n_trials=N_TRIALS
#     )
    # write_result("DOWN_SPLITK", down_splitk)

