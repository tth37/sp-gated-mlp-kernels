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

from sparsemm_kernels.autotune import autotune


parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str, required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--embed-dim", type=int, required=True)
parser.add_argument("--hidden-dim", type=int, required=True)
parser.add_argument("--Q", type=int, required=True)
parser.add_argument("--op", type=str, required=True)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
EMBED_DIM = args.embed_dim
HIDDEN_DIM = args.hidden_dim
P = 1
Q = args.Q

N_TRIALS = 150

logger.info(f"Tuning kernels for {args.arch} with batch size {BATCH_SIZE}, embed dim {EMBED_DIM}, hidden dim {HIDDEN_DIM}, Q {Q}")

def write_result(op, result):
    with open(f"sbatch-results/kernel-results.txt", "a") as f:
        f.write(
            f"ARCH={args.arch}\tBATCH_SIZE={BATCH_SIZE}\tEMBED_DIM={EMBED_DIM}\tHIDDEN_DIM={HIDDEN_DIM}\tQ={Q}\t"
            f"{op}={result:.6f}\n"
        )

if args.op == "up_dense":
    up_dense = bench_sparsemm_up_dense(BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q)
    write_result("UP_DENSE", up_dense)

if args.op == "up_dejavu":
    _, up_dejavu = autotune(
        bench_sparsemm_up_dejavu,
        (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
        {
            "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_K": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_Q": [16, 32, 64, 128, 256],
            "num_stages": [2, 3, 4, 5],
            "num_warps": [4, 8, 16, 32],
        },
        n_trials=N_TRIALS
    )
    write_result("UP_DEJAVU", up_dejavu)

if args.op == "up_neo":
    _, up_neo = autotune(
        bench_sparsemm_up_neo,
        (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
        {
            "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_K": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_Q": [16, 32, 64, 128, 256],
            "GROUP_SIZE_Q": [1, 2, 4, 8, 16],
            "num_stages": [2, 3, 4, 5],
            "num_warps": [4, 8, 16, 32],
        },
        n_trials=N_TRIALS
    )
    write_result("UP_NEO", up_neo)

if args.op == "up_cats":
    _, up_cats = autotune(
        bench_sparsemm_up_cats,
        (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
        {
            "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_K": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
            "GROUP_SIZE_N": [1, 2, 4, 8, 16],
            "num_stages": [2, 3, 4, 5],
            "num_warps": [4, 8, 16, 32],
        },
        n_trials=N_TRIALS
    )
    write_result("UP_CATS", up_cats)

if args.op == "down_dense": 
    down_dense = bench_sparsemm_down_dense(BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q)
    write_result("DOWN_DENSE", down_dense)

if args.op == "down_dejavu":
    _, down_dejavu = autotune(
        bench_sparsemm_down_dejavu,
        (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
        {
            "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_Q": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
            "num_stages": [2, 3, 4, 5],
            "num_warps": [4, 8, 16, 32],
        },
        n_trials=N_TRIALS
    )
    write_result("DOWN_DEJAVU", down_dejavu)

if args.op == "down_neo":
    _, down_neo = autotune(
        bench_sparsemm_down_neo,
        (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
        {
            "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_Q": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
            "GROUP_SIZE_N": [1, 2, 4, 8, 16],
            "num_stages": [2, 3, 4, 5],
            "num_warps": [4, 8, 16, 32],
        },
        n_trials=N_TRIALS
    )
    write_result("DOWN_NEO", down_neo)

if args.op == "down_splitk":
    _, down_splitk = autotune(
        bench_sparsemm_down_splitk,
        (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
        {
            "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_Q": [16, 32, 64, 128, 256],
            "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
            "GROUP_SIZE_Q": [1, 2, 4, 8, 16],
            "num_stages": [2, 3, 4, 5],
            "num_warps": [4, 8, 16, 32],
        },
        n_trials=N_TRIALS
    )
    write_result("DOWN_SPLITK", down_splitk)
