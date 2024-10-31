import argparse
from loguru import logger

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
args = parser.parse_args()

BATCH_SIZE = args.batch_size
EMBED_DIM = args.embed_dim
HIDDEN_DIM = args.hidden_dim
P = 1
Q = args.Q

logger.info(f"Tuning kernels for {args.arch} with batch size {BATCH_SIZE}, embed dim {EMBED_DIM}, hidden dim {HIDDEN_DIM}, Q {Q}")

up_dense = bench_sparsemm_up_dense(BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q)
_, up_dejavu = autotune(
    bench_sparsemm_up_dejavu,
    (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
    {
        "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
        "BLOCK_SIZE_K": [16, 32, 64, 128, 256],
        "BLOCK_SIZE_Q": [16, 32, 64, 128, 256],
        "num_stages": [2, 3, 4, 5],
        "num_warps": [4, 8],
    },
    n_trials=100
)
_, up_neo = autotune(
    bench_sparsemm_up_neo,
    (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
    {
        "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
        "BLOCK_SIZE_K": [16, 32, 64, 128, 256],
        "BLOCK_SIZE_Q": [16, 32, 64, 128, 256],
        "GROUP_SIZE_Q": [1, 2, 4, 8, 16],
        "num_stages": [2, 3, 4, 5],
        "num_warps": [4, 8],
    },
    n_trials=100
)
_, up_cats = autotune(
    bench_sparsemm_up_cats,
    (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
    {
        "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
        "BLOCK_SIZE_K": [16, 32, 64, 128, 256],
        "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
        "GROUP_SIZE_N": [1, 2, 4, 8, 16],
        "num_stages": [2, 3, 4, 5],
        "num_warps": [4, 8],
    },
    n_trials=100
)

down_dense = bench_sparsemm_down_dense(BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q)
_, down_dejavu = autotune(
    bench_sparsemm_down_dejavu,
    (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
    {
        "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
        "BLOCK_SIZE_Q": [16, 32, 64, 128, 256],
        "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
        "num_stages": [2, 3, 4, 5],
        "num_warps": [4, 8],
    },
    n_trials=100
)
_, down_neo = autotune(
    bench_sparsemm_down_neo,
    (BATCH_SIZE, EMBED_DIM, HIDDEN_DIM, P, Q),
    {
        "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
        "BLOCK_SIZE_Q": [16, 32, 64, 128, 256],
        "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
        "GROUP_SIZE_N": [1, 2, 4, 8, 16],
        "num_stages": [2, 3, 4, 5],
        "num_warps": [4, 8],
    },
    n_trials=100
)

logger.info(f"Up kernels: dense {up_dense:.6f}ms, dejavu {up_dejavu:.6f}ms, neo {up_neo:.6f}ms, cats {up_cats:.6f}ms")
logger.info(f"Down kernels: dense {down_dense:.6f}ms, dejavu {down_dejavu:.6f}ms, neo {down_neo:.6f}ms")

with open("sbatch-results/kernel-results.txt", "a") as f:
    f.write(
        f"ARCH={args.arch}\tBATCH_SIZE={BATCH_SIZE}\tEMBED_DIM={EMBED_DIM}\tHIDDEN_DIM={HIDDEN_DIM}\tQ={Q}\t"
        f"UP_DENSE={up_dense:.6f}\tUP_DEJAVU={up_dejavu:.6f}\tUP_NEO={up_neo:.6f}\tUP_CATS={up_cats:.6f}\t"
        f"DOWN_DENSE={down_dense:.6f}\tDOWN_DEJAVU={down_dejavu:.6f}\tDOWN_NEO={down_neo:.6f}\n"
    )
