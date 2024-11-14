import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=512, required=False)
parser.add_argument("--embed-dim", type=int, default=5120, required=False)
parser.add_argument("--hidden-dim", type=int, default=13824, required=False)
parser.add_argument("--block-size-m", type=int, default=64, required=False)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
EMBED_DIM = args.embed_dim
HIDDEN_DIM = args.hidden_dim
BLOCK_SIZE_M = args.block_size_m

def get_arch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = torch.cuda.get_device_properties(device)
    return props.name.lower().replace(" ", "-")

ARCH = [ get_arch() ]
OP = ['gemm_bcsr', 'gemm_dense']

SPARSITY_list = [
    0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
]


# Clear sbatch-results dir
if os.path.exists("sbatch-results"):
    for f in os.listdir("sbatch-results"):
        os.remove(os.path.join("sbatch-results", f))
else:
    os.makedirs("sbatch-results")

for arch in ARCH:
    for SPARSITY in SPARSITY_list:
        for op in OP:
            os.system(
                f"python3 tune.py --arch {arch} --batch-size {BATCH_SIZE} --embed-dim {EMBED_DIM} --hidden-dim {HIDDEN_DIM} --sparsity {SPARSITY} --op {op} --block-size-m {BLOCK_SIZE_M} "
                f"2> sbatch-results/tune-{arch}-b{BATCH_SIZE}-e{EMBED_DIM}-h{HIDDEN_DIM}-sp{SPARSITY}-b{BLOCK_SIZE_M}-{op}.err"
            )
