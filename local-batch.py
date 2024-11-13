import os

BATCH_SIZE = 512
EMBED_DIM = 5120
HIDDEN_DIM = 13824
ARCH = ['3090']
OP = ['gemm_bcsr', 'gemm_dense']

SPARSITY_list = [
    0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
]

# SBATCH_TEMPLATE = """#!/bin/bash
# #SBATCH -J tune-{arch}-b{BATCH_SIZE}-e{EMBED_DIM}-h{HIDDEN_DIM}-q{Q}
# #SBATCH -N 1
# #SBATCH -p {arch}
# #SBATCH -e sbatch-results/tune-{arch}-b{BATCH_SIZE}-e{EMBED_DIM}-h{HIDDEN_DIM}-q{Q}-{op}.err
# #SBATCH -o /dev/null
# #SBATCH --no-requeue
# #SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:1

# source /apps/soft/anaconda3/bin/activate
# conda info --envs
# conda activate /home/fit/renju/WORK/miniconda3/envs/tianhaodong-sparsity-db

# cd /home/fit/renju/WORK/tianhaodong/sp-gated-mlp-kernels

# python3 tune.py --arch {arch} --batch-size {BATCH_SIZE} --embed-dim {EMBED_DIM} --hidden-dim {HIDDEN_DIM} --Q {Q} --op {op}
# """

# Clear sbatch-results dir
if os.path.exists("sbatch-results"):
    for f in os.listdir("sbatch-results"):
        os.remove(os.path.join("sbatch-results", f))
else:
    os.makedirs("sbatch-results")

for arch in ARCH:
    for SPARSITY in SPARSITY_list:
        for op in OP:
            # tmp_file = f"/tmp/tune-{arch}-b{BATCH_SIZE}-e{EMBED_DIM}-h{HIDDEN_DIM}-q{Q}-{op}.sh"
            # with open(tmp_file, "w") as f:
            #     f.write(SBATCH_TEMPLATE.format(arch=arch, BATCH_SIZE=BATCH_SIZE, EMBED_DIM=EMBED_DIM, HIDDEN_DIM=HIDDEN_DIM, Q=Q, op=op))
            # os.system(f"sbatch {tmp_file}")
            # os.remove(tmp_file)
            os.system(
                f"python3 tune.py --arch {arch} --batch-size {BATCH_SIZE} --embed-dim {EMBED_DIM} --hidden-dim {HIDDEN_DIM} --sparsity {SPARSITY} --op {op} "
                f"2> sbatch-results/tune-{arch}-b{BATCH_SIZE}-e{EMBED_DIM}-h{HIDDEN_DIM}-sp{SPARSITY}-{op}.err"
            )
