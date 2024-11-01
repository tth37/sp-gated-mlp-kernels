import os

BATCH_SIZE = 512
EMBED_DIM = 5120
HIDDEN_DIM = 13824
ARCH = ['a01', 'h01']
OP = ['up_dense', 'up_dejavu', 'up_neo', 'up_cats', 'down_dense', 'down_dejavu', 'down_neo', 'down_splitk']

Q_list = [
    i for i in range(512, HIDDEN_DIM + 1, 1024)
]

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH -J tune-{arch}-b{BATCH_SIZE}-e{EMBED_DIM}-h{HIDDEN_DIM}-q{Q}
#SBATCH -N 1
#SBATCH -p {arch}
#SBATCH -e sbatch-results/tune-{arch}-b{BATCH_SIZE}-e{EMBED_DIM}-h{HIDDEN_DIM}-q{Q}-{op}.err
#SBATCH -o /dev/null
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source /apps/soft/anaconda3/bin/activate
conda info --envs
conda activate /home/fit/renju/WORK/miniconda3/envs/tianhaodong-sparsity-db

cd /home/fit/renju/WORK/tianhaodong/sp-gated-mlp-kernels

python3 tune.py --arch {arch} --batch-size {BATCH_SIZE} --embed-dim {EMBED_DIM} --hidden-dim {HIDDEN_DIM} --Q {Q} --op {op}
"""

# Clear sbatch-results dir
if os.path.exists("sbatch-results"):
    for f in os.listdir("sbatch-results"):
        os.remove(os.path.join("sbatch-results", f))
else:
    os.makedirs("sbatch-results")

for arch in ARCH:
    for Q in Q_list:
        for op in OP:
            tmp_file = f"/tmp/tune-{arch}-b{BATCH_SIZE}-e{EMBED_DIM}-h{HIDDEN_DIM}-q{Q}-{op}.sh"
            with open(tmp_file, "w") as f:
                f.write(SBATCH_TEMPLATE.format(arch=arch, BATCH_SIZE=BATCH_SIZE, EMBED_DIM=EMBED_DIM, HIDDEN_DIM=HIDDEN_DIM, Q=Q, op=op))
            os.system(f"sbatch {tmp_file}")
            os.remove(tmp_file)