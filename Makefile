sync:
	rsync -av \
		--exclude='__pycache__' \
		--exclude='sbatch-results' \
		./ gpucluster:~/WORK/tianhaodong/sp-gated-mlp-kernels

	rsync -av \
		gpucluster:~/WORK/tianhaodong/sp-gated-mlp-kernels/sbatch-results \
		./
