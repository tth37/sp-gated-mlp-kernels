sync:
	rsync -av \
		--exclude='__pycache__' \
		--exclude='sbatch-results' \
		--exclude='results' \
		./ gpucluster:~/WORK/tianhaodong/sp-gated-mlp-kernels

	rsync -av \
		gpucluster:~/WORK/tianhaodong/sp-gated-mlp-kernels/sbatch-results \
		./

	# if exists sbatch-results/kernel-results.txt:
	if [ -f sbatch-results/kernel-results.txt ]; then \
		sort sbatch-results/kernel-results.txt -o sbatch-results/kernel-results.txt; \
	fi
