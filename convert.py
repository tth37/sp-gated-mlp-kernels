import torch
import convert_bcsr_ext

a = torch.tensor([
    [1, 2, 0, 4],
    [5, 6, 0, 8],
    [9, 0, 0, 12],
    [13, 0, 0, 16],
], dtype=torch.float32, device='cuda')
mask = torch.tensor([
    [1, 1, 0, 1],
    [1, 1, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
], dtype=torch.int32, device='cuda')

bcsr = convert_bcsr_ext.forward(mask, a, 1, 4)

torch.Tensor.to_sparse_bsr