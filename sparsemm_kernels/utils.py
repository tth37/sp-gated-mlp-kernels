import triton
import triton.language as tl
import torch


@triton.jit
def act_fn(x, act_type: tl.constexpr):
    if act_type == "relu":
        return tl.maximum(x, 0.0)
    elif act_type == "fatrelu":
        threshold = 0.01
        return tl.where(x > threshold.to(tl.float16), x, 0.0)
    else:
        return x  # Default case: no activation
    
@triton.jit
def get_m_n(pid, M, N, GROUP_SIZE_M):
    num_groups_m = (M + GROUP_SIZE_M - 1) // GROUP_SIZE_M  # Ceiling division
    max_group_index = num_groups_m - 1  # Maximum valid group index
    group_pids_capacity = N * GROUP_SIZE_M
    g = min(pid // group_pids_capacity, max_group_index)
    cumulative_pids_before_group_g = group_pids_capacity * g
    group_start_row = GROUP_SIZE_M * g
    group_size_m = min(GROUP_SIZE_M, M - group_start_row)
    pid_in_group = pid - cumulative_pids_before_group_g
    n = pid_in_group // group_size_m
    delta_m = pid_in_group % group_size_m
    m = group_start_row + delta_m
    return (m, n) # equiv to blockIdx.m, blockIdx.n

def idx_to_mask(IDX, Q, HIDDEN_DIM):
    assert IDX.shape[1] == Q, "IDX must have the same number of columns as Q"
    IDX_int64 = IDX.to(torch.int64)
    MASK = torch.zeros((IDX.shape[0], HIDDEN_DIM), dtype=torch.int32, device=IDX.device)
    for i in range(IDX.shape[0]):
        MASK[i].scatter_(0, IDX_int64[i], 1)
    return MASK

def mask_to_idx(MASK, HIDDEN_DIM):
    IDX = torch.nonzero(MASK, as_tuple=False)
    return IDX[:, 1]