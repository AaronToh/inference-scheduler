import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_stage1(
        Q, 
        K_int8,
        V_int8,
        k_scale,
        v_scale,
        qk_scale,
        Mid_o,
        Mid_lse,
        seq_len,
        head_dim: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SPLIT_SIZE: tl.constexpr,
        num_splits: tl.constexpr
):
    cur_head = tl.program_id(0)
    cur_split = tl.program_id(1)

    kv_split_start = cur_split * SPLIT_SIZE
    kv_split_end = tl.minimum(kv_split_start + SPLIT_SIZE, seq_len)
    
    q_start = cur_head * head_dim
    kv_start = cur_head * seq_len * head_dim # start for this head
    kv_scale_start = cur_head * seq_len
    mid_o_start = cur_head * num_splits * head_dim + cur_split * head_dim
    mid_lse_start = cur_head * num_splits + cur_split

    q_offsets = q_start + tl.arange(0, BLOCK_D)
    q_mask = tl.arange(0, BLOCK_D) < head_dim
    q = tl.load(Q + q_offsets, mask=q_mask, other=0.0)

    m = float('-inf')
    d = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for n_start in range(kv_split_start, kv_split_end, BLOCK_N):
        n_mask = (n_start + tl.arange(0, BLOCK_N)) < kv_split_end

        k_offsets = kv_start + (n_start + tl.arange(0, BLOCK_N))[:, None] * head_dim + tl.arange(0, BLOCK_D)[None, :]
        k_mask = n_mask[:, None] & q_mask[None, :]
        s_offsets = kv_scale_start + n_start + tl.arange(0, BLOCK_N)

        k = tl.load(K_int8 + k_offsets, mask=k_mask, other=0)
        k_scale_vals = tl.load(k_scale + s_offsets, mask=n_mask, other=0.0)
        k = k.to(tl.float32) * k_scale_vals[:, None]

        v = tl.load(V_int8 + k_offsets, mask=k_mask, other=0)
        v_scale_vals = tl.load(v_scale + s_offsets, mask=n_mask, other=0.0)
        v = v.to(tl.float32) * v_scale_vals[:, None]

        qk = tl.sum(q[None, :] * k, 1)
        qk = qk * qk_scale
        qk = tl.where(n_mask, qk, float('-inf'))

        n_max = tl.max(qk, axis=0)
        new_m = tl.maximum(m, n_max)
        exp_qk = tl.exp(qk - new_m)
        n_sum = tl.sum(exp_qk, axis=0)

        acc = acc * tl.exp(m - new_m) + tl.sum(exp_qk[:, None] * v, axis=0)
        d = d * tl.exp(m - new_m) + n_sum
        m = new_m

    o = acc / d
    lse = m + tl.log(d)
    mid_o_offsets = mid_o_start + tl.arange(0, BLOCK_D)
    tl.store(Mid_o + mid_o_offsets, o, mask=q_mask)
    tl.store(Mid_lse + mid_lse_start, lse)

@triton.jit
def _fwd_kernel_stage2(Mid_o, Mid_lse, output, head_dim, BLOCK_D: tl.constexpr, num_splits:tl.constexpr):
    cur_head = tl.program_id(0)
    output_start = cur_head * head_dim

    m = float('-inf')
    d = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for split_id in range(num_splits):
        o_offsets = (cur_head * num_splits + split_id) * head_dim + tl.arange(0, BLOCK_D)
        o_mask = tl.arange(0, BLOCK_D) < head_dim 
        o = tl.load(Mid_o + o_offsets, mask=o_mask, other=0.0)
        lse = tl.load(Mid_lse + cur_head * num_splits + split_id)

        new_m = tl.maximum(m, lse)
        old_scale = tl.exp(m - new_m)
        exp_lse = tl.exp(lse - new_m)
        d = d * old_scale + exp_lse
        acc = acc * old_scale + exp_lse * o
        m = new_m
        
    d_offsets = output_start + tl.arange(0, BLOCK_D)
    d_mask = tl.arange(0, BLOCK_D) < head_dim
    tl.store(output + d_offsets, acc / d, mask=d_mask)
    
def decode_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
    seq_len: int,
    head_dim: int,
):
    qk_scale = head_dim ** -0.5
    BLOCK_N = 16
    BLOCK_D = max(16, triton.next_power_of_2(head_dim))
    SPLIT_SIZE = 512
    num_splits = triton.cdiv(seq_len, 512)
    grid1 = (num_heads, num_splits)
    Mid_o = torch.zeros(num_heads, num_splits, head_dim, dtype=torch.float32, device=Q.device)
    Mid_lse = torch.full((num_heads, num_splits), float('-inf'), dtype=torch.float32, device=Q.device)
    _fwd_kernel_stage1[grid1](
        Q, K_int8, V_int8, k_scale, v_scale, qk_scale, Mid_o, Mid_lse, seq_len, head_dim, BLOCK_N, BLOCK_D, SPLIT_SIZE, num_splits
    )
    grid2 = (num_heads,)
    _fwd_kernel_stage2[grid2](Mid_o, Mid_lse, output, head_dim, BLOCK_D, num_splits)
    return output
