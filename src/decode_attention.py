import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_stage1(
        Q, 
        K,
        V,
        qk_scale,
        Mid_o,
        Mid_lse,
        seq_lens,
        kv_offsets,
        num_heads,
        num_splits: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        head_dim: tl.constexpr,
        SPLIT_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_split = tl.program_id(2)

    seq_len = tl.load(seq_lens + cur_batch)

    kv_split_start = cur_split * SPLIT_SIZE
    kv_split_end = tl.minimum(kv_split_start + SPLIT_SIZE, seq_len)
    if kv_split_end <= kv_split_start:
        return
    
    q_start = (cur_batch * num_heads + cur_head) * head_dim

    kv_start = tl.load(kv_offsets + cur_batch * num_heads + cur_head) * head_dim # start for this head
    mid_o_start = ((cur_batch * num_heads + cur_head) * num_splits + cur_split) * head_dim
    mid_lse_start = (cur_batch * num_heads + cur_head) * num_splits + cur_split

    q_offsets = q_start + tl.arange(0, BLOCK_D)
    q_mask = tl.arange(0, BLOCK_D) < head_dim
    q = tl.load(Q + q_offsets, mask=q_mask, other=0.0)

    m = float('-inf')
    d = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for n_start in range(kv_split_start, kv_split_end, BLOCK_N):
        n_mask = (n_start + tl.arange(0, BLOCK_N)) < kv_split_end
        # todo: overlapping variable name
        kv_offsets_split = kv_start + (n_start + tl.arange(0, BLOCK_N))[:, None] * head_dim + tl.arange(0, BLOCK_D)[None, :]
        kv_mask = n_mask[:, None] & q_mask[None, :]

        # ()
        k = tl.load(K + kv_offsets_split, mask=kv_mask, other=0)
        v = tl.load(V + kv_offsets_split, mask=kv_mask, other=0)

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
def _fwd_kernel_stage2(
        Mid_o,
        Mid_lse,
        output,
        seq_lens,
        num_heads,
        head_dim,
        num_splits: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SPLIT_SIZE: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    output_start = cur_batch * num_heads * head_dim + cur_head * head_dim
    seq_len = tl.load(seq_lens + cur_batch)

    m = float('-inf')
    d = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    split_start = (cur_batch * num_heads + cur_head) * num_splits # start for this head

    for split_id in range(num_splits):
        if split_id * SPLIT_SIZE >= seq_len:
            break
        split = split_start + split_id
        o_offsets = split * head_dim + tl.arange(0, BLOCK_D)
        o_mask = tl.arange(0, BLOCK_D) < head_dim
        o = tl.load(Mid_o + o_offsets, mask=o_mask, other=0.0)
        lse = tl.load(Mid_lse + split)

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
    seq_lens: torch.Tensor,
    kv_offsets: torch.Tensor,
    num_heads: int,
    head_dim: int
):
    qk_scale = head_dim ** -0.5
    SPLIT_SIZE = 512
    BLOCK_N = 16 # intra-split split
    BLOCK_D = max(16, triton.next_power_of_2(head_dim)) # head_dim block round up to nearest power of 2

    num_batches = Q.shape[0]
    num_splits = triton.cdiv(torch.max(seq_lens).item(), SPLIT_SIZE)
    grid1 = (num_batches, num_heads, num_splits)
    Mid_o = torch.zeros(num_batches, num_heads, num_splits, head_dim, dtype=torch.float32, device=Q.device)
    Mid_lse = torch.full((num_batches, num_heads, num_splits), float('-inf'), dtype=torch.float32, device=Q.device)
    _fwd_kernel_stage1[grid1](
        Q, K, V, qk_scale, Mid_o, Mid_lse, seq_lens, kv_offsets, num_heads, num_splits, BLOCK_N, BLOCK_D, head_dim, SPLIT_SIZE
    )
    grid2 = (num_batches, num_heads,)
    _fwd_kernel_stage2[grid2](Mid_o, Mid_lse, output, seq_lens, num_heads, head_dim, num_splits, BLOCK_D, SPLIT_SIZE)
    return output
