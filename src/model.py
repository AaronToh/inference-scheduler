import torch
from decode_attention import decode_attention

class Model:
    def __init__(self, vocab_size: int, num_heads: int, head_dim: int, device: str = 'cuda'):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embedding = torch.randn(vocab_size, head_dim, device=device)
        self.w_q = torch.randn(num_heads, head_dim, head_dim, device=device)
        self.w_k = torch.randn(num_heads, head_dim, head_dim, device=device)
        self.w_v = torch.randn(num_heads, head_dim, head_dim, device=device)
        self.w_out = torch.randn(num_heads * head_dim, vocab_size, device=device)

    def forward_decode(self, token_ids: int, k_context: torch.Tensor, v_context: torch.Tensor, sequence_lengths: list[int], kv_offsets: list[int]):
        # (batch, head_dim) @ (num_heads, head_dim, head_dim) = (batch, num_heads, head_dim)
        batch_size = len(token_ids)
        q = (self.embedding[token_ids] @ self.w_q).permute(1, 0, 2).contiguous()

        output = torch.zeros(batch_size, self.num_heads, self.head_dim, device=k_context.device)
        decode_attention(q, k_context, v_context, output, sequence_lengths, kv_offsets, self.num_heads, self.head_dim)
        output = output.reshape(batch_size, -1)

        # (batch_size, num_heads * head_dim) @ (num_heads * head_dim, vocab_size) = (batch, vocab_size)
        logits = output @ self.w_out
        next_tokens = torch.argmax(logits, dim=1).tolist() # (batch, )

        next_embedding = self.embedding[next_tokens]
        k_new = (next_embedding @ self.w_k).permute(1, 0, 2).contiguous()
        v_new = (next_embedding @ self.w_v).permute(1, 0, 2).contiguous()

        return next_tokens, k_new, v_new
    
    def forward_prefill(self, token_ids: list[int], k_to_fill: torch.Tensor, v_to_fill: torch.Tensor):
        embeddings = self.embedding[token_ids] # (num_tokens, head_dim)
        k_to_fill[:] = (embeddings @ self.w_k).permute(1, 0, 2).contiguous()
        v_to_fill[:] = (embeddings @ self.w_v).permute(1, 0, 2).contiguous()

        return k_to_fill, v_to_fill
