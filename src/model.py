import torch
from decode_attention import decode_attention

class Model:
    def __init__(self, vocab_size: int, head_dim: int, device: str = 'cuda'):
        self.head_dim = head_dim
        self.embedding = torch.randn(vocab_size, head_dim, device=device)
        self.w_q = torch.randn(head_dim, head_dim, device=device)
        self.w_k = torch.randn(head_dim, head_dim, device=device)
        self.w_v = torch.randn(head_dim, head_dim, device=device)
        self.w_out = torch.randn(head_dim, vocab_size, device=device)

    def forward_decode(self, token_id: int, k_context: torch.Tensor, v_context: torch.Tensor):
        q = self.embedding[token_id] @ self.w_q

        output = torch.zeros(self.head_dim, device=k_context.device)
        decode_attention(q, k_context, v_context, output, 1, k_context.shape[0], self.head_dim)

        logits = output @ self.w_out
        next_token = torch.argmax(logits).item()

        next_embedding = self.embedding[next_token]
        k_new = next_embedding @ self.w_k
        v_new = next_embedding @ self.w_v

        return next_token, k_new, v_new
    
    def forward_prefill(self, token_ids: list[int], k_to_fill: torch.Tensor, v_to_fill: torch.Tensor):
        embeddings = self.embedding[token_ids] # (num_tokens, head_dim)
        k_to_fill[:] = (embeddings @ self.w_k)
        v_to_fill[:] = (embeddings @ self.w_v)

        return k_to_fill, v_to_fill
