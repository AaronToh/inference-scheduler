import torch

from model import Model
from request import Request, Status
from scheduler import DispatchMode, Scheduler


def test_basic():
      requests = [
          Request(request_id=0, max_output_tokens=5, input_tokens=[1, 2, 3]),
          Request(request_id=1, max_output_tokens=5, input_tokens=[1, 2, 4])
      ]
      scheduler = Scheduler(requests, 50, Model(1000, 1, 16))
      scheduler.run()

      assert all(r.status == Status.FINISHED for r in requests)
      assert all(len(r) == 8 for r in requests)

def test_retract():
      requests = [
            Request(request_id=0, max_output_tokens=30, input_tokens=list(range(10))),
            Request(request_id=1, max_output_tokens=30, input_tokens=list(range(10,20))),
      ]

      scheduler = Scheduler(requests, 50, Model(1000, 1, 16))
      scheduler.run()

      assert all(r.status == Status.FINISHED for r in requests)
      assert all(len(r) == 40 for r in requests)


def reference_decode_step(model, current_tokens, k_contexts, v_contexts):
    """Naive, non-flash attention computed independently of the Triton kernel,
    used to numerically check Model.forward_decode / Scheduler.dispatch_to_model."""
    next_tokens = []
    scale = model.head_dim ** -0.5
    for token_id, k_ctx, v_ctx in zip(current_tokens, k_contexts, v_contexts):
        embedding = model.embedding[token_id]
        q = torch.einsum('d,hde->he', embedding, model.w_q)

        k = k_ctx.permute(1, 0, 2)
        v = v_ctx.permute(1, 0, 2)

        scores = torch.einsum('hd,hsd->hs', q, k) * scale
        weights = torch.softmax(scores, dim=-1)
        attn_out = torch.einsum('hs,hsd->hd', weights, v)

        logits = attn_out.reshape(-1) @ model.w_out
        next_tokens.append(torch.argmax(logits).item())
    return next_tokens


def test_forward_decode_matches_reference():
    torch.manual_seed(0)
    model = Model(vocab_size=100, num_heads=2, head_dim=8)

    seq_len_a, seq_len_b = 5, 9
    token_a, token_b = 3, 17

    k_a = torch.randn(seq_len_a, model.num_heads, model.head_dim, device='cuda')
    v_a = torch.randn(seq_len_a, model.num_heads, model.head_dim, device='cuda')
    k_b = torch.randn(seq_len_b, model.num_heads, model.head_dim, device='cuda')
    v_b = torch.randn(seq_len_b, model.num_heads, model.head_dim, device='cuda')

    expected_tokens = reference_decode_step(model, [token_a, token_b], [k_a, k_b], [v_a, v_b])

    # build the batched kernel inputs the same way Scheduler.dispatch_to_model does
    k_context = torch.cat([k_a, k_b], dim=0).permute(1, 0, 2).contiguous()
    v_context = torch.cat([v_a, v_b], dim=0).permute(1, 0, 2).contiguous()
    sequence_lengths = torch.tensor([seq_len_a, seq_len_b], device='cuda')
    total_seq_len = seq_len_a + seq_len_b
    kv_offsets = torch.tensor([0, total_seq_len, seq_len_a, total_seq_len + seq_len_a], device='cuda')

    next_tokens, k_new, v_new = model.forward_decode(
        [token_a, token_b], k_context, v_context, sequence_lengths, kv_offsets
    )

    assert next_tokens == expected_tokens


def test_dispatch_to_model_decode_matches_reference():
    torch.manual_seed(1)
    model = Model(vocab_size=100, num_heads=2, head_dim=8)
    scheduler = Scheduler([], 50, model)

    seq_len_a, seq_len_b = 4, 6
    token_a, token_b = 11, 42

    request_a = Request(request_id=0, max_output_tokens=5, input_tokens=list(range(seq_len_a - 1)))
    request_a.output_tokens = [token_a]
    request_a.status = Status.DECODING

    request_b = Request(request_id=1, max_output_tokens=5, input_tokens=list(range(seq_len_b - 1)))
    request_b.output_tokens = [token_b]
    request_b.status = Status.DECODING

    for request, seq_len in [(request_a, seq_len_a), (request_b, seq_len_b)]:
        # seq_len slots for existing context + 1 reserved for the token this step produces
        indices = scheduler.memory_pool.allocate(seq_len + 1)
        request.kv_indices = indices
        scheduler.k_cache[indices[:-1]] = torch.randn(seq_len, model.num_heads, model.head_dim, device='cuda')
        scheduler.v_cache[indices[:-1]] = torch.randn(seq_len, model.num_heads, model.head_dim, device='cuda')

    k_a = scheduler.k_cache[request_a.kv_indices[:-1]].clone()
    v_a = scheduler.v_cache[request_a.kv_indices[:-1]].clone()
    k_b = scheduler.k_cache[request_b.kv_indices[:-1]].clone()
    v_b = scheduler.v_cache[request_b.kv_indices[:-1]].clone()

    expected_tokens = reference_decode_step(model, [token_a, token_b], [k_a, k_b], [v_a, v_b])

    output_tokens = scheduler.dispatch_to_model([request_a, request_b], DispatchMode.DECODE)

    assert output_tokens == expected_tokens
