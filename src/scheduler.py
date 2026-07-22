from collections import deque
from enum import Enum
from model import Model
from schedule_batch import ScheduleBatch
from trie import Trie
from memory_pool import MemoryPool
from request import Request
import torch

class DispatchMode(Enum):
    PREFILL = 1
    DECODE = 2

HEAD_DIM = 16
VOCAB_SIZE = 1000

class Scheduler:
    def __init__(self, requests: list[Request], total_slots: int, model: Model):
        self.request_queue = deque(requests)
        self.model = model
        self.k_cache = torch.zeros(total_slots, model.num_heads, model.head_dim, device='cuda')
        self.v_cache = torch.zeros(total_slots, model.num_heads, model.head_dim, device='cuda')
        self.memory_pool = MemoryPool(total_slots)
        self.prefix_cache = Trie(self.memory_pool)
        self.batch = ScheduleBatch([], self.memory_pool, self.prefix_cache)

    def dispatch_to_model(self, requests: list[Request], mode: DispatchMode):
        if mode == DispatchMode.DECODE:
            input_tokens = []
            k_parts = []
            v_parts = []
            sequence_lengths = []
            kv_offsets = []
            num_heads = self.model.num_heads
            total_sequence_length = sum((len(r)) for r in requests)
            base_offset = 0
            for request in requests:
                kv_indices = request.kv_indices
                input_tokens.append(request.output_tokens[-1])

                # creates a contiguous copy to match decode kernel
                # (batch_size, seq_len, num_heads, head_dim)
                k_parts.append(self.k_cache[kv_indices[:-1]])
                v_parts.append(self.v_cache[kv_indices[:-1]])
                sequence_length = len(request)
                sequence_lengths.append(sequence_length)
                head_offset = base_offset
                base_offset += sequence_length
                # (batch_size, num_heads)
                for i in range(num_heads):
                    kv_offsets.append(head_offset)
                    head_offset += total_sequence_length

            # (batch_size * seq_len, num_heads, head_dim)
            k_context = torch.cat(k_parts, dim=0).permute(1, 0, 2).contiguous()
            v_context = torch.cat(v_parts, dim=0).permute(1, 0, 2).contiguous()
            sequence_lengths = torch.tensor(sequence_lengths, device='cuda')
            kv_offsets = torch.tensor(kv_offsets, device='cuda')
            output_tokens, k_new, v_new = self.model.forward_decode(input_tokens, k_context, v_context, sequence_lengths, kv_offsets)

            kv_indices_to_fill = [r.kv_indices[-1] for r in requests]
            self.k_cache[kv_indices_to_fill] = k_new
            self.v_cache[kv_indices_to_fill] = v_new

        else:
            output_tokens = []
            for request in requests:
                num_cached = request.num_cached
                kv_indices = request.kv_indices
                kv_indices_to_fill = kv_indices[num_cached:-1]

                tokens_to_fill = request.input_tokens[num_cached:]
                k_to_fill = self.k_cache[kv_indices_to_fill]
                v_to_fill = self.v_cache[kv_indices_to_fill]

                k_new, v_new = self.model.forward_prefill(tokens_to_fill, k_to_fill, v_to_fill)

                self.k_cache[kv_indices_to_fill] = k_new
                self.v_cache[kv_indices_to_fill] = v_new

                current_token = request.input_tokens[-1]

                # creates a contiguous copy to match decode kernel; to be optimised
                sequence_length = len(kv_indices) - 1
                k_context = self.k_cache[kv_indices[:-1]].permute(1, 0, 2).contiguous()
                v_context = self.v_cache[kv_indices[:-1]].permute(1, 0, 2).contiguous()
                sequence_lengths = torch.tensor([sequence_length], device='cuda')
                kv_offsets = torch.tensor([i * sequence_length for i in range(self.model.num_heads)], device='cuda')

                new_tokens, k_new, v_new = self.model.forward_decode([current_token], k_context, v_context, sequence_lengths, kv_offsets)

                self.k_cache[kv_indices[-1]] = k_new[0]
                self.v_cache[kv_indices[-1]] = v_new[0]

                output_tokens.append(new_tokens[0])

        return output_tokens


    def run_batch_prefill(self, requests: list[Request]):
        tokens = self.dispatch_to_model(requests, DispatchMode.PREFILL)
        for request, token in zip(requests, tokens):
            request.output_tokens.append(token)
            self.prefix_cache.insert(request)
            request.num_cached = len(request.kv_indices)

    def _add_to_request_queue(self, retracted_requests: list[Request]):
        for request in retracted_requests:
            self.request_queue.append(request)
    
    def run_batch_decode(self, requests: list[Request]):
        tokens = self.dispatch_to_model(requests, DispatchMode.DECODE)
        for request, token in zip(requests, tokens):
            request.output_tokens.append(token)
            self.prefix_cache.insert(request)
            request.num_cached += 1

    def step(self):
        self.batch.prepare_prefill_requests(self.request_queue)

        prefill_requests = self.batch.prefill_requests()
        if prefill_requests:
            self.run_batch_prefill(prefill_requests)
            self.batch.mark_decoding(prefill_requests)
        else:
            # all requests are in decoding stage
            retracted_requests = self.batch.retract_decode()
            self._add_to_request_queue(retracted_requests)
            decode_requests = self.batch.decode_requests()
            prepared_decode_requests = self.batch.prepare_decode_requests(decode_requests)
            self.run_batch_decode(prepared_decode_requests)
        self.batch.filter_finished()

    def run(self):
        while self.request_queue or self.batch:
            self.step()
