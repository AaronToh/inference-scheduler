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

TOTAL_SLOTS = 100
HEAD_DIM = 16
VOCAB_SIZE = 1000

class Scheduler:
    def __init__(self, requests: list[Request]):
        self.request_queue = deque(requests)
        self.k_cache = torch.zeros(TOTAL_SLOTS, HEAD_DIM, device='cuda')
        self.v_cache = torch.zeros(TOTAL_SLOTS, HEAD_DIM, device='cuda')
        self.model = Model(VOCAB_SIZE, HEAD_DIM)
        self.memory_pool = MemoryPool(100)
        self.prefix_cache = Trie(self.memory_pool)
        self.batch = ScheduleBatch([], self.memory_pool, self.prefix_cache)

    def dispatch_to_model(self, requests: list[Request], mode: DispatchMode):
        tokens = []
        if mode == DispatchMode.DECODE:
            for request in requests:
                kv_indices = request.kv_indices
                current_token = request.output_tokens[-1]

                # creates a contiguous copy to match decode kernel
                k_context = self.k_cache[kv_indices[:-1]]
                v_context = self.v_cache[kv_indices[:-1]]

                new_token, k_new, v_new = self.model.forward_decode(current_token, k_context, v_context)

                self.k_cache[kv_indices[-1]] = k_new
                self.v_cache[kv_indices[-1]] = v_new

                tokens.append(new_token)
        else:
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
                k_context = self.k_cache[kv_indices[:-1]]
                v_context = self.v_cache[kv_indices[:-1]]

                new_token, k_new, v_new = self.model.forward_decode(current_token, k_context, v_context)

                self.k_cache[kv_indices[-1]] = k_new
                self.v_cache[kv_indices[-1]] = v_new

                tokens.append(new_token)

        return tokens


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
