from collections import deque
from enum import Enum
from schedule_batch import ScheduleBatch
from trie import Trie
from memory_pool import MemoryPool
from request import Request

class DispatchMode(Enum):
    PREFILL = 1
    DECODE = 2

class Scheduler:
    def __init__(self, requests: list[Request]):
        self.request_queue = deque(requests)
        self.memory_pool = MemoryPool(100)
        self.prefix_cache = Trie(self.memory_pool)
        self.batch = ScheduleBatch([], self.memory_pool, self.prefix_cache)
    
    def dispatch_to_model(self, requests: list[Request], mode: DispatchMode):
        return [0] * len(requests)

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
