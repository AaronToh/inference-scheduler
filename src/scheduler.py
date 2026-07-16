from batch import ScheduleBatch
from collections import deque
from memory_pool import MemoryPool

from request import Request
class Scheduler:
    def __init__(self, requests: list[Request]):
        self.request_queue = deque(requests)
        self.batch = ScheduleBatch([])
        self.memory_pool = MemoryPool(100)

    def run_batch_prefill(self, requests: list[Request]):
        for request in requests:
            request.output_tokens.append(0)
            request.mark_decoding()
    
    def run_batch_decode(self, requests: list[Request]):
        for request in requests:
            if not self.memory_pool.has_space(1):
                continue
            slots = self.memory_pool.allocate(1)
            request.output_tokens.append(0)
            request.kv_cache.append(slots[0])

    def process_batch_result(self, requests: list[Request]):
        for request in requests:
            if request.finished():
                request.mark_finished()
                self.memory_pool.free(request.kv_cache)
                self.batch.remove(request)

    def step(self):
        while self.request_queue and self.memory_pool.has_space(len(self.request_queue[0]) + 1):
            request = self.request_queue.popleft()
            request.kv_cache = self.memory_pool.allocate(len(request) + 1)
            self.batch.add(request)
            request.mark_prefilling()

        prefill_requests = self.batch.prefill_requests()
        if prefill_requests:
            self.run_batch_prefill()
            self.process_batch_result()
        else:
            decode_requests = self.batch.decode_requests()
            self.run_batch_decode()
            self.process_batch_result()

    def run(self):
        while self.request_queue or self.batch:
            self.step()
