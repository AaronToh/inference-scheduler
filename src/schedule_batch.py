from collections import deque
from request import Request, Status
from memory_pool import MemoryPool

class ScheduleBatch:
    def __init__(self, requests: list[Request], memory_pool: MemoryPool):
        self.requests = requests
        self.memory_pool = memory_pool

    def __len__(self):
        return len(self.requests)

    def add(self, request: Request):
        self.requests.append(request)

    def prepare_prefill_requests(self, request_queue: deque[Request]):
        for request in list(request_queue):
            if len(request) + 1 > self.memory_pool.get_capacity():
                request_queue.remove(request)
                request.mark_waiting_aborted()

        while request_queue and self.memory_pool.has_space(len(request_queue[0]) + 1):
            request = request_queue.popleft()
            request.kv_cache = self.memory_pool.allocate(len(request) + 1)
            self.add(request)
            request.mark_prefilling()
        
        
    def retract_decode(self):
        self.requests.sort(
            key=lambda r: (
                -len(r.output_tokens), 
                len(r.input_tokens)
            )
        )

        retracted_requests = []
        while len(self.requests) > 1 and not self.memory_pool.has_space(len(self.requests)):
            request = self.requests.pop()
            self.memory_pool.free(request.kv_cache)
            request.mark_waiting()
            retracted_requests.append(request)

        if len(self.requests) <= 1 and not self.memory_pool.has_space(len(self.requests)):
            request = self.requests.pop()
            self.memory_pool.free(request.kv_cache)
            request.mark_decoding_aborted()

        return retracted_requests

    def mark_decoding(self, requests: list[Request]):
        for request in requests:
            request.mark_decoding()

    def prepare_decode_requests(self, requests: list[Request]) -> list[Request]:
        prepared = []
        for request in requests:
            # guaranteed to have memory as retract_decode called earlier
            slots = self.memory_pool.allocate(1)
            request.kv_cache.append(slots[0])
            prepared.append(request)
        return prepared

    def filter_finished(self):
        for request in list(self.requests):
            if request.finished():
                request.mark_finished()
                self.memory_pool.free(request.kv_cache)
                self.remove(request)

    def prefill_requests(self):
        res = []
        for request in self.requests:
            if request.status == Status.PREFILLING:
                res.append(request)
        return res
    
    def decode_requests(self):
        res = []
        for request in self.requests:
            if request.status == Status.DECODING:
                res.append(request)
        return res

    def remove(self, request: Request):
        self.requests.remove(request)
