from schedule_batch import ScheduleBatch
from collections import deque
from memory_pool import MemoryPool

from request import Request
class Scheduler:
    def __init__(self, requests: list[Request]):
        self.request_queue = deque(requests)
        self.memory_pool = MemoryPool(100)
        self.batch = ScheduleBatch([], self.memory_pool)
    
    def dispatch_to_model(self, request: Request):
        return 0

    def run_batch_prefill(self, requests: list[Request]):
        for request in requests:
            token = self.dispatch_to_model(request)
            request.output_tokens.append(token)

    def _add_to_request_queue(self, retracted_requests: list[Request]):
        for request in retracted_requests:
            self.request_queue.append(request)
    
    def run_batch_decode(self, requests: list[Request]):
        for request in requests:
            token = self.dispatch_to_model(request)
            request.output_tokens.append(token)

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
