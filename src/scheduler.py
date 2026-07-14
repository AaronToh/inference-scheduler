from request import Request
from batch import ScheduleBatch
from collections import deque

class Scheduler:
    def __init__(self, requests: list[Request]):
        self.request_queue = deque(requests)
        self.batch = ScheduleBatch([])
        self.max_batch_size = 10

    def _is_full(self):
        return len(self.batch) >= self.max_batch_size

    def step(self):
        while not self._is_full() and self.request_queue:
            request = self.request_queue.popleft()
            self.batch.add(request)
            request.mark_prefilling()

        prefill_requests = self.batch.prefill_requests()
        decode_requests = self.batch.decode_requests()

        for request in prefill_requests:
            request.mark_decoding()

        for request in decode_requests:
            request.output_tokens.append(0)
            if request.finished():
                request.mark_finished()
                self.batch.remove(request)

