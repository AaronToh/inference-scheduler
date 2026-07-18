from enum import Enum
class Status(Enum):
    WAITING = 1
    PREFILLING = 2
    DECODING = 3
    FINISHED = 4
    ABORTED = 5

class Request:
    def __init__(self, request_id: int, max_output_tokens: int, input_tokens: list[int], status: Status=Status.WAITING):
        assert max_output_tokens > 0
        self.request_id = request_id
        self.max_output_tokens = max_output_tokens
        self.input_tokens = input_tokens
        self.output_tokens = []
        self.status = status
        self.kv_indices = None
        self.num_cached = 0

    def __len__(self):
        return len(self.input_tokens) + len(self.output_tokens)

    def finished(self):
        # Todo: Stop on EOS token instead
        return len(self.output_tokens) >= self.max_output_tokens
    
    def mark_waiting(self):
        assert self.status == Status.DECODING
        self.input_tokens.extend(self.output_tokens)
        self.max_output_tokens -= len(self.output_tokens)
        self.output_tokens = []
        self.kv_indices = []
        self.status = Status.WAITING

    def mark_prefilling(self):
        assert self.status == Status.WAITING
        self.status = Status.PREFILLING

    def mark_decoding(self):
        assert self.status == Status.PREFILLING
        self.status = Status.DECODING

    def mark_finished(self):
        assert self.status == Status.DECODING
        self.status = Status.FINISHED

    def mark_decoding_aborted(self):
        assert self.status == Status.DECODING
        self.status = Status.ABORTED

    def mark_waiting_aborted(self):
        assert self.status == Status.WAITING
        self.status = Status.ABORTED
