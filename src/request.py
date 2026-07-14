from enum import Enum
class Status(Enum):
    WAITING = 1
    PREFILLING = 2
    DECODING = 3
    FINISHED = 4

class Request:
    def __init__(self, request_id: int, max_output_tokens: int, input_tokens: list[int], status: Status=Status.WAITING):
        self.request_id = request_id
        self.max_output_tokens = max_output_tokens
        self.input_tokens = input_tokens
        self.output_tokens = []
        self.status = status
        self.kv_cache = None

    def finished(self):
        # Todo: Stop on EOS token instead
        return len(self.output_tokens) >= self.max_output_tokens

    def mark_prefilling(self):
        assert self.status == Status.WAITING
        self.status = Status.PREFILLING

    def mark_decoding(self):
        assert self.status == Status.PREFILLING
        self.status = Status.DECODING

    def mark_finished(self):
        assert self.status == Status.DECODING
        self.status = Status.FINISHED
