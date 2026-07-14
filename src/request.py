from enum import Enum
class Status(Enum):
    WAITING = 1
    PREFILLING = 2
    DECODING = 3
    FINISHED = 4

class Request:
    def __init__(self, requestId: int, maxOutputTokens: int, inputTokens: list[int], status: Status=Status.WAITING):
        self.requestId = requestId
        self.maxOutputTokens = maxOutputTokens
        self.inputTokens = inputTokens
        self.outputTokens = []
        self.status = status

    def finished(self):
        # Todo: Stop on EOS token instead
        return len(self.outputTokens) >= self.maxOutputTokens
