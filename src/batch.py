from request import Request, Status

class ScheduleBatch:
    def __init__(self, requests: list[Request]):
        self.requests = requests

    def __len__(self):
        return len(self.requests)

    def add(self, request: Request):
        self.requests.append(request)

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
