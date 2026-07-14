from request import Request, Status

class ScheduleBatch:
    def __init__(self, requests: list[Request]):
        self.requests = requests

    def prefillRequests(self):
        res = []
        for request in self.requests:
            if request.status == Status.PREFILLING:
                res.append(request)
        return res
    
    def decodeRequests(self):
        res = []
        for request in self.requests:
            if request.status == Status.DECODING:
                res.append(request)
        return res
