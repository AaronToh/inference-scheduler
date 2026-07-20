from request import Request, Status
from scheduler import Scheduler

def test_basic():
      requests = [
          Request(request_id=0, max_output_tokens=5, input_tokens=[1, 2, 3]),
          Request(request_id=1, max_output_tokens=5, input_tokens=[1, 2, 4]),
      ]
      scheduler = Scheduler(requests)
      scheduler.run()
      
      assert all(r.status == Status.FINISHED for r in requests)
      assert all(len(r.output_tokens) == 5 for r in requests)
