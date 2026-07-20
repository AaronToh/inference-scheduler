from request import Request, Status
from model import Model
from scheduler import Scheduler

def test_basic():
      requests = [
          Request(request_id=0, max_output_tokens=5, input_tokens=[1, 2, 3]),
          Request(request_id=1, max_output_tokens=5, input_tokens=[1, 2, 4])
      ]
      scheduler = Scheduler(requests, 50, Model(1000, 16))
      scheduler.run()
      
      assert all(r.status == Status.FINISHED for r in requests)
      assert all(len(r) == 8 for r in requests)

def test_retract():
      requests = [
            Request(request_id=0, max_output_tokens=30, input_tokens=list(range(10))),
            Request(request_id=1, max_output_tokens=30, input_tokens=list(range(10,20))),
      ]

      scheduler = Scheduler(requests, 50, Model(1000, 16))
      scheduler.run()

      assert all(r.status == Status.FINISHED for r in requests)
      assert all(len(r) == 40 for r in requests)
