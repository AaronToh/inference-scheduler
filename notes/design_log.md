# Design Log

Aaron's own design decisions and rationale, dated. Write reasoning here before comparing
against SGLang, not after.

## 2026-07-14 — Project kickoff

- Repo scaffolded (src/, tests/, notes/, scratch/).
- Next: design `Req` (src/request.py) — what state a request carries through its lifecycle
  (waiting → prefill → decode → finished), and `ScheduleBatch` (src/batch.py) — what a batch
  needs to track.
