# inference-scheduler

A GPU inference batching scheduler, built as a learning project to mirror SGLang's scheduler
architecture at the structural level.

Work so far: Implemented the scheduling loop that admits new requests depending on memory available. Performs prefilling or decoding at each step.
