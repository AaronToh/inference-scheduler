class MemoryPool:
    def __init__(self, total_slots: int):
        self.total_slots = total_slots
        self.free_slots = total_slots
        self.free_indices = set([i for i in range(total_slots)])

    def has_space(self, num_slots: int) -> bool:
        return self.free_slots >= num_slots

    def allocate(self, num_slots: int) -> list[int]:
        if not self.has_space(num_slots):
            raise ValueError("Not enough space")
        res = []
        for _ in range(num_slots):
            res.append(self.free_indices.pop())
            self.free_slots -= 1
        return res

    def free(self, slots: list[int]):
        for slot in slots:
            self.free_indices.add(slot)
            self.free_slots += 1
