from memory_pool import MemoryPool
from request import Request

class Node:
    def __init__(self, token: int, kv_index: int):
        self.children = {}
        self.token = token
        self.kv_index = kv_index
        self.lock_ref = 0

class Trie:
    def __init__(self, memory_pool: MemoryPool):
        self.root = Node(-1, -1)
        self.memory_pool = memory_pool
    
    def match_prefix(self, tokens: list[int]):
        node = self.root
        kv_indices = []
        i = 0
        while i < len(tokens) and tokens[i] in node.children:
            node = node.children[tokens[i]]
            kv_indices.append(node.kv_index)
            i += 1
        return kv_indices

    def insert(self, request: Request):
        node = self.root
        tokens = request.input_tokens + request.output_tokens
        kv_indices = request.kv_indices
        i = 0
        level = 0
        while i < len(tokens) and tokens[i] in node.children:
            node = node.children[tokens[i]]
            level += 1
            if level > request.num_cached:
                node.lock_ref += 1
            i += 1

        while i < len(tokens):
            node.children[tokens[i]] = Node(tokens[i], kv_indices[i])
            node = node.children[tokens[i]]
            node.lock_ref += 1
            i += 1

    def remove(self, request: Request):
        node = self.root
        tokens = request.input_tokens + request.output_tokens
        kv_indices = request.kv_indices
        i = 0
        while i < len(tokens):
            child = node.children[tokens[i]]
            if child.lock_ref == 1:
                del node.children[tokens[i]]
                self.memory_pool.free([kv_indices[i]])
                node = child
                i += 1
                break
            child.lock_ref -= 1
            node = child
            i += 1

        while i < len(tokens):
            node = node.children[tokens[i]]
            self.memory_pool.free([kv_indices[i]])
            i += 1
