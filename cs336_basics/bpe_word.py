import logging
from collections import Counter

logger = logging.getLogger(__name__)

class Word:
    def __init__(self, bytes_repr: bytes):
        self.bytes_repr = bytes_repr
        self.bytes_list: list[bytes] = []
        self._init_bytes_list()
    
    
    
    def _init_bytes_list(self):
        self.bytes_list = [bytes([byte]) for byte in self.bytes_repr]
    

    def merge(self, pair_merge: tuple[bytes, bytes]) -> Counter[tuple[bytes, bytes]]:
        new_bytes_list: list[bytes] = []
        pair_change_counter = Counter()
        idx: int = 0
        while idx < len(self.bytes_list):
            if idx == len(self.bytes_list) - 1:
                new_bytes_list.append(self.bytes_list[-1])
                break
            if (self.bytes_list[idx], self.bytes_list[idx + 1]) == pair_merge:
                if idx > 0:
                    pair_change_counter[(new_bytes_list[-1], self.bytes_list[idx] + self.bytes_list[idx + 1])] += 1
                    pair_change_counter[(new_bytes_list[-1], self.bytes_list[idx])] -= 1
                if idx < len(self.bytes_list) - 2:
                    pair_change_counter[(self.bytes_list[idx] + self.bytes_list[idx + 1], self.bytes_list[idx + 2])] += 1
                    pair_change_counter[(self.bytes_list[idx + 1], self.bytes_list[idx + 2])] -= 1
                new_bytes_list.append(self.bytes_list[idx] + self.bytes_list[idx + 1])
                idx += 1
            else:
                new_bytes_list.append(self.bytes_list[idx])
            idx += 1
        self.bytes_list = new_bytes_list
        return pair_change_counter
                
    @staticmethod
    def count_pair(bytes_repr: bytes):
        if len(bytes_repr) < 1:
            return Counter()
        pair_counter = Counter((bytes([left]), bytes([right])) for left, right in zip(bytes_repr[:-1], bytes_repr[1:]))
        return pair_counter
    
    @staticmethod
    def get_bytes_list(bytes_repr: bytes):
        return [bytes([byte]) for byte in bytes_repr]
    
    