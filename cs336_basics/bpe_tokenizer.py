from typing import Optional, List, Dict, BinaryIO
from utils import find_chunk_boundaries
from collections import defaultdict, Counter
from multiprocessing import Pool
import regex as re
import time
import os

class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else []
        self.vocab = {}
        self.merges = []
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def train(self, input_path: str):
        pass
        
    def parallel_pretokenize(self, input_path: str) -> Dict[str, int]:
        token_counts = defaultdict(int)
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(
                f, 4, "<|endoftext|>".encode("utf-8")
            )
            results = []
            with Pool(4) as p:
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    with open(input_path, 'rb') as f:
                        f.seek(start)
                        chunk = f.read(end - start)
                        results.append(p.apply_async(self.pretokenize_binary, (chunk,)))
                p.close()
                p.join()
            for r in results:
                token_counts.update(r.get())
        return token_counts    
    
    def pretokenize_binary(self, file: bytes):
        token_counts = defaultdict(int)
        chunk = file.decode('utf-8', errors='ignore')
        chunks = re.split(r'<|endoftext|>', chunk)
        for chunk in chunks:
                tokens = [re_match.group() for re_match in re.finditer(self.PAT, chunk)]
                counts = Counter(tokens)
                token_counts.update(counts)
        return token_counts
                
if __name__ == '__main__':
    BPE = BPETokenizer(30, [r'<|endoftext|>'])
    DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/TinyStoriesV2-GPT4-valid.txt')
    DATA_PATH = os.path.abspath(DATA_PATH)
    start = time.time()
    with open(DATA_PATH, 'rb') as f:
        token_counts = BPE.pretokenize_binary(f.read())
    end = time.time()
    print(token_counts)
    print(f'Time cost is {end - start}')