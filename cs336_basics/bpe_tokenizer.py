from typing import Optional, List, Dict, BinaryIO
from utils import find_chunk_boundaries
from collections import defaultdict, Counter
from multiprocessing import Pool
import regex as re
import time
import os

'''
Auxiliary functions
'''

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
        def process_chunk(input_path: str, sta: int, end: int):
            with open(input_path, 'rb') as f:
                f.seek(sta)
                chunk = f.read(end - sta)
            return self.pretokenize_binary(chunk)
        
        token_counts = Counter()
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(
                f, 32, "<|endoftext|>".encode("utf-8")
            )
        subprocess_args = [(input_path, sta, end) for sta, end in zip(boundaries[:-1], boundaries[1:])]
        with Pool(8) as p:
            results = p.map(process_chunk, subprocess_args)
        for r in results:
            token_counts.update(r)
        return token_counts    
    
    @staticmethod
    def pretokenize_binary(file: bytes, pattern: str, special_tokens: Optional[List[str]] = [r'<|endoftext|>']):
        token_counts = Counter()
        chunk = file.decode('utf-8', errors='ignore')
        chunks = re.split(re.escape('|'.join(special_tokens)), chunk)
        for chunk in chunks:
                tokens = [re_match.group() for re_match in re.finditer(pattern, chunk)]
                counts = Counter(tokens)
                token_counts.update(counts)
        return token_counts
    
    @staticmethod
    def parallel_pretokenize_auxiliary(input_path: str, pattern, special_tokens: Optional[list[str]] = [r'<|endoftext|>']):
        pass
                
if __name__ == '__main__':
    BPE = BPETokenizer(30, [r'<|endoftext|>'])
    DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/TinyStoriesV2-GPT4-valid.txt')
    DATA_PATH = os.path.abspath(DATA_PATH)
    start = time.time()
    with open(DATA_PATH, 'rb') as f:
        token_counts = BPETokenizer.pretokenize_binary(f.read(), BPE.PAT)
    end = time.time()
    for idx, (token, count) in enumerate(token_counts.most_common()):
        if idx < 100:
            print(token, count)
    print(f'Time cost is {end - start}')