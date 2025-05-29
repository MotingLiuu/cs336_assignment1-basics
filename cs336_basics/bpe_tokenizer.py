from typing import Optional, List, Dict, BinaryIO, Tuple
from utils import find_chunk_boundaries
from collections import defaultdict, Counter
from multiprocessing import Pool
import heapq
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
        token_counts = BPETokenizer.pretokenize_parallel(input_path, self.PAT, self.special_tokens)
        # reform the token_counts{bytes: int} to {bytes: (List, int)}
        token_counts = BPETokenizer._reform_tokens_counts(token_counts)
        # get the pair freqeuncy: Counter
        pair_counts = BPETokenizer._pair_frequency(token_counts)
        # for i < self.vocab
        #   find the most freqeunt
        #       merge token_counts
        #       change the pair frequency
        for i in range(self.vocab_size):
            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            # TODO: merge in token_counts
            # TODO: change the pair frequency
            self.merges.append(most_frequent_pair)
            pair_changed_counter = BPETokenizer._merge_pair_token_counts(token_counts, most_frequent_pair)
            pair_counts.update(pair_changed_counter)
            pair_counts.pop(most_frequent_pair)
            
            
    @staticmethod
    def _merge_pair_token_counts(token_counts: Dict[str, Tuple[List[Tuple[int]], int]], pair: Tuple[int]) -> Counter[Tuple[int]]:
        pair_frequency_change_counter = Counter()
        for _, (token_bytes, count) in token_counts.items():
            if len(token_bytes) > 1:
                idx = len(token_bytes) - 2
                while idx > -1:
                    if token_bytes[idx] + token_bytes[idx + 1] == pair:
                        if idx > 0:
                            pair_frequency_change_counter[token_bytes[idx - 1] + token_bytes[idx]] -= count
                            pair_frequency_change_counter[token_bytes[idx - 1] + token_bytes[idx] + token_bytes[idx + 1]] += count
                        if idx < len(token_bytes) - 2:
                            pair_frequency_change_counter[token_bytes[idx + 1] + token_bytes[idx + 2]] -= count
                            pair_frequency_change_counter[token_bytes[idx] + token_bytes[idx + 1] + token_bytes[idx + 2]] += count
                        token_bytes[idx] = token_bytes[idx] + token_bytes.pop(idx + 1)
                        idx -= 1
                    idx -= 1
        return pair_frequency_change_counter
                    
    
    @staticmethod
    def _pair_frequency(token_counts: Dict[str, Tuple[List[Tuple[int]], int]]) -> Counter[Tuple[int]]:
        pair_counter = Counter()
        for _, (token_bytes, count) in token_counts.items():
            lefts, rights = token_bytes[:-1], token_bytes[1:]
            pair_counter.update(Counter({left + right: count for left, right in zip(lefts, rights)}))
        return pair_counter
    
    @staticmethod
    def _reform_tokens_counts(token_counts: Counter[str]) -> Dict[str, Tuple[List[int], int]]:
        return {token: ([(byte,) for byte in token.encode('utf-8')], count) for token, count in token_counts.items()}
    
    @staticmethod
    def pretokenize_parallel(input_path: str, pattern, special_tokens: Optional[List[str]] = None) -> Counter:
        '''
        pretokenizes a file in parallel and returns token frequencies
        '''
        if special_tokens is None:
            special_tokens = [r'<|endoftext|>']
        token_counts = Counter()
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(
                f, 128, "<|endoftext|>".encode("utf-8")
            )
        subprocess_args = [(input_path, pattern, special_tokens, sta, end) for sta, end in zip(boundaries[:-1], boundaries[1:])]
        with Pool(8) as p:
            results = p.starmap(BPETokenizer._parallel_pretokenize_worker, subprocess_args)
        for r in results:
            token_counts.update(r)
        return token_counts    
    
    @staticmethod
    def pretokenize_binary(file: bytes, pattern: str, special_tokens: Optional[List[str]] = None) -> Counter:
        '''
        pretokenizes a file and returns token frequencies
        '''
        if special_tokens is None:
            special_tokens = [r'<|endoftext|>']
        token_counts = Counter()
        chunk = file.decode('utf-8', errors='ignore')
        chunks = re.split('|'.join(map(re.escape, special_tokens)), chunk)
        for chunk in chunks:
                tokens = [re_match.group() for re_match in re.finditer(pattern, chunk)]
                counts = Counter(tokens)
                token_counts.update(counts)
        return token_counts
    
    @staticmethod
    def _parallel_pretokenize_worker(input_path: str, pattern: str, special_tokens: Optional[List[str]] = None, sta: int = 0, end: int = 0) -> Counter:
        '''
        called by subprocesses in pretokenize_parallel, returns token frequencies
        '''
        if special_tokens is None:
            special_tokens = [r'<|endoftext|>']
        with open(input_path, 'rb') as f:
            f.seek(sta)
            chunk = f.read(end - sta)
        return BPETokenizer.pretokenize_binary(chunk, pattern, special_tokens)
                
if __name__ == '__main__':
    ''' # Here is some test for pretokenize
    BPE = BPETokenizer(30, [r'<|endoftext|>'])
    DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/TinyStoriesV2-GPT4-train.txt')
    DATA_PATH = os.path.abspath(DATA_PATH)
    start = time.time()
    #with open(DATA_PATH, 'rb') as f:
    #    token_counts = BPETokenizer.pretokenize_binary(f.read(), BPE.PAT)
    token_counts = BPETokenizer.pretokenize_parallel(DATA_PATH, BPE.PAT)
    end = time.time()
    for idx, (token, count) in enumerate(token_counts.most_common()):
        if idx < 100:
            print(token, count)
    print(f'Time cost is {end - start}')
    '''
    
    # ===
    # Test of BPETokenizer._merge_pair_token_counts
    # ===
    def test_merge_pair_token_counts():
        test_dict = {
            'a': ([(1,), (2,), (3,), (4,)], 2),
            'b': ([(2,), (3,)], 3),
            'c': ([(7,), (2,), (3,)], 1)
        }
        pair = (2, 3)
        pair_changed_counter = BPETokenizer._merge_pair_token_counts(test_dict, pair)
        print(pair_changed_counter)
    
    test_merge_pair_token_counts()