from .utils import find_chunk_boundaries
from collections import Counter
from multiprocessing import Pool
#from tqdm import tqdm
import regex as re
import os

class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str] | None = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else []
        self.vocab = {
            **{idx: special_token.encode('utf-8') for idx, special_token in enumerate(self.special_tokens)},
            **{num + len(self.special_tokens): bytes([num]) for num in range(256)}
        }
        self.merges = []
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def train(self, input_path: str, parallel: bool = True):
        if parallel:
            token_counts = BPETokenizer.pretokenize_parallel(input_path, self.PAT, self.special_tokens)
        else:
            token_counts = BPETokenizer.pretokenize(input_path, self.PAT, self.special_tokens)
        # reform the token_counts{bytes: int} to {bytes: (List, int)}
        token_counts = BPETokenizer._reform_tokens_counts(token_counts)
        # get the pair freqeuncy: Counter
        pair_counts = BPETokenizer._pair_frequency(token_counts)
        vocab_size_before_train = len(self.vocab)
        for i in range(vocab_size_before_train, self.vocab_size):
            most_frequent_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
            self.merges.append(most_frequent_pair)
            self.vocab[i] = most_frequent_pair[0] + most_frequent_pair[1]
            pair_changed_counter = BPETokenizer._merge_pair_token_counts(token_counts, most_frequent_pair)
            pair_counts.update(pair_changed_counter)
            pair_counts.pop(most_frequent_pair)

    @staticmethod
    def _merge_pair_token_counts(token_counts: dict[str, tuple[list[bytes], int]], pair: tuple[bytes]) -> Counter[tuple[bytes]]:
        pair_frequency_change_counter = Counter()
        for _, (token_bytes, count) in token_counts.items():
            if len(token_bytes) > 1:
                idx = 0
                while idx < len(token_bytes) - 1:
                    if (token_bytes[idx], token_bytes[idx + 1]) == pair:
                        if idx > 0:
                            pair_frequency_change_counter[(token_bytes[idx - 1], token_bytes[idx])] -= count
                            pair_frequency_change_counter[(token_bytes[idx - 1], token_bytes[idx] + token_bytes[idx + 1])] += count
                        if idx < len(token_bytes) - 2:
                            pair_frequency_change_counter[(token_bytes[idx + 1], token_bytes[idx + 2])] -= count
                            pair_frequency_change_counter[(token_bytes[idx] + token_bytes[idx + 1], token_bytes[idx + 2])] += count
                        token_bytes[idx] = token_bytes[idx] + token_bytes.pop(idx + 1)
                    idx += 1
        return pair_frequency_change_counter
                    
    
    @staticmethod
    def _pair_frequency(token_counts: dict[str, tuple[list[bytes], int]]) -> Counter[tuple[bytes]]:
        pair_counter = Counter()
        for _, (token_bytes, count) in token_counts.items():
            for idx in range(len(token_bytes) - 1):
                pair_counter[(token_bytes[idx], token_bytes[idx + 1])] += count
        return pair_counter
    
    @staticmethod
    def _reform_tokens_counts(token_counts: Counter[str]) -> dict[str, tuple[list[bytes], int]]:
        return {token: ([bytes([byte]) for byte in token.encode('utf-8')], count) for token, count in token_counts.items()}
    
    @staticmethod
    def pretokenize(input_path:str, pattern: str, special_tokens: list[str] | None = None) -> Counter:
        if not special_tokens:
            special_tokens = [r'<endoftext>']
        token_counts = Counter()
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(
                f, 64, b"<endoftext>"
            )
        print("Pretokenizing without parallel... \n")
        for sta, end in zip(boundaries[:-1], boundaries[1:]):
            with open(input_path, 'rb') as f:
                f.seek(sta)
                chunk = f.read(end -sta)
            token_counts.update(BPETokenizer.pretokenize_binary(chunk, pattern, special_tokens))
        return token_counts
    
    @staticmethod
    def pretokenize_parallel(input_path: str, pattern, special_tokens: list[str] | None = None) -> Counter:
        '''
        pretokenizes a file in parallel and returns token frequencies
        '''
        if not special_tokens:
            special_tokens = [r'<|endoftext|>']
        token_counts = Counter()
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(
                f, 64, b"<|endoftext|>"
            )
        subprocess_args = [(input_path, pattern, special_tokens, sta, end) for sta, end in zip(boundaries[:-1], boundaries[1:])]
        with Pool(64) as p:
            results = p.starmap(BPETokenizer._parallel_pretokenize_worker, subprocess_args) # 这里使用了硬编码，考虑将函数改为cls method？
        for r in results:
            token_counts.update(r)
        return token_counts    
    
    @staticmethod
    def pretokenize_binary(file: bytes, pattern: str, special_tokens: list[str] | None = None) -> Counter:
        '''
        pretokenizes a file and returns token frequencies
        '''
        if not special_tokens:
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
    def _parallel_pretokenize_worker(input_path: str, pattern: str, special_tokens: list[str] | None = None, sta: int = 0, end: int = 0) -> Counter:
        '''
        called by subprocesses in pretokenize_parallel, returns token frequencies
        '''
        if not special_tokens:
            special_tokens = [r'<|endoftext|>']
        with open(input_path, 'rb') as f:
            f.seek(sta)
            chunk = f.read(end - sta)
        return BPETokenizer.pretokenize_binary(chunk, pattern, special_tokens)
                
if __name__ == '__main__':
    
    def test_pretokenize_parallel():
        DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/corpus.en')
        DATA_PATH = os.path.abspath(DATA_PATH)
        print(BPETokenizer.pretokenize_parallel(DATA_PATH, r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""))
    
    # ===
    # Test of BPETokenizer.pretokenize
    # ===
    def test_pretokenize():
        DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/corpus.en')
        DATA_PATH = os.path.abspath(DATA_PATH)
        print(BPETokenizer.pretokenize(DATA_PATH, r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""))    
        
    # ===
    # Test of BPETokenizer._merge_pair_token_counts
    # ===
    def test_merge_pair_token_counts():
        test_dict = {
            'a': ([bytes([1]), bytes([2]), bytes([5]), bytes([3]), bytes([4]), bytes([4])], 1),
            'b': ([bytes([1]), bytes([2]), bytes([2]), bytes([3]), bytes([4])], 2),
            'c': ([bytes([5]), bytes([1]), bytes([2]), bytes([6]), bytes([3]), bytes([4])], 3),
            'd': ([bytes([7]), bytes([1]), bytes([2]), bytes([8]), bytes([3]), bytes([4])], 2),
            'e': ([bytes([1]), bytes([2]), bytes([3]), bytes([4]), bytes([9])], 1),
            'f': ([bytes([10]), bytes([1]), bytes([2]), bytes([3]), bytes([4])], 4),
            }
        pair = (bytes([2]), bytes([3])) 
        pair_changed_counter = BPETokenizer._merge_pair_token_counts(test_dict, pair)
        print(pair_changed_counter)
        print(test_dict)
    
    
    # ===
    # Test of BPETokenizer._pair_frequency
    # ===
    def test_pair_frequency():
        test_dict = {
            'a': ([bytes([1]), bytes([2]), bytes([5]), bytes([3]), bytes([4]), bytes([4])], 1),
            'b': ([bytes([1]), bytes([2]), bytes([2]), bytes([3]), bytes([4])], 2),
            'c': ([bytes([5]), bytes([1]), bytes([2]), bytes([6]), bytes([3]), bytes([4])], 3),
            'd': ([bytes([7]), bytes([1]), bytes([2]), bytes([8]), bytes([3]), bytes([4])], 2),
            'e': ([bytes([1]), bytes([2]), bytes([3]), bytes([4]), bytes([9])], 1),
            'f': ([bytes([10]), bytes([1]), bytes([2]), bytes([3]), bytes([4])], 4),
            }
        BPE = BPETokenizer(500, [r'<|endoftext|>'])
        DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/corpus.en')
        DATA_PATH = os.path.abspath(DATA_PATH)
        token_counts = BPETokenizer.pretokenize_parallel(DATA_PATH, BPE.PAT)
        token_counts = BPETokenizer._reform_tokens_counts(token_counts)
        pair_counts = BPETokenizer._pair_frequency(token_counts)
        print(pair_counts)

    
    # ===
    # Test of BPETokeinzer.train
    # ===
    def test_BPE_train():
        BPE = BPETokenizer(500, [r'<|endoftext|>'])
        DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/TinyStoriesV2-GPT4-valid.txt')
        DATA_PATH = os.path.abspath(DATA_PATH)
        BPE.train(DATA_PATH)
        print(BPE.vocab)
        print(BPE.merges)
    
    test_pretokenize()
    #test_pretokenize_parallel()
    #test_merge_pair_token_counts()
    #test_pair_frequency()
    #test_BPE_train()