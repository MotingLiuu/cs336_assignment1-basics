from .utils import find_chunk_boundaries
from collections import Counter, defaultdict
from multiprocessing import Pool
from tqdm import tqdm
import time
import logging
import regex as re
import os
import heapq

logger = logging.getLogger(__name__)

class ReversePair:
    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair
        
    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair

    def __eq__(self, other):
        return self.pair == other.pair
    
    def __hash__(self):
        return hash(self.pair)

    def __repr__(self):
        return f"ReversedPair({self.pair!r})"
        

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
        logger.info(f"Started Pretokenization: {input_path} (parallel={parallel}).\n")
        time_sta_pretokenization = time.time()
        if parallel:
            token_counts = BPETokenizer.pretokenize_parallel(input_path, self.PAT, self.special_tokens)
        else:
            token_counts = BPETokenizer.pretokenize(input_path, self.PAT, self.special_tokens)
        logger.info(f"Finished Pretokenization in {time.time() - time_sta_pretokenization:.2f} seconds.\n")
        # reform the token_counts{bytes: int} to {bytes: (List, int)}
        token_bytes_counts, pair2tokens = BPETokenizer._reform_tokens_counts(token_counts)
        # get the pair freqeuncy: Counter
        pair_counts, max_heapq = BPETokenizer._pair_frequency(token_bytes_counts)
        vocab_size_before_train = len(self.vocab)
        logger.info(f"Started Merging\n")
        time_sta_merging = time.time()
        for i in tqdm(range(vocab_size_before_train, self.vocab_size)):
            if i % 100 == 0:
                logger.info(f"Iteration {i}, vocab size: {len(self.vocab)}")
            most_frequent_pair = None
            while max_heapq:
                neg_count, reversed_pair = heapq.heappop(max_heapq)
                pair = reversed_pair.pair
                if pair_counts.get(pair) == -neg_count:
                    most_frequent_pair = pair
                    break
            if most_frequent_pair is None:
                logger.warning(f"No more pairs to merge at iteration {i}, stopping early.")
                break
            self.merges.append(most_frequent_pair)
            self.vocab[i] = most_frequent_pair[0] + most_frequent_pair[1]
            pair_changed_counter = BPETokenizer._merge_pair_token_counts(token_bytes_counts, pair2tokens, most_frequent_pair)
            pair_counts.update(pair_changed_counter)
            pair_counts.pop(most_frequent_pair, None)
            pairs_to_remove = []
            for pair, change in pair_changed_counter.items():
                new_count = pair_counts.get(pair)
                if new_count is not None and new_count > 0:
                    heapq.heappush(max_heapq, (-new_count, ReversePair(pair)))
                elif new_count is not None and new_count <= 0:
                    pairs_to_remove.append(pair)
            for pair in pairs_to_remove:
                pair_counts.pop(pair, None)
        logger.info(f"Finsished Merging in {time.time() - time_sta_merging:.2f} seconds, vocab size: {len(self.vocab)}\n")
        

    @staticmethod
    def _merge_pair_token_counts(token_bytes_counts: dict[str, tuple[list[bytes], int]], pair2tokens: dict[tuple[bytes], set[str]], pair: tuple[bytes]) -> Counter[tuple[bytes]]:
        '''
        merge the pair in token_bytes_counts and pair2tokens, return the change of pair frequency. This function modifies the token_bytes_counts and pair2tokens in place.
        '''
        pair_frequency_change_counter = Counter()
        token_with_pair = pair2tokens.get(pair, set())
        pair2tokens.pop(pair, None)
        for token in token_with_pair:
            token_bytes, count = token_bytes_counts[token]
            token_pairs = set([(token_bytes[idx], token_bytes[idx + 1]) for idx in range(len(token_bytes) - 1)])
            if len(token_bytes) > 1:
                idx = 0
                while idx < len(token_bytes) - 1:
                    if (token_bytes[idx], token_bytes[idx + 1]) == pair:
                        if idx > 0:
                            pair_frequency_change_counter[(token_bytes[idx - 1], token_bytes[idx])] -= count
                            pair_frequency_change_counter[(token_bytes[idx - 1], token_bytes[idx] + token_bytes[idx + 1])] += count
                            pair2tokens[(token_bytes[idx - 1], token_bytes[idx] + token_bytes[idx + 1])].add(token)
                        if idx < len(token_bytes) - 2:
                            pair_frequency_change_counter[(token_bytes[idx + 1], token_bytes[idx + 2])] -= count
                            pair_frequency_change_counter[(token_bytes[idx] + token_bytes[idx + 1], token_bytes[idx + 2])] += count
                            pair2tokens[(token_bytes[idx] + token_bytes[idx + 1], token_bytes[idx + 2])].add(token)
                        token_bytes[idx] = token_bytes[idx] + token_bytes.pop(idx + 1)
                    idx += 1
            token_pairs_changed = set([(token_bytes[idx], token_bytes[idx + 1]) for idx in range(len(token_bytes) - 1)])
            pairs_deleted = token_pairs - token_pairs_changed
            for pair_deleted in pairs_deleted:
                if token in pair2tokens[pair_deleted]:
                    pair2tokens[pair_deleted].remove(token)

        for idx in list(pair2tokens.keys()):
            if not pair2tokens[idx]:
                pair2tokens.pop(idx)
        return pair_frequency_change_counter
                    
    
    @staticmethod
    def _pair_frequency(token_counts: dict[bytes, tuple[list[bytes], int]]) -> Counter[tuple[bytes]]:
        pair_counter = Counter()
        max_heapq = []
        for _, (token_bytes, count) in token_counts.items():
            for idx in range(len(token_bytes) - 1):
                pair_counter[(token_bytes[idx], token_bytes[idx + 1])] += count
        for pair, count in pair_counter.items():
            if count > 0:
                heapq.heappush(max_heapq, (-count, ReversePair(pair)))
        return pair_counter, max_heapq
    
    @staticmethod
    def _reform_tokens_counts(token_counts: Counter[str]) -> tuple[dict[str, tuple[list[bytes], int]], dict[tuple[bytes], set[str]]]:
        '''
        reform the token_counts from {token: int} to {token: ([bytes], int)} and pair2tokens {[bytes, bytes]: set[str]}
        '''
        token_bytes_counts = {token: ([bytes([byte]) for byte in token.encode('utf-8')], count) for token, count in token_counts.items()}
        pair2tokens = defaultdict(set)
        for token, (token_bytes, count) in token_bytes_counts.items():
            for idx in range(len(token_bytes) - 1):
                pair = (token_bytes[idx], token_bytes[idx + 1])
                pair2tokens[pair].add(token)
        return token_bytes_counts, pair2tokens
    
    @staticmethod
    def pretokenize(input_path:str, pattern: str, special_tokens: list[str] | None = None) -> Counter:
        if not special_tokens:
            special_tokens = [r'<|endoftext|>']
        token_counts = Counter()
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(
                f, 64, b"<|endoftext|>"
            )
        print("Pretokenizing without parallel... \n")
        for sta, end in tqdm(zip(boundaries[:-1], boundaries[1:])):
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