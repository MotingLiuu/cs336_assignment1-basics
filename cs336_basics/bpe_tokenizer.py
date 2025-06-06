from .utils import find_chunk_boundaries
from collections import Counter, defaultdict
from multiprocessing import Pool
from tqdm import tqdm
import time
import logging
import regex as re
import os

logger = logging.getLogger(__name__)

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
        token_counts, pair2tokens = BPETokenizer._reform_tokens_counts(token_counts)
        # get the pair freqeuncy: Counter
        pair_counts = BPETokenizer._pair_frequency(token_counts)
        vocab_size_before_train = len(self.vocab)
        logger.info(f"Started Merging\n")
        time_sta_merging = time.time()
        for i in tqdm(range(vocab_size_before_train, self.vocab_size)):
            if i % 100 == 0:
                logger.info(f"Iteration {i}, vocab size: {len(self.vocab)}")
            most_frequent_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
            self.merges.append(most_frequent_pair)
            self.vocab[i] = most_frequent_pair[0] + most_frequent_pair[1]
            pair_changed_counter = BPETokenizer._merge_pair_token_counts(token_counts, most_frequent_pair)
            pair_counts.update(pair_changed_counter)
            pair_counts.pop(most_frequent_pair)
        logger.info(f"Finsished Merging in {time.time() - time_sta_merging:.2f} seconds, vocab size: {len(self.vocab)}\n")
        

    @staticmethod
    def _merge_pair_token_counts(token_counts: dict[bytes, tuple[list[bytes], int]],  pair2tokens: dict[tuple[bytes, bytes], set[bytes]], pair: tuple[bytes, bytes]) -> Counter[tuple[bytes]]:
        # merge would change token_counts[list[bytes]], pair2tokens, add some new pair, remove some pair eliminated during merge, change some pair's set
        # update the pair_counts, without accessing to pair_counts
        # TODO: merge list[bytes] in token_counts, find corresponding tokens from pair2tokens, then merge list[bytes] in token_counts, counts the side effect to pair2tokens, pair_counts
        # TODO: update pair2tokens and pair_counts, using update
        pair_change_counter = Counter()
        for token in pair2tokens[pair]:
            bytes_list, count = token_counts[token]
            new_bytes_list = []
            idx = 0
            while idx <= len(bytes_list) - 1:
                if idx == len(bytes_list) - 1:
                    new_bytes_list.append(bytes_list[-1])
                    break
                if (bytes_list[idx], bytes_list[idx + 1]) == pair:
                    if idx > 0:
                        pair_change_counter[(new_bytes_list[-1], bytes_list[idx] + bytes_list[idx + 1])] += count
                        pair2tokens[(new_bytes_list[-1], bytes_list[idx] + bytes_list[idx + 1])].add(token)
                        pair_change_counter[(new_bytes_list[-1], bytes_list[idx])] -= count
                        pair2tokens[(new_bytes_list[-1], bytes_list[idx])].remove(token)
                    if idx < len(bytes_list) - 2:
                        pair_change_counter[(bytes_list[idx] + bytes_list[idx + 1], bytes_list[idx + 2])] += count
                        pair2tokens[(bytes_list[idx] + bytes_list[idx + 1], bytes_list[idx + 2])].add(token)
                        pair_change_counter[(bytes_list[idx + 1], bytes_list[idx + 2])] -= count
                        pair2tokens[(bytes_list[idx + 1], bytes_list[idx + 2])].remove(token)
                    new_bytes_list.append(bytes_list[idx] + bytes_list[idx + 1])
                    idx += 1
                else:
                    new_bytes_list.append(bytes_list[idx])
                idx += 1
                token_counts[token] = (new_bytes_list, count)
        pair2tokens.pop(pair)
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
            
    
    @staticmethod
    def _pair_frequency(token_counts: dict[bytes, tuple[list[bytes], int]]) -> Counter[tuple[bytes]]:
        pair_counter = Counter()
        for _, (token_bytes, count) in token_counts.items():
            for idx in range(len(token_bytes) - 1):
                pair_counter[(token_bytes[idx], token_bytes[idx + 1])] += count
        return pair_counter
    
    @staticmethod
    def _reform_tokens_counts(token_counts: Counter[str]) -> tuple[dict[bytes, tuple[list[bytes], int]], dict[tuple[bytes,bytes], set[bytes]]]:
        token_counts_reformed = Counter()
        pair2tokens = defaultdict(set)
        for token, count in token_counts.items():
            token_bytes = token.encode('utf-8')
            token_counts_reformed[token_bytes] = (BPETokenizer.get_bytes_list(token_bytes), count)
            for pair in BPETokenizer.count_pair(token_bytes):
                pair2tokens[pair].add(token_bytes)
        return token_counts_reformed, pair2tokens
    
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