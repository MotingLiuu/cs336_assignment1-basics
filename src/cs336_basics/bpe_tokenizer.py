from .utils import find_chunk_boundaries
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import Iterable, Iterator
from tqdm import tqdm
import time
import logging
import regex as re
import os
import json
import heapq
import base64

logger = logging.getLogger(__name__)

class ReversePair:
    def __init__(self, pair: tuple[bytes, bytes]):
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
        if "<|endoftext|>" not in self.special_tokens:
            self.special_tokens.append("<|endoftext|>")
        self.vocab = {
            **{idx: special_token.encode('utf-8') for idx, special_token in enumerate(self.special_tokens)},
            **{num + len(self.special_tokens): bytes([num]) for num in range(256)}
        }
        self.merges = []
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.token2id = {}
        self.prog = re.compile(self.PAT)
        self.special_tokens_rearranged = sorted(self.special_tokens, key = lambda x: -len(x))
        self.token2id_buffer = {}
    
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
        self.token2id = {token: idx for idx, token in self.vocab.items()}
        logger.info(f"Finsished Merging in {time.time() - time_sta_merging:.2f} seconds, vocab size: {len(self.vocab)}\n")
        
    @classmethod
    def from_pretrained(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None):
        '''
        Loads a Tokenizer from pretrained vocab and merges json files.
        '''
        tokenizer = cls(0, special_tokens)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tokenizer.vocab = json.load(f)
            tokenizer.vocab = {int(idx): base64.b64decode(token_bytes.encode('utf-8')) for idx, token_bytes in tokenizer.vocab.items()}
        with open(merges_path, 'r', encoding='utf-8') as f:
            tokenizer.merges = json.load(f)
            tokenizer.merges = [(base64.b64decode(left.encode('utf-8')), base64.b64decode(right.encode('utf-8'))) for left, right in tokenizer.merges]
        tokenizer.vocab_size = len(tokenizer.vocab)
        tokenizer.token2id = {token: idx for idx, token in tokenizer.vocab.items()}
        special_tokens = [tok.encode('utf-8') for tok in special_tokens] if special_tokens else []
        for tok in special_tokens:
            if not tokenizer.token2id.get(tok):
                idx = len(tokenizer.vocab)
                tokenizer.vocab[idx] = tok
                tokenizer.token2id[tok] = idx
                
        return tokenizer
    
    @classmethod
    def from_dict_list(cls, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        '''
        Loads a Tokenizer from a dict of vocab and a list of merges.
        '''
        tokenizer = cls(0, special_tokens)
        tokenizer.vocab = vocab
        tokenizer.merges = merges
        tokenizer.vocab_size = len(tokenizer.vocab)
        tokenizer.token2id = {token: idx for idx, token in tokenizer.vocab.items()}
        special_tokens = [tok.encode('utf-8') for tok in special_tokens] if special_tokens else []
        for tok in special_tokens:
            if not tokenizer.token2id.get(tok):
                idx = len(tokenizer.vocab)
                tokenizer.vocab[idx] = tok
                tokenizer.token2id[tok] = idx
                
        return tokenizer
      
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
        logging.info("Pretokenizing without parallel... \n")
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
    
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        tokens = []
        sta, end = 0, len(text) - 1
        for match in re.finditer("|".join(map(re.escape, self.special_tokens_rearranged)), text):
            end = match.start()
            tokens.extend(self.prog.findall(text[sta:end]))
            tokens.append(match.group())
            sta = match.end()
        tokens.extend(self.prog.findall(text[sta:]))
        return self._encode_tokens(tokens)
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of strings into a Iterator of token IDs.
        For processing large datasets without loading everything into memory at once.
        """
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into a string.
        """
        tokens = [self.vocab[id] for id in ids]
        return b''.join(tokens).decode('utf-8', errors='ignore')
            
    
    def _encode_tokens(self, tokens:list[str]) -> list[int]:
        """
        Encode a list of tokens into a sequence of token IDs
        """
        tokens = [token.encode('utf-8') for token in tokens]
        token_ids = []
        for token in tokens:
            token_ids.extend(self.token_2_ids(token))
        return token_ids
            
    
    def token_2_ids(self, token: bytes) -> list[int]:
        """
        Convert a token(bytes) to a list of token IDs.
        """
        if self.token2id.get(token):
            return [self.token2id[token]]
        elif self.token2id_buffer.get(token):
            return self.token2id_buffer[token]
        else:
            tokens = [bytes([byte]) for byte in token]
            token_ids = []
            merge_completed = False
            while merge_completed is False:
                pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                pair_ids = [self.token2id.get(pair[0]+pair[1], -1) for pair in pairs]
                tmp_ids = [pair_id for pair_id in pair_ids if pair_id != -1]
                merge_id = min(tmp_ids) if tmp_ids else -1
                if merge_id == -1:
                    merge_completed = True
                else:
                    merge_completed = False
                    merge = pairs[pair_ids.index(merge_id)]
                    BPETokenizer._token_merge(tokens, merge)
            for tok in tokens:
                if self.token2id.get(tok) is not None:
                    token_ids.append(self.token2id[tok])
                else:
                    logger.warning(f"Token {tok} not found in vocabulary.")
        self.token2id_buffer[token] = token_ids
        return token_ids
                    
                    

    @classmethod
    def _token_merge(cls, token: list[bytes], merge: tuple[bytes, bytes]) -> list[bytes]:
        """
        Merge a token with a merge pair.
        """
        idx = 0
        while idx < len(token) - 1:
            if (token[idx], token[idx + 1]) == merge:
                token[idx] = token[idx] + token.pop(idx + 1)
            idx += 1


