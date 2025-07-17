import json
import time
import logging
from collections import defaultdict, Counter
import pytest
import cs336_basics
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

@pytest.fixture
def setup_tokens_counts():
    """
    Fixture to set up token counts for testing.
    """
    return Counter({
        "hello": 5,
        "world": 3,
        "test": 2,
        "bpe": 1,
        "token": 1,
        "ization": 1,
        "example": 1,
        "data": 4,
        "processing": 2,
        "python": 3,
        "code": 2,
        "tee": 2,
        "ababa": 1,
    })

@pytest.fixture
def setup_token_bytes_counts():
    return {
        "hello": ([b'h', b'e', b'l', b'l', b'o'], 5),
        "world": ([b'w', b'o', b'r', b'l', b'd'], 3),
        "test": ([b't', b'e', b's', b't'], 2),
        "bpe": ([b'b', b'p', b'e'], 1),
        "token": ([b't', b'o', b'k', b'e', b'n'], 1),
        "ization": ([b'i', b'z', b'a', b't', b'i', b'o', b'n'], 1),
        "example": ([b'e', b'x', b'a', b'm', b'p', b'l', b'e'], 1),
        "data": ([b'd', b'a', b't', b'a'], 4),
        "processing": ([b'p', b'r', b'o', b'c', b'e', b's', b's', b'i', b'n', b'g'], 2),
        "python": ([b'p', b'y', b't', b'h', b'o', b'n'], 3),
        "code": ([b'c', b'o', b'd', b'e'], 2),
        "tee": ([b't', b'e', b'e'], 2),
        "ababa": ([b'a', b'b', b'a', b'b', b'a'], 1),
    }
    
@pytest.fixture
def setup_pair2tokens():
    pair2tokens = defaultdict(set)
    pair2tokens.update({
        (b'h', b'e'): {'hello'},
        (b'e', b'l'): {'hello'},
        (b'l', b'l'): {'hello'},
        (b'l', b'o'): {'hello'},
        (b'w', b'o'): {'world'},
        (b'o', b'r'): {'world'},
        (b'r', b'l'): {'world'},
        (b'l', b'd'): {'world'},
        (b't', b'e'): {'test', 'tee'}, # 'te' 出现在 'test' 和 'tee' 中
        (b'e', b's'): {'test', 'processing'}, # 'es' 出现在 'test' 和 'processing' 中
        (b's', b't'): {'test'},
        (b'b', b'p'): {'bpe'},
        (b'p', b'e'): {'bpe'},
        (b't', b'o'): {'token'},
        (b'o', b'k'): {'token'},
        (b'k', b'e'): {'token'},
        (b'e', b'n'): {'token'},
        (b'i', b'z'): {'ization'},
        (b'z', b'a'): {'ization'},
        (b'a', b't'): {'data', 'ization'}, # 'at' 出现在 'data' 和 'ization' 中
        (b't', b'i'): {'ization'},
        (b'i', b'o'): {'ization'},
        (b'o', b'n'): {'python', 'ization'}, # 'on' 出现在这三个词中
        (b'e', b'x'): {'example'},
        (b'x', b'a'): {'example'},
        (b'a', b'm'): {'example'},
        (b'm', b'p'): {'example'},
        (b'p', b'l'): {'example'},
        (b'l', b'e'): {'example'},
        (b'd', b'a'): {'data'},
        (b't', b'a'): {'data'},
        (b'p', b'r'): {'processing'},
        (b'r', b'o'): {'processing'},
        (b'o', b'c'): {'processing'},
        (b'c', b'e'): {'processing'},
        (b's', b's'): {'processing'},
        (b's', b'i'): {'processing'},
        (b'i', b'n'): {'processing'},
        (b'n', b'g'): {'processing'},
        (b'p', b'y'): {'python'},
        (b'y', b't'): {'python'},
        (b't', b'h'): {'python'},
        (b'h', b'o'): {'python'},
        (b'c', b'o'): {'code'},
        (b'o', b'd'): {'code'},
        (b'd', b'e'): {'code'},
        (b'e', b'e'): {'tee'},
        (b'a', b'b'): {'ababa'},
        (b'b', b'a'): {'ababa'},
    })
    return pair2tokens


def test_reform_tokens_counts(setup_tokens_counts, setup_pair2tokens, setup_token_bytes_counts):
    """
    Test the _reform_tokens_counts method of BPETokenizer.
    """
    token_counts = setup_tokens_counts
    expected_token_bytes_counts = setup_token_bytes_counts
    expected_pair2tokens = setup_pair2tokens
  
    token_bytes_counts, pair2tokens = cs336_basics.BPETokenizer._reform_tokens_counts(token_counts)
    assert token_bytes_counts == expected_token_bytes_counts, "Token bytes counts do not match expected values."
    assert pair2tokens == expected_pair2tokens, "Pair to tokens mapping does not match expected values."
    
def test_merge_pair_token_counts(setup_token_bytes_counts, setup_pair2tokens):
    
    pair = (b'a', b't')
    token_bytes_counts = setup_token_bytes_counts
    pair2tokens = setup_pair2tokens
    
    expected_token_bytes_counts = {
        "hello": ([b'h', b'e', b'l', b'l', b'o'], 5),
        "world": ([b'w', b'o', b'r', b'l', b'd'], 3),
        "test": ([b't', b'e', b's', b't'], 2),
        "bpe": ([b'b', b'p', b'e'], 1),
        "token": ([b't', b'o', b'k', b'e', b'n'], 1),
        "ization": ([b'i', b'z', b'at', b'i', b'o', b'n'], 1),
        "example": ([b'e', b'x', b'a', b'm', b'p', b'l', b'e'], 1),
        "data": ([b'd', b'at', b'a'], 4),
        "processing": ([b'p', b'r', b'o', b'c', b'e', b's', b's', b'i', b'n', b'g'], 2),
        "python": ([b'p', b'y', b't', b'h', b'o', b'n'], 3),
        "code": ([b'c', b'o', b'd', b'e'], 2),
        "tee": ([b't', b'e', b'e'], 2),
        "ababa": ([b'a', b'b', b'a', b'b', b'a'], 1),
    }
    
    expected_pair2tokens = {
        (b'd', b'at'): {'data'},
        (b'at', b'a'): {'data'},
        (b'z', b'at'): {'ization'},
        (b'at', b'i'): {'ization'},
        (b'h', b'e'): {'hello'},
        (b'e', b'l'): {'hello'},
        (b'l', b'l'): {'hello'},
        (b'l', b'o'): {'hello'},
        (b'w', b'o'): {'world'},
        (b'o', b'r'): {'world'},
        (b'r', b'l'): {'world'},
        (b'l', b'd'): {'world'},
        (b't', b'e'): {'test', 'tee'}, # 'te' 出现在 'test' 和 'tee' 中
        (b'e', b's'): {'test', 'processing'}, # 'es' 出现在 'test' 和 'processing' 中
        (b's', b't'): {'test'},
        (b'b', b'p'): {'bpe'},
        (b'p', b'e'): {'bpe'},
        (b't', b'o'): {'token'},
        (b'o', b'k'): {'token'},
        (b'k', b'e'): {'token'},
        (b'e', b'n'): {'token'},
        (b'i', b'z'): {'ization'},
        (b'i', b'o'): {'ization'},
        (b'o', b'n'): {'python', 'ization'}, # 'on' 出现在这三个词中
        (b'e', b'x'): {'example'},
        (b'x', b'a'): {'example'},
        (b'a', b'm'): {'example'},
        (b'm', b'p'): {'example'},
        (b'p', b'l'): {'example'},
        (b'l', b'e'): {'example'},
        (b'p', b'r'): {'processing'},
        (b'r', b'o'): {'processing'},
        (b'o', b'c'): {'processing'},
        (b'c', b'e'): {'processing'},
        (b's', b's'): {'processing'},
        (b's', b'i'): {'processing'},
        (b'i', b'n'): {'processing'},
        (b'n', b'g'): {'processing'},
        (b'p', b'y'): {'python'},
        (b'y', b't'): {'python'},
        (b't', b'h'): {'python'},
        (b'h', b'o'): {'python'},
        (b'c', b'o'): {'code'},
        (b'o', b'd'): {'code'},
        (b'd', b'e'): {'code'},
        (b'e', b'e'): {'tee'},
        (b'a', b'b'): {'ababa'},
        (b'b', b'a'): {'ababa'},
    }
    
    expected_pair_frequency_change_counter = {
        (b'd', b'a'): -4,
        (b't', b'a'): -4,
        (b'z', b'a'): -1,
        (b't', b'i'): -1,
        (b'd', b'at'): 4,
        (b'at', b'a'): 4,
        (b'z', b'at'): 1,
        (b'at', b'i'): 1,
    }
    
    pair_frequency_change_counter = cs336_basics.BPETokenizer._merge_pair_token_counts(token_bytes_counts, pair2tokens, pair)
    assert pair_frequency_change_counter == expected_pair_frequency_change_counter, "Pair frequency change counter does not match expected values."
    assert token_bytes_counts == expected_token_bytes_counts, "Token bytes counts do not match expected values after merging."
    assert pair2tokens == expected_pair2tokens, "Pair to tokens mapping does not match expected values after merging."
    
@pytest.fixture
def setup_bpe_tokenizer():
    """
    Fixture to set up a BPETokenizer instance for testing.
    """
    vocab = {
        0: b' ',
        1: b'a',
        2: b'c',
        3: b'e',
        4: b'h',
        5: b't',
        6: b'th',
        7: b' c', 
        8: b' a', 
        9: b'the', 
        10: b' at' 
    }
    
    merges = [
        (b't', b'h'),  
        (b' ', b'c'),    
        (b' ', b'a'),    
        (b'th', b'e'),  
        (b' a', b't')
    ]
    
    special_tokens = ['<|endoftext|>']

    return cs336_basics.BPETokenizer.from_dict_list(vocab, merges, special_tokens)

@pytest.fixture
def setup_text():
    return "This is a test text for BPE tokenizer to test the functionality of encoding and decoding"

def test_from_pretrained():
    """
    Test the from_pretrained method of BPETokenizer.
    """
    vocab_path = str(Path(__file__).parent / "../experiments/vocab_TinyStories.json")
    merges_path = str(Path(__file__).parent / "../experiments/merges_TinyStories.json")

    # Assuming the files exist and are correctly formatted
    tokenizer = cs336_basics.BPETokenizer.from_pretrained(vocab_path, merges_path, None)
    
    assert isinstance(tokenizer, cs336_basics.BPETokenizer), "The returned object is not an instance of BPETokenizer."
    assert len(tokenizer.vocab) > 0, "The vocabulary should not be empty."
    for idx in range(10):
        logging.info(f"Token {idx}: {tokenizer.vocab.get(idx, 'Not Found')}")
    assert len(tokenizer.merges) > 0, "The merges should not be empty."
    for idx in range(10):
        logging.info(f"Merge left:{tokenizer.merges[idx][0]}, right: {tokenizer.merges[idx][1]}")

def test_encode(setup_text):
    """
    Test the encode method of BPETokenizer.
    """
    pass

def test_token_merge():
    """
    Test the _token_merge method of BPETokenizer.
    """
    token = [b'h', b'e', b'l', b'l', b'o', b'l', b'o']
    merge = (b'l', b'o')
    expected_result = [b'h', b'e', b'l', b'lo', b'lo']
    cs336_basics.BPETokenizer._token_merge(token, merge)
    assert token == expected_result, f"Expected {expected_result}, but got {token}"
    
def test_token_2_ids(setup_bpe_tokenizer):
    """
    Test the _token_2_ids method of BPETokenizer.
    """
    tokenizer = setup_bpe_tokenizer
    token1, token2, token3 = b"the", b" cat", b" ate"
    token1_ids = tokenizer.token_2_ids(token1)
    token2_ids = tokenizer.token_2_ids(token2)
    token3_ids = tokenizer.token_2_ids(token3)
    assert token1_ids == [9], f"Expected [9], but got {token1_ids}"
    assert token2_ids == [7, 1, 5], f"Expected [7], but got {token2_ids}"
    assert token3_ids == [10, 3], f"Expected [10], but got {token3_ids}"
    
def test_encode_tokens(setup_bpe_tokenizer):
    """
    Test the encode_tokens method of BPETokenizer.
    """
    tokenizer = setup_bpe_tokenizer
    tokens = ['the', ' cat', ' ate']
    expected_ids = [9, 7, 1, 5, 10, 3]
    encoded_ids = tokenizer._encode_tokens(tokens)
    assert encoded_ids == expected_ids, f"Expected {expected_ids}, but got {encoded_ids}"
    
def test_encode(setup_bpe_tokenizer):
    """
    Test the encode method of BPETokenizer.
    """
    tokenizer = setup_bpe_tokenizer
    text = "the cat ate"
    expected_ids = [9, 7, 1, 5, 10, 3]
    encoded_ids = tokenizer.encode(text)
    assert encoded_ids == expected_ids, f"Expected {expected_ids}, but got {encoded_ids}"
    
def test_encode_iterable(setup_bpe_tokenizer):
    """
    Test the encode_iterable method of BPETokenizer.
    """
    tokenizer = setup_bpe_tokenizer
    texts = ["the cat ate", "the cat ate", "the cat ate"]
    expected_ids = [9, 7, 1, 5, 10, 3, 9, 7, 1, 5, 10, 3, 9, 7, 1, 5, 10, 3]
    encoded_iterator = tokenizer.encode_iterable(texts)
    for value, expected_value in zip(encoded_iterator, expected_ids):
        assert value == expected_value, f"Expected {expected_value}, but got {value}"
        