import json
import time
import logging
from collections import defaultdict, Counter
import pytest
import cs336_basics

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
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
    
    