from collections import Counter, defaultdict
from cs336_basics import BPETokenizer

def test_reform_tokens_counts():
    token_counts = Counter({
        'abc': 2,
        'bcd': 12,
        'sadebzkjhg': 1,
        ',zncmvk': 5,
        ',n,nzh': 2,
    })
    expected_token_counts_reformed = {
        b'abc': ([b'a', b'b', b'c'], 2),
        b'bcd': ([b'b', b'c', b'd'], 12),
        b'sadebzkjhg': ([b's', b'a', b'd', b'e', b'b', b'z', b'k', b'j', b'h', b'g'], 1),
        b',zncmvk': ([b',', b'z', b'n', b'c', b'm', b'v', b'k'], 5),
        b',n,nzh': ([b',', b'n', b',', b'n', b'z', b'h'], 2),
    }
    expected_pair2tokens = {
    (b'a', b'b'): {b'abc'},
    (b'b', b'c'): {b'abc', b'bcd'}, 
    (b'c', b'd'): {b'bcd'},
    (b's', b'a'): {b'sadebzkjhg'},
    (b'a', b'd'): {b'sadebzkjhg'},
    (b'd', b'e'): {b'sadebzkjhg'},
    (b'e', b'b'): {b'sadebzkjhg'},
    (b'b', b'z'): {b'sadebzkjhg'},
    (b'z', b'k'): {b'sadebzkjhg'},
    (b'k', b'j'): {b'sadebzkjhg'},
    (b'j', b'h'): {b'sadebzkjhg'},
    (b'h', b'g'): {b'sadebzkjhg'},
    (b',', b'z'): {b',zncmvk'},
    (b'z', b'n'): {b',zncmvk'},
    (b'n', b'c'): {b',zncmvk'},
    (b'c', b'm'): {b',zncmvk'},
    (b'm', b'v'): {b',zncmvk'},
    (b'v', b'k'): {b',zncmvk'},
    (b',', b'n'): {b',n,nzh'}, 
    (b'n', b','): {b',n,nzh'},
    (b'n', b'z'): {b',n,nzh'},
    (b'z', b'h'): {b',n,nzh'},
}
    
    token_counts_reformed, pair2tokens = BPETokenizer._reform_tokens_counts(token_counts)

    assert token_counts_reformed == expected_token_counts_reformed
    assert pair2tokens == expected_pair2tokens
    
def test_merge_and_side_effects():
    token1 = b'abcd'
    token2 = b'bcbc'
    token_counts = {
        token1: ([b'a', b'b', b'c', b'd'], 5),
        token2: ([b'b', b'c', b'b', b'c'], 3)
    }
    pair2tokens = defaultdict(set)
    pair2tokens.update(
        {
        (b'a', b'b'): {token1},
        (b'b', b'c'): {token1, token2}, 
        (b'c', b'd'): {token1},
        (b'c', b'b'): {token2},
        }
    )
    
    pair_to_merge = (b'b', b'c')
    
    pair_change_counter = BPETokenizer._merge_pair_token_counts(
        token_counts, pair2tokens, pair_to_merge
    )
    
    expected_counter = Counter({
        (b'a', b'b'): -5,
        (b'c', b'd'): -5,
        (b'a', b'bc'): +5,
        (b'bc', b'd'): +5,
        (b'c', b'b'): -3,
        (b'bc', b'bc'): 3,
    })
    assert pair_change_counter == expected_counter

    expected_token_counts = {
        token1: ([b'a', b'bc', b'd'], 5),
        token2: ([b'bc', b'bc'], 3)
    }
    assert token_counts == expected_token_counts

    expected_pair2tokens = {
        (b'a', b'b'): set(),
        (b'bc', b'b'): set(),
        (b'c', b'b'): set(),
        (b'c', b'd'): set(),
        (b'a', b'bc'): {token1}, 
        (b'bc', b'd'): {token1}, 
        (b'bc', b'bc'): {token2}, 
    }
    assert pair2tokens == expected_pair2tokens
