from collections import Counter
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
    # 从 'abc' (b'abc')
    (b'a', b'b'): {b'abc'},
    (b'b', b'c'): {b'abc', b'bcd'}, # b'abc' 和 b'bcd' 共享 (b'b', b'c')
    # 从 'bcd' (b'bcd')
    (b'c', b'd'): {b'bcd'},
    # 从 'sadebzkjhg' (b'sadebzkjhg')
    (b's', b'a'): {b'sadebzkjhg'},
    (b'a', b'd'): {b'sadebzkjhg'},
    (b'd', b'e'): {b'sadebzkjhg'},
    (b'e', b'b'): {b'sadebzkjhg'},
    (b'b', b'z'): {b'sadebzkjhg'},
    (b'z', b'k'): {b'sadebzkjhg'},
    (b'k', b'j'): {b'sadebzkjhg'},
    (b'j', b'h'): {b'sadebzkjhg'},
    (b'h', b'g'): {b'sadebzkjhg'},
    # 从 ',zncmvk' (b',zncmvk')
    (b',', b'z'): {b',zncmvk'},
    (b'z', b'n'): {b',zncmvk'},
    (b'n', b'c'): {b',zncmvk'},
    (b'c', b'm'): {b',zncmvk'},
    (b'm', b'v'): {b',zncmvk'},
    (b'v', b'k'): {b',zncmvk'},
    # 从 ',n,nzh' (b',n,nzh')
    (b',', b'n'): {b',n,nzh'}, # 注意: ',n,nzh' 中 (b',',b'n') 出现两次，但集合中只包含一次 token_bytes
    (b'n', b','): {b',n,nzh'},
    # (b',', b'n') 再次出现，不改变集合
    (b'n', b'z'): {b',n,nzh'},
    (b'z', b'h'): {b',n,nzh'},
}
    
    token_counts_reformed, pair2tokens = BPETokenizer._reform_tokens_counts(token_counts)

    assert token_counts_reformed == expected_token_counts_reformed
    assert pair2tokens == expected_pair2tokens