from cs336_basics import Word
from collections import Counter

def test_init_bytes_list():
    word = Word(b"abracadabra")
    excepted_bytes_list = [b'a', b'b', b'r', b'a', b'c', b'a', b'd', b'a', b'b', b'r', b'a']
    assert word.bytes_list == excepted_bytes_list, "Fails on initialing bytes_list"

def test_simple_merge():
    word = Word(b"abracadabra")
    pair_to_merge = (b'a', b'b')
    
    pair_changes = word.merge(pair_to_merge)
    
    expected_bytes_list = [b'ab', b'r', b'a', b'c', b'a', b'd', b'ab', b'r', b'a']
    assert word.bytes_list == expected_bytes_list, "Fails on byte_list after merging\n"
    
    expected_changes = Counter({
        (b'ab', b'r'): 2,  
        (b'b', b'r'): -2, 
        (b'd', b'ab'): 1,  
        (b'd', b'a'): -1   
    })
    assert pair_changes == expected_changes, "Fails on expeted_changes after merging\n"

def test_merge_no_occurrence():
    original_bytes = b"hello"
    word = Word(original_bytes)
    pair_to_merge = (b'x', b'y')
    
    pair_changes = word.merge(pair_to_merge)
    
    assert word.bytes_list ==  [b'h', b'e', b'l', b'l', b'o'], "byte_list after merging is incorrect\n"
    assert pair_changes == Counter(), "pair_changes is incorrect\n"

def test_merge_at_ends():
    word = Word(b"xyx")
    # bytes_list: [b'x', b'y', b'x']
    pair_to_merge = (b'x', b'y')
    pair_changes = word.merge(pair_to_merge)
    
    expected_bytes_list = [b'xy', b'x']
    assert word.bytes_list == expected_bytes_list

    expected_changes = Counter({
        (b'xy', b'x'): 1,
        (b'y', b'x'): -1
    })
    assert pair_changes == expected_changes

    word2 = Word(b"zxy")

    pair_to_merge2 = (b'x',b'y')
    pair_changes2 = word2.merge(pair_to_merge2)
    expected_bytes_list2 = [b'z', b'xy']
    assert word2.bytes_list == expected_bytes_list2

    expected_changes2 = Counter({
        (b'z', b'xy'):1,
        (b'z', b'x'): -1
    })
    assert pair_changes2 == expected_changes2


def test_merge_all():
    word = Word(b"abab")
    pair_to_merge = (b'a', b'b')
    
    pair_changes = word.merge(pair_to_merge)
    
    expected_bytes_list = [b'ab', b'ab']
    assert word.bytes_list == expected_bytes_list
    
    final_expected_changes = Counter({
        (b'b', b'a'): -1,
        (b'ab', b'ab'): 1
    })
    assert pair_changes == final_expected_changes


def test_empty_word():
    word = Word(b"")
    pair_to_merge = (b'a', b'b')
    pair_changes = word.merge(pair_to_merge)
    assert word.bytes_list == []
    assert pair_changes == Counter()

def test_single_byte_word():
    word = Word(b"a")
    pair_to_merge = (b'a', b'b')
    pair_changes = word.merge(pair_to_merge)
    assert word.bytes_list == [b'a']
    assert pair_changes == Counter()

def test_banana_merge_an():
    word = Word(b"banana")
    pair_to_merge = (b'a', b'n')
    
    pair_changes = word.merge(pair_to_merge)
    
    expected_bytes_list = [b'b', b'an', b'an', b'a']
    assert word.bytes_list == expected_bytes_list
    
    expected_changes = Counter({
        (b'b', b'an'): 1,
        (b'b', b'a'): -1,
        (b'an', b'a'): 1, 
        (b'n', b'a'): -2,
        (b'an', b'an'): 1
    })
    assert pair_changes == expected_changes

def test_pair_count():
    word = Word(b"abrad")
    
    expected_counter = Counter({
        (b'a', b'b'): 1,  
        (b'b', b'r'): 1, 
        (b'r', b'a'): 1,  
        (b'a', b'd'): 1,   
    })
    
    assert Word.count_pair(word.bytes_repr) == expected_counter
