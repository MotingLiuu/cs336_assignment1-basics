from cs336_basics import BPETokenizer
from pathlib import Path
import time
import json

DATA_PATH = str((Path(__file__).parent / '../../data/TinyStoriesV2-GPT4-valid.txt').resolve())
bpe_TinyStories = BPETokenizer(20000, [r'<|endoftext|>'])
start = time.time()
bpe_TinyStories.train(DATA_PATH, parallel=False)
end = time.time()

vocab_to_save = {
    idx: token_bytes.decode('utf-8', errors='replace')
    for idx, token_bytes in bpe_TinyStories.vocab.items()
}

with open('vocab_TinyStories.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_to_save, f, ensure_ascii=False, indent=2)

longest_tokens = sorted(
    bpe_TinyStories.vocab.items(),
    key=lambda x: len(x[1]),
    reverse=True
)[:10]

print(f'train on TinyStories took {end - start}s\n')
print(f'longest_tokens: {longest_tokens}\n')
