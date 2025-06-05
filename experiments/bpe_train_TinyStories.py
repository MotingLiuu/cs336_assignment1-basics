from cs336_basics import BPETokenizer
from pathlib import Path
import time
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="train_TinyStories_valid.log",
    filemode="w",
)

DATA_PATH = str((Path(__file__).parent / '../../data/TinyStoriesV2-GPT4-valid.txt').resolve())
bpe_TinyStories = BPETokenizer(10000, [r'<|endoftext|>'])
start = time.time()
bpe_TinyStories.train(DATA_PATH, parallel=True)
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
