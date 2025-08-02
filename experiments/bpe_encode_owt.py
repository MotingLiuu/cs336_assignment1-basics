import mmap
import numpy as np
from cs336_basics import BPETokenizer
from pathlib import Path
import time
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

current_dir = Path(os.getcwd())

def tokenize_large_file(
    file_path: str,
    tokenizer: BPETokenizer,
):
    tokens = []
    with open(file_path, "r", encoding="utf-8") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            text = mm.read().decode('utf-8')
            tokens = tokenizer.encode(text)
    return np.array(tokens, dtype=np.uint16)

def save_list(file_path: str, data: list):
    np.save(file_path, data)
    logging.info(f"Saved tokenized data to {file_path}")
    
owt_valid_path = str((current_dir / "../../data/owt_valid.txt").resolve())
owt_train_path = str((current_dir / "../../data/owt_train.txt").resolve())
owt_vocab_path = str((current_dir / "vocab_owt.json").resolve())
owt_merges_path = str((current_dir / "merges_owt.json").resolve())

owt_tokenizer = BPETokenizer.from_pretrained(
    vocab_path=owt_vocab_path,
    merges_path=owt_merges_path,
    )

owt_valid_tokens = tokenize_large_file(
    file_path=owt_valid_path,
    tokenizer=owt_tokenizer,
)
save_list(
    file_path=str((current_dir / "owt_valid_tokens.npy").resolve()),
    data=owt_valid_tokens,
)
owt_train_tokens = tokenize_large_file(
    file_path=owt_train_path,
    tokenizer=owt_tokenizer,
)
save_list(
    file_path=str((current_dir / "owt_train_tokens.npy").resolve()),
    data=owt_train_tokens,
)