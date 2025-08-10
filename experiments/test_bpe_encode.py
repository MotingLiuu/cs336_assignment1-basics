from cs336_basics import BPETokenizer
from pathlib import Path
import time
import logging
import os
import random
from typing import BinaryIO

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

current_dir = Path(__file__).parent.resolve()

tiny_vocab_path = str((current_dir / "vocab_TinyStories.json").resolve())
tiny_merges_path = str((current_dir / "merges_TinyStories.json").resolve())

tiny_tokenizer = BPETokenizer.from_pretrained(
    vocab_path=tiny_vocab_path,
    merges_path=tiny_merges_path, 
    )

example_text = "<|endoftext|>"
example_encoded = tiny_tokenizer.encode(example_text)
endoftext_id = tiny_tokenizer.token2id[example_text.encode("utf-8")]

logger.debug(f"Example text is {example_text}")
logger.debug(f"Example text encoded is {example_encoded}")
logger.debug(f"End of text token ID is {endoftext_id}")
