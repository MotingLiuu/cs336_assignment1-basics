import json
import time
import logging
from collections import defaultdict, Counter
import pytest
from cs336_basics import model, bpe_tokenizer
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def test_transformer():
    transformer_lm = model.Transformer(
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        vocab_size=10000,
        max_seq_len=512,
        theta=10000
    )
    for name, module in transformer_lm.named_modules():
        logging.info(f"Module: {name}, Type: {type(module).__name__}")