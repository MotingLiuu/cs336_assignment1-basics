import json
import time
import logging
from collections import defaultdict, Counter
import pytest
from cs336_basics import model, bpe_tokenizer
from pathlib import Path
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def test_sgd():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = model.SGD([weights], lr=0.1)
    
    for t in range(100):
        opt.zero_grad()
        loss = (weights ** 2).sum()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
        logger.info(f"Step {t}, Loss: {loss.item()}")