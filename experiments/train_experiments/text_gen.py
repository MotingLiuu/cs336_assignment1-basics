from cs336_basics import model as mt_model
from cs336_basics import BPETokenizer
import torch
from pathlib import Path
import time
import logging
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

current_dir = Path(os.getcwd())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tiny_vocab_path = str((current_dir / "../vocab_TinyStories.json").resolve())
tiny_merges_path = str((current_dir / "../merges_TinyStories.json").resolve())
tiny_tokenizer = BPETokenizer.from_pretrained(
    vocab_path=tiny_vocab_path,
    merges_path=tiny_merges_path,
    )
print(f"Tokenizer loaded from {tiny_vocab_path} and {tiny_merges_path}")
print(f"Tokenizer vocabs from 0 to 100: {[tiny_tokenizer.vocab[i] for i in range(100)]}")
print(f"Tokenizer <|endoftext|> id: {tiny_tokenizer.token2id[b"<|endoftext|>"]}")

model_config = {
    "d_model": 512,
    "num_heads": 16,
    "d_ff": 1344,
    "num_layers": 4,
    "vocab_size": 10000,
    "max_seq_len": 256,
    "theta": 10000.0,
}
tiny_model = mt_model.Transformer(
    model_config["d_model"],
    model_config["num_heads"],
    model_config["d_ff"],
    model_config["num_layers"],
    model_config["vocab_size"],
    model_config["max_seq_len"],
    model_config["theta"],
)
tiny_model.to(device)
mt_model.load_checkpoint(
    src="checkpoints/checkpoint_iter_5000.pth",
    model=tiny_model,
    optimizer=None,
)

prompt = "I hate Mondays."
prompt_ids = tiny_tokenizer.encode(prompt)
prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)
generated_ids = tiny_model.generate(
    input_ids=prompt_tensor,
    max_length=256,
    temperature=1.0,
    top_p=0.9,
)
print(f"Prompt: {prompt}")
print(f"Prompt IDs: {prompt_ids}")
print(f"Generated IDs: {generated_ids.tolist()}")
generated_text = tiny_tokenizer.decode(generated_ids[0].tolist())
print(f"Generated text: {generated_text}")